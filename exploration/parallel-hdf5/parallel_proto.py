"""Prototype: split pynbody's HDF5 array loading into (a) plan generation and
(b) parallel plan execution.

(a) is a pure refactor of HDFArrayLoader.load_arrays: the existing sequential
    loop is a state machine over precomputed chunk maps, so the whole sequence
    of (file, source selection, disk offset, memory destination) tuples can be
    materialised before any I/O happens.

(b) executes the plan with a thread pool.  Because h5py serialises every call
    into libhdf5 behind its global `phil` lock, the workers do NOT use h5py for
    the bulk transfer; for datasets that are contiguous and unfiltered they
    pread the bytes straight out of the file at the dataset's own byte offset.
    Anything else falls back to the h5py path (correct, but serialised).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

from pynbody.snapshot.gadgethdf import _DummyHDFData, _HDFArrayFiller, HDFArrayLoader


# ---------------------------------------------------------------- plan generation

def plan_for_group(loader: HDFArrayLoader, hdf_group_name: str, i0_start: int,
                   num_files: int):
    """Materialise the read plan for one HDF particle group.

    Returns a list of ``(file_index, source_sel, disk_offset, mem_slice)`` and the
    memory write position reached at the end.  Touches no HDF5 file.
    """
    interrupts = loader._file_interrupt_points[hdf_group_name]
    plan = []
    state = {"file_index": 0, "offset": 0, "exhausted": num_files == 0}

    def select_file(_):
        if state["file_index"] + 1 >= num_files:
            state["exhausted"] = True
        else:
            state["file_index"] += 1
        state["offset"] = 0

    i0 = i0_start
    for readlen, buf_index, mem_index in loader._load_control.iterate_with_interrupts(
            hdf_group_name, hdf_group_name, interrupts, select_file):
        if mem_index is None or state["exhausted"]:
            state["offset"] += readlen
            continue
        i1 = i0 + mem_index.stop - mem_index.start
        plan.append((state["file_index"], buf_index, state["offset"], slice(i0, i1)))
        state["offset"] += readlen
        i0 = i1
    return plan, i0


# ------------------------------------------------------- raw-pread fast path

class _RawDatasetReader:
    """Reads rows of a contiguous, unfiltered HDF5 dataset with plain pread(2).

    This exists purely to escape h5py's global lock: pread releases the GIL, takes
    no h5py lock, and so several of these can be in flight at once.
    """

    def __init__(self, dataset: h5py.Dataset):
        self.dataset = dataset
        self.offset = dataset.id.get_offset()
        self.dtype = dataset.dtype
        self.row_items = int(np.prod(dataset.shape[1:])) if dataset.ndim > 1 else 1
        self.row_bytes = self.dtype.itemsize * self.row_items
        self.nrows = dataset.shape[0]
        self.filename = dataset.file.filename
        self._fd = None

    @classmethod
    def usable_for(cls, dataset) -> bool:
        if isinstance(dataset, _DummyHDFData) or dataset is None:
            return False
        dcpl = dataset.id.get_create_plist()
        if dcpl.get_layout() != h5py.h5d.CONTIGUOUS or dcpl.get_nfilters() != 0:
            return False
        if dataset.id.get_offset() is None:          # not yet allocated
            return False
        dt = dataset.dtype
        if dt.kind not in "fiub" or dt.names is not None:
            return False
        return dataset.id.get_space().get_simple_extent_ndims() <= 2

    def open(self):
        self._fd = os.open(self.filename, os.O_RDONLY)

    def close(self):
        if self._fd is not None:
            os.close(self._fd); self._fd = None

    def read_rows(self, row_start: int, nrows: int, out: np.ndarray):
        """Read ``nrows`` rows starting at ``row_start`` into ``out`` (file dtype)."""
        mv = memoryview(out).cast("B")
        want = nrows * self.row_bytes
        done = 0
        base = self.offset + row_start * self.row_bytes
        while done < want:
            n = os.preadv(self._fd, [mv[done:want]], base + done)
            if n == 0:
                raise OSError(f"short read from {self.filename}")
            done += n


def _fill_raw(reader: _RawDatasetReader, target: np.ndarray, source_sel, offset: int):
    """Same contract as _HDFArrayFiller.fill_array_from_hdf_dataset, but via pread.

    Mirrors _HDFArrayFiller, including the case where the file folds an n-vector
    into a flat 1-D dataset so that one row in memory spans several file rows.
    """
    sim_items = int(np.prod(target.shape[1:])) if target.ndim > 1 else 1
    scaling = sim_items / reader.row_items          # file rows per memory row

    if isinstance(source_sel, slice):
        sel = slice(source_sel.start + offset, source_sel.stop + offset)
    elif isinstance(source_sel, np.ndarray):
        sel = source_sel + offset
        if len(sel) > 1 and sel[-1] - sel[0] == len(sel) - 1:
            sel = slice(int(sel[0]), int(sel[-1]) + 1)
    else:
        sel = None

    if sel is None:
        row0, nrows, fancy = 0, reader.nrows, None
    elif isinstance(sel, slice):
        row0 = int(sel.start * scaling)
        nrows = int(sel.stop * scaling) - row0
        fancy = None
    else:
        id_min, id_max = int(sel[0]), int(sel[-1])
        row0 = int(id_min * scaling)
        nrows = int((id_max + 1) * scaling) - row0
        fancy = sel - id_min

    read_shape = (nrows, reader.row_items) if reader.row_items > 1 else (nrows,)

    if fancy is None and target.dtype == reader.dtype and target.size == nrows * reader.row_items:
        reader.read_rows(row0, nrows, target.reshape(read_shape))
        return

    tmp = np.empty(read_shape, reader.dtype)
    reader.read_rows(row0, nrows, tmp)
    if fancy is None:
        target.reshape(read_shape)[:] = tmp
        return
    # regroup the contiguous span into memory-shaped rows, then pick the wanted ones
    tmp = tmp.reshape((-1,) + target.shape[1:])
    data = tmp[fancy]
    target.reshape(data.shape)[:] = data


# ---------------------------------------------------------------- execution

def load_arrays_parallel(loader: HDFArrayLoader, all_fams_to_load, sim, array_name,
                         translated_names, nthreads=4):
    """Drop-in replacement for HDFArrayLoader.load_arrays that reads files in parallel.

    One task per file, so each file is read by a single thread front-to-back: that
    keeps the per-file access pattern sequential (good for readahead) while putting
    several files in flight at once (good for a striped parallel filesystem).
    """
    for loading_fam in all_fams_to_load:
        sim_fam_array, array_filler = loader._get_array_filler(
            array_name, loading_fam, sim, translated_names)

        i0 = 0
        tasks = {}      # (group, file_index) -> (source, is_raw, [(source_sel, offset, mem_slice), ...])
        readers = []
        for hdf_group_name in loader._family_to_group_map[loading_fam]:
            sl = loader._file_ptype_slice[hdf_group_name]
            if sl.stop <= sl.start:
                continue

            groups = list(loader._hdf_files.iter_particle_groups_with_name(hdf_group_name))
            datasets = [loader._get_dataset_from_translated_names(sim, g, translated_names)
                        for g in groups]

            plan, i0 = plan_for_group(loader, hdf_group_name, i0, len(groups))

            for file_index, source_sel, offset, mem_slice in plan:
                dset = datasets[file_index]
                if dset is None:
                    continue
                key = (hdf_group_name, file_index)
                if key not in tasks:
                    if _RawDatasetReader.usable_for(dset):
                        reader = _RawDatasetReader(dset)
                        readers.append(reader)
                        tasks[key] = (reader, True, [])
                    else:
                        tasks[key] = (dset, False, [])
                tasks[key][2].append((source_sel, offset, mem_slice))

        for r in readers:
            r.open()
        try:
            def run(task):
                src, is_raw, chunks = task
                for source_sel, offset, mem_slice in chunks:
                    target = sim_fam_array[mem_slice]
                    if is_raw:
                        _fill_raw(src, target, source_sel, offset)
                    else:
                        array_filler.fill_array_from_hdf_dataset(
                            target, src, source_sel=source_sel, offset=offset)

            work = list(tasks.values())
            if nthreads > 1 and len(work) > 1:
                with ThreadPoolExecutor(min(nthreads, len(work))) as ex:
                    list(ex.map(run, work))
            else:
                for task in work:
                    run(task)
        finally:
            for r in readers:
                r.close()
