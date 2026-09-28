"""Direct reading of virtual datasets, by reading their source datasets."""

from __future__ import annotations

import os
import threading
import typing

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from .common import (
    _THROUGH_H5PY,
    ReadProperties,
    _CannotReadDirectly,
    _supported_drivers,
)
from .datasets import _check_fill, _DirectReader
from .files import _file_identity

if typing.TYPE_CHECKING:
    from .reader import BulkReader

class _VirtualSourceBlock:
    """One mapping of a virtual dataset: rows [start, stop) of the virtual dataset come from rows
    [source_start, source_start + stop - start) of the source dataset."""

    def __init__(self, start, stop, filename, dataset_name, source_start, whole_source):
        self.start = start
        self.stop = stop
        self.filename = filename  # None if the source file could not be found
        self.dataset_name = dataset_name
        self.source_start = source_start
        self.whole_source = whole_source  # True if the mapping takes the entire source dataset
        self.reader = None
        self.reader_resolved = False
        self.lock = threading.Lock()  # pieces of one virtual dataset may be read concurrently


class _VirtualReader(_DirectReader):
    """A virtual dataset, read by reading each of its source datasets directly.

    Sources are opened lazily, so that a partial load touching only a few rows opens only the source files it needs.
    A source that cannot be found or read directly is read through the h5py virtual dataset instead, restricted to
    the rows that source supplies, which gives exactly the result HDF5 would.
    """

    def __init__(self, dataset, h5file, virtual_filename, blocks: list[_VirtualSourceBlock], bulk_reader: BulkReader,
                 fillvalue):
        super().__init__(dataset, bulk_reader)
        self._file = h5file
        self._filename = virtual_filename
        self._fillvalue = fillvalue
        self._fill = _Fill(fillvalue)
        self._blocks = blocks
        self._block_starts = np.array([b.start for b in blocks], dtype=np.int64)


    @property
    def filename(self) -> str:
        """The file holding the virtual dataset (its data come from the source files; see divide)"""
        return self._filename

    @classmethod
    def plan(cls, dataset, h5file, bulk_reader: BulkReader) -> _VirtualReader:
        """Work out how to read an h5py virtual dataset from its sources, or raise _CannotReadDirectly if its
        layout is not one this class handles (see the module docstring)."""
        shape = tuple(dataset.shape)
        unsupported = "its virtual dataset layout is not one pynbody can decompose"
        if os.environ.get('HDF5_VDS_PREFIX') or dataset.id.get_access_plist().get_virtual_prefix():
            raise _CannotReadDirectly("a search path for the sources of virtual datasets has been configured")
        # Source files are found relative to the directory of the virtual dataset's own file, so that must be known.
        # (If it was opened by a relative path, and the current directory has since changed to one holding a link to
        # the same file, this check cannot tell; pynbody opens files by absolute path, which avoids the question.)
        if h5file.driver not in _supported_drivers:
            raise _CannotReadDirectly(f"its file is open through the HDF5 '{h5file.driver}' driver, so pynbody "
                                      f"cannot confirm where its source files are")
        virtual_filename, _ = _file_identity(h5file)
        fillvalue = _check_fill(dataset)
        try:
            blocks = cls._plan_blocks(dataset, shape, virtual_filename, unsupported)
        except (RuntimeError, ValueError) as e:
            # e.g. HDF5 cannot describe a selection of unlimited extent as a set of points
            raise _CannotReadDirectly(unsupported) from e
        return cls(dataset, h5file, virtual_filename, blocks, bulk_reader, fillvalue)

    @staticmethod
    def _plan_blocks(dataset, shape, virtual_filename, unsupported) -> list[_VirtualSourceBlock]:
        blocks = []
        for source in dataset.virtual_sources():
            if '%' in source.file_name or '%' in source.dset_name:
                # printf-style patterns, which HDF5 expands for mappings with unlimited extents
                raise _CannotReadDirectly(unsupported)
            if source.vspace.get_select_npoints() == 0:
                continue  # e.g. a file holding no particles of this type, which supplies nothing

            virtual_box = _selection_box(source.vspace, shape)
            if virtual_box is None:
                raise _CannotReadDirectly(unsupported)
            v_start, v_stop = virtual_box
            if any(v_start[1:]) or tuple(v_stop[1:]) != shape[1:]:
                raise _CannotReadDirectly(unsupported)  # does not map whole rows

            if source.src_space.get_select_type() == h5py.h5s.SEL_ALL:
                source_start = 0
                whole_source = True
            else:
                source_box = _selection_box(source.src_space)
                if source_box is None:
                    raise _CannotReadDirectly(unsupported)
                s_start, s_stop = source_box
                if [b - a for a, b in zip(s_start, s_stop)] != [b - a for a, b in zip(v_start, v_stop)]:
                    raise _CannotReadDirectly(unsupported)
                if any(s_start[1:]):
                    raise _CannotReadDirectly(unsupported)
                source_start = s_start[0]
                whole_source = False

            filename = _resolve_virtual_source_filename(virtual_filename, source.file_name)
            blocks.append(_VirtualSourceBlock(v_start[0], v_stop[0], filename, source.dset_name,
                                              source_start, whole_source))

        blocks.sort(key=lambda b: b.start)
        for previous, following in zip(blocks[:-1], blocks[1:]):
            if previous.stop > following.start:
                raise _CannotReadDirectly(unsupported)  # overlapping mappings
        return blocks

    def _blocks_overlapping(self, start, stop):
        """The blocks that can overlap rows [start, stop), found without walking the whole list"""
        first = max(int(np.searchsorted(self._block_starts, start, side='right')) - 1, 0)
        last = int(np.searchsorted(self._block_starts, stop, side='left'))
        return (self._blocks[i] for i in range(first, last))

    def divide(self, rows: slice | np.ndarray, offset: int, destination: np.ndarray) -> list[tuple]:
        """Divide reading *rows* (each plus *offset*) into *destination* into parts, each read from one place.

        Returns (source, rows, offset, part of *destination*, the file the data come from) for each part, in order.
        The source is a direct reader of a source dataset (with rows counted in that dataset); for rows no source
        supplies, a _Fill; and for a source that cannot be read directly, the h5py virtual dataset itself (with
        rows counted in it), so that HDF5 reads it (filling it with the fill value if it cannot be found). *rows*
        is a slice (with *offset* 0), or a sorted array of rows, whose parts are returned as views of it, with
        offsets adjusted to their sources."""
        if self._use_h5py:
            return [(self._dataset, rows, offset, destination, self._filename)]
        if isinstance(rows, slice):
            start, stop = rows.start + offset, rows.stop + offset
        else:
            start, stop = int(rows[0]) + offset, int(rows[-1]) + 1 + offset

        # the intervals [lo, hi) of rows, in order, each with its source and the offset of its rows in the source
        intervals = []
        position = start
        for block in self._blocks_overlapping(start, stop):
            lo, hi = max(start, block.start), min(stop, block.stop)
            if hi <= lo:
                continue
            if lo > position:
                intervals.append((position, lo, self._fill, 0, self._filename))
            reader = self._get_source_reader(block)
            if reader is not None:
                intervals.append((lo, hi, reader, block.source_start - block.start, block.filename))
            else:
                intervals.append((lo, hi, self._dataset, 0, self._filename))
            position = hi
        if position < stop:
            intervals.append((position, stop, self._fill, 0, self._filename))

        parts = []
        if isinstance(rows, slice):
            for lo, hi, source, shift, filename in intervals:
                parts.append((source, slice(lo + shift, hi + shift), 0, destination[lo - start:hi - start],
                              filename))
        else:
            cuts = np.searchsorted(rows, [hi - offset for _, hi, _, _, _ in intervals])
            first = 0
            for (lo, hi, source, shift, filename), last in zip(intervals, cuts):
                if last > first:
                    parts.append((source, rows[first:last], offset + shift, destination[first:last], filename))
                first = last
        return parts

    def properties(self, start, stop):
        if self._use_h5py:
            return _THROUGH_H5PY
        direct, compressed, chunk_nbytes = True, False, 0
        for block in self._blocks_overlapping(start, stop):
            if min(stop, block.stop) <= max(start, block.start):
                continue
            reader = self._get_source_reader(block)
            if reader is None:
                if block.filename is not None:
                    direct = False  # read through h5py (a block with no source file is just filled in)
                continue
            source_start = block.source_start + max(start, block.start) - block.start
            source = reader.properties(source_start, source_start + min(stop, block.stop) - max(start, block.start))
            direct = direct and source.direct
            compressed = compressed or source.compressed
            chunk_nbytes = max(chunk_nbytes, source.chunk_nbytes)
        return ReadProperties(direct=direct, compressed=compressed, chunk_nbytes=chunk_nbytes)

    def prepare(self, start, stop):
        if self._use_h5py:
            return
        for block in self._blocks_overlapping(start, stop):
            piece_start, piece_stop = max(start, block.start), min(stop, block.stop)
            if piece_stop > piece_start:
                reader = self._get_source_reader(block)
                if reader is not None:
                    source_start = block.source_start + piece_start - block.start
                    reader.prepare(source_start, source_start + piece_stop - piece_start)

    def _get_source_reader(self, block: _VirtualSourceBlock):
        """Return a direct reader for the block's source dataset, or None if it must be read through h5py"""
        with block.lock:
            if not block.reader_resolved:
                block.reader = self._open_source_reader(block)
                block.reader_resolved = True
            return block.reader

    def _open_source_reader(self, block: _VirtualSourceBlock):
        if block.filename is None:
            return None  # HDF5 will fill the block with the fill value

        try:
            source = self._bulk_reader._open_source(block.filename, block.dataset_name, self._file, self._filename)
        except (OSError, KeyError) as e:
            self._bulk_reader._report_fallback(self._dataset, e, reading=True)
            return None
        try:
            if not isinstance(source, h5py.Dataset):
                raise _CannotReadDirectly("a source of its virtual dataset is not a dataset")
            if source.dtype != self.dtype:
                raise _CannotReadDirectly("a source of its virtual dataset has a different datatype, so would need "
                                          "converting")
            if tuple(source.shape[1:]) != self.shape[1:] or \
                    (block.whole_source and source.shape[0] != block.stop - block.start) or \
                    block.source_start + block.stop - block.start > source.shape[0]:
                raise _CannotReadDirectly("a source of its virtual dataset does not have the expected shape")
            reader = self._bulk_reader._plan(source)
            if isinstance(reader, _VirtualReader):
                raise _CannotReadDirectly("a source of its virtual dataset is itself virtual")
        except _CannotReadDirectly as e:
            self._bulk_reader._report_fallback(source if isinstance(source, h5py.Dataset) else self._dataset, e)
            return None
        return reader

    def _read_rows_into(self, out, start, stop):
        rows_covered = 0
        pieces = []
        for block in self._blocks_overlapping(start, stop):
            piece_start, piece_stop = max(start, block.start), min(stop, block.stop)
            if piece_stop > piece_start:
                pieces.append((block, piece_start, piece_stop))
                rows_covered += piece_stop - piece_start

        if rows_covered < stop - start:
            out[...] = self._fillvalue  # rows no source supplies

        for block, piece_start, piece_stop in pieces:
            dest = out[piece_start - start:piece_stop - start]
            reader = self._get_source_reader(block)
            if reader is None:
                dest[...] = self._dataset[piece_start:piece_stop]
            else:
                source_start = block.source_start + piece_start - block.start
                reader.read_direct(dest, source_sel=np.s_[source_start:source_start + piece_stop - piece_start])


def _selection_box(space, extent=None) -> tuple[tuple, tuple] | None:
    """Return (start, stop) of a dataspace selection that is a single box, or None if it is anything else."""
    select_type = space.get_select_type()
    if select_type == h5py.h5s.SEL_ALL:
        if extent is None:
            return None
        return (0,) * len(extent), tuple(extent)
    if select_type != h5py.h5s.SEL_HYPERSLABS:
        return None
    try:
        first, last = space.get_select_bounds()
        num_points = space.get_select_npoints()
    except Exception:
        return None  # e.g. an unlimited selection
    start = tuple(int(x) for x in first)
    stop = tuple(int(x) + 1 for x in last)
    if extent is not None and any(b > e for b, e in zip(stop, extent)):
        return None
    if num_points != np.prod([b - a for a, b in zip(start, stop)], dtype=np.int64):
        return None  # the bounding box contains points that are not selected
    return start, stop


def _hdf5_considers_absolute(name: str) -> bool:
    """Whether HDF5 treats a virtual dataset's source file name as absolute (H5_CHECK_ABSOLUTE).

    This is HDF5's rule, not Python's: on Windows only a name starting with a drive letter, a colon and a separator
    counts, whereas os.path.isabs has changed its answer for names like '/data/x.h5' between Python versions."""
    if os.name == 'nt':
        return len(name) >= 3 and name[0].isalpha() and name[1] == ':' and name[2] in '/\\'
    return name.startswith('/')


def _resolve_virtual_source_filename(virtual_filename: str, source_filename: str) -> str | None:
    """Find a virtual dataset's source file, as HDF5 would, or return None if it does not exist.

    With no search path configured (see :meth:`_VirtualReader.plan`), HDF5 tries, in order: an absolute name as
    given; the name relative to the directory of the virtual dataset's own file; and finally the name relative to the
    current directory. An absolute name that does not exist is reduced to its final component and searched for in
    the same way. ``.`` means the virtual dataset's own file. *virtual_filename* must be absolute, and so is the
    result.

    Raises _CannotReadDirectly for names whose treatment is not clear-cut: on Windows, names that start with a
    separator but no drive letter (including UNC paths), which HDF5 glues onto a directory in a way Windows then
    reinterprets.
    """
    if source_filename == '.':
        return virtual_filename

    absolute = _hdf5_considers_absolute(source_filename)
    if os.name == 'nt' and not absolute and (source_filename[:1] in ('/', '\\') or
                                             (len(source_filename) >= 2 and source_filename[1] == ':')):
        raise _CannotReadDirectly("a source file of its virtual dataset is named in a way HDF5 and Windows may "
                                  "interpret differently")

    origin = os.path.dirname(virtual_filename)
    candidates = []
    if absolute:
        candidates.append(source_filename)
        source_filename = os.path.basename(source_filename)
    candidates.append(os.path.join(origin, source_filename))
    candidates.append(source_filename)

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


class _Fill:
    """A source for rows of a virtual dataset that no source dataset supplies, which read as the fill value"""

    def __init__(self, value):
        self.value = value

    def read_direct(self, dest: np.ndarray, source_sel=None):
        dest[...] = self.value

    def properties(self, start, stop) -> ReadProperties:
        return ReadProperties(direct=True)
