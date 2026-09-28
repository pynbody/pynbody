"""Direct readers of contiguous and chunked datasets."""

from __future__ import annotations

import functools
import itertools
import threading
import typing
import zlib

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from . import decode, files
from .common import (
    _DEFLATE_FILTER,
    _FLETCHER32_FILTER,
    _SHUFFLE_FILTER,
    _THROUGH_H5PY,
    Job,
    ReadProperties,
    _CannotReadDirectly,
    _supported_filters,
    _UnexpectedData,
)
from .decode import _DecodedChunk, _shuffled_planes, _unshuffle_planes_into

if typing.TYPE_CHECKING:
    from .reader import BulkReader

# How many chunks of a dataset are looked up one by one before an index of all its chunks is made instead
_chunk_lookups_before_indexing = 8

# Data that must be converted on their way into the destination are read this many bytes at a time
_conversion_block_nbytes = 16 * 1024 * 1024

# Chunks smaller than this are fetched and decoded several to a job, up to about this many bytes decoded
_min_job_nbytes = 1024 * 1024


def _check_datatype(dataset):
    """Raise _CannotReadDirectly unless the dataset's stored bytes are exactly those of its numpy dtype"""
    dtype = dataset.dtype
    if dtype.kind not in 'iuf' or dtype.fields is not None or dtype.subdtype is not None:
        raise _CannotReadDirectly(f"its datatype ({dtype}) is not a plain integer or floating-point type",
                                  warn=False)
    if dtype.itemsize not in (1, 2, 4, 8):
        raise _CannotReadDirectly(f"its datatype ({dtype}) is an extended-precision type, whose layout varies")
    stored_type = dataset.id.get_type()
    try:
        standard_type = h5py.h5t.py_create(dtype)
    except TypeError:
        raise _CannotReadDirectly(f"its datatype ({dtype}) has no standard HDF5 equivalent")
    # H5Tequal: same class, size, byte order, precision, offset, padding and, for floats, bit fields and bias
    if not stored_type == standard_type:
        raise _CannotReadDirectly(f"its datatype is not stored in the standard way for {dtype}")


def _check_fill(dataset):
    """Raise _CannotReadDirectly unless space never written reads as a well-defined fill value; return it"""
    plist = dataset.id.get_create_plist()
    if plist.fill_value_defined() == h5py.h5d.FILL_VALUE_UNDEFINED or \
            plist.get_fill_time() == h5py.h5d.FILL_TIME_NEVER:
        raise _CannotReadDirectly("space in it that was never written has no defined value")
    try:
        return dataset.fillvalue
    except (RuntimeError, OSError, ValueError, TypeError):
        raise _CannotReadDirectly("its fill value could not be read")


def _row_range(source_sel, num_rows) -> tuple[int, int] | None:
    """Convert a selection into a (start, stop) range of rows, or return None if it is not a simple range."""
    if isinstance(source_sel, tuple) and len(source_sel) == 1:
        source_sel = source_sel[0]
    if source_sel is None or source_sel is Ellipsis or (isinstance(source_sel, tuple) and len(source_sel) == 0):
        return 0, num_rows
    if isinstance(source_sel, slice):
        start, stop, step = source_sel.indices(num_rows)
        if step == 1:
            return start, max(start, stop)
    return None


class _DirectReader:
    """Base class for direct readers of whole rows, presenting the subset of the h5py Dataset API pynbody uses.

    If a direct read goes wrong, the reader warns and passes that read, and all later ones, to h5py."""

    def __init__(self, dataset, bulk_reader: BulkReader):
        self._dataset = dataset
        self._bulk_reader = bulk_reader
        self.shape = tuple(dataset.shape)
        self.dtype = np.dtype(dataset.dtype)
        self._use_h5py = False

    @property
    def filename(self) -> str:
        """The file the data are read from"""
        return self._file.filename

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape, dtype=np.int64))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, sel):
        rows = _row_range(sel, self.shape[0])
        if rows is None or self._use_h5py:
            return self._dataset[sel]
        out = np.empty((rows[1] - rows[0],) + self.shape[1:], dtype=self.dtype)
        self._read(out, *rows)
        return out

    def read_direct(self, dest: np.ndarray, source_sel=None):
        """Read the selected rows into *dest*, which must have exactly the shape of the selection."""
        rows = _row_range(source_sel, self.shape[0])
        if rows is None or self._use_h5py or not _conversion_is_exact(self.dtype, dest.dtype):
            # a narrowing conversion might round or overflow differently in numpy and HDF5; leave it to HDF5
            self._dataset.read_direct(dest, source_sel=source_sel)
            return
        expected_shape = (rows[1] - rows[0],) + self.shape[1:]
        if dest.shape != expected_shape:
            raise ValueError(f"Destination has shape {dest.shape} but the selection has shape {expected_shape}")
        # Write through a plain ndarray view: slicing an ndarray subclass (such as pynbody's SimArray) runs its
        # __array_finalize__, which can be far more expensive than copying the data
        self._read(dest.view(np.ndarray), *rows)

    def _read(self, out, start, stop):
        if stop <= start:
            return
        try:
            self._read_rows_into(out, start, stop)
        except (_UnexpectedData, OSError, zlib.error) as e:
            self._bulk_reader._report_fallback(self._dataset, e, reading=True)
            self._use_h5py = True
            out[...] = self._dataset[start:stop]

    def _read_rows_into(self, out: np.ndarray, start: int, stop: int):
        """Read rows [start, stop) into *out*, here and now: for reads made one at a time. (BulkReader.read shares
        each chunk between all the requests that need it, and shares the work between threads.)"""
        target = out if out.flags.c_contiguous and out.flags.writeable else np.empty(out.shape, dtype=out.dtype)
        for job in self.jobs([(slice(start, stop), 0, target)]):
            job.run()
        if target is not out:
            out[...] = target

    def is_compressed(self) -> bool:
        """True if the data are compressed, so that decoding them takes appreciable CPU time"""
        return False

    def properties(self, start: int, stop: int, destination_dtype=None, gathered: bool = False) -> ReadProperties:
        """Describe reading rows [start, stop) (all of them, or if *gathered*, some of them) into a destination of
        the given dtype (if known), for deciding how to perform the reads (see ReadProperties)"""
        if self._use_h5py:
            return _THROUGH_H5PY
        return ReadProperties(direct=True, compressed=self.is_compressed())

    def prepare(self, start: int, stop: int):
        """Do, now, every HDF5 lookup that reading rows [start, stop) will need.

        Reads that are then made from several threads only move and decode data, rather than queueing for h5py's
        lock. Calling this is optional: anything not prepared is looked up when it is needed."""
        pass

    def jobs(self, targets: list[tuple]) -> typing.Iterator[Job]:
        """Yield the jobs (see common.Job) that read *targets*, in order.

        Each target is (rows, offset, destination): rows being a slice (with offset 0) or a sorted array of rows to
        which offset is added, and destination a plain C-contiguous array with room for them. Targets are best given
        in increasing order of rows, when consecutive targets needing the same chunk share one job. Anything that
        goes wrong in a job while reading directly is reported, and that job's rows (and everything later read from
        this dataset) are read through h5py instead."""
        raise NotImplementedError

    def _fall_back(self, error):
        """Report *error* (met while reading directly) and read through h5py from now on"""
        if not self._use_h5py:
            self._bulk_reader._report_fallback(self._dataset, error, reading=True)
            self._use_h5py = True

    def _read_through_h5py(self, rows, offset, destination, trailing=()):
        """Read rows (a slice, or a sorted array plus offset) of the dataset through h5py into *destination*,
        restricted along trailing axes to the slices *trailing*. HDF5 converts them to the destination's datatype,
        as it would reading them itself."""
        trailing = tuple(trailing)
        if isinstance(rows, slice):
            self._h5py_read_into((slice(rows.start + offset, rows.stop + offset),) + trailing, destination)
            return
        # h5py reads an index selection slowly, and needs it copied anyway: read the span a block at a time instead
        row_nbytes = max(destination.dtype.itemsize * int(np.prod(destination.shape[1:], dtype=np.int64)), 1)
        block_rows = max(_conversion_block_nbytes // row_nbytes, 1)
        i = 0
        while i < len(rows):
            first = int(rows[i])
            j = int(np.searchsorted(rows, first + block_rows, side='left'))
            last = int(rows[j - 1])
            block = np.empty((last + 1 - first,) + destination.shape[1:], dtype=destination.dtype)
            self._h5py_read_into((slice(first + offset, last + 1 + offset),) + trailing, block)
            destination[i:j] = block[rows[i:j] - first]
            del block
            i = j

    def _h5py_read_into(self, selection: tuple, destination):
        """Read a selection of the dataset through h5py into *destination* (in its datatype)"""
        if destination.flags.c_contiguous:
            self._dataset.read_direct(destination, source_sel=selection)
        else:
            buffer = np.empty(destination.shape, dtype=destination.dtype)
            self._dataset.read_direct(buffer, source_sel=selection)
            destination[...] = buffer

    def _h5py_job(self, rows, offset, destination) -> Job:
        def fetch():
            self._read_through_h5py(rows, offset, destination)
        return Job(fetch)


def _conversion_is_exact(from_dtype, to_dtype) -> bool:
    """True if numpy and HDF5 are known to agree bit for bit on converting from_dtype to to_dtype.

    That requires the conversion to be exact for every value. Floating-point conversions must also be between
    native-order types of at least 4 bytes: otherwise HDF5 uses its own conversion routines, which replace any NaN
    with a canonical one where numpy preserves its payload."""
    from_dtype, to_dtype = np.dtype(from_dtype), np.dtype(to_dtype)
    if from_dtype == to_dtype:
        return True
    if not np.can_cast(from_dtype, to_dtype, casting='safe'):
        return False
    if from_dtype.kind in 'iu' and to_dtype.kind == 'f':
        # numpy counts e.g. int64 to float64 as safe, though it rounds integers beyond 2**53
        value_bits = from_dtype.itemsize * 8 - (from_dtype.kind == 'i')
        return value_bits <= np.finfo(to_dtype).nmant + 1
    if from_dtype.kind == 'f':
        return from_dtype.itemsize >= 4 and from_dtype.isnative and to_dtype.isnative
    return True


class _ContiguousReader(_DirectReader):
    """A contiguous dataset, read directly from the file.

    Its jobs (see jobs) read rows of the right datatype straight into their destination; rows needing conversion,
    or selected by index, a block at a time into a buffer (the fetch), from which they are then put in place (the
    process)."""

    def __init__(self, dataset, file, bulk_reader):
        super().__init__(dataset, bulk_reader)
        plist = dataset.id.get_create_plist()
        if plist.get_external_count() > 0:
            raise _CannotReadDirectly("it is stored in external files")
        self._offset = dataset.id.get_offset()
        if self._offset is None:
            raise _CannotReadDirectly("it has never been written", warn=False)
        self._row_nbytes = int(np.prod(self.shape[1:], dtype=np.int64)) * self.dtype.itemsize
        nbytes = self.shape[0] * self._row_nbytes
        if dataset.id.get_storage_size() != nbytes:
            raise _CannotReadDirectly("its storage size does not match its shape and datatype")
        if self._offset + nbytes > file.size:
            raise _CannotReadDirectly("its data extend beyond the end of the file")
        self._file = file

    def properties(self, start, stop, destination_dtype=None, gathered=False):
        if self._use_h5py:
            return _THROUGH_H5PY
        if destination_dtype is not None and np.dtype(destination_dtype) == self.dtype and not gathered:
            return ReadProperties(direct=True)  # (read straight into place, holding nothing)
        # (a read needing conversion or gathering is made a block at a time, each block held until put in place)
        return ReadProperties(direct=True,
                              job_nbytes=min(_conversion_block_nbytes, max(stop - start, 0) * self._row_nbytes))

    def jobs(self, targets):
        for rows, offset, destination in targets:
            if self._use_h5py or not _conversion_is_exact(self.dtype, destination.dtype):
                yield self._h5py_job(rows, offset, destination)
            elif isinstance(rows, slice) and destination.dtype == self.dtype:
                yield Job(functools.partial(self._fetch_into, rows, destination))
            elif isinstance(rows, slice):
                rows_per_block = max(1, _conversion_block_nbytes // max(self._row_nbytes, 1))
                for block_start in range(rows.start, rows.stop, rows_per_block):
                    block = slice(block_start, min(rows.stop, block_start + rows_per_block))
                    part = destination[block.start - rows.start:block.stop - rows.start]
                    yield Job(functools.partial(self._fetch_block, block, 0, part),
                              functools.partial(self._convert_block, part), nbytes=max(part.nbytes, 1))
            else:
                rows_per_block = max(1, _conversion_block_nbytes // max(self._row_nbytes, 1))
                i = 0
                while i < len(rows):
                    first = int(rows[i])
                    j = int(np.searchsorted(rows, first + rows_per_block, side='left'))
                    span = slice(first + offset, int(rows[j - 1]) + 1 + offset)
                    yield Job(functools.partial(self._fetch_block, span, rows[i:j], destination[i:j]),
                              functools.partial(self._gather_block, rows[i:j], first, destination[i:j]),
                              nbytes=max((span.stop - span.start) * self._row_nbytes, 1))
                    i = j

    def _fetch_into(self, rows: slice, destination):
        """Read rows of the dataset's own datatype straight into *destination*"""
        if not self._use_h5py:
            try:
                files._read_bytes(self._file, self._offset + rows.start * self._row_nbytes,
                                  (rows.stop - rows.start) * self._row_nbytes,
                                  into=memoryview(destination.view(np.uint8)))
                return None
            except (_UnexpectedData, OSError) as e:
                self._fall_back(e)
        self._read_through_h5py(rows, 0, destination)

    def _fetch_block(self, span: slice, selected, destination):
        """Read the stored bytes of rows *span*, for _convert_block or _gather_block. If they cannot be read
        directly, read the rows wanted through h5py instead (all of *span*, or if *selected* is an array, those
        rows of it, counted so that its first is span.start), and return None."""
        if not self._use_h5py:
            try:
                return files._read_bytes(self._file, self._offset + span.start * self._row_nbytes,
                                         (span.stop - span.start) * self._row_nbytes)
            except (_UnexpectedData, OSError) as e:
                self._fall_back(e)
        if isinstance(selected, np.ndarray):
            self._read_through_h5py(selected, span.start - int(selected[0]), destination)
        else:
            self._read_through_h5py(span, 0, destination)
        return None

    def _convert_block(self, destination, data):
        destination[...] = np.frombuffer(data, dtype=self.dtype).reshape(destination.shape)

    def _gather_block(self, rows, first, destination, data):
        block = np.frombuffer(data, dtype=self.dtype).reshape((-1,) + self.shape[1:])
        destination[...] = block[rows - first]


class _ChunkedReader(_DirectReader):
    """A chunked dataset, whose chunks are read directly from the file and decoded here.

    Chunk positions are looked up through h5py (see prepare). HDF5's chunk cache is not involved, and there is no
    cache here either: snapshot writers routinely use chunks of many megabytes (sometimes one for a whole dataset),
    while reads come in pieces (pynbody's of at most ``_max_buf`` rows), so instead each chunk is read by one job,
    which copies out of it the rows every piece wants (see jobs). Pieces read one at a time, through read_direct,
    decode any chunk they share once for each piece.
    """

    def __init__(self, dataset, file, bulk_reader):
        super().__init__(dataset, bulk_reader)
        plist = dataset.id.get_create_plist()
        self._pipeline = []
        for i in range(plist.get_nfilters()):
            filter_id, _flags, client_data, name = plist.get_filter(i)
            if filter_id not in _supported_filters:
                name = name.decode(errors='replace') if isinstance(name, bytes) else name
                raise _CannotReadDirectly(f"it uses the HDF5 filter '{name}' (id {filter_id}), which pynbody "
                                          f"cannot decode itself")
            if filter_id == _SHUFFLE_FILTER and tuple(client_data)[:1] != (self.dtype.itemsize,):
                raise _CannotReadDirectly("its shuffle filter does not match its datatype")
            self._pipeline.append({'filter_id': filter_id, 'client_data': tuple(client_data)})

        self._chunk_shape = tuple(int(c) for c in dataset.chunks)
        if len(self._chunk_shape) != len(self.shape):
            raise _CannotReadDirectly("its chunks do not have the same number of dimensions as the dataset")
        self._chunk_nbytes = int(np.prod(self._chunk_shape, dtype=np.int64)) * self.dtype.itemsize
        self._fillvalue = _check_fill(dataset)
        self._file = file
        self._file_size = file.size

        # Where each chunk is stored, as (byte offset or None if never written, size, filter mask): looked up chunk by
        # chunk, until enough have been wanted that an index of all of them is worth making (see _chunk_location)
        self._chunk_info = {}  # chunk origin -> location, for chunks looked up one by one
        self._chunk_index = None  # an index of every chunk (see _build_chunk_index), or False if one cannot be made
        self._chunk_index_lock = threading.Lock()
        self._chunk_lookups = 0

    def is_compressed(self):
        return any(f['filter_id'] == _DEFLATE_FILTER for f in self._pipeline)

    def properties(self, start, stop, destination_dtype=None, gathered=False):
        if self._use_h5py:
            return _THROUGH_H5PY
        rows_per_chunk = self._chunk_shape[0]
        trailing_chunks = int(np.prod([-(-n // c) for n, c in zip(self.shape[1:], self._chunk_shape[1:])]))
        num_chunks = (-(-stop // rows_per_chunk) - start // rows_per_chunk) * trailing_chunks if stop > start else 0
        # (small chunks are fetched and decoded several to a job; see jobs)
        job_nbytes = -(-_min_job_nbytes // self._chunk_nbytes) * self._chunk_nbytes if self._chunk_nbytes else 0
        return ReadProperties(direct=True, compressed=self.is_compressed(), chunk_nbytes=self._chunk_nbytes,
                              job_nbytes=job_nbytes, num_chunks=num_chunks)

    def _chunk_origins(self, start, stop):
        """Origins of all chunks holding any of rows [start, stop), each with the part of those rows it holds"""
        rows_per_chunk = self._chunk_shape[0]
        # chunk origins along each trailing axis, all of which are needed since reads are of whole rows
        trailing_origins = [range(0, n, c) for n, c in zip(self.shape[1:], self._chunk_shape[1:])]
        for row_origin in range((start // rows_per_chunk) * rows_per_chunk, stop, rows_per_chunk):
            for trailing_origin in itertools.product(*trailing_origins):
                yield (row_origin,) + trailing_origin

    def prepare(self, start, stop):
        if self._use_h5py:
            return
        origins = [origin for origin in self._chunk_origins(start, stop) if origin not in self._chunk_info]
        try:
            if self._chunk_index is None and self._chunk_lookups + len(origins) > _chunk_lookups_before_indexing:
                self._build_chunk_index()
            if not self._chunk_index:
                for origin in origins:
                    self._chunk_location(origin)
        except _UnexpectedData as e:
            self._bulk_reader._report_fallback(self._dataset, e, reading=True)
            self._use_h5py = True

    def _chunk_location(self, origin) -> tuple[int | None, int, int]:
        """Return (byte offset, or None if never written; size; filter mask) for the chunk at *origin*.

        HDF5 finds a chunk by its coordinates by searching the dataset's chunk index, taking time proportional to the
        number of chunks, so reading a dataset of many chunks that way would take time proportional to the square of
        their number. The first few chunks wanted are looked up that way; after that, an index of all of them is
        made in one pass."""
        location = self._chunk_info.get(origin)
        if location is not None:
            return location
        if self._chunk_index is None and self._chunk_lookups >= _chunk_lookups_before_indexing:
            self._build_chunk_index()
        if self._chunk_index:
            positions, offsets, sizes, masks = self._chunk_index
            i = int(positions[tuple(o // c for o, c in zip(origin, self._chunk_shape))])
            return (None, 0, 0) if i < 0 else (int(offsets[i]), int(sizes[i]), int(masks[i]))
        self._chunk_lookups += 1
        info = self._dataset.id.get_chunk_info_by_coord(origin)
        if info.byte_offset is not None and tuple(info.chunk_offset) != origin:
            raise _UnexpectedData(f"HDF5 reported the chunk at {origin} as being at {tuple(info.chunk_offset)}")
        location = (info.byte_offset, info.size, info.filter_mask)
        self._chunk_info[origin] = location
        return location

    def _build_chunk_index(self):
        """Index every stored chunk, in one pass through HDF5's chunk index (where h5py and HDF5 support that).

        The index is an array over the grid of chunk positions, giving for each the position in arrays of byte
        offsets, sizes and filter masks of the stored chunks, or -1 for a chunk never written."""
        with self._chunk_index_lock:
            if self._chunk_index is not None:
                return
            grid = tuple(-(-n // c) for n, c in zip(self.shape, self._chunk_shape))
            positions = np.full(grid, -1, dtype=np.int64 if np.prod(grid, dtype=np.int64) >= 2 ** 31 else np.int32)
            offsets, sizes, masks = [], [], []

            def visit(info):
                origin = tuple(info.chunk_offset)
                if any(o % c for o, c in zip(origin, self._chunk_shape)):
                    raise _UnexpectedData(f"HDF5 reported a chunk at {origin}, which is not on the grid of chunks")
                position = tuple(o // c for o, c in zip(origin, self._chunk_shape))
                if all(p < g for p, g in zip(position, grid)):  # (chunks beyond the current extent are never read)
                    if positions[position] >= 0:
                        raise _UnexpectedData(f"HDF5 reported two chunks at {origin}")
                    positions[position] = len(offsets)
                    offsets.append(info.byte_offset)
                    sizes.append(info.size)
                    masks.append(info.filter_mask)

            try:
                self._dataset.id.chunk_iter(visit)
            except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
                # (chunk_iter needs h5py 3.8 and HDF5 1.14, or 1.12.3 and later; without it, chunks are looked up
                # one by one)
                self._chunk_index = False
                return
            self._chunk_index = (positions, np.array(offsets, dtype=np.int64), np.array(sizes, dtype=np.int64),
                                 np.array(masks, dtype=np.int64))

    def jobs(self, targets):
        """Yield jobs reading the chunks needed, in order, each job fetching one or more chunks and then decoding each
        into every target that wants rows of it (see _DirectReader.jobs). Small chunks are fetched and decoded
        several to a job, so that handing jobs between threads costs little beside the work they do."""
        rows_per_chunk = self._chunk_shape[0]
        batch = []
        pending_origin, pending = None, []  # a row of chunks, and the pieces of targets it holds
        for rows, offset, destination in targets:
            if self._use_h5py or not _conversion_is_exact(self.dtype, destination.dtype):
                if pending:
                    yield from self._add_to_batch(batch, pending_origin, pending)
                    pending_origin, pending = None, []
                if batch:
                    yield self._batch_job(batch)
                    batch = []
                yield self._h5py_job(rows, offset, destination)
                continue
            for row_origin, piece in self._pieces(rows, offset, destination, rows_per_chunk):
                if row_origin != pending_origin and pending:
                    yield from self._add_to_batch(batch, pending_origin, pending)
                    pending = []
                pending_origin = row_origin
                pending.append(piece)
        if pending:
            yield from self._add_to_batch(batch, pending_origin, pending)
        if batch:
            yield self._batch_job(batch)

    def _add_to_batch(self, batch, row_origin, pieces):
        """Add the chunks of a row of chunks (one along each trailing axis), holding the given pieces of targets, to
        *batch*, a list of (origin, trailing slices, pieces); yield a job of the batch, and empty it, whenever it
        holds enough"""
        trailing_origins = [range(0, n, c) for n, c in zip(self.shape[1:], self._chunk_shape[1:])]
        for trailing_origin in itertools.product(*trailing_origins):
            origin = (row_origin,) + trailing_origin
            trailing = tuple(slice(o, min(o + c, n)) for o, c, n in
                             zip(trailing_origin, self._chunk_shape[1:], self.shape[1:]))
            batch.append((origin, trailing, pieces))
            if len(batch) * self._chunk_nbytes >= _min_job_nbytes:
                yield self._batch_job(batch[:])
                batch.clear()

    def _batch_job(self, batch) -> Job:
        return Job(functools.partial(self._fetch_batch, batch), functools.partial(self._decode_batch, batch),
                   nbytes=len(batch) * self._chunk_nbytes)

    def _fetch_batch(self, batch):
        """Fetch the chunks of a batch (see _fetch_for_pieces), returning a list of (index in the batch, location,
        stored bytes) for those needing decoding; or None if there are none"""
        fetched = []
        for i, (origin, trailing, pieces) in enumerate(batch):
            result = self._fetch_for_pieces(origin, trailing, pieces)
            if result is not None:
                fetched.append((i,) + result)
        return fetched or None

    def _decode_batch(self, batch, fetched):
        """Decode the chunks fetched by _fetch_batch into place, letting go of each chunk's bytes once decoded"""
        fetched.reverse()
        while fetched:
            # (the chunk popped is passed on without being kept here, so that it is freed as soon as it is decoded)
            self._decode_into_pieces(*batch[fetched[-1][0]], fetched.pop())

    @staticmethod
    def _pieces(rows, offset, destination, rows_per_chunk):
        """Yield (row origin of a chunk, piece) for each row of chunks a target needs, in order. A piece is (rows
        relative to the chunk's origin: a slice, or an array and the shift to add to it; the part of *destination*
        they go to)."""
        if isinstance(rows, slice):
            start, stop = rows.start, rows.stop
            for row_origin in range((start // rows_per_chunk) * rows_per_chunk, stop, rows_per_chunk):
                lo, hi = max(start, row_origin), min(stop, row_origin + rows_per_chunk)
                yield row_origin, (slice(lo - row_origin, hi - row_origin), 0, destination[lo - start:hi - start])
            return
        i = 0
        while i < len(rows):
            row_origin = ((int(rows[i]) + offset) // rows_per_chunk) * rows_per_chunk
            j = int(np.searchsorted(rows, row_origin + rows_per_chunk - offset, side='left'))
            yield row_origin, (rows[i:j], offset - row_origin, destination[i:j])
            i = j

    def _fetch_for_pieces(self, origin, trailing, pieces):
        """Fetch the chunk at *origin*, returning (its location, its stored bytes); or, if the pieces can be filled
        without decoding it (it was never written, or only HDF5 can read it, or reading it directly failed), fill
        them and return None"""
        if not self._use_h5py:
            try:
                location = self._locate_chunk(origin)
                if location is None:
                    for _, _, destination in pieces:
                        destination[(slice(None),) + trailing] = self._fillvalue
                    return None
                if location is not _READ_THROUGH_H5PY:
                    return location, self._fetch_chunk(location)
                self._pieces_through_h5py(origin, trailing, pieces)
                return None
            except (_UnexpectedData, OSError) as e:
                self._fall_back(e)
        self._pieces_through_h5py(origin, trailing, pieces)
        return None

    def _decode_into_pieces(self, origin, trailing, pieces, fetched):
        """Decode a chunk fetched by _fetch_batch, (index in the batch, location, stored bytes), and copy the rows each
        piece wants into place"""
        _, location, raw = fetched
        del fetched
        try:
            chunk = self._decode_chunk(origin, location, raw)
            del raw
            extent = tuple(t.stop - t.start for t in trailing)
            for chunk_rows, shift, destination in pieces:
                if not isinstance(chunk_rows, slice):
                    chunk_rows = chunk_rows + shift  # (the size of the piece)
                self._copy_rows(chunk, destination[(slice(None),) + trailing],
                                chunk_rows, extent)
        except (_UnexpectedData, OSError, zlib.error) as e:
            self._fall_back(e)
            self._pieces_through_h5py(origin, trailing, pieces)

    def _pieces_through_h5py(self, origin, trailing, pieces):
        for chunk_rows, shift, destination in pieces:
            part = destination[(slice(None),) + trailing]
            if isinstance(chunk_rows, slice):
                self._read_through_h5py(chunk_rows, origin[0], part, trailing)
            else:
                self._read_through_h5py(chunk_rows, shift + origin[0], part, trailing)

    def _copy_rows(self, chunk: _DecodedChunk, dest, rows, trailing_extent):
        """Copy rows of a decoded chunk (a slice, or an array, counted from its origin) into *dest*, which has room
        for those rows and, along each trailing axis, the first *trailing_extent* elements of the chunk."""
        trailing_sel = tuple(slice(0, n) for n in trailing_extent)
        if chunk.planes is None:
            selected = chunk.array.view(self.dtype).reshape(self._chunk_shape)[rows]
            dest[...] = selected[(slice(None),) + trailing_sel]
            return
        # The chunk is still shuffled, with byte b of element i at planes[b, i]. Unshuffle just the elements of the
        # rows needed, straight into *dest* if its memory is laid out exactly like theirs, and otherwise into a
        # buffer the size of those rows.
        elements_per_row = int(np.prod(self._chunk_shape[1:], dtype=np.int64))
        if isinstance(rows, slice):
            planes = chunk.planes[:, rows.start * elements_per_row:rows.stop * elements_per_row]
            num_rows = rows.stop - rows.start
        else:
            elements = rows if elements_per_row == 1 else \
                (rows[:, np.newaxis] * elements_per_row + np.arange(elements_per_row)).reshape(-1)
            planes = chunk.planes[:, elements]  # (the bytes of just the elements selected)
            num_rows = len(rows)
        whole_rows = trailing_extent == self._chunk_shape[1:]
        if whole_rows and dest.dtype == self.dtype and dest.flags.c_contiguous:
            _unshuffle_planes_into(planes, dest.reshape(-1).view(np.uint8))
        else:
            selected = np.empty((num_rows,) + self._chunk_shape[1:], dtype=self.dtype)
            _unshuffle_planes_into(planes, selected.reshape(-1).view(np.uint8))
            dest[...] = selected[(slice(None),) + trailing_sel]

    def _locate_chunk(self, origin):
        """Return (byte offset, size, filter mask) of the chunk at *origin*; None if it has never been written (so
        reads as the fill value); or _READ_THROUGH_H5PY if only HDF5 can be sure how to read it."""
        byte_offset, size, filter_mask = self._chunk_location(origin)
        if byte_offset is None:
            return None
        if byte_offset + size > self._file_size:
            raise _UnexpectedData(f"the chunk at {origin} extends beyond the end of the file")
        if size == self._chunk_nbytes and self._applies_filters(filter_mask) and \
                any(o + c > n for o, c, n in zip(origin, self._chunk_shape, self.shape)):
            # A partial chunk at the edge of the dataset, of exactly the size it would have unfiltered. HDF5 can be
            # told (H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) to store such chunks unfiltered, without recording that it
            # has, and h5py offers no way to find out whether it was; so only HDF5 can be sure how to read it.
            return _READ_THROUGH_H5PY
        return byte_offset, size, filter_mask

    def _fetch_chunk(self, location):
        """Read the stored bytes of a chunk located by _locate_chunk"""
        byte_offset, size, _ = location
        return files._read_bytes(self._file, byte_offset, size)

    def _decode_chunk(self, origin, location, raw) -> _DecodedChunk:
        """Undo the filters of a chunk's stored bytes *raw* (see _fetch_chunk)"""
        _, _, filter_mask = location
        deferred = self._deferred_filters(filter_mask)
        decoded = decode.decode_chunk(raw, filter_mask, self._pipeline, self.dtype.itemsize,
                                      nbytes=self._chunk_nbytes, first_filter=deferred)
        expected_nbytes = self._chunk_nbytes + (4 if deferred == 2 else 0)
        if len(decoded) != expected_nbytes:
            raise _UnexpectedData(f"the chunk at {origin} decoded to {len(decoded)} bytes, not {expected_nbytes}")
        if deferred:
            return _DecodedChunk(planes=_shuffled_planes(decoded, self.dtype.itemsize,
                                                         self._chunk_nbytes // self.dtype.itemsize,
                                                         checksummed=deferred == 2))
        return _DecodedChunk(array=decoded)

    def _deferred_filters(self, filter_mask) -> int:
        """How many of the first filters applied on writing to leave undone until rows are copied out of a chunk.

        If the first filter was a shuffle, undoing it as rows are copied out saves a pass over the chunk and a buffer
        the size of it. That is also possible if the first two filters were fletcher32 and then a shuffle (as SWIFT
        sometimes writes), since the checksum can be verified without unshuffling. Returns 0, 1 or 2."""
        if self.dtype.itemsize not in (2, 4, 8):
            return 0
        pipeline = [f['filter_id'] for f in self._pipeline]
        if pipeline[:1] == [_SHUFFLE_FILTER] and not filter_mask & 1:
            return 1
        if pipeline[:2] == [_FLETCHER32_FILTER, _SHUFFLE_FILTER] and not filter_mask & 3:
            return 2
        return 0

    def _applies_filters(self, filter_mask):
        """True if any filter of the pipeline applies to a chunk with the given filter mask"""
        return any(not filter_mask & (1 << i) for i in range(len(self._pipeline)))


_READ_THROUGH_H5PY = object()  # returned by _ChunkedReader._locate_chunk for chunks only HDF5 can be sure of


def is_direct_reader(reader) -> bool:
    """True if *reader* (as returned by BulkReader.open) reads directly, rather than being an h5py dataset"""
    return isinstance(reader, _DirectReader)
