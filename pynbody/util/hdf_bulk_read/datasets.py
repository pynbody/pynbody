"""Direct readers of contiguous and chunked datasets."""

from __future__ import annotations

import collections
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
    ReadProperties,
    _CannotReadDirectly,
    _default_cache_nbytes,
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
        raise NotImplementedError

    def is_compressed(self) -> bool:
        """True if the data are compressed, so that decoding them takes appreciable CPU time"""
        return False

    def properties(self, start: int, stop: int) -> ReadProperties:
        """Describe reading rows [start, stop), for deciding how to perform the reads (see ReadProperties)"""
        if self._use_h5py:
            return _THROUGH_H5PY
        return ReadProperties(direct=True, compressed=self.is_compressed())

    def chunk_containing(self, row: int):
        """Something identifying the chunk that holds *row*, equal for rows in the same chunk (so that reads sharing
        a chunk can be kept together); or None if the data are not chunked"""
        return None

    def prepare(self, start: int, stop: int):
        """Do, now, every HDF5 lookup that reading rows [start, stop) will need.

        Reads that are then made from several threads only move and decode data, rather than queueing for h5py's
        lock. Calling this is optional: anything not prepared is looked up when it is needed."""
        pass


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
    """A contiguous dataset, read directly from the file."""

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

    def _read_rows_into(self, out, start, stop):
        if out.dtype == self.dtype and out.flags.c_contiguous:
            offset = self._offset + start * self._row_nbytes
            files._read_bytes(self._file, offset, (stop - start) * self._row_nbytes,
                              into=memoryview(out.view(np.uint8)))
            return
        # The data must be converted (or scattered) on their way into *out*: read them a block of rows at a time, so
        # that the extra memory needed stays small however many rows are read
        rows_per_block = max(1, _conversion_block_nbytes // max(self._row_nbytes, 1))
        for block_start in range(start, stop, rows_per_block):
            block_stop = min(stop, block_start + rows_per_block)
            data = files._read_bytes(self._file, self._offset + block_start * self._row_nbytes,
                               (block_stop - block_start) * self._row_nbytes)
            out[block_start - start:block_stop - start] = \
                np.frombuffer(data, dtype=self.dtype).reshape((block_stop - block_start,) + self.shape[1:])


class _ChunkedReader(_DirectReader):
    """A chunked dataset, whose chunks are read directly from the file and decoded here.

    Chunk positions are looked up through h5py as they are needed. HDF5's chunk cache is not involved, so this class
    keeps its own: snapshot writers routinely use chunks of many megabytes (sometimes one for a whole dataset), while
    reads come in pieces (pynbody's of at most ``_max_buf`` rows), so without a cache a partial load would decompress
    the same chunk over and over. Chunks that extend beyond the end of a read are therefore kept, least recently used
    first out, up to a total of *cache_nbytes* (or one chunk, if a chunk is larger than that). Reads are expected to
    come in increasing order of rows (see ReadRequest), so a chunk is dropped again as soon as a read consumes the
    rest of it; reads in another order are still correct, but may decode chunks more than once.
    """

    def __init__(self, dataset, file, bulk_reader, cache_nbytes: int = _default_cache_nbytes):
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

        self._cache_nbytes = cache_nbytes
        self._cache = collections.OrderedDict()
        self._cache_lock = threading.Lock()
        # Where each chunk is stored, as (byte offset or None if never written, size, filter mask): looked up chunk by
        # chunk, until enough have been wanted that an index of all of them is worth making (see _chunk_location)
        self._chunk_info = {}  # chunk origin -> location, for chunks looked up one by one
        self._chunk_index = None  # an index of every chunk (see _build_chunk_index), or False if one cannot be made
        self._chunk_index_lock = threading.Lock()
        self._chunk_lookups = 0

    def is_compressed(self):
        return any(f['filter_id'] == _DEFLATE_FILTER for f in self._pipeline)

    def properties(self, start, stop):
        if self._use_h5py:
            return _THROUGH_H5PY
        return ReadProperties(direct=True, compressed=self.is_compressed(), chunk_nbytes=self._chunk_nbytes)

    def chunk_containing(self, row):
        return row // self._chunk_shape[0]

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

    def _read_rows_into(self, out, start, stop):
        rows_per_chunk = self._chunk_shape[0]
        # chunk origins along each trailing axis, all of which are needed since reads are of whole rows
        trailing_origins = [range(0, n, c) for n, c in zip(self.shape[1:], self._chunk_shape[1:])]

        for row_origin in range((start // rows_per_chunk) * rows_per_chunk, stop, rows_per_chunk):
            row_lo, row_hi = max(start, row_origin), min(stop, row_origin + rows_per_chunk)
            keep = row_origin + rows_per_chunk > stop and stop < self.shape[0]
            for trailing_origin in itertools.product(*trailing_origins):
                origin = (row_origin,) + trailing_origin
                chunk = self._get_chunk(origin, keep)
                trailing = tuple(slice(o, min(o + c, n)) for o, c, n in
                                 zip(trailing_origin, self._chunk_shape[1:], self.shape[1:]))
                dest = out[(slice(row_lo - start, row_hi - start),) + trailing]
                if chunk is None:
                    dest[...] = self._fillvalue
                elif chunk is _READ_THROUGH_H5PY:
                    dest[...] = self._dataset[(slice(row_lo, row_hi),) + trailing]
                else:
                    self._copy_rows(chunk, dest, row_lo - row_origin, row_hi - row_origin,
                                    tuple(s.stop - s.start for s in trailing))
                chunk = None  # so that this chunk (unless cached) is freed before the next one is read

    def _copy_rows(self, chunk: _DecodedChunk, dest, row_lo, row_hi, trailing_extent):
        """Copy rows [row_lo, row_hi) of a decoded chunk (counted from its origin) into *dest*, which has room for
        those rows and, along each trailing axis, the first *trailing_extent* elements of the chunk."""
        trailing_sel = tuple(slice(0, n) for n in trailing_extent)
        if chunk.planes is None:
            rows = chunk.array.view(self.dtype).reshape(self._chunk_shape)[row_lo:row_hi]
            dest[...] = rows[(slice(None),) + trailing_sel]
            return
        # The chunk is still shuffled, with byte b of element i at planes[b, i]. Unshuffle just the elements of the
        # rows needed, straight into *dest* if its memory is laid out exactly like theirs, and otherwise into a
        # buffer the size of those rows.
        elements_per_row = int(np.prod(self._chunk_shape[1:], dtype=np.int64))
        planes = chunk.planes[:, row_lo * elements_per_row:row_hi * elements_per_row]
        whole_rows = trailing_extent == self._chunk_shape[1:]
        if whole_rows and dest.dtype == self.dtype and dest.flags.c_contiguous:
            _unshuffle_planes_into(planes, dest.reshape(-1).view(np.uint8))
        else:
            rows = np.empty((row_hi - row_lo,) + self._chunk_shape[1:], dtype=self.dtype)
            _unshuffle_planes_into(planes, rows.reshape(-1).view(np.uint8))
            dest[...] = rows[(slice(None),) + trailing_sel]

    def _get_chunk(self, origin, keep):
        """Return the decoded chunk at *origin*; None if it has never been written; or _READ_THROUGH_H5PY.

        Getting a chunk has three steps, kept separate so that they could be done by different threads: locating it
        (HDF5 metadata), fetching its bytes (input only) and decoding them (computation only)."""
        chunk = self._take_from_cache(origin, keep)
        if chunk is not None:
            return chunk
        location = self._locate_chunk(origin)
        if location is None or location is _READ_THROUGH_H5PY:
            return location
        raw = self._fetch_chunk(location)
        chunk = self._decode_chunk(origin, location, raw)
        del raw
        if keep:
            self._keep_in_cache(origin, chunk)
        return chunk

    def _take_from_cache(self, origin, keep):
        with self._cache_lock:
            if keep:
                chunk = self._cache.get(origin)
                if chunk is not None:
                    self._cache.move_to_end(origin)
                return chunk
            # this read consumes the rest of the chunk, and reads come in increasing order, so no later read will
            # want it
            return self._cache.pop(origin, None)

    def _keep_in_cache(self, origin, chunk):
        # A chunk larger than the whole cache is still kept, alone, for the next read (which would otherwise decode it
        # all again): its memory is taken already, and it is dropped once a read consumes the rest of it
        with self._cache_lock:
            while self._cache and (len(self._cache) + 1) * self._chunk_nbytes > self._cache_nbytes:
                self._cache.popitem(last=False)
            self._cache[origin] = chunk

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


_READ_THROUGH_H5PY = object()  # returned by _ChunkedReader._get_chunk for chunks only HDF5 can be sure of


def is_direct_reader(reader) -> bool:
    """True if *reader* (as returned by BulkReader.open) reads directly, rather than being an h5py dataset"""
    return isinstance(reader, _DirectReader)
