"""Bulk reads of HDF5 datasets that bypass libhdf5, so that reads of different files can overlap.

pynbody reads HDF5 snapshots through h5py, whose interpreter-wide lock serialises every call into libhdf5, even
calls that concern different files. This module lets the bulk particle data be read without going through
libhdf5. h5py is still used for everything it does cheaply: finding the dataset, its datatype, its storage layout,
its filters, and the position in the file of its data or of each of its chunks. pynbody then reads the bytes itself
and, for chunked datasets, undoes the filters itself (zlib, which releases the GIL, for deflate; numpy for shuffle
and for verifying fletcher32 checksums).

Reading bytes from a file behind HDF5's back is only correct if pynbody understands exactly how they are stored, so
:meth:`BulkReader.open` checks that before anything is read, and hands the dataset back to be read through h5py
unless every check passes. The checks are:

* the file is open read-only (and not in SWMR mode), through HDF5's default ``sec2`` driver (whose file addresses
  are plain byte offsets), and the path pynbody would read is the very file HDF5 has open -- which is checked
  again on every read;
* the dataset has a simple dataspace of at least one dimension, and its datatype is exactly the standard HDF5
  representation of a numpy integer or floating-point type of 1, 2, 4 or 8 bytes (so no padding, unusual
  precision, non-IEEE or extended-precision floats, enumerations, compound or string types);
* contiguous data lies wholly within the file, is not held in external files, and occupies exactly as many bytes
  as its elements;
* chunked data uses no filters other than deflate, shuffle and fletcher32, and a shuffle filter's element size is
  the datatype's (a partial chunk at the edge of a dataset whose stored size is that of an unfiltered chunk is read
  through h5py, since HDF5 can be told to leave such chunks unfiltered without recording that it has);
* virtual datasets map contiguous blocks of whole rows from source datasets that pass the same checks, and have
  the same datatype (see below);
* the fill value is defined and is written into unallocated space (the default), so that parts of a dataset never
  written read as that value.

Where a check fails because of how the file was written, a :class:`BulkReadFallbackWarning` says so, once per
reason per snapshot, and h5py is used. Datasets h5py is simply the better tool for (compact storage, which only
very small datasets use; scalars; datasets never written) are handed back without a warning.

Reads are checked as they happen, too: every read must return the number of bytes expected, every chunk must
decode to exactly the size of a chunk, and fletcher32 checksums are verified. If anything is amiss, a warning is
issued and that dataset is read through h5py from then on, so any disagreement between pynbody and HDF5 about a
file is settled by HDF5.

Virtual datasets, such as those in the single-file view SWIFT writes of a multi-file snapshot, are decomposed into
their source datasets, which are then read like any other. This is supported where every mapping places a
contiguous block of whole rows of the virtual dataset, which covers the layouts written by SWIFT and by
:class:`pynbody.util.hdf_vds.HdfVdsMaker`. Source files are found by following HDF5's rules (see
:func:`_resolve_virtual_source_filename`), except that if a search path has been configured (through the
``HDF5_VDS_PREFIX`` environment variable or a dataset access property) the virtual dataset is read through h5py,
since HDF5's handling of those paths varies between versions. Any other layout is read through h5py, as is any source that cannot be
found (HDF5 then fills its rows with the fill value) or that does not itself pass the checks above.
"""

from __future__ import annotations

import collections
import itertools
import logging
import os
import threading
import warnings
import zlib

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

logger = logging.getLogger('pynbody.util.hdf_bulk_read')

_DEFLATE_FILTER = 1
_SHUFFLE_FILTER = 2
_FLETCHER32_FILTER = 3

_supported_filters = {_DEFLATE_FILTER, _SHUFFLE_FILTER, _FLETCHER32_FILTER}

# HDF5 drivers whose file addresses are byte offsets into a single ordinary file, which pynbody can check against
# the file descriptor HDF5 holds. ('windows' is an alias of sec2 in some builds.)
_supported_drivers = {'sec2', 'windows'}

_default_cache_nbytes = 64 * 1024 * 1024


class BulkReadFallbackWarning(UserWarning):
    """Issued when pynbody reads a dataset through h5py because it cannot be sure of reading it correctly itself.

    The data are still read correctly; but through h5py, reads cannot overlap."""


class _CannotReadDirectly(Exception):
    """Raised while planning a read when a dataset should be read through h5py instead.

    *warn* is False when that is merely the better choice (e.g. for tiny datasets) rather than a limitation."""

    def __init__(self, reason: str, warn: bool = True):
        super().__init__(reason)
        self.reason = reason
        self.warn = warn


class _UnexpectedData(Exception):
    """Raised during a read when the file does not contain what its metadata led us to expect."""


class BulkReader:
    """Opens HDF5 datasets for bulk reading, reading them directly wherever that is safe.

    Each multi-file manager owns one of these. It holds open any source files of virtual datasets, which
    :meth:`close` releases.
    """

    def __init__(self, enabled: bool = True, cache_nbytes: int = _default_cache_nbytes):
        """Create a bulk reader.

        Parameters
        ----------
        enabled : bool
            If False, every dataset is read through h5py.
        cache_nbytes : int
            The most decoded chunk data each chunked dataset keeps between reads. See :class:`_ChunkedReader`.
        """
        self._enabled = enabled and h5py is not None
        self._cache_nbytes = cache_nbytes
        self._source_files = {}
        self._source_files_lock = threading.Lock()
        self._warned_reasons = set()

    @property
    def enabled(self) -> bool:
        """True if datasets are read directly where possible, False if everything is read through h5py."""
        return self._enabled

    def open(self, dataset):
        """Return an object from which the data in *dataset* can be read.

        Parameters
        ----------
        dataset : h5py.Dataset
            The dataset to be read. Anything else is returned unchanged.

        Returns
        -------
        An object with ``shape``, ``dtype`` and ``ndim`` attributes, indexable by a slice along the first axis, and
        with an h5py-style ``read_direct(dest, source_sel=None)`` method, in which *source_sel* is None or a slice
        along the first axis. It reads the data directly where that has been found to be safe; otherwise it is
        *dataset* itself. Either way, it may only be used while the file containing *dataset* remains open,
        since chunk positions are looked up through h5py as they are needed.
        """
        if not self._enabled or not isinstance(dataset, h5py.Dataset):
            return dataset
        try:
            return self._plan(dataset)
        except _CannotReadDirectly as e:
            self._report_fallback(dataset, e)
            return dataset

    def _plan(self, dataset):
        """Return a direct reader for *dataset*, or raise _CannotReadDirectly"""
        if dataset.id.get_space().get_simple_extent_type() != h5py.h5s.SIMPLE or dataset.ndim == 0:
            raise _CannotReadDirectly("it is a scalar", warn=False)
        _check_datatype(dataset)

        layout = dataset.id.get_create_plist().get_layout()
        if layout == h5py.h5d.VIRTUAL:
            return _VirtualReader.plan(dataset, self)

        file = _check_file(dataset.file)
        if layout == h5py.h5d.CONTIGUOUS:
            return _ContiguousReader(dataset, file, self)
        elif layout == h5py.h5d.CHUNKED:
            return _ChunkedReader(dataset, file, self, self._cache_nbytes)
        elif layout == h5py.h5d.COMPACT:
            raise _CannotReadDirectly("it uses compact storage", warn=False)
        else:
            raise _CannotReadDirectly(f"it uses HDF5 storage layout {layout}, which pynbody does not know")

    def _open_source(self, filename, dataset_name, virtual_dataset):
        """Open a source dataset of a virtual dataset through h5py"""
        if filename == os.path.abspath(virtual_dataset.file.filename):
            return virtual_dataset.file[dataset_name]  # a source in the virtual dataset's own file ('.')
        with self._source_files_lock:
            source_file = self._source_files.get(filename)
            if source_file is None:
                source_file = h5py.File(filename, 'r')
                self._source_files[filename] = source_file
        return source_file[dataset_name]

    def _report_fallback(self, dataset, error: _CannotReadDirectly | _UnexpectedData | Exception, reading=False):
        if reading:
            message = (f"pynbody could not read {dataset.name} in {dataset.file.filename} itself ({error}), "
                       f"so is reading it through h5py instead")
            key = (dataset.file.filename, dataset.name)
        elif getattr(error, 'warn', True):
            message = (f"pynbody is reading {dataset.name} in {dataset.file.filename}, and any other dataset for "
                       f"the same reason, through h5py (which prevents reads from overlapping) because "
                       f"{error.reason}")
            key = error.reason
        else:
            logger.debug("Reading %s in %s through h5py because %s", dataset.name, dataset.file.filename, error)
            return
        if key not in self._warned_reasons:
            self._warned_reasons.add(key)
            warnings.warn(message, BulkReadFallbackWarning, stacklevel=3)

    def close(self):
        """Close any source files of virtual datasets opened by this reader.

        The reader remains usable, and reopens files as needed. Readers returned by :meth:`open` before the call
        should not be used afterwards."""
        with self._source_files_lock:
            for f in self._source_files.values():
                f.close()
            self._source_files = {}


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


def _file_identity(h5file) -> tuple[str, tuple]:
    """Return the absolute path of the file HDF5 has open, and its (device, inode), checking that they agree"""
    filename = os.path.abspath(h5file.filename)
    try:
        held = os.fstat(h5file.id.get_vfd_handle())
        on_disk = os.stat(filename)
    except (OSError, TypeError, ValueError) as e:
        raise _CannotReadDirectly(f"pynbody could not confirm which file HDF5 has open ({e})")
    if (held.st_dev, held.st_ino) != (on_disk.st_dev, on_disk.st_ino):
        raise _CannotReadDirectly(f"the file at {filename} is not the one HDF5 has open (it may have been "
                                  f"replaced, or opened by a relative path from another directory)")
    return filename, (held.st_dev, held.st_ino)


def _check_file(h5file) -> tuple[str, tuple]:
    """Raise _CannotReadDirectly unless pynbody can read the file's bytes itself; return its path and identity"""
    if h5file.mode != 'r':
        raise _CannotReadDirectly("the file is open for writing", warn=False)
    if h5file.swmr_mode:
        raise _CannotReadDirectly("the file is open in SWMR mode, so may be growing", warn=False)
    if h5file.driver not in _supported_drivers:
        raise _CannotReadDirectly(f"the file is open through the HDF5 '{h5file.driver}' driver")
    return _file_identity(h5file)


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


def _read_bytes(file, offset, nbytes, into=None):
    """Read *nbytes* at *offset* from *file*, a (path, identity) pair, into the writable buffer *into* if given, else
    returning bytes.

    Each read opens its own handle, so that reads may proceed in parallel, and checks that the handle is to the
    file HDF5 has open, in case the file at that path has been replaced since."""
    filename, identity = file
    with open(filename, 'rb') as f:
        st = os.fstat(f.fileno())
        if (st.st_dev, st.st_ino) != identity:
            raise _UnexpectedData(f"the file at {filename} has been replaced since HDF5 opened it")
        f.seek(offset)
        if into is None:
            data = f.read(nbytes)
            got = len(data)
        else:
            data = None
            got = f.readinto(into)
    if got != nbytes:
        raise _UnexpectedData(f"expected {nbytes} bytes at offset {offset}, but the file supplied {got}")
    return data


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
    if from_dtype.kind == 'f':
        return from_dtype.itemsize >= 4 and from_dtype.isnative and to_dtype.isnative
    return True


class _ContiguousReader(_DirectReader):
    """A contiguous dataset, read directly from the file."""

    def __init__(self, dataset, file, bulk_reader):
        super().__init__(dataset, bulk_reader)
        filename = file[0]
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
        if self._offset + nbytes > os.path.getsize(filename):
            raise _CannotReadDirectly("its data extend beyond the end of the file")
        self._file = file

    def _read_rows_into(self, out, start, stop):
        offset = self._offset + start * self._row_nbytes
        nbytes = (stop - start) * self._row_nbytes
        if out.dtype == self.dtype and out.flags.c_contiguous:
            _read_bytes(self._file, offset, nbytes, into=memoryview(out.view(np.uint8)))
        else:
            data = _read_bytes(self._file, offset, nbytes)
            out[...] = np.frombuffer(data, dtype=self.dtype).reshape(out.shape)


class _ChunkedReader(_DirectReader):
    """A chunked dataset, whose chunks are read directly from the file and decoded here.

    Chunk positions are looked up through h5py as they are needed. HDF5's chunk cache is not involved, so this class
    keeps its own: snapshot writers routinely use chunks of many megabytes (sometimes one for a whole dataset), while
    pynbody reads in pieces of at most ``_max_buf`` rows, so without a cache a partial load would decompress the same
    chunk over and over. Chunks that extend beyond the end of a read are therefore kept, least recently used first
    out, up to a total of *cache_nbytes*. Chunks that a read consumes entirely are not kept, since pynbody reads each
    file in increasing order of rows.
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
        self._file_size = os.path.getsize(file[0])

        self._cache_nbytes = cache_nbytes
        self._cache = collections.OrderedDict()
        self._cache_lock = threading.Lock()

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
                    dest[...] = chunk[(slice(row_lo - row_origin, row_hi - row_origin),) +
                                      tuple(slice(0, s.stop - s.start) for s in trailing)]

    def _get_chunk(self, origin, keep):
        """Return the decoded chunk at *origin*; None if it has never been written; or _READ_THROUGH_H5PY."""
        with self._cache_lock:
            chunk = self._cache.get(origin)
            if chunk is not None:
                self._cache.move_to_end(origin)
                return chunk

        info = self._dataset.id.get_chunk_info_by_coord(origin)
        if info.byte_offset is None:
            return None  # never written, so reads as the fill value
        if tuple(info.chunk_offset) != origin:
            raise _UnexpectedData(f"HDF5 reported the chunk at {origin} as being at {tuple(info.chunk_offset)}")
        if info.byte_offset + info.size > self._file_size:
            raise _UnexpectedData(f"the chunk at {origin} extends beyond the end of the file")
        if info.size == self._chunk_nbytes and self._applies_filters(info.filter_mask) and \
                any(o + c > n for o, c, n in zip(origin, self._chunk_shape, self.shape)):
            # A partial chunk at the edge of the dataset, of exactly the size it would have unfiltered. HDF5 can be
            # told (H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) to store such chunks unfiltered, without recording that it
            # has, and h5py offers no way to find out whether it was; so only HDF5 can be sure how to read it.
            return _READ_THROUGH_H5PY

        raw = _read_bytes(self._file, info.byte_offset, info.size)
        decoded = decode_chunk(raw, info.filter_mask, self._pipeline, self.dtype.itemsize)
        if len(decoded) != self._chunk_nbytes:
            raise _UnexpectedData(f"the chunk at {origin} decoded to {len(decoded)} bytes, "
                                  f"not {self._chunk_nbytes}")
        chunk = np.frombuffer(decoded, dtype=self.dtype).reshape(self._chunk_shape)

        if keep and self._chunk_nbytes <= self._cache_nbytes:
            with self._cache_lock:
                while self._cache and (len(self._cache) + 1) * self._chunk_nbytes > self._cache_nbytes:
                    self._cache.popitem(last=False)
                self._cache[origin] = chunk
        return chunk

    def _applies_filters(self, filter_mask):
        """True if any filter of the pipeline applies to a chunk with the given filter mask"""
        return any(not filter_mask & (1 << i) for i in range(len(self._pipeline)))


_READ_THROUGH_H5PY = object()  # returned by _ChunkedReader._get_chunk for chunks only HDF5 can be sure of


def decode_chunk(raw, filter_mask: int, pipeline: list, itemsize: int) -> np.ndarray:
    """Undo the filter pipeline applied to a raw HDF5 chunk, returning its bytes as a uint8 array.

    Parameters
    ----------
    raw : bytes
        The chunk as stored in the file.
    filter_mask : int
        Bit *i* is set if filter *i* of the pipeline was skipped for this chunk.
    pipeline : list of dict
        The filters, in the order they were applied on writing, each with at least a ``filter_id``; shuffle
        filters may also carry their element size as ``client_data[0]``.
    itemsize : int
        The dataset's element size, used by the shuffle filter if its client data does not give one.
    """
    data = np.frombuffer(raw, dtype=np.uint8)
    for i in reversed(range(len(pipeline))):
        if filter_mask & (1 << i):
            continue
        filter_id = pipeline[i]['filter_id']
        if filter_id == _DEFLATE_FILTER:
            data = np.frombuffer(zlib.decompress(data), dtype=np.uint8)
        elif filter_id == _SHUFFLE_FILTER:
            client_data = pipeline[i].get('client_data') or ()
            data = _unshuffle(data, int(client_data[0]) if len(client_data) > 0 else itemsize)
        elif filter_id == _FLETCHER32_FILTER:
            _verify_fletcher32(data)
            data = data[:-4]
        else:
            raise NotImplementedError(f"HDF5 filter {filter_id} is not supported")
    return data


def _unshuffle(data: np.ndarray, element_size: int) -> np.ndarray:
    """Undo HDF5's shuffle filter.

    As in HDF5, only the largest whole number of elements is shuffled; any bytes left over (for instance, a
    checksum appended by a filter applied before the shuffle) are passed through unchanged."""
    num_elements = len(data) // element_size
    if element_size <= 1 or num_elements <= 1:
        return data
    shuffled_nbytes = num_elements * element_size
    out = np.empty_like(data)
    out[:shuffled_nbytes].reshape(num_elements, element_size)[...] = \
        data[:shuffled_nbytes].reshape(element_size, num_elements).T
    out[shuffled_nbytes:] = data[shuffled_nbytes:]
    return out


def fletcher32(data) -> int:
    """Return the checksum HDF5's fletcher32 filter computes for *data* (bytes or a uint8 array).

    HDF5 reads the data as big-endian 16-bit words (padding an odd final byte with zero) and keeps the two running
    sums below 2**16 by end-around carry, ``x = (x & 0xffff) + (x >> 16)``. That is arithmetic modulo 65535, except
    that a nonzero sum divisible by 65535 is represented as 0xffff rather than 0. Both sums are nonzero unless every
    word is, so each can be found from its exact value modulo 65535. The second sum accumulates the first after every
    word, making it sum_i w_i (n - i) for words w_0 ... w_{n-1}.
    """
    data = np.frombuffer(data, dtype=np.uint8) if not isinstance(data, np.ndarray) else data
    if len(data) % 2:
        data = np.concatenate((data, np.zeros(1, dtype=np.uint8)))
    words = data.view('>u2')
    num_words = len(words)

    # Both sums need only be known modulo 65535. The second is sum_i w_i (n - i) = n S - sum_i i w_i, where S is the
    # first. To compute sum_i i w_i with vectorised reductions, view the words as a matrix W of `width` columns: then
    # it is width * sum_r r R_r + sum_c c C_c, where R and C are the row and column sums of W. Every term is reduced
    # modulo 65535 before being multiplied, so nothing can overflow int64.
    width = 1024
    num_rows = num_words // width
    total, index_weighted = 0, 0
    if num_rows > 0:
        matrix = words[:num_rows * width].reshape(num_rows, width)
        row_sums = matrix.sum(axis=1, dtype=np.int64)
        column_sums = matrix.sum(axis=0, dtype=np.int64)
        total = int(row_sums.sum())
        index_weighted = width * int(np.dot(np.arange(num_rows, dtype=np.int64) % 65535, row_sums % 65535)) \
                         + int(np.dot(np.arange(width, dtype=np.int64), column_sums % 65535))
    remainder = words[num_rows * width:].astype(np.int64)
    total += int(remainder.sum())
    index_weighted += int(np.dot(np.arange(num_rows * width, num_words, dtype=np.int64), remainder))
    weighted_total = (num_words * total - index_weighted) % 65535

    if total == 0:
        return 0
    sum1 = (total - 1) % 65535 + 1
    sum2 = (weighted_total - 1) % 65535 + 1
    return (sum2 << 16) | sum1


def _verify_fletcher32(data: np.ndarray):
    """Check the fletcher32 checksum at the end of a chunk, raising OSError if it does not match.

    The checksum is stored little-endian. Like HDF5, this also accepts the checksum with the bytes of each 16-bit
    half swapped, as written by some old versions of the library."""
    if len(data) < 4:
        raise OSError("Chunk is too short to carry a fletcher32 checksum")
    stored = int(data[-4:].view('<u4')[0])
    computed = fletcher32(data[:-4])
    reversed_bytes = ((computed & 0x00ff00ff) << 8) | ((computed >> 8) & 0x00ff00ff)
    if stored != computed and stored != reversed_bytes:
        raise OSError("fletcher32 checksum of HDF5 chunk is invalid; the file may be corrupt")


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


class _VirtualReader(_DirectReader):
    """A virtual dataset, read by reading each of its source datasets directly.

    Sources are opened lazily, so that a partial load touching only a few rows opens only the source files it needs.
    A source that cannot be found or read directly is read through the h5py virtual dataset instead, restricted to
    the rows that source supplies, which gives exactly the result HDF5 would.
    """

    def __init__(self, dataset, blocks: list[_VirtualSourceBlock], bulk_reader: BulkReader, fillvalue):
        super().__init__(dataset, bulk_reader)
        self._fillvalue = fillvalue
        self._blocks = blocks
        self._block_starts = np.array([b.start for b in blocks], dtype=np.int64)

    @classmethod
    def plan(cls, dataset, bulk_reader: BulkReader) -> _VirtualReader:
        """Work out how to read an h5py virtual dataset from its sources, or raise _CannotReadDirectly if its
        layout is not one this class handles (see the module docstring)."""
        shape = tuple(dataset.shape)
        unsupported = "its virtual dataset layout is not one pynbody can decompose"
        if os.environ.get('HDF5_VDS_PREFIX') or dataset.id.get_access_plist().get_virtual_prefix():
            raise _CannotReadDirectly("a search path for the sources of virtual datasets has been configured")
        # Source files are found relative to the directory of the virtual dataset's own file, so that must be known
        virtual_filename, _ = _file_identity(dataset.file)
        fillvalue = _check_fill(dataset)
        try:
            blocks = cls._plan_blocks(dataset, shape, virtual_filename, unsupported)
        except (RuntimeError, ValueError) as e:
            # e.g. HDF5 cannot describe a selection of unlimited extent as a set of points
            raise _CannotReadDirectly(unsupported) from e
        return cls(dataset, blocks, bulk_reader, fillvalue)

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

    def _get_source_reader(self, block: _VirtualSourceBlock):
        """Return a direct reader for the block's source dataset, or None if it must be read through h5py"""
        if block.reader_resolved:
            return block.reader
        block.reader_resolved = True
        if block.filename is None:
            return None  # HDF5 will fill the block with the fill value

        try:
            source = self._bulk_reader._open_source(block.filename, block.dataset_name, self._dataset)
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
        block.reader = reader
        return reader

    def _read_rows_into(self, out, start, stop):
        first = max(int(np.searchsorted(self._block_starts, start, side='right')) - 1, 0)
        rows_covered = 0
        pieces = []
        for block in self._blocks[first:]:
            if block.start >= stop:
                break
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


def _resolve_virtual_source_filename(virtual_filename: str, source_filename: str) -> str | None:
    """Find a virtual dataset's source file, as HDF5 would, or return None if it does not exist.

    With no search path configured (see :meth:`_VirtualReader.plan`), HDF5 tries, in order: an absolute name as
    given; the name relative to the directory of the virtual dataset's own file; and finally the name relative to the
    current directory. An absolute name that does not exist is reduced to its final component and searched for in
    the same way. ``.`` means the virtual dataset's own file. *virtual_filename* must be absolute, and so is the
    result.
    """
    if source_filename == '.':
        return virtual_filename

    origin = os.path.dirname(virtual_filename)
    candidates = []
    if os.path.isabs(source_filename):
        candidates.append(source_filename)
        source_filename = os.path.basename(source_filename)
    candidates.append(os.path.join(origin, source_filename))
    candidates.append(source_filename)

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None
