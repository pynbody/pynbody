"""Bulk reads of HDF5 datasets through pyfive, falling back to h5py where pyfive cannot be trusted.

pynbody discovers the structure of an HDF5 snapshot (groups, attributes, dataset shapes) through h5py, but reads
the bulk particle data through `pyfive <https://github.com/NCAS-CMS/pyfive>`_ wherever it safely can. pyfive
parses the HDF5 container in Python and reads data with ordinary file I/O, opening a fresh file handle for each
read. Unlike h5py, whose interpreter-wide lock serialises every call into libhdf5 (even calls on different
files), it therefore places no obstacle in the way of reads overlapping.

pyfive is used to parse each file's metadata, to find where the data lie, and to read contiguous datasets (which
it does by memory mapping). Chunked datasets are decoded here rather than by pyfive, for two reasons: pyfive
verifies fletcher32 checksums in a pure-Python loop over every 16-bit word, which makes it more than ten times
slower than h5py on typical SWIFT output; and as of version 1.2.1 it rejects valid chunks whose checksum sums
fold to ``0xFFFF``, which happens to roughly one chunk in every 30,000. Only pyfive's public, h5py-compatible
chunk API (``get_chunk_info_by_coord`` and ``read_direct_chunk``) is used for this.

A dataset is read through h5py instead if any of the following hold:

* pyfive is not installed, or has been disabled through the ``bulk-read-backend`` option in the ``[gadgethdf]``
  section of the configuration;
* it has a compound, string, reference or other non-numeric datatype;
* it is chunked with a filter other than deflate, shuffle and fletcher32;
* it is contiguous but has no storage address in the file, as happens with external storage (which pyfive would
  silently read as fill values) and with datasets that were never written;
* it uses compact storage, which only very small datasets do;
* pyfive fails to parse it, or reports a different shape or dtype from h5py.

Virtual datasets, such as those in the single-file view SWIFT writes of a multi-file snapshot, are decomposed
into their source datasets. Each source is opened directly in pyfive and read like any other dataset, so a read
of a virtual dataset never goes through libhdf5. This is supported where every mapping places a contiguous block
of whole rows of the virtual dataset, which covers the layouts written by SWIFT and by
:class:`pynbody.util.hdf_vds.HdfVdsMaker`. Any other virtual layout is read through h5py. A source file that
cannot be found is also read through h5py, which fills the missing region with the dataset's fill value,
exactly as HDF5 itself would.
"""

from __future__ import annotations

import collections
import itertools
import logging
import os
import zlib

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    import pyfive
except ImportError:
    pyfive = None

logger = logging.getLogger('pynbody.util.hdf_bulk_read')

_DEFLATE_FILTER = 1
_SHUFFLE_FILTER = 2
_FLETCHER32_FILTER = 3

_supported_filters = {_DEFLATE_FILTER, _SHUFFLE_FILTER, _FLETCHER32_FILTER}

_COMPACT_LAYOUT = 0
_CONTIGUOUS_LAYOUT = 1
_CHUNKED_LAYOUT = 2

_UNDEFINED_ADDRESS = 2 ** 64 - 1

_default_cache_nbytes = 64 * 1024 * 1024


def pyfive_available() -> bool:
    """Return True if pyfive is installed and so can be used for bulk reads."""
    return pyfive is not None


class BulkReader:
    """Opens HDF5 datasets for bulk reading, preferring pyfive over h5py wherever it is safe.

    Each multi-file manager owns one of these. It caches the pyfive file objects, so that each file's metadata is
    parsed only once, and must be closed (see :meth:`close`) before any of the files are modified through h5py.
    """

    def __init__(self, use_pyfive: bool = True, cache_nbytes: int = _default_cache_nbytes):
        """Create a bulk reader.

        Parameters
        ----------
        use_pyfive : bool
            If False, or if pyfive is not installed, every dataset is read through h5py.
        cache_nbytes : int
            The most decoded chunk data each chunked dataset keeps between reads. See :class:`_ChunkedDataset`.
        """
        self._use_pyfive = use_pyfive and pyfive is not None
        self._cache_nbytes = cache_nbytes
        self._pyfive_files = {}

    @property
    def uses_pyfive(self) -> bool:
        """True if this reader reads through pyfive where it can, False if it always uses h5py."""
        return self._use_pyfive

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
        along the first axis. It is backed by pyfive where possible; otherwise it is *dataset* itself.
        """
        if not self._use_pyfive or h5py is None or not isinstance(dataset, h5py.Dataset):
            return dataset
        if not _is_plain_numeric(dataset.dtype) or dataset.ndim == 0:
            return dataset
        if dataset.is_virtual:
            reader = _VirtualDataset.from_h5py(dataset, self)
        elif dataset.external is not None:
            reader = None
        else:
            reader = self.open_pyfive(dataset.file.filename, dataset.name, dataset.shape, dataset.dtype)
        return dataset if reader is None else reader

    def open_pyfive(self, filename: str, dataset_name: str, shape: tuple | None = None,
                    dtype: np.dtype | None = None) -> _RowReader | None:
        """Open the named dataset through pyfive, or return None if pyfive cannot be trusted to read it.

        If *shape* or *dtype* are given, pyfive's view of the dataset must match them.
        """
        if not self._use_pyfive:
            return None
        try:
            pyfive_dataset = self._get_pyfive_file(filename)[dataset_name]
        except Exception as e:
            # pyfive has not been able to parse the file, or at least not this part of it. Whatever went wrong,
            # h5py is still able to read the dataset, so there is no reason to fail here.
            logger.debug("pyfive could not open %s in %s (%r); falling back to h5py", dataset_name, filename, e)
            return None

        if not _pyfive_can_read(pyfive_dataset):
            return None
        if shape is not None and tuple(pyfive_dataset.shape) != tuple(shape):
            return None
        if dtype is not None and np.dtype(pyfive_dataset.dtype) != np.dtype(dtype):
            return None
        if pyfive_dataset.id.layout_class == _CHUNKED_LAYOUT:
            return _ChunkedDataset(pyfive_dataset, self._cache_nbytes)
        else:
            return _ContiguousDataset(pyfive_dataset)

    def _get_pyfive_file(self, filename):
        f = self._pyfive_files.get(filename)
        if f is None:
            f = pyfive.File(filename)
            self._pyfive_files[filename] = f
        return f

    def close(self):
        """Close every pyfive file this reader holds open.

        The reader remains usable, and reopens files as needed. Datasets returned by :meth:`open` before the
        call should not be used afterwards."""
        for f in self._pyfive_files.values():
            f.close()
        self._pyfive_files = {}


def _is_plain_numeric(dtype) -> bool:
    dtype = np.dtype(dtype)
    return dtype.kind in 'biuf' and dtype.fields is None and dtype.subdtype is None


def _pyfive_can_read(pyfive_dataset) -> bool:
    """Return True if pyfive can be trusted to read this dataset (see the module docstring)."""
    if not isinstance(pyfive_dataset, pyfive.Dataset):
        return False
    if not _is_plain_numeric(pyfive_dataset.dtype):
        return False

    dataset_id = pyfive_dataset.id
    layout = dataset_id.layout_class
    if layout == _CONTIGUOUS_LAYOUT:
        # External storage has no address within the file, and pyfive would read it as fill values. The only
        # other way to lack an address is never to have been written, which h5py handles equally well.
        return getattr(dataset_id, 'data_offset', _UNDEFINED_ADDRESS) != _UNDEFINED_ADDRESS
    elif layout == _CHUNKED_LAYOUT:
        filters = {f['filter_id'] for f in (dataset_id.filter_pipeline or [])}
        return filters <= _supported_filters
    else:
        # compact data is too small to be worth reading any other way than the simplest; and anything else
        # (e.g. a virtual dataset nested inside a virtual dataset) is beyond what we handle here
        return False


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


class _RowReader:
    """Base class for datasets read in ranges of whole rows, presenting the subset of the h5py API pynbody uses."""

    shape: tuple
    dtype: np.dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, sel):
        rows = _row_range(sel, self.shape[0])
        if rows is None:
            raise TypeError(f"{type(self).__name__} can only be indexed by a contiguous range of rows")
        out = np.empty((rows[1] - rows[0],) + tuple(self.shape[1:]), dtype=self.dtype)
        self._read_rows_into(out, *rows)
        return out

    def read_direct(self, dest: np.ndarray, source_sel=None):
        """Read the selected rows into *dest*, which must have exactly the shape of the selection."""
        rows = _row_range(source_sel, self.shape[0])
        if rows is None:
            raise TypeError(f"{type(self).__name__} can only read a contiguous range of rows")
        expected_shape = (rows[1] - rows[0],) + tuple(self.shape[1:])
        if dest.shape != expected_shape:
            raise ValueError(f"Destination has shape {dest.shape} but the selection has shape {expected_shape}")
        # Write through a plain ndarray view: slicing an ndarray subclass (such as pynbody's SimArray) runs its
        # __array_finalize__, which can be far more expensive than copying the data
        self._read_rows_into(dest.view(np.ndarray), *rows)

    def _read_rows_into(self, out: np.ndarray, start: int, stop: int):
        raise NotImplementedError


class _ContiguousDataset(_RowReader):
    """A contiguous dataset, read through pyfive (which memory maps it)."""

    def __init__(self, pyfive_dataset):
        self._dataset = pyfive_dataset
        self.shape = tuple(pyfive_dataset.shape)
        self.dtype = np.dtype(pyfive_dataset.dtype)

    def _read_rows_into(self, out, start, stop):
        if stop > start:
            out[...] = self._dataset[start:stop]


class _ChunkedDataset(_RowReader):
    """A chunked dataset, whose raw chunks are fetched by pyfive and decoded here (see the module docstring).

    pyfive has no equivalent of HDF5's chunk cache. Snapshot writers routinely use chunks of many megabytes
    (sometimes one chunk for a whole dataset), while pynbody reads in pieces of at most ``_max_buf`` rows, so
    without a cache a partial load would decompress the same chunk over and over. Chunks that extend beyond the
    end of a read are therefore kept, least recently used first out, up to a total of *cache_nbytes*. Chunks
    that a read consumes entirely are not kept, since pynbody reads each file in increasing order of rows.
    """

    def __init__(self, pyfive_dataset, cache_nbytes: int = _default_cache_nbytes):
        self._id = pyfive_dataset.id
        self.shape = tuple(pyfive_dataset.shape)
        self.dtype = np.dtype(pyfive_dataset.dtype)
        self._chunk_shape = tuple(int(c) for c in pyfive_dataset.chunks)
        self._chunk_nbytes = int(np.prod(self._chunk_shape, dtype=np.int64)) * self.dtype.itemsize
        self._pipeline = list(self._id.filter_pipeline or [])
        fillvalue = pyfive_dataset.fillvalue
        self._fillvalue = 0 if fillvalue is None else fillvalue

        self._cache_nbytes = cache_nbytes
        self._cache = collections.OrderedDict()

    def _read_rows_into(self, out, start, stop):
        if stop <= start:
            return
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
                else:
                    dest[...] = chunk[(slice(row_lo - row_origin, row_hi - row_origin),) +
                                      tuple(slice(0, s.stop - s.start) for s in trailing)]

    def _get_chunk(self, origin, keep):
        """Return the decoded chunk at *origin*, or None if it has never been written."""
        chunk = self._cache.get(origin)
        if chunk is not None:
            self._cache.move_to_end(origin)
            return chunk

        try:
            self._id.get_chunk_info_by_coord(origin)
        except KeyError:
            return None  # a chunk that has never been written, which reads as the fill value
        filter_mask, raw = self._id.read_direct_chunk(origin)
        decoded = decode_chunk(raw, filter_mask, self._pipeline, self.dtype.itemsize)
        if len(decoded) != self._chunk_nbytes:
            raise OSError(f"Chunk at {origin} decoded to {len(decoded)} bytes, but {self._chunk_nbytes} were expected")
        chunk = np.frombuffer(decoded, dtype=self.dtype).reshape(self._chunk_shape)

        if keep and self._chunk_nbytes <= self._cache_nbytes:
            while self._cache and (len(self._cache) + 1) * self._chunk_nbytes > self._cache_nbytes:
                self._cache.popitem(last=False)
            self._cache[origin] = chunk
        return chunk


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


class _VirtualDataset(_RowReader):
    """A virtual dataset, read by reading each of its source datasets through pyfive.

    Sources are opened lazily, so that a partial load touching only a few rows opens only the source files it
    needs. A source pyfive cannot read (or cannot find) is read through the h5py virtual dataset instead, restricted
    to the rows that source supplies, which gives exactly the result HDF5 would.
    """

    def __init__(self, h5py_dataset, blocks: list[_VirtualSourceBlock], bulk_reader: BulkReader):
        self._h5py_dataset = h5py_dataset
        self.shape = tuple(h5py_dataset.shape)
        self.dtype = np.dtype(h5py_dataset.dtype)
        self._fillvalue = h5py_dataset.fillvalue
        self._blocks = blocks
        self._bulk_reader = bulk_reader
        self._block_starts = np.array([b.start for b in blocks], dtype=np.int64)

    @classmethod
    def from_h5py(cls, dataset, bulk_reader: BulkReader) -> _VirtualDataset | None:
        """Work out how to read an h5py virtual dataset through pyfive, or return None if its layout is not one
        this class handles (see the module docstring)."""
        shape = tuple(dataset.shape)
        virtual_filename = dataset.file.filename
        blocks = []

        for source in dataset.virtual_sources():
            if source.vspace.get_select_npoints() == 0:
                continue  # e.g. a file holding no particles of this type, which supplies nothing

            if '%' in source.file_name or '%' in source.dset_name:
                # printf-style patterns, which HDF5 expands for mappings with unlimited extents
                return None

            virtual_box = _selection_box(source.vspace, shape)
            if virtual_box is None:
                return None
            v_start, v_stop = virtual_box
            if any(v_start[1:]) or tuple(v_stop[1:]) != shape[1:]:
                return None  # does not map whole rows

            if source.src_space.get_select_type() == h5py.h5s.SEL_ALL:
                source_start = 0
                whole_source = True
            else:
                source_box = _selection_box(source.src_space)
                if source_box is None:
                    return None
                s_start, s_stop = source_box
                if [b - a for a, b in zip(s_start, s_stop)] != [b - a for a, b in zip(v_start, v_stop)]:
                    return None
                if any(s_start[1:]):
                    return None
                source_start = s_start[0]
                whole_source = False

            filename = _resolve_virtual_source_filename(virtual_filename, source.file_name)
            blocks.append(_VirtualSourceBlock(v_start[0], v_stop[0], filename, source.dset_name,
                                              source_start, whole_source))

        blocks.sort(key=lambda b: b.start)
        for previous, following in zip(blocks[:-1], blocks[1:]):
            if previous.stop > following.start:
                return None  # overlapping mappings

        return cls(dataset, blocks, bulk_reader)

    def _get_source_reader(self, block: _VirtualSourceBlock):
        if not block.reader_resolved:
            block.reader_resolved = True
            if block.filename is not None:
                reader = self._bulk_reader.open_pyfive(block.filename, block.dataset_name)
                if reader is not None:
                    if reader.shape[1:] != self.shape[1:]:
                        reader = None
                    elif block.whole_source and reader.shape[0] != block.stop - block.start:
                        reader = None
                    elif block.source_start + block.stop - block.start > reader.shape[0]:
                        reader = None
                block.reader = reader
        return block.reader

    def _read_rows_into(self, out, start, stop):
        if stop <= start:
            return

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
                dest[...] = self._h5py_dataset[piece_start:piece_stop]
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

    HDF5 tries, in order: an absolute name as given; each directory listed in the ``HDF5_VDS_PREFIX`` environment
    variable, with ``${ORIGIN}`` standing for the directory of the virtual dataset's own file; that directory
    itself; and finally the name relative to the current directory. An absolute name that does not exist is
    reduced to its final component and searched for in the same way. ``.`` means the virtual dataset's own file.
    """
    if source_filename == '.':
        return virtual_filename

    origin = os.path.dirname(os.path.abspath(virtual_filename))
    candidates = []
    if os.path.isabs(source_filename):
        candidates.append(source_filename)
        source_filename = os.path.basename(source_filename)

    for prefix in os.environ.get('HDF5_VDS_PREFIX', '').split(os.pathsep):
        if prefix:
            candidates.append(os.path.join(prefix.replace('${ORIGIN}', origin), source_filename))
    candidates.append(os.path.join(origin, source_filename))
    candidates.append(source_filename)

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None
