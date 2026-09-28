""":class:`BulkReader`, which reads requests (see :class:`.plan.ReadRequest`), directly wherever that is safe."""

from __future__ import annotations

import logging
import threading
import typing
import warnings

try:
    import h5py
except ImportError:
    h5py = None

from . import execute, files, plan, strategy
from .common import (
    BulkReadFallbackWarning,
    _CannotReadDirectly,
    _default_cache_nbytes,
    _UnexpectedData,
)
from .datasets import _check_datatype, _ChunkedReader, _ContiguousReader
from .files import _check_file, _FileHandle
from .plan import ReadRequest
from .virtual import _VirtualReader

logger = logging.getLogger('pynbody.util.hdf_bulk_read')


class BulkReader:
    """Reads requests of HDF5 datasets (see :meth:`read`), directly wherever that is safe.

    A bulk reader remembers which files it has checked can be read directly, and holds open the files it reads them
    through and any source files of virtual datasets, until :meth:`close`. So a program reading from a set of files
    over and over (as pynbody does for each snapshot) should keep one bulk reader for them.
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
        self._file_checks = {}  # filename -> _FileHandle, or the _CannotReadDirectly it raised
        self._files_kept_open = 0
        self._warned_reasons = set()
        self._warned_lock = threading.Lock()

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

    def read(self, requests: typing.Iterable[ReadRequest]) -> strategy.ReadStrategy:
        """Read every request, in whatever way is judged best, and return how they were read.

        The requests are turned into units of work (see :mod:`.plan`), the rules in :mod:`.strategy` decide from
        a description of the work how many threads to read with, and the work is then done (see :mod:`.execute`).
        Datasets that cannot be read directly are read through h5py, with a :class:`BulkReadFallbackWarning`.

        Parameters
        ----------
        requests : iterable of ReadRequest
            What to read. Their destinations must not overlap. Passing an iterator (rather than, say, a list),
            and keeping no other reference to the datasets, lets each dataset be freed as soon as it has been read.

        Returns
        -------
        strategy.ReadStrategy
            How the requests were read, and why.
        """
        works = plan.plan(requests, self.open)
        chosen = strategy.choose_read_strategy(plan.summarise(works))
        logger.debug("Reading %d pieces with %d thread(s) because %s", len(works), chosen.threads, chosen.reason)
        execute.perform(works, chosen)
        return chosen

    def _plan(self, dataset):
        """Return a direct reader for *dataset*, or raise _CannotReadDirectly"""
        if dataset.id.get_space().get_simple_extent_type() != h5py.h5s.SIMPLE or dataset.ndim == 0:
            raise _CannotReadDirectly("it is a scalar", warn=False)
        _check_datatype(dataset)

        h5file = dataset.file  # (h5py makes a new File object every time this is accessed, so do it once)
        layout = dataset.id.get_create_plist().get_layout()
        if layout == h5py.h5d.VIRTUAL:
            return _VirtualReader.plan(dataset, h5file, self)

        file = self._check_file(h5file)
        if layout == h5py.h5d.CONTIGUOUS:
            return _ContiguousReader(dataset, file, self)
        elif layout == h5py.h5d.CHUNKED:
            return _ChunkedReader(dataset, file, self, self._cache_nbytes)
        elif layout == h5py.h5d.COMPACT:
            raise _CannotReadDirectly("it uses compact storage", warn=False)
        else:
            raise _CannotReadDirectly(f"it uses HDF5 storage layout {layout}, which pynbody does not know")

    def _check_file(self, h5file) -> _FileHandle:
        """Check that a file can be read directly (see _check_file), and return a handle through which to read it.

        The answer, and the handle, are shared by every dataset in the file, which matters when a snapshot's virtual
        datasets draw on many source files. Up to files._max_files_kept_open() files are kept open for positioned
        reads; beyond that, files are opened afresh for each read."""
        key = h5file.filename
        result = self._file_checks.get(key)
        if result is None:
            try:
                filename, identity = _check_file(h5file)
                keep_open = self._files_kept_open < files._max_files_kept_open()
                try:
                    result = _FileHandle(filename, identity, keep_open)
                except OSError:
                    if not keep_open:
                        raise
                    # e.g. the process has too many files open; try opening the file only for each read instead
                    keep_open = False
                    result = _FileHandle(filename, identity, keep_open)
                if keep_open:
                    self._files_kept_open += 1
            except _CannotReadDirectly as e:
                result = e
            except OSError as e:
                result = _CannotReadDirectly(f"pynbody could not open the file itself ({e})")
            self._file_checks[key] = result
        if isinstance(result, _CannotReadDirectly):
            raise result
        return result

    def _open_source(self, filename, dataset_name, virtual_file, virtual_filename):
        """Open a source dataset of a virtual dataset through h5py"""
        if filename == virtual_filename:
            return virtual_file[dataset_name]  # a source in the virtual dataset's own file ('.')
        source_file = self._source_files.get(filename)
        if source_file is None:
            # Opened outside the lock, so that threads opening different files do not queue for each other. If two
            # threads open the same file at once, one copy is kept and the other closed.
            opened = h5py.File(filename, 'r')
            with self._source_files_lock:
                source_file = self._source_files.setdefault(filename, opened)
            if source_file is not opened:
                opened.close()
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
        with self._warned_lock:
            if key in self._warned_reasons:
                return
            self._warned_reasons.add(key)
        warnings.warn(message, BulkReadFallbackWarning, stacklevel=3)

    def close(self):
        """Close every file this reader has opened (whether to read data, or as sources of virtual datasets).

        The reader remains usable, and reopens files as needed. Readers returned by :meth:`open` before the call
        should not be used afterwards."""
        with self._source_files_lock:
            for f in self._source_files.values():
                f.close()
            self._source_files = {}
            for handle in self._file_checks.values():
                if isinstance(handle, _FileHandle):
                    handle.close()
            self._file_checks = {}
            self._files_kept_open = 0
