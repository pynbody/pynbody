"""Memoised, read-only access to the filesystem for identifying file formats.

When :func:`pynbody.load` or :meth:`pynbody.snapshot.simsnap.SimSnap.halos` is called, each candidate class in turn
is asked whether it can load the data. Many of those checks need the same information: whether a path exists,
whether it is a directory, the first few bytes of a file, whether it is an HDF5 file and which groups it contains,
or what files sit alongside it. On network filesystems, every ``stat``, ``open`` or directory listing can cost
milliseconds, so repeating these operations for each candidate class makes format identification slow.

This module provides a :class:`ProbeCache`, which is created once per identification attempt and hands out
:class:`FileProbe` objects. Each probe remembers the result of every operation performed on it, so that
successive candidate classes can inspect the same file at almost no cost.

The cache is deliberately short-lived (one call to ``load`` or ``halos``), so there is no need to worry about
its contents becoming stale.
"""

from __future__ import annotations

import fnmatch
import glob
import gzip
import os
import pathlib
import stat as stat_module

try:
    import h5py
except ImportError:
    h5py = None

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"

# HDF5 places its signature at offset 0, or at 512 * 2^k if there is a user block
_HDF5_SIGNATURE_OFFSETS_IN_HEAD = (0, 512, 1024, 2048)

_DEFAULT_HEAD_BYTES = 4096


class FileProbe:
    """A memoising, read-only view of a single path.

    Probes should be obtained from :meth:`ProbeCache.probe`, not constructed directly, so that the cache is shared.
    All methods are safe to call on paths that do not exist; they then return ``False``, ``None`` or empty values as
    appropriate.
    """

    def __init__(self, path: pathlib.Path, cache: ProbeCache):
        self.path = path
        self.cache = cache
        self._stat = None
        self._stat_done = False
        self._head = None
        self._head_request = 0
        self._head_decompressed = None
        self._is_hdf5 = None
        self._hdf5 = None
        self._hdf5_done = False
        self._hdf5_adoptable = False
        self._listing = None
        self._listing_done = False

    def __repr__(self):
        return f"<FileProbe {self.path}>"

    def __fspath__(self):
        return os.fspath(self.path)

    def sibling(self, name: str) -> FileProbe:
        """Return a probe for a file with the given name, in the same directory as this one"""
        return self.cache.probe(self.path.parent / name)

    def with_appended(self, suffix: str) -> FileProbe:
        """Return a probe for a path formed by appending a string to this path's name, e.g. ``.0`` or ``.gz``"""
        return self.sibling(self.path.name + suffix)

    def child(self, name: str) -> FileProbe:
        """Return a probe for a path inside this one (which is presumably a directory)"""
        return self.cache.probe(self.path / name)

    def parent(self) -> FileProbe:
        """Return a probe for the directory containing this path"""
        return self.cache.probe(self.path.parent)

    def stat(self) -> os.stat_result | None:
        """Return the result of os.stat on the path, or None if it cannot be accessed"""
        if not self._stat_done:
            try:
                self._stat = os.stat(self.path)
            except (OSError, ValueError):
                self._stat = None
            self._stat_done = True
        return self._stat

    def exists(self) -> bool:
        return self.stat() is not None

    def is_dir(self) -> bool:
        st = self.stat()
        return st is not None and stat_module.S_ISDIR(st.st_mode)

    def is_file(self) -> bool:
        st = self.stat()
        return st is not None and stat_module.S_ISREG(st.st_mode)

    def size(self) -> int | None:
        st = self.stat()
        return None if st is None else st.st_size

    def head(self, nbytes: int = _DEFAULT_HEAD_BYTES) -> bytes:
        """Return up to *nbytes* bytes from the start of the file (fewer if the file is shorter).

        Returns an empty bytes object if the path is not a readable regular file. The first call reads at least
        4096 bytes, so that later calls asking for a few bytes are served from memory.
        """
        if self._head is None or (len(self._head) < nbytes and len(self._head) == self._head_request):
            self._head_request = max(nbytes, _DEFAULT_HEAD_BYTES)
            self._head = b""
            if self.is_file():
                try:
                    with open(self.path, "rb") as f:
                        self._head = f.read(self._head_request)
                except OSError:
                    pass
        return self._head[:nbytes]

    def head_decompressed(self, nbytes: int = _DEFAULT_HEAD_BYTES) -> bytes:
        """Like :meth:`head`, but transparently decompresses the file if its name ends in ``.gz``"""
        if not self.path.name.endswith(".gz"):
            return self.head(nbytes)
        if self._head_decompressed is None or len(self._head_decompressed) < nbytes:
            self._head_decompressed = b""
            if self.is_file():
                try:
                    with gzip.open(self.path, "rb") as f:
                        self._head_decompressed = f.read(max(nbytes, _DEFAULT_HEAD_BYTES))
                except (OSError, EOFError):
                    pass
        return self._head_decompressed[:nbytes]

    def is_hdf5(self) -> bool:
        """Return True if the path is an HDF5 file.

        The HDF5 signature is first sought in the bytes already read by :meth:`head`; only if it is not found
        there (and the file is large enough to have a bigger user block) is h5py consulted, at most once.
        """
        if self._is_hdf5 is None:
            self._is_hdf5 = False
            if self.is_file():
                head = self.head()
                if any(head[o:o + len(_HDF5_SIGNATURE)] == _HDF5_SIGNATURE for o in _HDF5_SIGNATURE_OFFSETS_IN_HEAD):
                    self._is_hdf5 = True
                elif len(head) == self._head_request and h5py is not None:
                    # file is longer than what we have read; may have a large user block
                    try:
                        self._is_hdf5 = bool(h5py.is_hdf5(self.path))
                    except (OSError, ValueError):
                        pass
        return self._is_hdf5

    def hdf5(self):
        """Return an open, read-only h5py.File for this path, or None if it is not a readable HDF5 file.

        The file is opened at most once and remains open until the owning :class:`ProbeCache` is closed. Callers
        must not close it themselves. To keep the handle beyond the lifetime of the cache, use
        :meth:`ProbeCache.adopt_hdf5`.
        """
        if not self._hdf5_done:
            self._hdf5_done = True
            if h5py is not None and self.is_hdf5():
                try:
                    self._hdf5 = self.cache._hdf5_opener(self.path)
                    self._hdf5_adoptable = True
                except OSError:
                    # HDF5 refuses to open a file twice with inconsistent locking flags, so if the file is already
                    # open elsewhere (e.g. by a halo catalogue that uses locking=False), try the alternative.
                    # A file opened this way is not suitable for adoption by a loader.
                    try:
                        self._hdf5 = h5py.File(self.path, "r", locking=False)
                        self._hdf5_adoptable = False
                    except OSError:
                        self._hdf5 = None
        return self._hdf5

    def listdir(self) -> dict[str, os.DirEntry] | None:
        """Return a dictionary mapping names to os.DirEntry objects for the contents of this directory.

        The dictionary preserves the order returned by the operating system (as for :func:`os.listdir`).
        DirEntry objects can usually report ``is_dir()`` and ``is_file()`` without a further ``stat``.
        Returns None if this path is not a readable directory.
        """
        if not self._listing_done:
            self._listing_done = True
            try:
                with os.scandir(self.path) as it:
                    self._listing = {e.name: e for e in it}
            except (OSError, ValueError):
                self._listing = None
        return self._listing

    def describe(self) -> str:
        """Return a short human-readable description of what is known about this path, for error messages"""
        if not self.exists():
            return "path does not exist"
        if self.is_dir():
            return "path is a directory"
        if self.is_hdf5():
            f = self.hdf5()
            if f is not None:
                keys = list(f.keys())
                if len(keys) > 10:
                    keys = keys[:10] + ["..."]
                return "file is HDF5 with top-level entries: " + ", ".join(keys)
            return "file is HDF5 but could not be opened"
        return "file is %d bytes and does not appear to be HDF5" % self.size()

    def _close(self):
        if self._hdf5 is not None:
            try:
                self._hdf5.close()
            except Exception:
                pass
            self._hdf5 = None


def _default_hdf5_opener(path):
    return h5py.File(path, "r")


class ProbeCache:
    """A cache of :class:`FileProbe` objects, scoped to a single attempt at identifying a file format.

    Use as a context manager; on exit, any HDF5 files opened by probes (and not adopted) are closed.

    >>> with ProbeCache() as cache:
    ...     probe = cache.probe("snapshot.hdf5")
    ...     if probe.is_hdf5() and "Header" in probe.hdf5():
    ...         ...
    """

    def __init__(self, hdf5_opener=None):
        """Create a new cache.

        Parameters
        ----------
        hdf5_opener : callable, optional
            A function taking a path and returning an open, read-only h5py.File. If not specified, files are opened
            with h5py's defaults. Specifying an opener allows the eventual loader to adopt the open file (see
            :meth:`adopt_hdf5`) without needing to reopen it with different settings.
        """
        self._probes: dict[str, FileProbe] = {}
        self._hdf5_opener = hdf5_opener or _default_hdf5_opener

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def probe(self, path) -> FileProbe:
        """Return the (shared) probe for the given path"""
        if isinstance(path, FileProbe):
            return path
        path = pathlib.Path(path)
        key = os.fspath(path)
        p = self._probes.get(key)
        if p is None:
            p = FileProbe(path, self)
            self._probes[key] = p
        return p

    def exists(self, path) -> bool:
        return self.probe(path).exists()

    def is_dir(self, path) -> bool:
        return self.probe(path).is_dir()

    def is_file(self, path) -> bool:
        return self.probe(path).is_file()

    def listdir(self, path) -> list[str]:
        """Return the names in the given directory (as :func:`os.listdir`), or raise OSError if it is not one"""
        listing = self.probe(path).listdir()
        if listing is None:
            raise FileNotFoundError(f"Cannot list directory {path}")
        return list(listing.keys())

    def glob(self, pattern) -> list[str]:
        """Equivalent to :func:`glob.glob` for patterns whose wildcards are only in the final path component.

        The directory listing is cached, so that multiple patterns in the same directory need only one listing.
        Other patterns are passed through to :func:`glob.glob`.
        """
        pattern = os.fspath(pattern)
        dirname, basename = os.path.split(pattern)
        if glob.has_magic(dirname) or not glob.has_magic(basename):
            return glob.glob(pattern)

        listing = self.probe(dirname or os.curdir).listdir()
        if listing is None:
            return []
        names = listing.keys()
        if not basename.startswith("."):
            # follow glob.glob in not matching hidden files unless the pattern explicitly asks for them
            names = [n for n in names if not n.startswith(".")]
        return [os.path.join(dirname, n) for n in fnmatch.filter(names, basename)]

    def adopt_hdf5(self, path, mode="r"):
        """Remove and return the open h5py.File for *path*, if the cache has one and *mode* is read-only.

        Once adopted, the file is no longer closed when the cache closes; responsibility passes to the caller.
        Returns None if there is no suitable handle, in which case the caller should open the file itself.
        """
        if mode != "r":
            return None
        p = self._probes.get(os.fspath(pathlib.Path(path)))
        if p is None or p._hdf5 is None or not p._hdf5_adoptable:
            return None
        f = p._hdf5
        p._hdf5 = None
        p._hdf5_done = False
        return f

    def close(self):
        """Close any HDF5 files that remain open (other than those adopted)"""
        for p in self._probes.values():
            p._close()


def _defining_index(cls, name):
    for i, k in enumerate(cls.__mro__):
        if name in k.__dict__:
            return i
    return len(cls.__mro__)


def legacy_can_load_overrides_probe(cls) -> bool:
    """Return True if *cls* defines ``_can_load`` more specifically than ``_can_load_from_probe``.

    This allows subclasses written before the introduction of probes (which override only ``_can_load``) to keep
    working, even if they derive from a class that has been converted to use ``_can_load_from_probe``.
    """
    return _defining_index(cls, "_can_load") < _defining_index(cls, "_can_load_from_probe")
