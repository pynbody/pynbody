"""Access to the bytes of the files HDF5 has open, checked to be the very files HDF5 has open."""

from __future__ import annotations

import os
import threading

from .common import _CannotReadDirectly, _supported_drivers, _UnexpectedData


def _file_identity(h5file) -> tuple[str, tuple]:
    """Return the absolute path of the file HDF5 has open, and its (device, inode), checking that they agree"""
    filename = os.path.abspath(h5file.filename)
    try:
        held = os.fstat(h5file.id.get_vfd_handle())
        on_disk = os.stat(filename)
    except (OSError, TypeError, ValueError, RuntimeError) as e:
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


def _max_files_kept_open() -> int:
    """How many files a BulkReader keeps open at once, leaving most of the process's allowance for everything else"""
    try:
        import resource
        soft_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if soft_limit == resource.RLIM_INFINITY:
            soft_limit = 4096
    except (ImportError, ValueError, OSError):
        soft_limit = 512  # e.g. on Windows, where the C runtime allows 512 by default
    return max(min(256, soft_limit // 4), 0)


class _FileHandle:
    """A file whose bytes are read directly.

    Where it can, this opens the file once and makes positioned reads (os.pread), which threads can share because
    each read names its own position. That matters on parallel filesystems: opening a file costs a round trip to a
    metadata server, and readahead (which makes sequential reads fast) is tracked per open file, so a file opened
    afresh for every read is read far more slowly. Having opened the file, the handle also stays attached to it, so
    that replacing the file on disk cannot change what is read.

    Where os.pread is unavailable (Windows), the handle keeps a small pool of open files: each read borrows one
    (opening another only if all are in use), so there are never more than there are reads in progress at once. If
    *keep_open* is False (when too many files are open already), the file is instead opened for each read. Any file
    opened after this handle was created is checked to be the file HDF5 has open.
    """

    def __init__(self, filename: str, identity: tuple, keep_open: bool = True):
        self.filename = filename
        self.identity = identity
        self._fd = None
        self._idle_files = []  # open descriptors not in use by any read (where os.pread is unavailable)
        self._all_thread_files = []  # every file opened for the pool, so that close() can close them
        self._lock = threading.Lock()
        self._keep_open = keep_open
        if keep_open and hasattr(os, 'pread'):
            self._fd = os.open(filename, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
            try:
                self._check(os.fstat(self._fd))
            except _CannotReadDirectly:
                os.close(self._fd)
                raise
            self.size = os.fstat(self._fd).st_size
        else:
            self.size = os.path.getsize(filename)

    def _check(self, st):
        if (st.st_dev, st.st_ino) != self.identity:
            raise _CannotReadDirectly(f"the file at {self.filename} is not the one HDF5 has open")

    def read(self, offset: int, nbytes: int, into=None):
        """Read *nbytes* at *offset*, into the writable buffer *into* if given, else returning bytes"""
        if self._fd is not None:
            return self._pread(offset, nbytes, into)
        if self._keep_open:
            with self._lock:
                fd = self._idle_files.pop() if self._idle_files else None
            if fd is None:
                fd = self._open_and_verify()
                with self._lock:
                    self._all_thread_files.append(fd)
            try:
                return self._read_from(fd, offset, nbytes, into)
            finally:
                with self._lock:
                    self._idle_files.append(fd)
        fd = self._open_and_verify()
        try:
            return self._read_from(fd, offset, nbytes, into)
        finally:
            os.close(fd)

    def _open_and_verify(self):
        # Raw descriptors rather than Python file objects, so that a pool left to the garbage collector is closed by
        # __del__ without file objects' ResourceWarnings
        fd = os.open(self.filename, os.O_RDONLY | getattr(os, 'O_BINARY', 0))
        try:
            self._verify(os.fstat(fd))
        except _UnexpectedData:
            os.close(fd)
            raise
        return fd

    def _verify(self, st):
        if (st.st_dev, st.st_ino) != self.identity:
            raise _UnexpectedData(f"the file at {self.filename} has been replaced since HDF5 opened it")

    def _pread(self, offset, nbytes, into):
        if into is None:
            parts = []
            got = 0
            while got < nbytes:
                part = os.pread(self._fd, nbytes - got, offset + got)
                if not part:
                    break
                parts.append(part)
                got += len(part)
            data = parts[0] if len(parts) == 1 else b''.join(parts)
        else:
            view = memoryview(into).cast('B')
            got = 0
            while got < nbytes:
                if hasattr(os, 'preadv'):
                    n = os.preadv(self._fd, [view[got:]], offset + got)
                else:
                    part = os.pread(self._fd, nbytes - got, offset + got)
                    n = len(part)
                    view[got:got + n] = part
                if n == 0:
                    break
                got += n
            data = None
        if got != nbytes:
            raise _UnexpectedData(f"expected {nbytes} bytes at offset {offset}, but the file supplied {got}")
        return data

    @staticmethod
    def _read_from(fd, offset, nbytes, into):
        buffer = bytearray(nbytes) if into is None else into
        view = memoryview(buffer).cast('B')
        got = 0
        with open(fd, 'rb', buffering=0, closefd=False) as f:
            f.seek(offset)
            while got < nbytes:
                n = f.readinto(view[got:])
                if not n:
                    break
                got += n
        data = buffer if into is None else None  # (a bytearray, which serves wherever bytes would)
        if got != nbytes:
            raise _UnexpectedData(f"expected {nbytes} bytes at offset {offset}, but the file supplied {got}")
        return data

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        with self._lock:
            for fd in self._all_thread_files:
                os.close(fd)
            self._all_thread_files = []
            self._idle_files = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _read_bytes(file: _FileHandle, offset, nbytes, into=None):
    """Read *nbytes* at *offset* from *file*, into the writable buffer *into* if given, else returning bytes"""
    return file.read(offset, nbytes, into)
