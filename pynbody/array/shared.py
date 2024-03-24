"""Support for numpy arrays in shared memory."""
import atexit
import functools
import mmap
import os
import random
import signal
import time
from functools import reduce

import numpy as np
import posix_ipc

from ..configuration import config_parser
from . import SimArray

DELAY_BEFORE_CLOSING_SHARED_MEM = float(config_parser.get('shared-array', 'cleanup-delay'))

_owned_shared_memory_names = []

class SharedArrayNotFound(OSError):
    pass

class SharedMemorySimArray(SimArray):
    __slots__ = ['_shared_fname']
    _shared_fname: str

    def __del__(self):
        global _owned_shared_memory_names
        if hasattr(self, '_shared_fname'):
            if self._shared_fname in _owned_shared_memory_names:
                _owned_shared_memory_names.remove(self._shared_fname)
                posix_ipc.unlink_shared_memory(self._shared_fname)

def make_shared_array(dims, dtype, zeros=False, fname=None, create=True,
                      offset = None, strides = None) -> SharedMemorySimArray:
    """Create an array of dimensions *dims* with the given numpy *dtype*.

    Parameters
    ----------

    dims: tuple | int
       The dimensions of the numpy array to create
    dtype:
       The data type to use
    zeros:
        If True, zero the array; otherwise leave uninitialised
    fname: str, optional
        The shared memory name to use. If None, a random name will be created. This is only
        valid with create=True.
    create: bool
        Whether to create the shared array, or to open existing shared memory. If the latter,
        the fname must be specified and the caller is responsible for making sure the dtype
        and dims match the original array.
    offset: int, optional
        The offset into the shared memory to use. This is only valid with create=False
    strides: tuple, optional
        The strides to use. This is only valid with create=False
    """
    if fname is None:
        if not create:
            raise ValueError("When opening an existing shared array, fname must be specified")
        random.seed(os.getpid() * time.time())
        fname = "pynbody-" + \
                ("".join([random.choice('abcdefghijklmnopqrstuvwxyz')
                          for _ in range(10)]))

    if not hasattr(dims, '__len__'):
        dims = (dims,)

    zero_size = False
    if dims[0] == 0:
        zero_size = True
        dims = (1,) + dims[1:]
    if hasattr(dims, '__len__'):
        size = reduce(np.multiply, dims)
    else:
        size = dims

    if create:
        _register_sigterm_handler()
        if offset is not None:
            raise ValueError("Offset only valid when opening an existing shared array")
        if strides is not None:
            raise ValueError("Strides only valid when opening an existing shared array")
        mem = posix_ipc.SharedMemory(fname, posix_ipc.O_CREX, size=int(np.dtype(dtype).itemsize*size))
        _owned_shared_memory_names.append(fname)

    else:
        try:
            mem = posix_ipc.SharedMemory(fname)
        except posix_ipc.ExistentialError:
            raise SharedArrayNotFound(f"No shared memory found with name {fname}") from None

    mapped_mem = mmap.mmap(mem.fd, mem.size)
    mem.close_fd()

    if offset is None:
        offset = 0

    ret_ar = np.frombuffer(mapped_mem, dtype=dtype, count=size, offset=offset).reshape(dims).view(SharedMemorySimArray)
    ret_ar._shared_fname = fname

    if strides:
        ret_ar.strides = strides

    if zero_size:
        ret_ar = ret_ar[1:]

    if zeros:
        ret_ar[:] = 0

    return ret_ar

@atexit.register
def delete_dangling_shared_memory():
    """Ensures that all shared memory has been cleaned up."""
    global _owned_shared_memory_names

    for fname in _owned_shared_memory_names:
        try:
            posix_ipc.unlink_shared_memory(fname)
        except posix_ipc.ExistentialError:
            pass
    _owned_shared_memory_names = []

_sigterm_handler_is_registered = False

def _sigterm_handler(signum, frame):
    delete_dangling_shared_memory()

def _register_sigterm_handler():
    """Registers a handler to clean up shared memory in the event of a SIGTERM signal."""
    global _sigterm_handler_is_registered
    if not _sigterm_handler_is_registered:
        signal.signal(signal.SIGTERM, _sigterm_handler)
        _sigterm_handler_is_registered = True

def get_num_shared_arrays_owned():
    """Returns the number of shared arrays currently owned by this process.

    A shared array is only considered owned if this process is reponsible for unlinking it on exit.
    """
    return len(_owned_shared_memory_names)

class _deconstructed_shared_array(tuple):
    pass

def _shared_array_deconstruct(ar, transfer_ownership=False):
    """Deconstruct an array backed onto shared memory into something that can be
    passed between processes efficiently. If *transfer_ownership* is True,
    also transfers responsibility for deleting the underlying memory (if this
    process has it) to the reconstructing process."""

    assert isinstance(ar, SimArray)
    ar_base = ar
    while isinstance(ar_base.base, SimArray):
        ar_base = ar_base.base

    assert isinstance(ar_base, SharedMemorySimArray), "Cannot prepare an array for shared use unless it was created in shared memory"

    ownership_out = transfer_ownership and ar_base._shared_del
    if transfer_ownership:
        ar_base._shared_del = False

    offset = ar.__array_interface__['data'][0] - \
              ar_base.__array_interface__['data'][0]

    return _deconstructed_shared_array((ar.dtype, ar.shape, ar_base._shared_fname, ownership_out,
                                        offset, ar.strides))


def _shared_array_reconstruct(X):
    dtype, dims, fname, ownership, offset, strides = X

    assert not ownership # transferring ownership not actually supported in current implementation

    new_ar = make_shared_array(dims, dtype, fname=fname, create=False, offset=offset, strides=strides)

    return new_ar


def _recursive_shared_array_deconstruct(input, transfer_ownership=False) :
    """Works through items in input, deconstructing any shared memory arrays
    into transferrable references"""
    output = []
    for item in input:
        if isinstance(item, SimArray):
            item = _shared_array_deconstruct(item, transfer_ownership)
        elif isinstance(item, list) or isinstance(item, tuple):
            item = _recursive_shared_array_deconstruct(item, transfer_ownership)
        output.append(item)
    return output


def _recursive_shared_array_reconstruct(input):
    """Works through items in input, reconstructing any shared memory arrays
    from transferrable references"""
    output = []
    for item in input:
        if isinstance(item, _deconstructed_shared_array):
            item = _shared_array_reconstruct(item)
        elif isinstance(item, list) or isinstance(item, tuple):
            item = _recursive_shared_array_reconstruct(item)
        output.append(item)
    return output


class RemoteKeyboardInterrupt(Exception):
    pass


def shared_array_remote(fn):
    """A decorator for functions that are expected to run on a 'remote' process, accepting shared memory inputs.

    The decorator reconstructs any shared memory arrays that have been 'deconstructed' into a reference by remote_map
    """

    @functools.wraps(fn)
    def new_fn(args, **kwargs):
        try:
            import signal
            assert hasattr(
                args, '__len__'), "Function must be called from remote_map to use shared arrays"
            assert args[0] == '__pynbody_remote_array__', "Function must be called from remote_map to use shared arrays"
            args = _recursive_shared_array_reconstruct(args)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            output = fn(*args[1:], **kwargs)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            return _recursive_shared_array_deconstruct([output], True)[0]
        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            raise RemoteKeyboardInterrupt()
    new_fn.__pynbody_remote_array__ = True

    return new_fn


def remote_map(pool, fn, *iterables):
    """Equivalent to pool.map, but turns any shared memory arrays into a reference that can be passed between processes.

    The function *fn* must be wrapped with the *shared_array_remote* decorator for this to work correctly."""

    assert getattr(fn, '__pynbody_remote_array__',
                   False), "Function must be wrapped with shared_array_remote to use shared arrays"
    iterables_deconstructed = _recursive_shared_array_deconstruct(
        iterables)
    try:
        results = pool.map(fn, list(zip(
            ['__pynbody_remote_array__'] * len(iterables_deconstructed[0]), *iterables_deconstructed)))
    except RemoteKeyboardInterrupt:
        raise KeyboardInterrupt
    return _recursive_shared_array_reconstruct(results)
