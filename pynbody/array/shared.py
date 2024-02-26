"""Support for numpy arrays in shared memory."""
import atexit
import functools
import os
import random
import time
import weakref
from functools import reduce
from multiprocessing import shared_memory as shmem
from threading import Timer

import numpy as np

from ..configuration import config_parser
from . import SimArray

DELAY_BEFORE_CLOSING_SHARED_MEM = float(config_parser.get('shared-array', 'cleanup-delay'))

# weakrefs to all buffers created from the shared memory, so we can close the shared memory when they are garbage collected:
_buf_weakrefs: list[weakref.ref[memoryview]] = []
_mem_weakrefs_to_unlink: list[weakref.ref[shmem.SharedMemory]] = []
_pending_cleanups: list[Timer] = [] # a list of timers that are waiting to clean up shared memory


class QuietSharedMemory(shmem.SharedMemory):
    """A wrapper around SharedMemory that doesn't print an error message if __del__ is called and mem is still mapped"""
    def __del__(self):
        try:
            shmem.SharedMemory.__del__(self)
        except BufferError:
            # When python exits, there is simply no way to avoid __del__ being called, but the numpy arrays
            # are still alive and so __del__ generates this exception. We can't do anything about it.
            # This isn't really such a problem. When python exits, it will eventually close the memory anyway.
            # The more important thing is that the shared memory gets unlinked, which is handled by our
            # at exit handler below.
            pass

def make_shared_array(dims, dtype, zeros=False, fname=None, create=True):
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
    """
    if fname is None:
        if not create:
            raise ValueError("When opening an existing shared array, fname must be specified")
        random.seed(os.getpid() * time.time())
        fname = "pynbody-" + \
                ("".join([random.choice('abcdefghijklmnopqrstuvwxyz')
                          for _ in range(10)]))

    _ensure_shared_memory_clean()

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

    mem = QuietSharedMemory(fname, create=create, size=int(np.dtype(dtype).itemsize*size))
    _mem_weakrefs_to_unlink.append(weakref.ref(mem))

    buffer = _get_auto_closing_shared_memory_buffer(mem, unlink=create, dtype=dtype, size=size)

    ret_ar = buffer.reshape(dims).view(SimArray)

    ret_ar._shared_fname = fname

    if zero_size:
        ret_ar = ret_ar[1:]

    if zeros:
        ret_ar[:] = 0

    return ret_ar

@atexit.register
def atex():
    import gc
    gc.collect()

    memory_to_unlink = [m() for m in _mem_weakrefs_to_unlink if m() is not None]

    _ensure_shared_memory_clean(stop_tracking=True)

    # if numpy arrays are still alive, the above fails to clean them up.
    # here is the last line of defence against leaving shared memory in place.
    # we don't try to close() because that throws an error when the memory is still
    # 'claimed' by a numpy object
    for m in memory_to_unlink:
        try:
            m.unlink()
        except FileNotFoundError:
            pass


def _ensure_shared_memory_clean(stop_tracking=False):
    """Ensures that all shared memory has been cleaned up. This is called
    automatically by the shared array code, but can be called manually if
    required

    If stop_tracking is True, then we stop tracking the shared memory
    (used when the shared memory is about to be forcibly cleaned up at exit)"""
    global _pending_cleanups

    for t in _pending_cleanups:
        if t.is_alive():
            t.join()
    _pending_cleanups = []

    global _buf_weakrefs
    if stop_tracking:
        _buf_weakrefs = []
    else:
        _buf_weakrefs = [w for w in _buf_weakrefs if w() is not None]


def get_num_shared_arrays():
    """Returns the number of shared arrays currently in use."""
    _ensure_shared_memory_clean()
    global _buf_weakrefs
    return len(_buf_weakrefs)


def _get_auto_closing_shared_memory_buffer(mem: shmem.SharedMemory, unlink=False, dtype=np.float32, size=0, offset=0) -> memoryview:
    """Returns a buffer object for the shared memory, and sets up a callback so that the shared memory will be closed
    when the buffer is garbage collected.

    If *unlink* is True, the shared memory will also be unlinked at the same time."""
    global _buf_weakrefs

    # mem.buf is stored internally to mem, so we won't be able to detect when it's no longer referenced while also
    # keeping the mem alive. So we create our own buffer that we can keep a weakref to, and then we can close the
    # shared memory when our own buffer is garbage collected.
    buf = np.frombuffer(mem.buf, dtype=dtype, count=size, offset=offset)

    def clearup_callback(_):
        def clearup_after_delay():
            try:
                mem.close()
            except BufferError:
                # this shouldn't happen, but might do if DELAY_BEFORE_CLOSING_SHARED_MEM is too short.
                # sleep a while and try again. If it fails again, something is really wrong.
                time.sleep(DELAY_BEFORE_CLOSING_SHARED_MEM)
                mem.close()
            if unlink:
                mem.unlink()

        # the following "callback" approach is necessitated because at the time when clearup is called, the numpy
        # reference to the buffer is still alive, and mem.close() will fail with an exception. It is
        # really ugly, and could in principle fail if the deletion of the numpy object proceeds too slowly.
        # But have tried very hard to find an alternative approach, and this is the only one that works
        # reliably. The other alternative would be a C extension that manually detaches the buffer from
        # the underlying memory, but that would risk segmentation faults if actually the buffer is still in
        # use.
        t = Timer(DELAY_BEFORE_CLOSING_SHARED_MEM, clearup_after_delay)
        t.start()
        _pending_cleanups.append(t) # a record of the timer is kept just in case we want to verify that it has been called

    _buf_weakrefs.append(weakref.ref(buf, clearup_callback))
    # nb the closure of clearup_callback keeps the mem alive, so we don't need to keep a reference to it explicitly
    # (we don't want mem to be garbage collected until the buffer is garbage collected, otherwise
    # its __del__ method spews an error, for similar reasons to those discussed above)

    return buf


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

    assert hasattr(ar_base,'_shared_fname'), "Cannot prepare an array for shared use unless it was created in shared memory"

    ownership_out = transfer_ownership and ar_base._shared_del
    if transfer_ownership:
        ar_base._shared_del = False

    offset = ar.__array_interface__['data'][0] - \
              ar_base.__array_interface__['data'][0]

    return _deconstructed_shared_array((ar.dtype, ar.shape, ar_base._shared_fname, ownership_out,
                                        offset, ar.strides))


def _shared_array_reconstruct(X):
    _ensure_shared_memory_clean()
    dtype, dims, fname, ownership, offset, strides = X

    assert not ownership # transferring ownership not actually supported in current implementation

    mem = QuietSharedMemory(fname)

    size = reduce(np.multiply, dims)
    buf = _get_auto_closing_shared_memory_buffer(mem, unlink=False, dtype=dtype, size=size, offset=offset)

    new_ar = buf.reshape(dims).view(SimArray)

    new_ar.strides = strides
    new_ar._shared_fname = fname

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
    """A decorator for functions returning a new function that is
    suitable for use remotely. Inputs to and outputs from the function
    can be transferred efficiently if they are backed onto shared
    memory. Ownership of any shared memory returned by the function
    is transferred."""

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
    """A replacement for python's in-built map function, sending out tasks
    to the pool and performing the magic required to transport shared memory arrays
    correctly. The function *fn* must be wrapped with the *shared_array_remote*
    decorator to interface correctly with this magic."""

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
