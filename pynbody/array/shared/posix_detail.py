"""POSIX-specific shared memory implementation using libc POSIX shared memory routines."""

import ctypes
import errno
import mmap
import os
import sys

try:
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None)  # Load the default C library on macOS
    else:
        libc = ctypes.CDLL("libc.so.6")  # Linux
except OSError:
    # Fallback to loading the default C library
    libc = ctypes.CDLL(None)

def create_shared_memory(name, size):
    """Create POSIX shared memory segment"""
    
    print("create:", name, size)
    
    # Set error return type for shm_open
    libc.shm_open.restype = ctypes.c_int
    libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
    
    # shm_open(name, oflag, mode)
    fd = libc.shm_open(
        ctypes.c_char_p(name.encode('utf-8')),
        ctypes.c_int(os.O_RDWR | os.O_CREAT | os.O_EXCL),  # Use os module constants
        ctypes.c_uint(0o600)
    )
    
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, f"Failed to create shared memory segment {name}: {os.strerror(err)}")
    
    # Set the size of the shared memory object
    try:
        os.ftruncate(fd, size)
    except OSError as e:
        os.close(fd)
        raise OSError(f"Failed to set size of shared memory segment {name}: {e}")
    
    # Memory map the file descriptor
    mapped_mem = mmap.mmap(fd, size)
    os.close(fd)
    
    return mapped_mem


def open_shared_memory(name, size=None):
    """Open existing POSIX shared memory segment"""
    print("open:", name, size)
    # Set error return type for shm_open
    libc.shm_open.restype = ctypes.c_int
    libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
    
    # shm_open(name, oflag, mode) - open existing without O_CREAT
    fd = libc.shm_open(
        ctypes.c_char_p(name.encode('utf-8')),
        ctypes.c_int(os.O_RDWR),  # Just read/write, no create
        ctypes.c_uint(0o600)
    )
    
    if fd == -1:
        err = ctypes.get_errno()
        from . import SharedArrayNotFound
        raise SharedArrayNotFound(f"No shared memory found with name {name}: {os.strerror(err)}") from None
    
    # Get the size if not provided
    if size is None:
        stat = os.fstat(fd)
        size = stat.st_size
    
    # Memory map the file descriptor
    mapped_mem = mmap.mmap(fd, size)
    os.close(fd)
    
    return mapped_mem


def unlink_shared_memory(name):
    """Unlink POSIX shared memory segment"""
    # Set error return type for shm_unlink
    libc.shm_unlink.restype = ctypes.c_int
    libc.shm_unlink.argtypes = [ctypes.c_char_p]
    
    # shm_unlink(name)
    result = libc.shm_unlink(ctypes.c_char_p(name.encode('utf-8')))
    
    if result == -1:
        err = ctypes.get_errno()
        # Don't raise error if the shared memory doesn't exist
        if err != errno.ENOENT:
            raise OSError(err, f"Failed to unlink shared memory segment {name}: {os.strerror(err)}")


def cleanup_all_shared_memory():
    """Clean up all POSIX shared memory segments - handled by individual unlinks"""
    # POSIX shared memory doesn't need global cleanup like Windows
    pass


def get_shared_memory_info(name):
    """Get information about a shared memory segment - not needed for POSIX"""
    return None


def reconstruct_shared_memory(name, dims, dtype):
    """Reconstruct shared memory for cross-process sharing"""
    # POSIX doesn't need size information to open existing shared memory
    return open_shared_memory(name)
