"""POSIX-specific shared memory implementation.

Uses Python's internal _posixshmem module instead of ctypes calling libc directly.
This is necessary because on macOS, ctypes has issues with shm_open permission handling
that cause "Permission denied" errors when opening existing shared memory segments.
The reason for this is unclear. However, the _posixshmem module (used internally by 
multiprocessing.shared_memory) handles the system calls correctly on all POSIX platforms.
"""

import mmap
import os

import _posixshmem


def create_shared_memory(name, size):
    """Create POSIX shared memory segment using _posixshmem module."""
    
    # Use Python's internal _posixshmem module
    try:
        fd = _posixshmem.shm_open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            mode=0o600
        )
    except OSError as e:
        raise OSError(e.errno, f"Failed to create shared memory segment {name}: {e.strerror}")
    
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
    """Open existing POSIX shared memory segment using _posixshmem module."""
    
    try:
        fd = _posixshmem.shm_open(
            name,
            os.O_RDWR,
            mode=0o600
        )
    except OSError as e:
        from . import SharedArrayNotFound
        raise SharedArrayNotFound(f"No shared memory found with name {name}: {e.strerror}") from None
    
    # Get the size if not provided
    if size is None:
        stat = os.fstat(fd)
        size = stat.st_size
    
    # Memory map the file descriptor
    mapped_mem = mmap.mmap(fd, size)
    os.close(fd)
    
    return mapped_mem


def unlink_shared_memory(name):
    """Unlink POSIX shared memory segment using _posixshmem module."""
    try:
        _posixshmem.shm_unlink(name)
    except OSError as e:
        # Don't raise error if the shared memory doesn't exist
        if e.errno != 2:  # ENOENT
            raise OSError(e.errno, f"Failed to unlink shared memory segment {name}: {e.strerror}")


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
