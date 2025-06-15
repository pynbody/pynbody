"""POSIX-specific shared memory implementation using posix_ipc."""

import mmap
from functools import reduce
import numpy as np

try:
    import posix_ipc
except ImportError:
    posix_ipc = None


def create_shared_memory(name, size):
    """Create POSIX shared memory segment"""
    if posix_ipc is None:
        raise ImportError("posix_ipc not available")
    
    mem = posix_ipc.SharedMemory(name, posix_ipc.O_CREX, size=size)
    mapped_mem = mmap.mmap(mem.fd, mem.size)
    mem.close_fd()
    return mapped_mem


def open_shared_memory(name, size=None):
    """Open existing POSIX shared memory segment"""
    if posix_ipc is None:
        raise ImportError("posix_ipc not available")
    
    try:
        mem = posix_ipc.SharedMemory(name)
        mapped_mem = mmap.mmap(mem.fd, mem.size)
        mem.close_fd()
        return mapped_mem
    except posix_ipc.ExistentialError:
        from . import SharedArrayNotFound
        raise SharedArrayNotFound(f"No shared memory found with name {name}") from None


def unlink_shared_memory(name):
    """Unlink POSIX shared memory segment"""
    if posix_ipc is not None:
        posix_ipc.unlink_shared_memory(name)


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