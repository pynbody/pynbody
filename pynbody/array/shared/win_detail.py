"""Windows-specific shared memory implementation using native Windows APIs."""

import ctypes
import ctypes.wintypes
from ctypes import wintypes
from functools import reduce

import numpy as np

# Windows API constants
FILE_MAP_READ = 0x0004
FILE_MAP_ALL_ACCESS = 0x000F001F
PAGE_READWRITE = 0x04
INVALID_HANDLE_VALUE = -1

# Windows API functions
kernel32 = ctypes.windll.kernel32

CreateFileMappingW = kernel32.CreateFileMappingW
CreateFileMappingW.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, 
                               wintypes.DWORD, wintypes.DWORD, wintypes.LPCWSTR]
CreateFileMappingW.restype = wintypes.HANDLE

OpenFileMappingW = kernel32.OpenFileMappingW
OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
OpenFileMappingW.restype = wintypes.HANDLE

MapViewOfFile = kernel32.MapViewOfFile
MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, 
                          wintypes.DWORD, ctypes.c_size_t]
MapViewOfFile.restype = ctypes.c_void_p

UnmapViewOfFile = kernel32.UnmapViewOfFile
UnmapViewOfFile.argtypes = [ctypes.c_void_p]
UnmapViewOfFile.restype = wintypes.BOOL

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL

GetLastError = kernel32.GetLastError
GetLastError.restype = wintypes.DWORD

GetLastError = kernel32.GetLastError
GetLastError.argtypes = []
GetLastError.restype = wintypes.DWORD

FormatMessageW = kernel32.FormatMessageW
FormatMessageW.argtypes = [wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                          wintypes.DWORD, wintypes.LPWSTR, wintypes.DWORD, ctypes.c_void_p]
FormatMessageW.restype = wintypes.DWORD

# Constants for FormatMessage
FORMAT_MESSAGE_FROM_SYSTEM = 0x00001000
FORMAT_MESSAGE_ALLOCATE_BUFFER = 0x00000100
FORMAT_MESSAGE_IGNORE_INSERTS = 0x00000200

ERROR_ALREADY_EXISTS = 183  # Error code for "already exists"

def get_error_string(error_code):
    """Get a human-readable string for the last Windows error."""
    if error_code == 0:
        return "No error"
    
    # Buffer to hold the error message
    buffer = ctypes.create_unicode_buffer(256)
    
    # Format the error message
    length = FormatMessageW(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        None,  # lpSource
        error_code,  # dwMessageId
        0,  # dwLanguageId (0 = default)
        buffer,  # lpBuffer
        len(buffer),  # nSize
        None  # Arguments
    )
    
    if length == 0:
        return f"Unknown error (code: {error_code})"
    
    # Remove trailing newlines and return
    return buffer.value.rstrip('\r\n')

# Windows-specific: store references to SharedMemory objects
_windows_shared_memory_objects = {}


class WindowsSharedMemory:
    """Windows native shared memory implementation"""
    def __init__(self, name, create=True, size=None):
        self.name = name
        self.size = size
        self._handle = None
        self._view = None
        self._buf = None
        
        if create:
            if size is None:
                raise ValueError("Size must be specified when creating shared memory")

            # Create new shared memory
            self._handle = CreateFileMappingW(
                INVALID_HANDLE_VALUE,  # Use paging file
                None,                  # Default security
                PAGE_READWRITE,        # Read/write access
                0,                     # High-order DWORD of size
                size,                  # Low-order DWORD of size
                name                   # Name
            )

            if self._handle !=0 and GetLastError() == ERROR_ALREADY_EXISTS:
                CloseHandle(self._handle)
                raise OSError(f"Shared memory '{name}' already exists.")
            
            if not self._handle:
                error = GetLastError()
                
                raise OSError(f"Failed to create shared memory '{name}': Error {get_error_string(error)}")
        else:
            # Open existing shared memory
            self._handle = OpenFileMappingW(
                FILE_MAP_ALL_ACCESS,   # Access mode
                False,                 # Don't inherit handle
                name                   # Name
            )
            if not self._handle:
                error = GetLastError()
                raise FileNotFoundError(f"Failed to open shared memory '{name}': Error {get_error_string(error)}")
        
        # Map the view
        self._view = MapViewOfFile(
            self._handle,          # Handle to file mapping
            FILE_MAP_ALL_ACCESS,   # Access mode
            0,                     # High-order DWORD of offset
            0,                     # Low-order DWORD of offset
            size or 0              # Number of bytes to map (0 = entire mapping)
        )
        
        if not self._view:
            error = GetLastError()
            CloseHandle(self._handle)
            raise OSError(f"Failed to map view of shared memory '{name}': Error {error}")
        
        # Create buffer interface
        if size is None:
            # For opening existing memory, we need to determine the size
            # This is a limitation - we'll need to track size separately
            # For now, assume a reasonable default or pass size when opening
            raise ValueError("Size must be known when opening existing shared memory")
        
        self._buf = (ctypes.c_char * size).from_address(self._view)
    
    @property
    def buf(self):
        return self._buf
    
    def close(self):
        if self._view:
            UnmapViewOfFile(self._view)
            self._view = None
        if self._handle:
            CloseHandle(self._handle)
            self._handle = None
    
    def unlink(self):
        # On Windows, shared memory is automatically cleaned up when all handles are closed
        self.close()
    
    def __del__(self):
        self.close()


def create_shared_memory(name, size):
    """Create Windows shared memory segment"""
    mem = WindowsSharedMemory(name, create=True, size=size)
    _windows_shared_memory_objects[name] = {'memory': mem, 'size': size}
    return mem.buf


def open_shared_memory(name, size=None):
    """Open existing Windows shared memory segment"""
    global _windows_shared_memory_objects
    
    # Check if we already have this shared memory object
    if name in _windows_shared_memory_objects:
        mem_info = _windows_shared_memory_objects[name]
        return mem_info['memory'].buf
    else:
        if size is None:
            from . import SharedArrayNotFound
            raise SharedArrayNotFound(f"No cached shared memory found with name {name}. "
                                    "Windows requires size information to open existing shared memory.")
        # Try to open with Windows API directly
        try:
            mem = WindowsSharedMemory(name, create=False, size=size)
            _windows_shared_memory_objects[name] = {'memory': mem, 'size': size}
            return mem.buf
        except (FileNotFoundError, OSError):
            from . import SharedArrayNotFound
            raise SharedArrayNotFound(f"No shared memory found with name {name}")


def unlink_shared_memory(name):
    """Unlink Windows shared memory segment"""
    global _windows_shared_memory_objects
    
    if name in _windows_shared_memory_objects:
        try:
            _windows_shared_memory_objects[name]['memory'].unlink()
            del _windows_shared_memory_objects[name]
        except:
            pass


def cleanup_all_shared_memory():
    """Clean up all Windows shared memory segments"""
    global _windows_shared_memory_objects
    
    for name in list(_windows_shared_memory_objects.keys()):
        try:
            _windows_shared_memory_objects[name]['memory'].unlink()
            del _windows_shared_memory_objects[name]
        except:
            pass
    _windows_shared_memory_objects.clear()


def get_shared_memory_info(name):
    """Get information about a shared memory segment"""
    if name in _windows_shared_memory_objects:
        return _windows_shared_memory_objects[name]
    return None


def reconstruct_shared_memory(name, dims, dtype):
    """Reconstruct shared memory for cross-process sharing"""
    size = int(reduce(np.multiply, dims))
    total_size = int(np.dtype(dtype).itemsize * size)
    return open_shared_memory(name, total_size)
