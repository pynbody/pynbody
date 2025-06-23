"""Pure Python implementation of FortranFile as a fallback for the Cython version.

This module provides the same API as the Cython implementation but using pure Python
and the standard library. This can help isolate Windows-specific issues that may be
related to Cython's C integration.
"""

import struct
from typing import Any, Dict, List, Tuple, Union

import numpy as np


class FortranFile:
    """Pure Python implementation of FortranFile with the same API as the Cython version.
    
    This class provides facilities to interact with files written in fortran-record format.
    Since this is a non-standard file format, whose contents depend on the compiler and 
    the endianness of the machine, caution is advised. This code assumes that the record 
    header is written as a 32bit (4byte) signed integer. The code also assumes that the 
    records use the system's local endianness.
    """
    
    def __init__(self, fname: str):
        """Initialize the FortranFile.
        
        Parameters
        ----------
        fname : str
            The filename to open
        """
        self._fname = fname
        self._file = None
        self._closed = True
        
        try:
            self._file = open(fname, 'rb')
            self._closed = False
        except OSError as e:
            raise OSError(f"Cannot open '{fname}': {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close the file descriptor.
        
        This method has no effect if the file is already closed.
        """
        if not self._closed and self._file is not None:
            self._file.close()
            self._closed = True
            self._file = None
    
    def __del__(self):
        self.close()
    
    def _check_open(self):
        """Check if file is open and raise appropriate exception if not."""
        if self._closed or self._file is None:
            raise ValueError("I/O operation on closed file.")
    
    def tell(self) -> int:
        """Return current stream position."""
        self._check_open()
        return self._file.tell()
    
    def seek(self, pos: int, whence: int = 0) -> int:
        """Change stream position.
        
        Parameters
        ----------
        pos : int
            Change the stream position to the given byte offset. The offset is
            interpreted relative to the position indicated by whence.
        whence : int
            Determine how pos is interpreted. Can be any of:
            * 0 -- start of stream (the default); offset should be zero or positive
            * 1 -- current stream position; offset may be negative  
            * 2 -- end of stream; offset is usually negative
            
        Returns
        -------
        pos : int
            The new absolute position.
        """
        self._check_open()
        if whence < 0 or whence > 2:
            raise ValueError(f"whence argument can be 0, 1, or 2. Got {whence}")
        
        self._file.seek(pos, whence)
        return self._file.tell()
    
    def skip(self, n: int = 1) -> int:
        """Skip records.
        
        Parameters
        ----------
        n : int
            The number of records to skip
            
        Returns
        -------
        value : int
            Returns 0 on success.
        """
        self._check_open()
        
        for _ in range(n):
            # Read record header
            header_data = self._file.read(4)
            if len(header_data) != 4:
                raise OSError("Failed to read record header")
            
            s1 = struct.unpack('<i', header_data)[0]
            
            # Skip record data
            self._file.seek(s1, 1)  # 1 = SEEK_CUR
            
            # Read record footer
            footer_data = self._file.read(4)
            if len(footer_data) != 4:
                raise OSError("Failed to read record footer")
            
            s2 = struct.unpack('<i', footer_data)[0]
            
            if s1 != s2:
                raise OSError(f'Sizes do not agree in the header and footer for '
                             f'this record - check header dtype. Got {s1} and {s2}')
        
        return 0
    
    def get_size(self, dtype: str) -> int:
        """Return the size of an element given its datatype.
        
        Parameters
        ----------
        dtype : str
           The dtype, see note for details about the values of dtype.
           
        Returns
        -------
        size : int
           The size in byte of the dtype
           
        Note
        ----
        See https://docs.python.org/3.5/library/struct.html#format-characters
        for details about the formatting characters.
        """
        if dtype == 'i':
            return 4
        elif dtype == 'd':
            return 8
        elif dtype == 'f':
            return 4
        elif dtype == 'q':
            return 8
        elif dtype == 'l':
            raise ValueError("FortranFile does not support 'l' (long) as it is not platform-independent. Use 'i' for int32 or 'q' for int64.")
        else:
            # Fallback to numpy to compute the size
            return np.dtype(dtype).itemsize
    
    def read_vector(self, dtype: str) -> np.ndarray:
        """Reads a record from the file and return it as numpy array.
        
        Parameters
        ----------
        dtype : str
            This is the datatype (from the struct module) that we should read.
            
        Returns
        -------
        data : numpy.ndarray
            This is the vector of values read from the file.
        """
        self._check_open()
        
        # Read record header
        header_data = self._file.read(4)
        if len(header_data) != 4:
            raise OSError("Failed to read record header")
        
        s1 = struct.unpack('<i', header_data)[0]
        
        # Determine element size and count
        size = self.get_size(dtype)
        
        # Check record is compatible with data type
        if s1 % size != 0:
            raise ValueError(f'Size obtained ({s1}) does not match with the expected '
                           f'size ({size}) of multi-item record')
        
        count = s1 // size
        
        # Read the data
        data_bytes = self._file.read(s1)
        if len(data_bytes) != s1:
            raise OSError("Failed to read record data")
        
        # Convert to numpy array
        if dtype == 'i':
            fmt = f'<{count}i'
        elif dtype == 'd':
            fmt = f'<{count}d'
        elif dtype == 'f':
            fmt = f'<{count}f'
        elif dtype == 'q':
            fmt = f'<{count}q'  # 'q' is 8-byte signed int
        else:
            # For other dtypes, use numpy directly
            data = np.frombuffer(data_bytes, dtype=np.dtype(dtype))
            # Read footer after creating array
            footer_data = self._file.read(4)
            if len(footer_data) != 4:
                raise OSError("Failed to read record footer")
            s2 = struct.unpack('<i', footer_data)[0]
            if s1 != s2:
                raise OSError('Sizes do not agree in the header and footer for '
                             'this record - check header dtype')
            return data
        
        # Unpack using struct for standard types
        values = struct.unpack(fmt, data_bytes)
        data = np.array(values, dtype=dtype)
        
        # Read record footer
        footer_data = self._file.read(4)
        if len(footer_data) != 4:
            raise OSError("Failed to read record footer")
        
        s2 = struct.unpack('<i', footer_data)[0]
        
        if s1 != s2:
            raise OSError('Sizes do not agree in the header and footer for '
                         'this record - check header dtype')
        
        return data
    
    def peek_record_size(self) -> int:
        """This function returns the size of the next record and then rewinds 
        the file to the previous position.
        
        Returns
        -------
        int
            Number of bytes in the next record
        """
        self._check_open()
        
        pos = self.tell()
        
        # Read record size
        header_data = self._file.read(4)
        if len(header_data) != 4:
            raise OSError("Failed to read record size")
        
        s1 = struct.unpack('<i', header_data)[0]
        
        # Rewind
        self.seek(pos)
        
        return s1
    
    def read_int(self) -> int:
        """Reads a single int32 from the file and return it.
        
        Returns
        -------
        data : int
            The value.
        """
        self._check_open()
        
        # Read record header
        header_data = self._file.read(4)
        if len(header_data) != 4:
            raise OSError("Failed to read record header")
        
        s1 = struct.unpack('<i', header_data)[0]
        
        if s1 != 4:  # sizeof(int32)
            raise ValueError(f'Size obtained ({s1}) does not match with the expected '
                           f'size (4) of record')
        
        # Read the int32 data
        data_bytes = self._file.read(4)
        if len(data_bytes) != 4:
            raise OSError("Failed to read record data")
        
        data = struct.unpack('<i', data_bytes)[0]
        
        # Read record footer
        footer_data = self._file.read(4)
        if len(footer_data) != 4:
            raise OSError("Failed to read record footer")
        
        s2 = struct.unpack('<i', footer_data)[0]
        
        if s1 != s2:
            raise OSError('Sizes do not agree in the header and footer for '
                         'this record - check header dtype')
        
        return data
    
    def read_int64(self) -> int:
        """Reads a single int64 from the file and return it.
        
        Returns
        -------
        data : int
            The value.
        """
        self._check_open()
        
        # Read record header
        header_data = self._file.read(4)
        if len(header_data) != 4:
            raise OSError("Failed to read record header")
        
        s1 = struct.unpack('<i', header_data)[0]
        
        if s1 != 8:  # sizeof(int64)
            raise ValueError(f'Size obtained ({s1}) does not match with the expected '
                           f'size (8) of record')
        
        # Read the int64 data
        data_bytes = self._file.read(8)
        if len(data_bytes) != 8:
            raise OSError("Failed to read record data")
        
        data = struct.unpack('<q', data_bytes)[0]
        
        # Read record footer
        footer_data = self._file.read(4)
        if len(footer_data) != 4:
            raise OSError("Failed to read record footer")
        
        s2 = struct.unpack('<i', footer_data)[0]
        
        if s1 != s2:
            raise OSError('Sizes do not agree in the header and footer for '
                         'this record - check header dtype')
        
        return data
    
    def read_int32_or_64(self) -> int:
        """Reads a single int32 or int64 (based on length of record) from the
        file and return it as an int64.
        
        Returns
        -------
        data : int
            The value.
        """
        self._check_open()
        
        # Read record header
        header_data = self._file.read(4)
        if len(header_data) != 4:
            raise OSError("Failed to read record header")
        
        s1 = struct.unpack('<i', header_data)[0]
        
        if s1 == 4:  # int32
            data_bytes = self._file.read(4)
            if len(data_bytes) != 4:
                raise OSError("Failed to read int32 data")
            data = struct.unpack('<i', data_bytes)[0]
        elif s1 == 8:  # int64
            data_bytes = self._file.read(8)
            if len(data_bytes) != 8:
                raise OSError("Failed to read int64 data")
            data = struct.unpack('<q', data_bytes)[0]
        else:
            raise ValueError(f'Size obtained ({s1}) does not match with the expected '
                           f'size (4 or 8) of record')
        
        # Read record footer
        footer_data = self._file.read(4)
        if len(footer_data) != 4:
            raise OSError("Failed to read record footer")
        
        s2 = struct.unpack('<i', footer_data)[0]
        
        if s1 != s2:
            raise OSError('Sizes do not agree in the header and footer for '
                         'this record - check header dtype')
        
        return data
    
    def read_attrs(self, attrs) -> Dict[str, Any]:
        """This function reads from that file according to a definition of attributes, 
        returning a dictionary.
        
        Fortran unformatted files provide total bytesize at the beginning and end of a 
        record. By correlating the components of that record with attribute names, we 
        construct a dictionary that gets returned.
        
        Parameters
        ----------
        attrs : iterable of iterables
            This object should be an iterable of one of the formats:
            [ (attr_name, count, struct type), ... ].
            [ ((name1,name2,name3),count, vector type]
            [ ((name1,name2,name3),count, 'type type type']
            [ (attr_name, count, struct type, optional)]

            `optional` : boolean.
                If True, the attribute can be stored as an empty Fortran record.
                
        Returns
        -------
        values : dict
            This will return a dict of iterables of the components of the values in
            the file.
        """
        self._check_open()
        
        data = {}
        
        for a in attrs:
            if len(a) == 3:
                key, n, dtype = a
                optional = False
            else:
                key, n, dtype, optional = a
                
            if n == 1:
                tmp = self.read_vector(dtype)
                if len(tmp) == 0 and optional:
                    continue
                elif len(tmp) == 1:
                    data[key] = tmp[0]
                else:
                    raise ValueError(f"Expected a record of length {n}, got {len(tmp)} ({key})")
            else:
                tmp = self.read_vector(dtype)
                if (len(tmp) == 0 and optional) or (n == -1):
                    continue
                elif len(tmp) != n:
                    raise ValueError(f"Expected a record of length {n}, got {len(tmp)} ({key})")
                
                if isinstance(key, tuple):
                    # There are multiple keys
                    for ikey in range(n):
                        data[key[ikey]] = tmp[ikey]
                else:
                    data[key] = tmp
        
        return data
    
    def direct_read_vector(self, dtype: str, count: int) -> np.ndarray:
        """Reads directly elements from the file and return it as numpy array.
        
        Parameters
        ----------
        dtype : str
            This is the datatype (from the struct module) that we should read.
        count : int
            Number of elements to read
            
        Returns
        -------
        data : numpy.ndarray
            This is the vector of values read from the file.
        """
        self._check_open()
        
        size = self.get_size(dtype)
        total_bytes = size * count
        
        # Read data directly without Fortran record headers/footers
        data_bytes = self._file.read(total_bytes)
        if len(data_bytes) != total_bytes:
            raise OSError("Failed to read the requested number of elements")
        
        # Convert to numpy array
        if dtype == 'i':
            fmt = f'<{count}i'
            values = struct.unpack(fmt, data_bytes)
            data = np.array(values, dtype=np.int32)
        elif dtype == 'd':
            fmt = f'<{count}d'
            values = struct.unpack(fmt, data_bytes)
            data = np.array(values, dtype=np.float64)
        elif dtype == 'f':
            fmt = f'<{count}f'
            values = struct.unpack(fmt, data_bytes)
            data = np.array(values, dtype=np.float32)
        elif dtype == 'q':
            fmt = f'<{count}q'
            values = struct.unpack(fmt, data_bytes)
            data = np.array(values, dtype=np.int64)
        else:
            # For other dtypes, use numpy directly
            data = np.frombuffer(data_bytes, dtype=np.dtype(dtype))
        
        return data
