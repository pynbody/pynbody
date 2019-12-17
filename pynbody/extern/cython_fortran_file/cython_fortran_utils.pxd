from libc.stdio cimport FILE
cimport numpy as np

ctypedef np.int32_t INT32_t
ctypedef np.int64_t INT64_t
ctypedef np.float64_t DOUBLE_t

cdef class FortranFile:
    cdef FILE* cfile
    cdef bint _closed

    cpdef INT64_t skip(self, INT64_t n=*) except -1
    cdef INT64_t get_size(self, str dtype)
    cpdef peek_record_size(self)
    cpdef INT32_t read_int(self) except? -1
    cpdef INT64_t read_int64(self) except? -1
    cpdef np.ndarray read_vector(self, str dtype)
    cpdef np.ndarray direct_read_vector(self, str dtype, int len)
    cpdef INT64_t tell(self) except -1
    cpdef INT64_t seek(self, INT64_t pos, INT64_t whence=*) except -1
    cpdef void close(self)
