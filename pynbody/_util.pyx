import numpy as np
cimport numpy as np
cimport cython
cimport libc.math as cmath
from libc.math cimport atan, pow
from libc.stdlib cimport malloc, free

ctypedef np.float32_t image_output_type
np_image_output_type = np.float32
ctypedef np.float64_t input_quantities_type

        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_boundaries(np.ndarray[np.long_t, ndim=1] ordered) :
    """Given an ascending-ordered integer array starting at zero, return an array that gives the first 
    element for each number. For example, calling with [0,0,0,1,2,2,3] should return [0,3,4,6]."""
    
    cdef np.ndarray[np.long_t, ndim=1] boundaries = np.zeros(ordered[len(ordered)-1]+1,dtype=int) - 1
    cdef int n, size = len(ordered), current=ordered[0]-1

    ordered[0] = 0
    with nogil :
        for n in range(size) :
            if current<ordered[n] :
                current = ordered[n]
                boundaries[current] = n

                
    return boundaries

        
