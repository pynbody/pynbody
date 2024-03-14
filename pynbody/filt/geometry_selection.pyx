import numpy as np

cimport cython
cimport numpy as np
cimport openmp
from cython.parallel cimport prange
from libc.math cimport atan, pow
from libc.stdlib cimport free, malloc

from .. import config

cdef extern from "geometry_selection.hpp" nogil:
    void wrapfn[T](T & x, T & y, T & z, const T & wrap, const T & wrap_by_two)

ctypedef fused fused_float:
    np.float32_t
    np.float64_t
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sphere(np.ndarray[fused_float, ndim=2] pos_ar,
           np.ndarray[fused_float, ndim=1] cen,
           double r_max, fused_float wrap):
    """OpenMP sphere selection algorithm.

    Returns an array of booleans, True where the distance from
    pos_ar to cen is less than r_max."""

    cdef long i
    cdef long N=len(pos_ar)
    cdef fused_float cx,cy,cz,x,y,z,r2
    cdef fused_float r_max_2
    cdef np.ndarray[np.uint8_t, ndim=1] output = np.empty(len(pos_ar),dtype=np.uint8)
    cdef fused_float wrap_by_two = wrap/2
    cdef int num_threads = config['number_of_threads']

    r_max_2 = r_max*r_max

    assert pos_ar.shape[1]==3
    assert len(cen)==3

    cx = cen[0]
    cy = cen[1]
    cz = cen[2]

    for i in prange(N,nogil=True,schedule='static',num_threads=num_threads):
        x=pos_ar[i,0]-cx
        y=pos_ar[i,1]-cy
        z=pos_ar[i,2]-cz
        if wrap>0:
            wrapfn(x,y,z,wrap, wrap_by_two)
        output[i]=(x*x+y*y+z*z)<r_max_2

    return output