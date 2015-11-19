import numpy as np
cimport numpy as np
cimport cython
cimport libc.math as cmath
from libc.math cimport atan, pow
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange

ctypedef fused fused_float:
    np.float32_t
    np.float64_t

ctypedef fused fused_float_2:
    np.float32_t
    np.float64_t

ctypedef fused fused_float_3:
    np.float32_t
    np.float64_t

ctypedef fused fused_int:
    np.int32_t
    np.int64_t

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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _grid_gen_from_slice(sl, int nx, int ny, int nz, np.ndarray[fused_float, ndim=2] pos):
    cdef float x,y,z
    cdef int i,n, start, stop, step

    start = sl.start
    stop = sl.stop
    step = sl.step

    if step is None:
        step = 1

    with nogil:
        n = start
        i=0
        while n<stop :
            x=n%nx
            y=(n//nx)%ny
            z=(n//(nx*ny))%nz
            pos[i,0]=(<fused_float>(x)+0.5)/nx
            pos[i,1]=(<fused_float>(y)+0.5)/ny
            pos[i,2]=(<fused_float>(z)+0.5)/nz
            i+=1
            n+=step

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _grid_gen_from_indices(np.ndarray[fused_int, ndim=1] ind, int nx, int ny, int nz, np.ndarray[fused_float, ndim=2] pos):
    cdef float x,y,z
    cdef int i,n_i,N=len(ind)


    with nogil:
        for i in range(N):
            n_i = ind[i]
            x=n_i%nx
            y=(n_i//nx)%ny
            z=(n_i//(nx*ny))%nz
            pos[i,0]=(<fused_float>(x)+0.5)/nx
            pos[i,1]=(<fused_float>(y)+0.5)/ny
            pos[i,2]=(<fused_float>(z)+0.5)/nz



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def grid_gen(indices_or_slice,  nx,  ny,  nz, pos=None):
    """Generate the x,y,z grid coordinates in the interval (0,1) for the
    specified indices (relative to the start of a GrafIC file) or slice of the
    file. nx,ny,nz are the number of particles in each dimension (presumably
    the same for all sane cases, but the file format allows for different
    values). If *pos* is not None, copy the results into the array; otherwise
    create a new array for the results and return it."""

    from . import util

    if pos is None:
        pos = np.empty((util.indexing_length(indices_or_slice), 3),dtype=float)

    if isinstance(indices_or_slice, slice):
        _grid_gen_from_slice(indices_or_slice,nx,ny,nz,pos)
    else:
        _grid_gen_from_indices(np.asarray(indices_or_slice),nx,ny,nz,pos)

    return pos

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sum(np.ndarray[fused_float, ndim=1] ar):
    """OpenMP summation algorithm equivalent to numpy.sum(ar)"""
    cdef fused_float v
    cdef long i
    cdef long N=len(ar)
    for i in prange(N,nogil=True,schedule='static'):
        v+=ar[i]
    return v

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sum_if_gt(np.ndarray[fused_float, ndim=1] ar,
                   np.ndarray[fused_float_2, ndim=1] cmp_ar,
                   double cmp_ar_val):
    """OpenMP summation algorithm equivalent to numpy.sum(ar*(cmp_ar>cmp_ar_val))"""
    cdef fused_float v
    cdef long i
    cdef long N=len(ar)
    assert len(cmp_ar)==len(ar)
    for i in prange(N,nogil=True,schedule='static'):
        if cmp_ar[i]>cmp_ar_val:
           v+=(ar[i])
    return v

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sum_if_lt(np.ndarray[fused_float, ndim=1] ar,
                   np.ndarray[fused_float_2, ndim=1] cmp_ar,
                   double cmp_ar_val):
    """OpenMP summation algorithm equivalent to numpy.sum(ar*(cmp_ar<cmp_ar_val))"""
    cdef fused_float v
    cdef long i
    cdef long N=len(ar)
    assert len(cmp_ar)==len(ar)
    for i in prange(N,nogil=True,schedule='static'):
        v+=(ar[i])*(cmp_ar[i]<cmp_ar_val)
    return v

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _sphere_selection(np.ndarray[fused_float, ndim=2] pos_ar,
                     np.ndarray[fused_float, ndim=1] cen,
                     double r_max, double wrap):
    """OpenMP sphere selection algorithm.

    Returns an array of booleans, True where the distance from
    pos_ar to cen is less than r_max."""

    cdef long i
    cdef long N=len(pos_ar)
    cdef fused_float cx,cy,cz,x,y,z,r2
    cdef fused_float r_max_2
    cdef np.ndarray[np.uint8_t, ndim=1] output = np.empty(len(pos_ar),dtype=np.uint8)
    cdef double wrap_by_two = wrap/2

    r_max_2 = r_max*r_max

    assert pos_ar.shape[1]==3
    assert len(cen)==3

    cx = cen[0]
    cy = cen[1]
    cz = cen[2]


    for i in prange(N,nogil=True,schedule='static'):
        x=pos_ar[i,0]-cx
        y=pos_ar[i,1]-cy
        z=pos_ar[i,2]-cz
        if wrap>0:
            if x>wrap_by_two:
                x=x-wrap
            if y>wrap_by_two:
                y=y-wrap
            if z>wrap_by_two:
                z=z-wrap
            if x<-wrap_by_two:
                x=x+wrap
            if y<-wrap_by_two:
                y=y+wrap
            if z<-wrap_by_two:
                z=z+wrap
        output[i]=(x*x+y*y+z*z)<r_max_2
    
    return output


__all__ = ['grid_gen','find_boundaries', 'sum', 'sum_if_gt', 'sum_if_lt',
           '_sphere_selection']
