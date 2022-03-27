import numpy as np

cimport cython
cimport libc.math as cmath
cimport numpy as np
cimport openmp
from cython.parallel cimport prange
from libc.math cimport atan, pow
from libc.stdlib cimport free, malloc

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

ctypedef fused fused_int_2:
    np.int32_t
    np.int64_t

ctypedef fused fused_int_3:
    np.int32_t
    np.int64_t

ctypedef fused int_or_float:
    np.float32_t
    np.float64_t
    np.int32_t
    np.int64_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_boundaries(np.ndarray[fused_int, ndim=1] ordered) :
    """Given an ascending-ordered integer array starting at zero, return an array that gives the first
    element for each number. For example, calling with [0,0,0,1,2,2,3] should return [0,3,4,6]."""

    cdef np.ndarray[fused_int, ndim=1] boundaries = np.zeros(ordered[len(ordered)-1]+1,dtype=ordered.dtype) - 1
    cdef int n, size = len(ordered), current=ordered[0]-1

    with nogil :
        for n in range(size) :
            if ordered[n]>=0 and current<ordered[n] :
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

    if sl.step is None:
        step = 1
    else:
        step = sl.step

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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.int64_t search(fused_int a, fused_int_2[:] B,
                       fused_int_3[:] sorter,
                       fused_int_3 ileft, fused_int_3 iright) nogil:
    cdef fused_int_2 b
    cdef fused_int_3 imid
    while ileft <= iright:
        imid = (ileft + iright) // 2
        b = B[sorter[imid]]
        if b < a:
            ileft = imid + 1
        elif b > a:
            iright = imid - 1
        else:
            return imid
    return -1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[ndim=1, dtype=fused_int] binary_search(
        fused_int[:] a, fused_int_2[:] b,
        np.ndarray[fused_int_3, ndim=1] sorter, int num_threads=-1):
    """Search elements of a in b, assuming a, b[sorter] are sorted in increasing order.

    Parameters
    ----------
    a : int array (N, )
    b : int array (M, )
    sorter : int array(M, )
        The input arrays

    num_threads : int, optional
        If greater than zero, use that many parallel threads.


    Returns
    -------
    indices : array(N, )
        An array such that b[indices] == a, for elements found in b.
        Elements of a that cannot be found have the index M.

    Notes
    -----
    The code does *not* check that a and b[sorted] are effectively sorted in increasing order.

    This functions can be used in place of np.searchsorted (note however that the order of the argument differ).
    The algorithm performs particularly well when a is much smaller than b and can be found at close locations in b.

    Algorithm
    ---------
    The algorithm implemented is a binary search algorithm of the elements of a into b. The algorithm consumes elements
    of a from both ends. Since a is sorted, the index of a[0] and a[-1] give boundaries to look for the next elements
    of a in b, resulting in the next binary search being faster.
    If the elements of a are almost contiguous in b, the algorithm scales as N log(N), with N = len(a) instead of
    N log(M) with M = len(b).

    Note that the algorithm performs similarly to np.searchsorted when N~M.
    """

    cdef int Na = len(a), Nb = len(b)

    cdef int ileft=0, iright=Nb - 1
    cdef int i, ii, j, pivot
    cdef int index

    cdef fused_int_3[:] indices = np.empty(Na, dtype=sorter.dtype)

    # HACK: prevent cython from complaining about "buffer source array is read-only"
    cdef bint write_flag = sorter.flags['WRITEABLE']
    sorter.setflags(write=True)

    cdef fused_int_3[:] sorter_mview = sorter

    cdef int ichunk, chunk_size, Nchunk = openmp.omp_get_num_threads(), this_chunk

    if num_threads > 0:
        Nchunk = num_threads
    openmp.omp_set_num_threads(Nchunk)

    if Na % Nchunk == 0:
        chunk_size = Na // Nchunk
    else:
        chunk_size = Na // Nchunk + 1

    for ichunk in prange(Nchunk, nogil=True, chunksize=1, schedule='static', num_threads=Nchunk):
        ileft = 0
        iright = Nb - 1
        this_chunk = min(chunk_size, Na-ichunk*chunk_size)
        pivot = (this_chunk + 1) // 2

        for ii in range(pivot):
            i = ichunk * chunk_size + ii
            j = ichunk * chunk_size + this_chunk - 1 - ii

            index = search(a[i], b, sorter_mview, ileft, iright)
            if index > -1:
                ileft = index
                indices[i] = sorter_mview[index]
            else:
                indices[i] = Nb


            if j > i:
                index = search(a[j], b, sorter_mview, ileft, iright)
                if index > -1:
                    iright = index
                    indices[j] = sorter_mview[index]
                else:
                    indices[j] = Nb

    # Restore write flag on sorter array
    sorter.setflags(write=write_flag)
    return np.asarray(indices)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int is_sorted(int_or_float[:] A):
    """Check whether input is sorted in ascending order.

    Arguments
    ---------
    A : array

    Returns
    -------
    ret : int
        +1 if A is in ascending order
        -1 if A is in descending order
        0 otherwise
    """
    cdef int Na = len(A), i, i0
    cdef int ret = 0

    # Special case for single-valued arrays
    if Na <= 1:
        return 1

    # Iterate until finding two consecutive non-null elements
    i0 = 1
    while i0 < Na:
        if A[i0] != A[0]:
            break
        i0 += 1

    # Special case if array is constant
    if i0 == Na:
        return 1

    if A[i0] >= A[0]:
        for i in range(i0, Na):
            if A[i] < A[i-1]:
                return 0
        return 1
    else:
        for i in range(i0, Na):
            if A[i] > A[i-1]:
                return 0
        return -1

__all__ = ['grid_gen','find_boundaries', 'sum', 'sum_if_gt', 'sum_if_lt',
           '_sphere_selection', 'binary_search', 'is_sorted']
