import numpy as np
cimport numpy as npc
cimport cython

from cython cimport integral


# The following slightly odd repetitiveness is to force Cython to generate
# code for different permutations of the possible integer inputs.
#
# Using just one fused type requires the types to be synchronized across all
# inputs.

ctypedef fused integral_1:
    npc.int32_t
    npc.int64_t

ctypedef fused integral_2:
    npc.int32_t
    npc.int64_t



@cython.boundscheck(False)
@cython.wraparound(False)
def bridge(npc.ndarray[integral_1, ndim=1] iord_to,
            npc.ndarray[integral_2, ndim=1] iord_from) :

    cdef npc.ndarray[npc.int64_t, ndim=1] output_index
    cdef int i=0, i_to=0, j=0
    cdef int length = len(iord_from)
    cdef int length_to = len(iord_to)


    output_index = np.empty(length_to,dtype=np.int64)

    for i in range(length) :
        i_from = iord_from[i]
        while i_from>iord_to[i_to] and i_to<length_to :
            i_to+=1
        if i_from==iord_to[i_to] :
            output_index[j] = i_to
            j+=1

    return output_index[:j]


@cython.boundscheck(False)
@cython.wraparound(False)
def match(npc.ndarray[integral_1, ndim=1] group_list_1,
          npc.ndarray[integral_2, ndim=1] group_list_2,
          int imin, int imax):
    cdef int i, l = len(group_list_1)
    cdef int g1, g2
    cdef npc.ndarray[npc.int64_t, ndim=2] output = np.zeros((imax+1-imin,imax+1-imin),dtype=np.int64)

    assert len(group_list_2)==l

    with nogil:
        for i in range(l):
            g1 = group_list_1[i]
            g2 = group_list_2[i]
            if g1<=imax and g2<=imax and g1>=imin and g2>=imin :
                output[g1-imin,g2-imin]+=1

    return output
