import sys

import numpy as np

cimport cython
cimport numpy as npc
from cython cimport integral

# The following slightly odd repetitiveness is to force Cython to generate
# code for different permutations of the possible integer inputs.
#
# Using just one fused type requires the types to be synchronized across all
# inputs.

ctypedef fused integral_1:
    npc.int32_t
    npc.uint32_t
    npc.int64_t
    npc.uint64_t

ctypedef fused integral_2:
    npc.int32_t
    npc.uint32_t
    npc.int64_t
    npc.uint64_t



@cython.boundscheck(False)
@cython.wraparound(False)
def bridge(npc.ndarray[integral_1, ndim=1] iord_to,
            npc.ndarray[integral_2, ndim=1] iord_from) :

    cdef npc.ndarray[npc.int64_t, ndim=1] output_index
    cdef npc.ndarray[npc.uint8_t, ndim=1] found_match
    cdef npc.int64_t i=0, i_to=0, j=0
    cdef npc.int64_t length = len(iord_from)
    cdef npc.int64_t length_to = len(iord_to)

    found_match = np.empty(length,dtype=np.uint8)
    output_index = np.empty(length,dtype=np.int64)
    for i in range(length) :
        i_from = iord_from[i]
        while i_from>iord_to[i_to] and i_to<length_to :
            i_to+=1
        if i_from==iord_to[i_to] :
            output_index[i] = i_to
            found_match[i] = 1 # true
        else:
            output_index[i] = 0
            found_match[i] = 0 # false

    return (output_index, found_match.astype(np.bool_))


@cython.boundscheck(False)
@cython.wraparound(False)
def match(npc.ndarray[integral_1, ndim=1] group_list_1,
          npc.ndarray[integral_2, ndim=1] group_list_2,
          int imin, int imax):
    cdef size_t i, l = len(group_list_1)
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
