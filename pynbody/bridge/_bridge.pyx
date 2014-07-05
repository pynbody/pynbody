cimport numpy as npc
import numpy as np

def bridge(npc.ndarray[npc.int64_t, ndim=1] iord_to,
            npc.ndarray[npc.int64_t, ndim=1] iord_from) :
    
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
        
