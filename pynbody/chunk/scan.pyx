import numpy as np
cimport numpy as np

def scan_for_next_stop(np.ndarray[np.int64_t, ndim=1, mode="c"] ids not None,
                       np.int64_t offset_start, np.int64_t id_maximum) :
    if len(ids)==0 :
            return 0
    if ids[-1]<=id_maximum :
        return len(ids)
    if ids[0]>id_maximum :
        return 0

    cdef np.int64_t left, right, mid, itr
    left,right,mid,itr = 0,0,0,0

    left = offset_start
    right = len(ids)-1
    mid = (left+right)/2

    while ids[mid-1]>id_maximum or ids[mid]<=id_maximum :
        if ids[mid]<=id_maximum :
            left = mid
        else :
            right = mid-1
        mid = (left+right+1)/2
        itr+=1
        if itr>1000 :
            break

    if itr>1000 :
        return -1
    else :
        return mid
    
   

