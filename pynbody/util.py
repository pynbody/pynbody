import numpy as np

def open_(filename, *args) :
    """Open a file, determining from the filename whether to use
    gzip decompression"""
    
    if filename[-3:]==".gz" :
	import gzip
	return gzip.open(filename, *args)
    else :
	return open(filename, *args)

def gcf(a,b) :
    while b>0 : a,b = b,a%b
    return a

def lcm(a,b) :
    return a*b/gcf(a,b)

def intersect_slices(s1, s2, array_length=None) :
    """Given two python slices s1 and s2, return a new slice which
    will extract the data of an array d which is in both d[s1] and
    d[s2].

    Note that it may not be possible to do this without information on
    the length of the array referred to, hence all slices with
    end-relative indexes are first converted into begin-relative
    indexes. This means that the slice returned may be specific to
    the length specified."""

    assert array_length is not None or \
	   (s1.start>=0 and s2.start>=0 and s1.stop>=0 and s2.start>=0)

    s1_start = s1.start
    s2_start = s2.start
    s1_stop = s1.stop
    s2_stop = s2.stop
    s1_step = s1.step
    s2_step = s2.step

    if s1_step==None : s1_step=1
    if s2_step==None : s2_step=1

    assert s1_step>0 and s2_step>0

    
    if s1_start<0 : s1_start = array_length+s1_start
    if s1_start<0 : return slice(0,0)

    if s2_start<0 : s2_start = array_length+s2_start
    if s2_start<0 : return slice(0,0)

    if s1_stop<0 : s1_stop = array_length+s1_stop
    if s1_stop<0 : return slice(0,0)

    if s2_stop<0 : s2_stop = array_length+s2_stop
    if s2_stop<0 : return slice(0,0)


    step = lcm(s1_step, s2_step)


    start = max(s1_start,s2_start)
    stop = min(s1_stop,s2_stop)

    if stop<=start :
	return slice(0,0)
    
    s1_offset = start - s1_start
    s2_offset = start - s2_start
    s1_offset_x = int(s1_offset)
    s2_offset_x = int(s2_offset)

    if s1_step==s2_step and s1_offset%s1_step!=s2_offset%s1_step :
	# slices are mutually exclusive
	return slice(0,0)
    
    # There is surely a more efficient way to do the following, but
    # it eludes me for the moment
    while s1_offset%s1_step!=0 or s2_offset%s2_step!=0 :
	start+=1
	s1_offset+=1
	s2_offset+=1
	if s1_offset%s1_step==s1_offset_x%s1_step and s2_offset%s2_step==s2_offset_x%s2_step :
	    # slices are mutually exclusive
	    return slice(0,0) 
	
    if step==1 : step = None
    
    return slice(start,stop,step)


def relative_slice(s_relative_to, s) :
    """Given a slice s, return a slice s_prime with the property that
    array[s_relative_to][s_prime] == array[s]. Clearly this will
    not be possible for arbitrarily chosen s_relative_to and s, but
    it should be possible for s=intersect_slices(s_relative_to, s_any)
    which is the use case envisioned here (and used by SubSim).
    This code currently does not work with end-relative (i.e. negative)
    start or stop positions."""

    
    assert  (s_relative_to.start>=0 and s.start>=0 and s.stop>=0)

    if s.start==s.stop :
	return slice(0,0,None)
    
    s_relative_to_step = s_relative_to.step if s_relative_to.step is not None else 1
    s_step = s.step if s.step is not None else 1
    
    if (s.start-s_relative_to.start)%s_relative_to_step!=0 :
	raise ValueError, "Incompatible slices"
    if s_step%s_relative_to_step!=0 :
	raise ValueError, "Incompatible slices"
    
    start = (s.start-s_relative_to.start)/s_relative_to_step
    step = s_step/s_relative_to_step
    stop = start+(s_relative_to_step-1+s.stop-s.start)/s_relative_to_step

    if step==1 : step=None
    
    return slice(start,stop,step)


def arrays_are_same(a1, a2) :
    """Returns True if a1 and a2 are numpy views pointing to the exact
    same underlying data; False otherwise."""
    try:
	return a1.__array_interface__['data']==a2.__array_interface__['data'] \
	       and a1.strides==a2.strides 
    except AttributeError :
	return False

def set_array_if_not_same(a_store, a_in) :
    """This routine checks whether a_store and a_in ultimately point to the
    same buffer; if not, the contents of a_in are copied into a_store."""
    if not arrays_are_same(a_store, a_in) :
	a_store[:] = a_in
	

def index_of_first(array, find) :
    """Returns the index to the first element in array
    which satisfies array[index]>=find. The array must
    be sorted in ascending order."""

    
    left = 0
    right = len(array)-1

    if array[left]>=find :
	return 0
    
    if array[right]<find :
	return len(array)

    while right-left>1 :
	mid = (left+right)/2
	if array[mid]>=find :
	    right = mid
	else :
	    left = mid

    
    return right
