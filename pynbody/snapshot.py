import array
import numpy as np
import copy


class SimSnap(object) :
    """The abstract holder for a simulation snapshot. Derived classes
    should implement

    __init__(self, filename) -> sets up object. May or may not load any actual data.
    _read_array(self, arrayname) -> attempts to load the named array into self._arrays, and returns it
    @staticmethod _can_load(filename) -> determines whether the specified file can be loaded
    ...
    """

    # Particle types:
    dm = 1
    star = 2
    gas = 3
    
    def __init__(self) :
	"""Initialize an empty, zero-length SimSnap."""

	
	self._arrays = {'pos': array.SimArray([]), 'vel': array.SimArray([])}
	self._num_particles = 0
	
	# not sure whether this paradigm is quite right, but it's a possibility
	self._calculated_arrays = {}
	for i,c in enumerate(('x','y','z')) :
	    self._calculated_arrays[c] = lambda s, j=i : s["pos"][:,j]
	    self._calculated_arrays['v'+c] = lambda s, j=i : s["vel"][:,j]
	    # The odd 'j=i' syntax is to value-bind (rather than
	    # object-bind) the dimension
    
    def __getitem__(self, i) :
	"""Given a SimSnap object s, the following should be implemented:

	s[string] -> return array of name string

	s[slice] -> return an object which returns subarrays of this
	object with the specified slicing

	s[numpy.where(...)] -> return an object which represents the
	subset of particles specified by the where clause

	s[abstract_condition] -> return an object which represents
	the particles satisfying abstract_condition, a bit like
	siman filters."""

	
	if type(i) is str :
	    if i in self._calculated_arrays :
		return self._calculated_arrays[i](self)	    
	    try:
		return self._arrays[i]
	    except KeyError :
		try:
		    return self._read_array(i)
		except IOError :
		    raise KeyError(i)

	elif type(i) is slice :
	    return SubSnap(self, i)

	raise TypeError

    def __setitem__(self, name, item) :
	if name in self._calculated_arrays :
	    # remove obscuring calculated array
	    del self._calculated_arrays[name]
	    
	if type(item) is array.SimArray :
	    if len(item)==len(self) :
		self._arrays[name] = item
	    else :
		raise ValueError, "Shape mismatch between simulation and array"
	else :
	    raise TypeError, "Incorrect type"

    def __delitem__(self, name) :
	if name in self._calculated_arrays :
	    del self._calculated_arrays[name]
	else :
	    # Assume to be a normal array and raise exception if not
	    # present
	    del self._arrays[name]
	
    def keys(self) :
	"""Return the accessible array names (in memory)"""
	return self._calculated_arrays.keys()+self._arrays.keys()

    def __len__(self) :
	return self._num_particles


    def _create_array(self, array_name, ndim=1, dtype=None) :
	if ndim==1 :
	    dims = self._num_particles
	else :
	    dims = (self._num_particles, ndim)
	    
	self._arrays[array_name] = np.zeros(dims,dtype=dtype).view(array.SimArray)
	
    
    def _create_arrays(self, array_list, n=1, dtype=None) :
	for array in array_list :
	    self._create_array(array, n, dtype)

    def assert_consistent(self) :
	for array_name in self.keys() :
	    assert len(self[array_name]) == len(self)


    # The following are hack-arounds to allow access to different
    # particle types until the proper family system is developed
    @property
    def dm(self) :
	return self[self._dm_slice]

    @property
    def star(self) :
	return self[self._star_slice]

    @property
    def gas(self) :
	return self[self._gas_slice]

class SubSnap(SimSnap) :
    def __init__(self, parent, _slice) :
	self.parent = parent
	self._slice = _slice
	self._num_particles = len(self["pos"]) 
	
    def __getitem__(self, i) :
	if type(i) is str :
	    return self.parent[i][self._slice]
  

from . import gadget, tipsy
_snap_classes = [gadget.GadgetSnap, tipsy.TipsySnap]


def load(filename, *args) :
    """Loads a file using the appropriate class, returning a SimSnap
    instance."""
    for c in _snap_classes :
	if c._can_load(filename) : return c(filename,*args)
	
    raise RuntimeError("File format not understood")

    
