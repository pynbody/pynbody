from . import array
from . import family, util
from . import decorate
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
    
    def __init__(self) :
	"""Initialize an empty, zero-length SimSnap."""

	
	self._arrays = {'pos': array.SimArray([]), 'vel': array.SimArray([])}
	self._num_particles = 0
	self._family_slice = {}
	self._family_arrays = {}
	self._unifamily = None
	self.filename=""
        self.properties = {}
    
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

	if isinstance(i, str) :
	    self._assert_not_family_array(i)
	    try:
		return self._get_array(i)
	    except KeyError :
		try:
		    return self._read_array(i)
		except IOError :
		    raise KeyError(i)
	
	elif isinstance(i,slice) :
	    return SubSnap(self, i)
	elif isinstance(i, family.Family) :
	    return FamilySubSnap(self, i)
	elif isinstance(i, tuple) or isinstance(i,np.ndarray) :
	    return IndexedSubSnap(self, i)
	

	raise TypeError

   
    def __setitem__(self, name, item) :
	self._assert_not_family_array(name)
	ax = np.asarray(item).view(array.SimArray)
	    
	if name not in self.keys() :
	    # Array needs to be created. We do this through the
	    # private _create_array method, so that if we are operating
	    # within a particle-specific subview we automatically create
	    # a particle-specific array
	    try:
		ndim = len(ax[0])
	    except TypeError :
		ndim = 1
	    self._create_array(name, ndim)

	# Copy in contents if the contents isn't actually pointing to
	# the same data (which will be the case following operations like
	# += etc, since these call __setitem__).
	util.set_array_if_not_same(self[name], ax)
	


    def __delitem__(self, name) :
	self._assert_not_family_array(name)
	del self._arrays[name]

    def __getattribute__(self, name) :
	"""Implements getting particles of a specified family name"""

	for c in family._registry :
	    if c.name==name :
		return self[c]

	return object.__getattribute__(self, name)

    def __setattr__(self, name, val) :
	"""Raise an error if an attempt is made to overwrite
	existing families"""
	if name in family.family_names() : raise AttributeError, "Cannot assign family name "+name
	return object.__setattr__(self, name, val)
	
    def keys(self) :
	"""Return the directly accessible array names (in memory)"""
	return self._arrays.keys()

    def __len__(self) :
	return self._num_particles

    def _create_family_array(self, array_name, family, ndim=1, dtype=None) :
	"""Create a single array of dimension len(self.<family.name>) x ndim,
	with a given numpy dtype, belonging to the specified family"""
	if ndim==1 :
	    dims = self[family]._num_particles
	else :
	    dims = (self[family]._num_particles, ndim)

	new_ar = np.zeros(dims,dtype=dtype).view(array.SimArray)
	
	try:
	    self._family_arrays[array_name][family] = new_ar
	except KeyError :
	    self._family_arrays[array_name] = dict({family : new_ar})

    def _del_family_array(self, array_name, family) :
	del self._family_arrays[array_name][family]
	if len(self._family_arrays[array_name])==0 :
	    del self._family_arrays[array_name]
	    
    def _create_array(self, array_name, ndim=1, dtype=None) :
	"""Create a single array of dimension len(self) x ndim, with
	a given numpy dtype"""
	if ndim==1 :
	    dims = self._num_particles
	else :
	    dims = (self._num_particles, ndim)
	    
	self._arrays[array_name] = np.zeros(dims,dtype=dtype).view(array.SimArray)

    def _get_array(self, name) :
	return self._arrays[name]
    
    def _create_arrays(self, array_list, ndim=1, dtype=None) :
	"""Create a set of arrays of dimension len(self) x ndim, with
	a given numpy dtype."""
	for array in array_list :
	    self._create_array(array, ndim, dtype)

    def assert_consistent(self) :
	"""Consistency checks, currently just checks that the length of all
	stored arrays is consistent."""
	for array_name in self.keys() :
	    assert len(self[array_name]) == len(self)

    def _get_family_slice(self, fam) :
	"""Turn a specified Family object into a concrete slice which describes
	which particles in this SimSnap belong to that family."""
	try :
	    return self._family_slice[fam]
	except KeyError :
	    return slice(0,0 )

    def _assert_not_family_array(self, name) :
	"""Raises a ValueError if the specified array name is connected to
	a family-specific array"""
	if name in self.family_keys() :
	    raise KeyError, "Array "+name+" is a family-level property"
	
    def _get_family_array(self, name, fam) :
	"""Retrieve the array with specified name for the given particle family
	type."""

	try:
	    return self._family_arrays[name][fam]
	except KeyError :
	    raise KeyError("No array "+name+" for family "+fam.name)

    def family_keys(self, fam=None) :
	"""Return list of arrays which are not accessible from this
	view, but can be accessed from family-specific sub-views."""
	if fam is not None :
	    return [x for x in self._family_arrays if fam in self._family_arrays[x]]
	else :
	    return self._family_arrays.keys()

    
    def __repr__(self) :
	if self._filename!="" :
	    return "<SimSnap \""+self._filename+"\" len="+str(len(self))+">"
	else :
	    return "<SimSnap len="+str(len(self))+">"


@decorate.sim_decorator
def put_1d_slices(sim) :
    for i, a in enumerate(["x","y","z"]) :
	sim._arrays[a] = sim._arrays["pos"][:,i]
	sim._arrays["v"+a] = sim._arrays["vel"][:,i]

class SubSnap(SimSnap) :
    """Represent a sub-view of a SimSnap, initialized by specifying a
    slice.  Arrays accessed through __getitem__ are automatically
    sub-viewed using the given slice."""
    
    def __init__(self, base, _slice) :
	self.base = base
	self._unifamily = base._unifamily
	
	if isinstance(_slice,slice) :
	    # Various slice logic later (in particular taking
	    # subsnaps-of-subsnaps) requires having positive
	    # (i.e. start-relative) slices, so if we have been passed a
	    # negative (end-relative) index, fix that now.
	    if _slice.start is None :
		_slice = slice(0, _slice.stop, _slice.step)
	    if _slice.start<0 :
		_slice = slice(len(base)+_slice.start, _slice.stop, _slice.step)
	    if _slice.stop is None :
		_slice = slice(_slice.start, len(base), _slice.step)
	    if _slice.stop<0 :
		_slice = slice(_slice.start, len(base)+_slice.stop, _slice.step)

	    self._slice = _slice

	    descriptor = "["+str(_slice.start)+":"+str(_slice.stop)
	    if _slice.step is not None :
		descriptor+=":"+str(_slice.step)
	    descriptor+="]"

	else :
	    raise TypeError("Unknown SubSnap slice type")
			
	# Work out the length by inspecting the guaranteed-available
	# array 'pos'.
	self._num_particles = len(self["pos"])
	self._descriptor = descriptor
        self.properties = base.properties
	

    def _get_array(self, name) :
	return self.base._get_array(name)[self._slice]

    @property
    def _filename(self) :
	return self.base._filename+":"+self._descriptor

    def keys(self) :
	return self.base.keys()

    def _get_family_slice(self, fam) :
	sl= util.relative_slice(self._slice,
	    util.intersect_slices(self._slice,self.base._get_family_slice(fam),len(self.base)))
	return sl

    def _get_family_array(self, name, fam) :
	base_family_slice = self.base._get_family_slice(fam)
	sl = util.relative_slice(base_family_slice,
				 util.intersect_slices(self._slice, base_family_slice, len(self.base)))
	return self.base._get_family_array(name, fam)[sl]

    def _read_array(self, array_name, fam=None) :
	self.base._read_array(array_name, fam)

    def family_keys(self, fam=None) :
	return self.base.family_keys(fam)

class IndexedSubSnap(SubSnap) :
    """Represents a subset of the simulation particles according
    to an index array."""
    def __init__(self, base, index_array) :
	if isinstance(index_array, tuple) :
	    if isinstance(index_array[0], np.ndarray) :
		index_array = index_array[0]

	# Check the index array is monotonically increasing
	# If not, the family slices cannot be implemented
	if not all(np.diff(index_array)>0) :
	    raise ValueError, "Index array must be monotonically increasing"

	self._slice = index_array
	self._descriptor = "indexed"
	self.properties = base.properties
	self._family_slice = {}
	self._family_indices = {}
	self._num_particles = len(index_array)
	self.base = base
	
	# Find the locations of the family slices
	for fam in family._registry :
	    base_slice = base._get_family_slice(fam)
	    start = util.index_of_first(index_array,base_slice.start)
	    stop = util.index_of_first(index_array, base_slice.stop)
	    new_slice=slice(start,stop)
	    self._family_slice[fam] = new_slice
	    self._family_indices[fam] = np.asarray(index_array[new_slice])-base_slice.start

    def _get_family_slice(self, fam) :
	# A bit messy: jump out the SubSnap inheritance chain
	# and call SimSnap method directly...
	return SimSnap._get_family_slice(self, fam)

    def _get_family_array(self, name, fam) :
	return self.base._get_family_array(name, fam)[self._family_indices[fam]]
    
    

	    

class FamilySubSnap(SubSnap) :
    """Represents a one-family portion of a parent snap object"""
    def __init__(self, base, fam) :
	self.base = base
	self.properties = base.properties
	self._slice = base._get_family_slice(fam)
	self._unifamily = fam
	self._descriptor = ":"+fam.name
	self._num_particles = len(self["pos"])


    def __delitem__(self, name) :
	if name in self.base.keys() :
	    raise ValueError("Cannot delete global simulation property from sub-view")
	elif name in self.base.family_keys(self._unifamily) :
	    self.base._del_family_array(name, self._unifamily)
	    
    def keys(self) :
	global_keys = self.base.keys()
	family_keys = self.base.family_keys(self._unifamily)
	return list(set(global_keys).union(family_keys))

    def family_keys(self, fam=None) :
	# We now define there to be no family-specific subproperties,
	# because all properties can be accessed through standard
	# __setitem__, __getitem__ methods
	return []

    def _get_family_slice(self, fam) :
	if fam is self._unifamily :
	    return slice(0,len(self))
	else :
	    return slice(0,0)

    def _get_array(self, name) :
	try:
	    return self.base._get_array(name)[self._slice]
	except KeyError :
	    return self.base._get_family_array(name, self._unifamily)

    def _create_array(self, array_name, ndim=1, dtype=None) :
	# Array creation now maps into family-array creation in the parent
	self.base._create_family_array(array_name, self._unifamily, ndim, dtype)


    def _create_family_array(self, array_name, family, ndim, dtype) :
	self.base._create_family_array(array_name, family, ndim, dtype)
	
    def _read_array(self, array_name, fam=None) :
	if fam is self._unifamily or fam is None :
	    self.base._read_array(array_name, self._unifamily)
