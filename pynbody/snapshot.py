from . import array
from . import family, util
from . import filt
from . import halo
import numpy as np
import copy
import weakref


class SimSnap(object) :
    """The abstract holder for a simulation snapshot. Derived classes
    should implement

    __init__(self, filename) -> sets up object. May or may not load any actual data. 
    _read_array(self, arrayname) -> attempts to load the named array into self._arrays
    @staticmethod _can_load(filename) -> determines whether the specified file can be loaded

    @staticmethod derived_quantity(qty) -> calculates a derived quantity, i.e. radius 'r'
    ...
    """

    _derived_quantity_registry = {}
    _calculating = [] # maintains a list of currently-being-calculated lazy evaluations
                      # to prevent circular references

    _decorator_registry = {}
    
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
            return self._get_array(i)
        
	    try:
		return self._get_array(i)
	    except KeyError :
		try:
		    self._read_array(i)
		    return self._get_array(i)
		except IOError :
		    try :
			self._derive_array(i)
			return self._get_array(i)
		    except (ValueError, KeyError) :
			raise KeyError(i)
		    
                    
	elif isinstance(i,slice) :
	    return SubSnap(self, i)
	elif isinstance(i, family.Family) :
	    return FamilySubSnap(self, i)
	elif isinstance(i, tuple) or isinstance(i,np.ndarray) or isinstance(i,filt.Filter) :
	    return IndexedSubSnap(self, i)
	elif isinstance(i, int) or isinstance(i, np.int32) or isinstance(i, np.int64) :
	    return IndexedSubSnap(self, (i,))
	raise TypeError

   
    def __setitem__(self, name, item) :
	if isinstance(name, tuple) or isinstance(name, list) :
	    index = name[1]
	    name = name[0]
	else :
	    index = None
	    
	self._assert_not_family_array(name)
	
	if isinstance(item, array.SimArray) :
	    ax = item
	else :
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
	self._set_array(name, ax, index)
	
    def halos(self, *args) :
	 """Tries to instantiate a halo catalogue object for the given
	 snapshot, using the first available method."""
	 
	 for c in halo._halo_classes :
	     if c._can_load(self) : return c(self, *args)
	     
	 raise RuntimeError("No halo catalogue found")

    def conversion_context(self) :	
	d = {}
	wanted = ['a','h']
	for x in wanted :
	    if self.properties.has_key(x) :
		d[x] = self.properties[x]
	return d
	
	

    def __delitem__(self, name) :
	self._assert_not_family_array(name)
	del self._arrays[name]

    def __getattribute__(self, name) :
	"""Implements getting particles of a specified family name"""

	try:
	    return self[family.get_family(name)]
	except ValueError :
	    pass

	return object.__getattribute__(self, name)

    def __setattr__(self, name, val) :
	"""Raise an error if an attempt is made to overwrite
	existing families"""
	if name in family.family_names() : raise AttributeError, "Cannot assign family name "+name
	return object.__setattr__(self, name, val)
	
    def keys(self) :
	"""Return the directly accessible array names (in memory)"""
	return self._arrays.keys()

    def has_key(self, name) :
	"""Returns True if the array name is accessible (in memory)"""
	return name in self.keys()
    
    def families(self) :
	"""Return the particle families which have representitives in this SimSnap."""
	out = []
	for fam in family._registry :
	    sl = self._get_family_slice(fam)
	    if sl.start!=sl.stop :
		out.append(fam)
	return out

    def transform(self, matrix) :
	"""Transforms the snapshot according to the 3x3 matrix given."""

	pos = self['pos']
	vel = self['vel']
	
	self['pos'] = np.dot(matrix, pos.transpose()).transpose().view(array.SimArray)
	self['vel'] = np.dot(matrix, vel.transpose()).transpose().view(array.SimArray)
	
	# could search for other 3D arrays here too?


    def rotate_x(self, angle):
        """Rotates the snapshot about the current x-axis by 'angle' degrees."""
        angle *= np.pi/180
        self.transform(np.matrix([[1,      0,             0],
                                  [0, np.cos(angle), -np.sin(angle)],
                                  [0, np.sin(angle),  np.cos(angle)]]))

    def rotate_y(self, angle):
        """Rotates the snapshot about the current y-axis by 'angle' degrees."""
        angle *= np.pi/180
        self.transform(np.matrix([[np.cos(angle),    0,   np.sin(angle)],
                                  [0,                1,        0       ],
                                  [-np.sin(angle),   0,   np.cos(angle)]]))

    def rotate_z(self, angle):
        """Rotates the snapshot about the current z-axis by 'angle' degrees."""
        angle *= np.pi/180
        self.transform(np.matrix([[np.cos(angle), -np.sin(angle), 0],
                                  [np.sin(angle),  np.cos(angle), 0],
                                  [      0,             0,        1]]))
    
    
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
	new_ar._sim = weakref.ref(self)
	new_ar._name = array_name
	
	try:
	    self._family_arrays[array_name][family] = new_ar
	except KeyError :
	    self._family_arrays[array_name] = dict({family : new_ar})

    def _del_family_array(self, array_name, family) :
	del self._family_arrays[array_name][family]
	if len(self._family_arrays[array_name])==0 :
	    del self._family_arrays[array_name]

    def _set_family_array(self, name, family, value, index=None) :
	util.set_array_if_not_same(self._family_arrays[name][family],
				   value, index)
	    
    def _create_array(self, array_name, ndim=1, dtype=None) :
	"""Create a single array of dimension len(self) x ndim, with
	a given numpy dtype"""
	if ndim==1 :
	    dims = self._num_particles
	else :
	    dims = (self._num_particles, ndim)

	new_array = np.zeros(dims,dtype=dtype).view(array.SimArray)
	new_array._sim = weakref.ref(self)
	new_array._name = array_name	
	self._arrays[array_name] = new_array
	
    def _get_array(self, name) :
        return self._arrays[name]
        
                
    def _set_array(self, name, value, index=None) :
	util.set_array_if_not_same(self._arrays[name], value, index)

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


    def loadable_keys(self) :
	"""Returns a list of arrays which can be lazy-evaluated or
	lazy-loaded from the underlying file."""
	res = []
	for cl in type(self).__mro__ :
	    if self._derived_quantity_registry.has_key(cl) :
		res+=self._derived_quantity_registry[cl].keys()
	return res
    
    # Methods for derived arrays
    
    @classmethod
    def derived_quantity(cl,fn):
	if not SimSnap._derived_quantity_registry.has_key(cl) :
	    SimSnap._derived_quantity_registry[cl] = {}
	SimSnap._derived_quantity_registry[cl][fn.__name__]=fn	   
        return fn

    def _derive_array(self, name) :
	"""Calculate and store, for this SnapShot, the derivable array 'name'.

	This searches the registry of @X.derived_quantity functions
	for all X in the inheritance path of the current class. The first
	"""
	
	if name in SimSnap._calculating :
	    raise ValueError, "Circular reference in derived quantity"
	else :
	    try:
		SimSnap._calculating.append(name)
		for cl in type(self).__mro__ :
		    if self._derived_quantity_registry.has_key(cl) \
			   and self._derived_quantity_registry[cl].has_key(name) :
			self[name] = SimSnap._derived_quantity_registry[cl][name](self)
			break
	    finally:
		assert SimSnap._calculating[-1]==name
		del SimSnap._calculating[-1]

    def derive(self, name) :
	"""Force a calculation of the derived_quantity specified by name.

	Unlike calling sim[name], which just gets the array, even if
	the array already exists, it is recalculated and re-stored."""

	self._derive_array(name)
	return self[name]
    
    # Methods for snapshot decoration
    
    @classmethod
    def decorator(cl, fn) :
	if not SimSnap._decorator_registry.has_key(cl) :
	    SimSnap._decorator_registry[cl]=[]
	SimSnap._decorator_registry[cl].append(fn)
	return fn


    def _decorate(self) :
	for cl in type(self).__mro__ :
	    if self._decorator_registry.has_key(cl) :
		for fn in self._decorator_registry[cl] :
		    fn(self)
		    
		
@SimSnap.decorator
def put_1d_slices(sim) :
    if not hasattr(sim, '_arrays') :
	return
    for i, a in enumerate(["x","y","z"]) :
	sim._arrays[a] = sim._arrays["pos"][:,i]
	sim._arrays[a]._name = a
	sim._arrays["v"+a] = sim._arrays["vel"][:,i]
	sim._arrays["v"+a]._name = "v"+a

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

    def _set_array(self, name, value, index=None) :
	self.base._set_array(name,value,util.concatenate_indexing(self._slice, index))

    def _set_family_array(self, name, family, value, index=None) :
	fslice = self._get_family_slice(family)
	self.base._set_family_array(name, family, value, util.concatenate_indexing(fslice, index))
	    

    def __delitem__(self, name) :
        # is this the right behaviour?
        raise RuntimeError, "Arrays can only be deleted from the base snapshot"

    def _del_family_array(self, name, family) :
        # is this the right behaviour?
        raise RuntimeError, "Arrays can only be deleted from the base snapshot"
    
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

    def loadable_keys(self) :
	return self.base.loadable_keys()

    def _create_array(self, *args) :
	self.base._create_array(*args)

    def _create_family_array(self, *args) :
	self.base._create_family_array(*args)

class IndexedSubSnap(SubSnap) :
    """Represents a subset of the simulation particles according
    to an index array."""
    def __init__(self, base, index_array) :

	self._descriptor = "indexed"
	
	if isinstance(index_array, filt.Filter) :
	    self._descriptor = index_array._descriptor
	    index_array = index_array.where(base)
	    

	if isinstance(index_array, tuple) :
	    if isinstance(index_array[0], np.ndarray) :
		index_array = index_array[0]
	    else :
		index_array = np.array(index_array)

	# Check the index array is monotonically increasing
	# If not, the family slices cannot be implemented
	if not all(np.diff(index_array)>0) :
	    raise ValueError, "Index array must be monotonically increasing"

	self._slice = index_array
	
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

    """def __setitem__(self, name, item) :
	# This is required because numpy indexing creates a new array
	# rather than a view on the old one. I.e. the data
	# returned from SubSnap._get_array will be a copy, and
	# so this provides a mechaism for mirroring changes back
	# into the main simulation data arrays.
	self.base._get_array(name)[self._slice] = item"""
	
    def _get_family_slice(self, fam) :
	# A bit messy: jump out the SubSnap inheritance chain
	# and call SimSnap method directly...
	return SimSnap._get_family_slice(self, fam)

    def _get_family_array(self, name, fam) :
	return self.base._get_family_array(name, fam)[self._family_indices[fam]]

    def _set_family_array(self, name, family, value, index=None) :
	self.base._set_family_array(name, family, value,
				    util.concatenate_indexing(self._family_indices[family], index)) 
    
    def _create_array(self, *args) :
	self.base._create_array(*args)
	    

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

    def _set_array(self, name, value, index=None) :
	if name in self.base.keys() :
	    self.base._set_array(name, value, util.concatenate_indexing(self._slice, index))
	else :
	    self.base._set_family_array(name, self._unifamily, value, index)



    def _create_family_array(self, array_name, family, ndim, dtype) :
	self.base._create_family_array(array_name, family, ndim, dtype)
	
    def _read_array(self, array_name, fam=None) :
	if fam is self._unifamily or fam is None :
	    self.base._read_array(array_name, self._unifamily)
