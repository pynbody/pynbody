
"""
snapshot
========

This module implements the  :class:`~pynbody.snapshot.SimSnap` class which manages and stores snapshot data.
It also implements the :class:`~pynbody.snapshot.SubSnap` class (and relatives) which
represent different views of an existing :class:`~pynbody.snapshot.SimSnap`.

"""

from . import array
from . import family, util
from . import filt
from . import units
from . import config
from . import simdict

import numpy as np
import copy
import weakref
import sys
import hashlib
import time
import warnings
import threading
import re
import gc
import traceback

from units import has_units


class SimSnap(object):
    """The class for managing simulation snapshots.

    For most purposes, SimSnaps should be initialized through
    :func:`~pynbody.load` or :func:`~pynbody.new`.

    For a basic tutorial explaining how to load a file as a SimSnap
    see :doc:`tutorials/data_access`.

    *Getting arrays or subsnaps*

    Once a :class:`SimSnap` object ``f`` is instantiated, it can
    be used in various ways. The most common operation is to
    access something with the code ``f[x]``.  Depending on the
    type of ``x``, various behaviours result:

    - If ``x`` is a string, the array named by ``x`` is returned. If
      no such array exists, the framework attempts to load or
      derive an array of that name (in that order). If this is
      unsuccessful, a `KeyError` is raised.

    - If ``x`` is a python `slice` (e.g. ``f[5:100:3]``) or an array of
      integers (e.g. ``f[[1,5,100,200]]``) a subsnap containing only the
      mentioned particles is returned.

      See :doc:`tutorials/data_access` for more information.

    - If ``x`` is a numpy array of booleans, it is interpreted as a mask and
      a subsnap containing only those particles for which x[i] is True.
      This means that f[condition] is a shortcut for f[np.where(condition)].

    - If ``x`` is a :class:`pynbody.filt.Filter` object, a subsnap
      containing only the particles which pass the filter condition
      is returned.

      See :doc:`tutorials/data_access` for more information.

    - If ``x`` is a :class:`pynbody.family.Family` object, a subsnap
      containing only the particles in that family is returned. In practice
      for most code it is more convenient to write e.g. ``f.dm`` in place of
      the equivalent syntax f[pynbody.family.dm].

    *Getting metadata*

    The property `filename` gives the filename of a snapshot.

    There is also a `properties` dictionary which
    contains further metadata about the snapshot. See :ref:`subsnaps`.
    """

    _derived_quantity_registry = {}

    _dependency_chain = {}
        # key is an array name. Value gives the set of arrays which depend on
        # it.

    _decorator_registry = {}

    _loadable_keys_registry = {}
    _persistent = ["kdtree", "_immediate_cache"]

    # The following will be objects common to a SimSnap and all its SubSnaps
    _inherited = [
        "lazy_off", "lazy_derive_off", "lazy_load_off", "auto_propagate_off",
        "_getting_array_lock", "properties", "_derived_array_track", "_family_derived_array_track",
        "_dependency_track", "_calculating", "immediate_mode", "delay_promotion"]

    # These 3D arrays get four views automatically created, one reflecting the
    # full Nx3 data, the others reflecting Nx1 slices of it
    #
    # TO DO: This should probably be read in from a config file
    _split_arrays = {'pos': ('x', 'y', 'z'),
                     'vel': ('vx', 'vy', 'vz')}

    @classmethod
    def _array_name_1D_to_ND(self, name):
        """Map a 1D array name to a corresponding 3D array name, or return None
        if no such mapping is possible.

        e.g. 'vy' -> 'vel'; 'acc_z' -> 'acc'; 'mass' -> None"""
        for k, v in self._split_arrays.iteritems():
            if name in v:
                return k

        generic_match = re.findall("([A-z]+)_[xyz]$", name)
        if len(generic_match) is 1 and generic_match[0] not in self._split_arrays:
            return generic_match[0]

        return None

    @classmethod
    def _array_name_ND_to_1D(self, array_name):
        """Give the 3D array names derived from a 3D array.

        This routine makes no attempt to establish whether the array
        name passed in should indeed be a 3D array. It just returns
        the 1D slice names on the assumption that it is. This is an
        important distinction between this procedure and the reverse
        mapping as implemented by _array_name_1D_to_ND."""

        if array_name in self._split_arrays:
            array_name_1D = self._split_arrays[array_name]
        else:
            array_name_1D = [array_name+"_"+i for i in 'x', 'y', 'z']

        return array_name_1D

    def __init__(self):
        """Initialize an empty, zero-length SimSnap.

        For most purposes SimSnaps should instead be initialized through
       :func:`~pynbody.load` or :func:`~pynbody.new`.
       """

        self._arrays = {}
        self._num_particles = 0
        self._family_slice = {}
        self._family_arrays = {}
        self._derived_array_track = []
        self._autoconvert = None
        self._family_derived_array_track = {}
        for i in family._registry:
            self._family_derived_array_track[i] = []

        self._calculating = []
            # maintains a list of currently-being-calculated lazy evaluations
                               # to prevent circular references

        self._dependency_track = [
        ]  # getitem automatically adds any requested arrays to sets within
                                    # this list, which allows for automatic calculation of the dependencies
                                    # of derived arrays

        self._persistent_objects = {}

        self._unifamily = None

        self._shared_arrays = False  # If True, when new arrays are created they are in shared memory by default

        self.lazy_off = util.ExecutionControl()
        # use 'with lazy_off :' blocks to disable all hidden/lazy behaviour

        self.lazy_derive_off = util.ExecutionControl()
        # use 'with lazy_derive_off : ' blocks to disable lazy-derivation

        self.lazy_load_off = util.ExecutionControl()
        # use 'with lazy_load_off : ' blocks to disable lazy-loading

        self.auto_propagate_off = util.ExecutionControl()
        # use 'with auto_propagate_off : ' blocks to disable auto-flagging changes
        # (i.e. prevent lazy-evaluated arrays from auto-re-evaluating when their
        # dependencies change)

        self.immediate_mode = util.ExecutionControl()
        # use 'with immediate_mode: ' to always return actual numpy arrays, rather
        # than IndexedSubArrays which point to sub-parts of numpy arrays
        self.immediate_mode.on_exit = lambda: self._clear_immediate_mode()

        self.delay_promotion = util.ExecutionControl()
        # use 'with delay_promotion: ' to prevent any family arrays being promoted
        # into simulation arrays (which can cause confusion because the array returned
        # from create_family_array might have properties you don't expect)

        self.delay_promotion.on_exit = lambda: self._delayed_array_promotions(
        )
        self.__delayed_promotions = []

        self._getting_array_lock = threading.RLock()

        self.properties = simdict.SimDict({})
        self._file_units_system = []

    ############################################
    # THE BASICS: SIMPLE INFORMATION
    ############################################

    @property
    def filename(self):
        return self._filename

    def __len__(self):
        return self._num_particles

    def __repr__(self):
        if self._filename != "":
            return "<SimSnap \""+self._filename+"\" len="+str(len(self))+">"
        else:
            return "<SimSnap len="+str(len(self))+">"

    def families(self):
        """Return the particle families which have representitives in this SimSnap.
        The families are ordered by their appearance in the snapshot."""
        out = []
        start = {}
        for fam in family._registry:
            sl = self._get_family_slice(fam)
            if sl.start != sl.stop:
                out.append(fam)
                start[fam] = (sl.start)
        out.sort(key=start.__getitem__)
        return out

    ############################################
    # THE BASICS: GETTING AND SETTING
    ############################################

    def __getitem__(self, i):
        """Return either a specific array or a subview of this simulation. See
        the class documentation (:class:`SimSnap`) for more information."""

        if isinstance(i, str):
            with self._getting_array_lock:

                if len(self._dependency_track) > 0:
                    self._dependency_track[-1].add(i)

                if i in self.family_keys():
                    if self.is_derived_array(i):
                        del self[i]
                    else:
                        in_fam = []
                        out_fam = []
                        for x in self.families():
                            if i in self[x]:
                                in_fam.append(x)
                            else:
                                out_fam.append(x)
                        # it's possible that we computed this array for the derived
                        # family previously and saved it on disk
                        try:
                            for fam in out_fam:
                                self.__load_array_with_magic(i, fam=fam)
                        except IOError:
                            raise KeyError("""%r is a family-level array for %s. To use it over the whole simulation you need either to delete it first, or create it separately for %s.""" % (
                                i, in_fam, out_fam))

                try:
                    return self._get_array(i)
                except KeyError:
                    if not self.lazy_off:
                        try:
                            if not self.lazy_load_off:
                                self.__load_array_with_magic(i)
                            else:
                                raise IOError

                            if i in self.family_keys():
                                self._promote_family_array(i)

                            return self._get_array(i)

                        except IOError:
                            if not self.lazy_derive_off:
                                self._derive_array(i)
                                return self._get_array(i)

                    # All available methods of getting this array have failed
            raise KeyError("No such array %r" % i)

        elif isinstance(i, slice):
            return SubSnap(self, i)
        elif isinstance(i, family.Family):
            return FamilySubSnap(self, i)
        elif isinstance(i, np.ndarray) and np.issubdtype(np.bool, i.dtype):
            if len(i.shape) > 1 or i.shape[0] > len(self):
                raise ValueError("Incorrect shape for masking array")
            else:
                return self[np.where(i)]
        elif isinstance(i, (list, tuple, np.ndarray, filt.Filter)):
            return IndexedSubSnap(self, i)
        elif isinstance(i, int) or isinstance(i, np.int32) or isinstance(i, np.int64):
            return IndexedSubSnap(self, (i,))

        raise TypeError

    def __setitem__(self, name, item):
        """Set the contents of an array in this snapshot"""
        if self.is_derived_array(name) and not self.auto_propagate_off:
            raise RuntimeError("Derived array is not writable")

        if isinstance(name, tuple) or isinstance(name, list):
            index = name[1]
            name = name[0]
        else:
            index = None

        self._assert_not_family_array(name)

        if isinstance(item, array.SimArray):
            ax = item
        else:
            ax = np.asanyarray(item).view(array.SimArray)

        if name not in self.keys():
            # Array needs to be created. We do this through the
            # private _create_array method, so that if we are operating
            # within a particle-specific subview we automatically create
            # a particle-specific array
            try:
                ndim = len(ax[0])
            except TypeError:
                ndim = 1
            except IndexError:
                ndim = ax.shape[-1] if len(ax.shape) > 1 else 1

            # The dtype will be the same as an existing family array if
            # one exists, or the dtype of the source array we are copying
            dtype = self._get_preferred_dtype(name)
            if dtype is None:
                dtype = getattr(item, 'dtype', None)

            self._create_array(name, ndim, dtype=dtype)

        # Copy in contents if the contents isn't actually pointing to
        # the same data (which will be the case following operations like
        # += etc, since these call __setitem__).
        self._set_array(name, ax, index)

    def __delitem__(self, name):
        if name in self._family_arrays:
            assert name not in self._arrays  # mustn't have simulation-level array of this name
            del self._family_arrays[name]

            for v in self._family_derived_array_track.itervalues():
                if name in v:
                    del v[v.index(name)]

        else:
            del self._arrays[name]
            if name in self._derived_array_track:
                del self._derived_array_track[
                    self._derived_array_track.index(name)]

    def _get_persist(self, hash, name):
        try:
            return self._persistent_objects[hash][name]
        except:
            return None

    def _set_persist(self, hash, name, obj=None):
        if hash not in self._persistent_objects:
            self._persistent_objects[hash] = {}
        self._persistent_objects[hash][name] = obj

    def _clear_immediate_mode(self):
        for k, v in self._persistent_objects.iteritems():
            if '_immediate_cache' in v:
                del v['_immediate_cache']

    def __getattr__(self, name):
        """This function overrides the behaviour of f.X where f is a SimSnap object.

        It serves two purposes; first, it provides the family-handling behaviour
        which makes f.dm equivalent to f[pynbody.family.dm]. Second, it implements
        persistent objects -- properties which are shared between two equivalent SubSnaps."""
        if name in SimSnap._persistent:
            obj = self.ancestor._get_persist(self._inclusion_hash, name)
            if obj:
                return obj

        try:
            return self[family.get_family(name)]
        except ValueError:
            pass

        raise AttributeError("%r object has no attribute %r" % (
            type(self).__name__, name))

    def __setattr__(self, name, val):
        """This function overrides the behaviour of setting f.X where f is a SimSnap object.

        It serves two purposes; first it prevents overwriting of family names (so you can't
        write to, for instance, f.dm). Second, it implements persistent objects -- properties
        which are shared between two equivalent SubSnaps."""
        if name in family.family_names():
            raise AttributeError("Cannot assign family name "+name)

        if name in SimSnap._persistent:
            self.ancestor._set_persist(self._inclusion_hash, name, val)
        else:
            return object.__setattr__(self, name, val)

    def __delattr__(self, name):
        """This function allows persistent objects (as shared between two equivalent SubSnaps)
        to be permanently deleted."""
        if name in SimSnap._persistent:
            obj = self.ancestor._get_persist(self._inclusion_hash, name)
            if obj:
                self.ancestor._set_persist(self._inclusion_hash, name, None)
                try:
                    object.__delattr__(self, name)
                except AttributeError:
                    pass
                return
        object.__delattr__(self, name)

    ############################################
    # DICTIONARY EMULATION FUNCTIONS
    ############################################
    def keys(self):
        """Return the directly accessible array names (in memory)"""
        return self._arrays.keys()

    def has_key(self, name):
        """Returns True if the array name is accessible (in memory)"""
        return name in self.keys()

    def values(self):
        """Returns a list of the actual arrays in memory"""
        x = []
        for k in self.keys():
            x.append(self[k])
        return x

    def items(self):
        """Returns a list of tuples describing the array
        names and their contents in memory"""
        x = []
        for k in self.keys():
            x.append((k, self[k]))
        return x

    def get(self, key, alternative=None):
        """Standard python get method, returns self[key] if
        key in self else alternative"""
        try:
            return self[key]
        except KeyError:
            return alternative

    def iterkeys(self):
        for k in self.keys():
            yield k

    __iter__ = iterkeys

    def itervalues(self):
        for k in self:
            yield self[k]

    def iteritems(self):
        for k in self:
            yield (k, self[k])

    ############################################
    # DICTIONARY-LIKE FUNCTIONS
    # (not in the normal interface for dictionaries,
    # but serving similar purposes)
    ############################################

    def has_family_key(self, name):
        """Returns True if the array name is accessible (in memory) for at least one family"""
        return name in self.family_keys()

    def loadable_keys(self, fam=None):
        """Returns a list of arrays which can be lazy-loaded from
        an auxiliary file."""
        return []

    def derivable_keys(self):
        """Returns a list of arrays which can be lazy-evaluated."""
        res = []
        for cl in type(self).__mro__:
            if cl in self._derived_quantity_registry:
                res += self._derived_quantity_registry[cl].keys()
        return res

    def all_keys(self):
        """Returns a list of all arrays that can be either lazy-evaluated
        or lazy loaded from an auxiliary file."""
        return self.derivable_keys() + self.loadable_keys()

    def family_keys(self, fam=None):
        """Return list of arrays which are not accessible from this
        view, but can be accessed from family-specific sub-views.

        If *fam* is not None, only those keys applying to the specific
        family will be returned (equivalent to self.fam.keys())."""
        if fam is not None:
            return [x for x in self._family_arrays if fam in self._family_arrays[x]]
        else:
            return self._family_arrays.keys()

    ############################################
    # ANCESTRY FUNCTIONS
    ############################################

    def is_ancestor(self, other):
        """Returns true if other is a subview of self"""

        if other is self:
            return True
        elif hasattr(other, 'base'):
            return self.is_ancestor(other.base)
        else:
            return False

    def is_descendant(self, other):
        """Returns true if self is a subview of other"""
        return other.is_ancestor(self)

    @property
    def ancestor(self):
        """The original SimSnap from which this view is derived (potentially self)"""
        if hasattr(self, 'base'):
            return self.base.ancestor
        else:
            return self

    def get_index_list(self, relative_to, of_particles=None):
        """Get a list specifying the index of the particles in this view relative
        to the ancestor *relative_to*, such that relative_to[get_index_list(relative_to)]==self."""

        # Implementation for base snapshot

        if self is not relative_to:
            raise RuntimeError("Not a descendant of the specified simulation")
        if of_particles is None:
            of_particles = np.arange(len(self))

        return of_particles

    ############################################
    # SET-LIKE OPERATIONS FOR SUBSNAPS
    ############################################
    def intersect(self, other, op=np.intersect1d):
        """Returns the set intersection of this simulation view with another view
        of the same simulation"""

        anc = self.ancestor
        if not anc.is_ancestor(other):
            raise RuntimeError("Parentage is not suitable")

        a = self.get_index_list(anc)
        b = other.get_index_list(anc)
        return anc[op(a, b)]

    def union(self, other):
        """Returns the set union of this simulation view with another view
        of the same simulation"""

        return self.intersect(other, op=np.union1d)

    def setdiff(self, other):
        """Returns the set difference of this simulation view with another view
        of the same simulation"""

        return self.intersect(other, op=np.setdiff1d)

    ############################################
    # UNIT MANIPULATION
    ############################################
    def conversion_context(self):
        """Return a dictionary containing a (scalefactor) and h
        (Hubble constant in canonical units) for this snapshot, ready for
        passing into unit conversion functions."""
        d = {}
        wanted = ['a', 'h']
        for x in wanted:
            if x in self.properties:
                d[x] = self.properties[x]
        return d

    def set_units_system(self, velocity = None, distance = None, mass = None, temperature = None) : 
        """Set the unit system for the snapshot by specifying any or
        all of `velocity`, `distance`, `mass` and `temperature`
        units. The units can be given as strings or as pynbody `Unit`
        objects.

        If any of the units are not specified and a previous
        `file_units_system` does not exist, the defaults are used.
        """
        from . import config_parser
        import ConfigParser

        # if the units system doesn't exist (if this is a new snapshot), create one
        if len(self._file_units_system) < 3 : 
            warnings.warn("Previous unit system incomplete -- using defaults")
            self._file_units_system = [units.Unit(x) for x in ('G', '1 kpc', '1e10 Msol')]
        
        else : 
            # we want to change the base units -- so convert to original
            # units first and then set all arrays to new unit system
            self.original_units()

        new_units = []
        for x in [velocity,distance,mass,temperature] : 
            if x is not None : 
                new_units.append(units.Unit(x))

        new_system = list(self._file_units_system)

        for i,x in enumerate(new_system) :
            if x is units.K : continue
            try : 
                d = x.dimensional_project(new_units)
                new_system[i] = reduce(lambda x, y: x*y, [
                          a**b for a, b in zip(new_units, d)])

            except units.UnitsException :
                pass

        # check that the new unit system is linearly independent
        for test in ['kpc','Msol','km s^-1'] : 
            test = units.Unit(test)
            try : 
                test.dimensional_project(new_system)
            except units.UnitsException : 
                raise units.UnitsException("New units are not linearly independent")
        
        self._file_units_system = new_system
        
        # set new units for all known arrays
        for arr_name in self.keys() : 
            arr = self[arr_name]
            # if the array has units, then use the current units, else
            # check if a default dimension for this array exists in
            # the configuration
            if arr.units != units.NoUnit() :
                ref_unit = arr.units
            else : 
                try : 
                    ref_unit = config_parser.get('default-array-dimensions',arr_name) 
                except ConfigParser.NoOptionError : 
                    # give up -- no applicable dimension found
                    continue
                    
            arr.set_units_like(ref_unit)


    def original_units(self):
        """Converts all arrays'units to be consistent with the units of
        the original file."""
        self.physical_units(distance=self.infer_original_units('km'),
                            velocity=self.infer_original_units('km s^-1'),
                            mass=self.infer_original_units('Msol'), persistent=False)

    def _autoconvert_array_unit(self, ar, dims=None, ucut=3):
        """Given an array ar, convert its units such that the new units span
        dims[:ucut]. dims[ucut:] are evaluated in the conversion (so will be things like
        a, h etc).

        If dims is None, use the internal autoconvert state to perform the conversion."""

        if dims is None:
            dims = self.ancestor._autoconvert
        if dims is None:
            return
        if ar.units is not None:
            try:
                d = ar.units.dimensional_project(dims)
            except units.UnitsException:
                return

            new_unit = reduce(lambda x, y: x*y, [
                              a**b for a, b in zip(dims, d[:ucut])])
            if new_unit != ar.units:
                if config['verbose']:
                    print>>sys.stderr, "SimSnap: converting", ar.name, "units from", ar.units, "to", new_unit
                ar.convert_units(new_unit)

    def physical_units(self, distance='kpc', velocity='km s^-1', mass='Msol', persistent=True):
        """
        Converts all array's units to be consistent with the
        distance, velocity, mass basis units specified.

        Base units can be specified using keywords.

        **Optional Keywords**:

           *distance*: string (default = 'kpc')

           *velocity*: string (default = 'km s^-1')

           *mass*: string (default = 'Msol')

           *persistent*: boolean (default = True); apply units change to future lazy-loaded arrays if True

        """

        global config

        dims = [units.Unit(x) for x in distance, velocity, mass, 'a', 'h']

        all = self._arrays.values()
        for x in self._family_arrays:
            all += self._family_arrays[x].values()

        for ar in all:
            self._autoconvert_array_unit(ar.ancestor, dims)

        for k in self.properties.keys():
            v = self.properties[k]
            if isinstance(v, units.UnitBase):
                try:
                    new_unit = v.dimensional_project(dims)
                except units.UnitsException:
                    continue
                new_unit = reduce(lambda x, y: x*y, [
                                  a**b for a, b in zip(dims, new_unit[:3])])
                new_unit *= v.ratio(new_unit, **self.conversion_context())
                self.properties[k] = new_unit

        if persistent:
            self._autoconvert = dims
        else:
            self._autoconvert = None

    def infer_original_units(self, dimensions):
        """Given a unit (or string) `dimensions`, returns a unit with the same
        physical dimensions which is in the unit schema of the current file."""
        dimensions = units.Unit(dimensions)
        d = dimensions.dimensional_project(self._file_units_system+["a", "h"])
        new_unit = reduce(lambda x, y: x*y, [
                          a**b for a, b in zip(self._file_units_system, d)])
        return new_unit

    def _default_units_for(self, array_name):
        """Attempt to construct and return the units for the named array
        on disk, using what we know about the purpose of arrays (in config.ini)
        and the original unit system (via infer_original_units)."""
        array_name = self._array_name_1D_to_ND(array_name) or array_name
        u = units._default_units.get(array_name, None)
        if u is not None:
            u = self.infer_original_units(u)
        return u

    ############################################
    # HALO CATALOGUES
    ############################################
    def halos(self, *args, **kwargs):
        """Tries to instantiate a halo catalogue object for the given
        snapshot, using the first available method (as defined in the
        configuration files)."""

        from . import halo

        for c in config['halo-class-priority']:
            try:
                if c._can_load(self,*args,**kwargs):
                    return c(self, *args, **kwargs)
            except TypeError:
                pass

        for c in config['halo-class-priority']:
            try:
                if c._can_run(self,*args,**kwargs):
                    return c(self, *args, **kwargs)
            except TypeError:
                pass
                
        raise RuntimeError("No halo catalogue found for %r" % str(self))

    ############################################
    # BRIDGES
    ############################################

    def bridge(self, other):
        """Tries to construct a bridge function between this SimSnap
        and another one.

        This function calls :func:`pynbody.bridge.bridge_factory`. For
        more information see :ref:`bridge-tutorial`, or the reference
        documentation for :py:mod:`pynbody.bridge`.

        """
        from . import bridge
        return bridge.bridge_factory(self, other)

    ############################################
    # HELPER FUNCTIONS FOR LAZY LOADING
    ############################################
    def _load_array(self, array_name, fam=None):
        """This function is called by the framework to load an array
        from disk and should be overloaded by child classes.

        If *fam* is not None, the array should be loaded only for the
        specified family.
        """
        raise IOError("No lazy-loading implemented")

    def __load_array_with_magic(self, array_name, fam=None):
        """Calls _load_array for the appropriate subclass, but also attempts to convert
        units of anything that gets loaded and automatically loads the whole ND array
        if this is a subview of an ND array"""
        array_name = self._array_name_1D_to_ND(array_name) or array_name

        # keep a record of every array in existence before load (in case it
        # triggers loading more than we expected, e.g. coupled pos/vel fields
        # etc)
        anc = self.ancestor

        pre_keys = set(anc.keys())

        # the following function builds a dictionary mapping families to a set of the
        # named arrays defined for them.
        fk = lambda: dict([(fami, set([k for k in anc._family_arrays.keys() if fami in anc._family_arrays[k]]))
                           for fami in family._registry])
        pre_fam_keys = fk()

        with self.delay_promotion:
            # delayed promotion is required here, otherwise units get messed up when
            # a simulation array gets promoted mid-way through our loading process.
            #
            # see the gadget unit test, test_unit_persistence
            if fam is not None:
                self._load_array(array_name, fam)
            else:
                try:
                    self._load_array(array_name, fam)
                except IOError:
                    for fam_x in self.families():
                        self._load_array(array_name, fam_x)

            # Find out what was loaded
            new_keys = set(anc.keys())-pre_keys
            new_fam_keys = fk()
            for fami in new_fam_keys:
                new_fam_keys[fami] = new_fam_keys[fami]-pre_fam_keys[fami]

            # If the loader hasn't given units already, try to determine the defaults
            # Then, attempt to convert what was loaded into friendly units
            for v in new_keys:
                if not units.has_units(anc[v]) :
                    anc[v].units = anc._default_units_for(v)
                anc._autoconvert_array_unit(anc[v])
            for f, vals in new_fam_keys.iteritems():
                for v in vals:
                    if not units.has_units(anc[f][v]) :
                        anc[f][v].units = anc._default_units_for(v)
                    anc._autoconvert_array_unit(anc[f][v])

    ############################################
    # VECTOR TRANSFORMATIONS OF THE SNAPSHOT
    ############################################
    def transform(self, matrix, ortho_tol=1.e-8):
        """Transforms the snapshot according to the 3x3 matrix given."""

        # Check that the matrix is orthogonal
        resid = np.dot(matrix, np.asarray(matrix).T) - np.eye(3)
        resid = (resid**2).sum()
        if resid > ortho_tol or resid != resid:
            raise ValueError("Transformation matrix is not orthogonal")
        for x in self.keys():
            ar = self[x]
            if len(ar.shape) == 2 and ar.shape[1] == 3:
                self[x] = np.dot(matrix, ar.transpose()).transpose()

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
                                  [0,                1,        0],
                                  [-np.sin(angle),   0,   np.cos(angle)]]))

    def rotate_z(self, angle):
        """Rotates the snapshot about the current z-axis by 'angle' degrees."""
        angle *= np.pi/180
        self.transform(np.matrix([[np.cos(angle), -np.sin(angle), 0],
                                  [np.sin(angle),  np.cos(angle), 0],
                                  [0,             0,        1]]))

    def wrap(self, boxsize=None):
        """Wraps the positions of the particles in the box to lie between
        [-boxsize/2, boxsize/2].

        If no boxsize is specified, self.properties["boxsize"] is used."""

        if boxsize is None:
            boxsize = self.properties["boxsize"]

        if isinstance(boxsize, units.UnitBase):
            boxsize = float(boxsize.ratio(self[
                            "pos"].units, **self.conversion_context()))

        for coord in "x", "y", "z":
            self[coord][np.where(self[coord] < -boxsize/2)] += boxsize
            self[coord][np.where(self[coord] > boxsize/2)] -= boxsize

    ############################################
    # WRITING FUNCTIONS
    ############################################
    def write(self, fmt=None, filename=None, **kwargs):
        if filename is None and "<" in self.filename:
            raise RuntimeError(
                'Cannot infer a filename; please provide one (use obj.write(filename="filename"))')

        if fmt is None:
            if not hasattr(self, "_write"):
                raise RuntimeError(
                    'Cannot infer a file format; please provide one (e.g. use obj.write(filename="filename", fmt=pynbody.tipsy.TipsySnap)')

            self._write(self, filename, **kwargs)
        else:
            fmt._write(self, filename, **kwargs)

    def write_array(self, array_name, fam=None, overwrite=False, **kwargs):
        """
        Write out the array with the specified name.

        Some of the functionality is available via the
        :func:`pynbody.array.SimArray.write` method, which calls the
        present function with appropriate arguments.

        **Input**

        *array_name* - the name of the array to write

        **Optional Keywords**

        *fam* (None) - Write out only one family; or provide a list to
         write out a set of families.
         """

        # Determine whether this is a write or an update
        if fam is None:
            fam = self.families()

        # It's an update if we're not fully replacing the file on
        # disk, i.e. there exists a family f in self.families() but
        # not in fam for which array_name is loadable
        is_update = any([array_name in self[
                        f].loadable_keys() and f not in fam for f in self.families()])

        if not hasattr(self, "_write_array"):
            raise IOError(
                "The underlying file format class does not support writing individual arrays back to disk. See http://code.google.com/p/pynbody/wiki/WritingFiles for further information.")

        if is_update and not hasattr(self, "_update_array"):
            raise IOError(
                "The underlying file format class does not support partially updating arrays on disk. See http://code.google.com/p/pynbody/wiki/WritingFiles for further information.")

        # It's an overwrite if we're writing over something loadable
        is_overwriting = any([array_name in self[
                             f].loadable_keys() for f in fam])

        if is_overwriting and not overwrite:
            # User didn't specifically say overwriting is OK
            raise IOError(
                "This operation would overwrite existing data on disk. Call again setting overwrite=True if you want to enable this behaviour. See http://code.google.com/p/pynbody/wiki/WritingFiles for further information.")

        if is_update:
            self._update_array(array_name, fam=fam, **kwargs)
        else:
            self._write_array(self, array_name, fam=fam, **kwargs)

    ############################################
    # LOW-LEVEL ARRAY MANIPULATION
    ############################################
    def _get_preferred_dtype(self, array_name):
        """Return the 'preferred' numpy datatype for a named array.

        This is mainly useful when creating family arrays for new families, to be
        sure the datatype chosen matches"""

        if hasattr(self, 'base'):
            return self.base._get_preferred_dtype(array_name)
        elif array_name in self.keys():
            return self[array_name].dtype
        elif array_name in self.family_keys():
            return self._family_arrays[array_name][self._family_arrays[array_name].keys()[0]].dtype
        else:
            return None

    def _create_array(self, array_name, ndim=1, dtype=None, zeros=True, derived=False, shared=None):
        """Create a single snapshot-level array of dimension len(self) x ndim, with
        a given numpy dtype.

        *kwargs*:

          - *ndim*: the number of dimensions for each particle
          - *dtype*: a numpy datatype for the new array
          - *zeros*: if True, zeros the array (which takes a bit of time); otherwise
            the array is uninitialized
          - *derived*: if True, this new array will be flagged as a derived array
            which makes it read-only
          - *shared*: if True, the array will be built on top of a shared-memory array
            to make it possible to access from another process
        """

        # Does this actually correspond to a slice into a 3D array?
        NDname = self._array_name_1D_to_ND(array_name)
        if NDname:
            self._create_array(
                NDname, ndim=3, dtype=dtype, zeros=zeros, derived=derived)
            return

        if ndim == 1:
            dims = self._num_particles
        else:
            dims = (self._num_particles, ndim)

        if shared is None:
            shared = self._shared_arrays

        new_array = array._array_factory(dims, dtype, zeros, shared)
        new_array._sim = weakref.ref(self)
        new_array._name = array_name
        new_array.family = None
        # new_array.set_default_units(quiet=True)
        self._arrays[array_name] = new_array

        if derived:
            if array_name not in self._derived_array_track:
                self._derived_array_track.append(array_name)

        if ndim is 3:
            array_name_1D = self._array_name_ND_to_1D(array_name)

            for i, a in enumerate(array_name_1D):
                self._arrays[a] = new_array[:, i]
                self._arrays[a]._name = a

    def _create_family_array(self, array_name, family, ndim=1, dtype=None, derived=False, shared=None):
        """Create a single array of dimension len(self.<family.name>) x ndim,
        with a given numpy dtype, belonging to the specified family. For arguments
        other than *family*, see the documentation for :func:`~pynbody.snapshot.SimSnap._create_array`.

        Warning: Do not assume that the family array will be available after
        calling this funciton, because it might be a 'completion' of existing
        family arrays, at which point the routine will actually be creating
        a simulation-level array, e.g.

        sim._create_family_array('bla', dm)
        sim._create_family_array('bla', star)
        'bla' in sim.family_keys() # -> True
        'bla' in sim.keys() # -> False
        sim._create_family_array('bla', gas)
        'bla' in sim.keys() # -> True
        'bla' in sim.family_keys() # -> False

        sim[gas]['bla'] *is* guaranteed to exist, however, it just might
        be a view on a simulation-length array.

        """

        NDname = self._array_name_1D_to_ND(array_name)
        if NDname:
            self._create_family_array(
                NDname, family, ndim=3, dtype=dtype, derived=derived)
            return

        self_families = self.families()

        if len(self_families) == 1 and family in self_families:
            # If the file has only one family, just go ahead and create
            # a normal array
            self._create_array(
                array_name, ndim=ndim, dtype=dtype, derived=derived)
            return

        if ndim == 1:
            dims = self[family]._num_particles
        else:
            dims = (self[family]._num_particles, ndim)

        # Determine what families already have an array of this name
        fams = []
        dtx = None
        try:
            fams = self._family_arrays[array_name].keys()
            dtx = self._family_arrays[array_name][fams[0]].dtype
        except KeyError:
            pass

        fams.append(family)

        if dtype is not None and dtx is not None and dtype != dtx:

            # We insist on the data types being the same for, e.g. sim.gas['my_prop'] and sim.star['my_prop']
            # This makes promotion to simulation-level arrays possible.
            raise ValueError("Requested data type %r is not consistent with existing data type %r for family array %r" % (
                str(dtype), str(dtx), array_name))

        if all([x in fams for x in self_families]):
            # If, once we created this array, *all* families would have
            # this array, just create a simulation-level array
            if self._promote_family_array(array_name, ndim=ndim, derived=derived, shared=shared) is not None:
                return None

        # if we get here, either the array cannot be promoted to simulation level, or that would
        # not be appropriate, so actually go ahead and create the family array

        if shared is None:
            shared = self._shared_arrays
        new_ar = array._array_factory(dims, dtype, False, shared)
        new_ar._sim = weakref.ref(self)
        new_ar._name = array_name
        new_ar.family = family

        def sfa(n, v):
            try:
                self._family_arrays[n][family] = v
            except KeyError:
                self._family_arrays[n] = dict({family: v})

        sfa(array_name, new_ar)
        if derived:
            if array_name not in self._family_derived_array_track[family]:
                self._family_derived_array_track[family].append(array_name)

        if ndim is 3:
            array_name_1D = self._array_name_ND_to_1D(array_name)
            for i, a in enumerate(array_name_1D):
                sfa(a, new_ar[:, i])
                self._family_arrays[a][family]._name = a

    def _del_family_array(self, array_name, family):
        """Delete the array with the specified name for the specified family"""
        del self._family_arrays[array_name][family]
        if len(self._family_arrays[array_name]) == 0:
            del self._family_arrays[array_name]

        derive_track = self._family_derived_array_track[family]
        if array_name in derive_track:
            del derive_track[derive_track.index(array_name)]

    def _get_from_immediate_cache(self, name, fn) :
        """Retrieves the named numpy array from the immediate cache associated
        with this snapshot. If the array does not exist in the immediate
        cache, function fn is called with no arguments and must generate
        it."""
        
        if not hasattr(self, '_immediate_cache'):
                self._immediate_cache = [{}]
        cache = self._immediate_cache[0]
        hx = hash(name)
        if hx not in cache:
            cache[hx] = fn()

        return cache[hx]
                
    def _get_array(self, name, index=None, always_writable=False):
        """Get the array of the specified *name*, optionally
        for only the particles specified by *index*.

        If *always_writable* is True, the returned array is
        writable. Otherwise, it is still normally writable, but
        not if the array is flagged as derived by the framework."""

        x = self._arrays[name]
        if x.derived and not always_writable:
            x = x.view()
            x.flags['WRITEABLE'] = False

        if index is not None:
            if type(index) is slice:
                ret = x[index]
            else:
                ret = array.IndexedSimArray(x, index)

            ret.family = None
            return ret

        else:
            return x

    def _get_family_array(self, name, fam, index=None, always_writable=False):
        """Get the family-level array with specified *name* for the family *fam*,
        optionally for only the particles specified by *index* (relative to the
        family slice).

        If *always_writable* is True, the returned array is writable. Otherwise
        it is still normally writable, but not if the array is flagged as derived
        by the framework.
        """

        try:
            x = self._family_arrays[name][fam]
        except KeyError:
            raise KeyError("No array "+name+" for family "+fam.name)

        if x.derived and not always_writable:
            x = x.view()
            x.flags['WRITEABLE'] = False

        if index is not None:
            if type(index) is slice:
                x = x[index]
            else:
                if _subarray_immediate_mode or self.immediate_mode :
                    x = self._get_from_immediate_cache(name,
                                                       lambda : x[index])
                else :
                    x = array.IndexedSimArray(x, index)
        return x

    def _set_array(self, name, value, index=None):
        """Update the contents of the snapshot-level array to that
        specified by *value*. If *index* is not None, update only that
        subarray specified."""
        util.set_array_if_not_same(self._arrays[name], value, index)

    def _set_family_array(self, name, family, value, index=None):
        """Update the contents of the family-level array to that
        specified by *value*. If *index* is not None, update only that
        subarray specified."""
        util.set_array_if_not_same(self._family_arrays[name][family],
                                   value, index)

    def _create_arrays(self, array_list, ndim=1, dtype=None, zeros=True):
        """Create a set of arrays *array_list* of dimension len(self) x ndim, with
        a given numpy dtype."""
        for array in array_list:
            self._create_array(array, ndim, dtype, zeros)

    def _get_family_slice(self, fam):
        """Turn a specified Family object into a concrete slice which describes
        which particles in this SimSnap belong to that family."""
        try:
            return self._family_slice[fam]
        except KeyError:
            return slice(0, 0)

    def _family_index(self):
        """Return an array giving the family number of each particle in this snapshot,
        something like 0,0,0,0,1,1,2,2,2, ... where 0 means self.families()[0] etc"""

        if hasattr(self, "_family_index_cached"):
            return self._family_index_cached

        ind = np.empty((len(self),), dtype='int8')
        for i, f in enumerate(self.families()):
            ind[self._get_family_slice(f)] = i

        self._family_index_cached = ind

        return ind

    def _assert_not_family_array(self, name):
        """Raises a ValueError if the specified array name is connected to
        a family-specific array"""
        if name in self.family_keys():
            raise KeyError("Array "+name+" is a family-level property")

    def _delayed_array_promotions(self):
        """Called automatically to catch up with pending array promotions"""
        for x in self.__delayed_promotions:
            self._promote_family_array(*x)

        self.__delayed_promotions = []

    def _promote_family_array(self, name, ndim=1, dtype=None, derived=False, shared=None):
        """Create a simulation-level array (if it does not exist) with
        the specified name. Copy in any data from family-level arrays
        of the same name."""

        if ndim == 1 and self._array_name_1D_to_ND(name):
            return self._promote_family_array(self._array_name_1D_to_ND(name), 3, dtype)

        if self.delay_promotion:
            # if array isn't already scheduled for promotion, do so now
            if not any([x[0] == name for x in self.__delayed_promotions]):
                self.__delayed_promotions.append(
                    [name, ndim, dtype, derived, shared])
            return None

        if dtype is None:
            try:
                x = self._family_arrays[name].keys()[0]
                dtype = self._family_arrays[name][x].dtype
                for x in self._family_arrays[name].values():
                    if x.dtype != dtype:
                        warnings.warn("Data types of family arrays do not match; assuming "+str(
                            dtype),  RuntimeWarning)

            except IndexError:
                pass

        dmap = [name in self._family_derived_array_track[
            i] for i in self._family_arrays[name]]
        some_derived = any(dmap)
        all_derived = all(dmap)

        if derived:
            some_derived = True
        if not derived:
            all_derived = False

        if name not in self._arrays:
            self._create_array(
                name, ndim=ndim, dtype=dtype, derived=all_derived, shared=shared)
        try:
            for fam in self._family_arrays[name]:
                if has_units(self._family_arrays[name][fam]) and not has_units(self._arrays[name]):
                    self._arrays[name].units = self._family_arrays[
                        name][fam].units
                    # inherits the units from the first dimensional family array found.
                    # Note that future copies, once the units are set, invoke the correct conversion
                    # and raise a UnitsException if such a conversion is
                    # impossible.

                try:
                    self._arrays[name][self._get_family_slice(
                        fam)] = self._family_arrays[name][fam]
                except units.UnitsException:
                    # There is a problem getting everything into the same units. The trouble is
                    # that having got here if we let the exception propagate, we're going to
                    # end up with the SimSnap in an inconsistent state. So force the copy
                    # ignoring the units and raise a warning
                    warnings.warn(
                        "When conjoining family arrays to create a snapshot level array, the units could not be unified. You will now have a snapshot-level array %r with inconsistent unit information" % name)
                    self._arrays[name].base[self._get_family_slice(
                        fam)] = self._family_arrays[name][fam].base

            del self._family_arrays[name]
            if ndim == 3:
                for v in self._array_name_ND_to_1D(name):
                    del self._family_arrays[v]
            gc.collect()
            
        except KeyError:
            pass

        if some_derived:
            if all_derived:
                self._derived_array_track.append(name)
            else:
                warnings.warn(
                    "Conjoining derived and non-derived arrays. Assuming result is non-derived, so no further updates will be made.", RuntimeWarning)
            for v in self._family_derived_array_track.itervalues():
                if name in v:
                    del v[v.index(name)]

        return self._arrays[name]

    ############################################
    # DERIVED ARRAY SYSTEM
    ############################################
    @classmethod
    def derived_quantity(cl, fn):
        if cl not in SimSnap._derived_quantity_registry:
            SimSnap._derived_quantity_registry[cl] = {}
        SimSnap._derived_quantity_registry[cl][fn.__name__] = fn
        fn.__stable__ = False
        return fn

    @classmethod
    def stable_derived_quantity(cl, fn):
        if cl not in SimSnap._derived_quantity_registry:
            SimSnap._derived_quantity_registry[cl] = {}
        SimSnap._derived_quantity_registry[cl][fn.__name__] = fn
        fn.__stable__ = True

        return fn

    def _derive_array(self, name, fam=None):
        """Calculate and store, for this SnapShot, the derivable array 'name'.
        If *fam* is not None, derive only for the specified family.

        This searches the registry of @X.derived_quantity functions
        for all X in the inheritance path of the current class.
        """
        global config

        calculated = False
        if name in self._calculating:
            raise ValueError("Circular reference in derived quantity")
        else:
            try:
                self._calculating.append(name)
                self._dependency_track.append(set())
                for cl in type(self).__mro__:
                    if cl in self._derived_quantity_registry \
                            and name in self._derived_quantity_registry[cl]:
                        if config['verbose']:
                            print>>sys.stderr, "SimSnap: deriving array", name
                        with self.auto_propagate_off:
                            fn = self._derived_quantity_registry[cl][name]
                            if fam is None:
                                result = fn(self)
                                ndim = result.shape[-1] if len(
                                    result.shape) > 1 else 1
                                self._create_array(
                                    name, ndim, dtype=result.dtype, derived=not fn.__stable__)
                                write_array = self._get_array(
                                    name, always_writable=True)
                            else:
                                result = fn(self[fam])
                                ndim = result.shape[-1] if len(
                                    result.shape) > 1 else 1
                                self[fam]._create_array(
                                    name, ndim, dtype=result.dtype, derived=not fn.__stable__)
                                write_array = self[fam]._get_array(
                                    name, always_writable=True)

                            self.ancestor._autoconvert_array_unit(result)

                            write_array[:] = result
                            if units.has_units(result):
                                write_array.units = result.units

                        calculated = True
                        for x in self._dependency_track[-1]:
                            if x != name:
                                if x not in self._dependency_chain:
                                    self._dependency_chain[x] = set()

                                self._dependency_chain[x].add(name)

                        break
            finally:
                assert self._calculating[-1] == name
                del self._calculating[-1]
                del self._dependency_track[-1]

            if not calculated:
                raise KeyError("No derivation rule for "+name)

    def _dirty(self, name):
        """Declare a given array as changed, so deleting any derived
        quantities which depend on it"""

        if not self.auto_propagate_off:
            name = self._array_name_1D_to_ND(name) or name
            if name in self._dependency_chain:
                for d_ar in self._dependency_chain[name]:
                    if d_ar in self or self.has_family_key(d_ar):
                        if self.is_derived_array(d_ar):
                            del self[d_ar]
                            self._dirty(d_ar)

    def is_derived_array(self, name, fam=None):
        """Returns True if the array or family array of given name is
        auto-derived (and therefore read-only)."""
        fam = fam or self._unifamily
        if fam:
            return (name in self._family_derived_array_track[fam]) or name in self._derived_array_track
        elif name in self.keys():
            return name in self._derived_array_track
        elif name in self.family_keys():
            return all([name in self._family_derived_array_track[i] for i in self._family_arrays[name]])
        else:
            return False

    def unlink_array(self, name):
        """If the named array is auto-derived, this destroys the link so that
        the array becomes editable but no longer auto-updates."""

        if self.is_derived_array(name) :
            if name in self.family_keys():                
                for fam in self._family_arrays[name] : 
                    track = self._family_derived_array_track[fam]
                    
                    if name in track :
                        del track[track.index(name)]
                
            else: 
                del self._derived_array_track[
                    self._derived_array_track.index(name)]
        
        else:
            raise RuntimeError("Not a derived array")

    ############################################
    # CONVENIENCE FUNCTIONS
    ############################################
    def mean_by_mass(self, name):
        """Calculate the mean by mass of the specified array."""
        m = np.asanyarray(self["mass"])
        ret = array.SimArray((self[name].transpose()*m).transpose().mean(axis=0)/m.mean(), self[name].units)

        return ret

    ############################################
    # SNAPSHOT DECORATION
    ############################################

    @classmethod
    def decorator(cl, fn):
        if cl not in SimSnap._decorator_registry:
            SimSnap._decorator_registry[cl] = []
        SimSnap._decorator_registry[cl].append(fn)
        return fn

    def _decorate(self):
        for cl in type(self).__mro__:
            if cl in self._decorator_registry:
                for fn in self._decorator_registry[cl]:
                    fn(self)

    ############################################
    # HASHING AND EQUALITY TESTING
    ############################################

    @property
    def _inclusion_hash(self):
        try:
            rval = self.__inclusion_hash
        except AttributeError:
            try:
                index_list = self.get_index_list(self.ancestor)
                hash = hashlib.md5(index_list.data)
                self.__inclusion_hash = hash.digest()
            except:
                print "Encountered a problem while calculating your inclusion hash. The original traceback follows."
                traceback.print_exc()
            rval = self.__inclusion_hash
        return rval

    def __hash__(self):
        return hash((object.__hash__(self.ancestor), self._inclusion_hash))

    def __eq__(self, other):
        """Equality test for Snapshots. Returns true if both sides of the
        == operator point to the same data."""

        if self is other:
            return True
        return hash(self) == hash(other)

    ############################################
    # COPYING
    ############################################
    def __deepcopy__(self, memo=None):
        create_args = {}
        for fam in family._registry:
            sl = self._get_family_slice(fam)
            if sl.start != sl.stop:
                create_args[fam.name] = len(self[fam])

        new = _new(**create_args)

        # ordering fix
        for k in copy.copy(new._family_slice.keys()):
            new._family_slice[k] = copy.copy(self._get_family_slice(k))

        for k in self.keys():
            new[k] = self[k]

        for k in self.family_keys():
            for fam in family._registry:
                if len(self[fam]) > 0:
                    self_fam = self[fam]
                    if k in self_fam.keys() and not self_fam.is_derived_array(k):
                        new[fam][k] = self_fam[k]

        new.properties = copy.deepcopy(self.properties, memo)
        new._file_units_system = copy.deepcopy(self._file_units_system, memo)

        return new


_subarray_immediate_mode = False
# Set this to True to always get copies of data when indexing is
# necessary. This is mainly a bug testing/efficiency checking mode --
# shouldn't be necessary


class SubSnap(SimSnap):
    """Represent a sub-view of a SimSnap, initialized by specifying a
    slice.  Arrays accessed through __getitem__ are automatically
    sub-viewed using the given slice."""

    def __init__(self, base, _slice):
        self.base = base
        self._file_units_system = base._file_units_system
        self._unifamily = base._unifamily

        self._inherit()

        if isinstance(_slice, slice):
            # Various slice logic later (in particular taking
            # subsnaps-of-subsnaps) requires having positive
            # (i.e. start-relative) slices, so if we have been passed a
            # negative (end-relative) index, fix that now.
            if _slice.start is None:
                _slice = slice(0, _slice.stop, _slice.step)
            if _slice.start < 0:
                _slice = slice(len(
                    base)+_slice.start, _slice.stop, _slice.step)
            if _slice.stop is None or _slice.stop > len(base):
                _slice = slice(_slice.start, len(base), _slice.step)
            if _slice.stop < 0:
                _slice = slice(_slice.start, len(
                    base)+_slice.stop, _slice.step)

            self._slice = _slice

            descriptor = "["+str(_slice.start)+":"+str(_slice.stop)
            if _slice.step is not None:
                descriptor += ":"+str(_slice.step)
            descriptor += "]"

        else:
            raise TypeError("Unknown SubSnap slice type")

        self._num_particles = util.indexing_length(_slice)

        self._descriptor = descriptor

    def _inherit(self):
        for x in self._inherited:
            setattr(self, x, getattr(self.base, x))

    def _get_array(self, name, index=None, always_writable=False):
        if _subarray_immediate_mode or self.immediate_mode:
            return self._get_from_immediate_cache(name, 
                   lambda : self.base._get_array(
                            name, None, always_writable)[self._slice])

        else:
            ret = self.base._get_array(name, util.concatenate_indexing(
                self._slice, index), always_writable)
            ret.family = self._unifamily
            return ret

    def _set_array(self, name, value, index=None):
        self.base._set_array(
            name, value, util.concatenate_indexing(self._slice, index))

    def _get_family_array(self, name, fam, index=None, always_writable=False):
        base_family_slice = self.base._get_family_slice(fam)
        sl = util.relative_slice(base_family_slice,
                                 util.intersect_slices(self._slice, base_family_slice, len(self.base)))
        sl = util.concatenate_indexing(sl, index)
        if _subarray_immediate_mode or self.immediate_mode:
            return self._get_from_immediate_cache((name,fam),
                                                  lambda : self.base._get_family_array(
                        name, fam, None, always_writable)[sl])
        else:
            return self.base._get_family_array(name, fam, sl, always_writable)

    def _set_family_array(self, name, family, value, index=None):
        fslice = self._get_family_slice(family)
        self.base._set_family_array(
            name, family, value, util.concatenate_indexing(fslice, index))

    def _promote_family_array(self, *args, **kwargs):
        self.base._promote_family_array(*args, **kwargs)

    def __delitem__(self, name):
        # is this the right behaviour?
        raise RuntimeError("Arrays can only be deleted from the base snapshot")

    def _del_family_array(self, name, family):
        # is this the right behaviour?
        raise RuntimeError("Arrays can only be deleted from the base snapshot")

    @property
    def _filename(self):
        return self.base._filename+":"+self._descriptor

    def keys(self):
        return self.base.keys()

    def loadable_keys(self, fam=None):
        if self._unifamily:
            return self.base.loadable_keys(self._unifamily)
        else:
            return self.base.loadable_keys(fam)

    def derivable_keys(self):
        return self.base.derivable_keys()

    def infer_original_units(self, *args):
        """Return the units on disk for a quantity with the specified dimensions"""
        return self.base.infer_original_units(*args)

    def _get_family_slice(self, fam):
        sl = util.relative_slice(self._slice,
                                 util.intersect_slices(self._slice, self.base._get_family_slice(fam), len(self.base)))
        return sl

    def _load_array(self, array_name, fam=None, **kwargs):
        self.base._load_array(array_name, fam, **kwargs)

    def write_array(self, array_name, fam=None, **kwargs):
        fam = fam or self._unifamily
        if not fam or self._get_family_slice(fam) != slice(0, len(self)):
            raise IOError(
                "Array writing is available for entire simulation arrays or family-level arrays, but not for arbitrary subarrays. See http://code.google.com/p/pynbody/wiki/WritingFiles")

        self.base.write_array(array_name, fam=fam, **kwargs)

    def _derive_array(self, array_name, fam=None):
        self.base._derive_array(array_name, fam)

    def family_keys(self, fam=None):
        return self.base.family_keys(fam)

    def _create_array(self, *args, **kwargs):
        self.base._create_array(*args, **kwargs)

    def _create_family_array(self, *args, **kwargs):
        self.base._create_family_array(*args, **kwargs)

    def physical_units(self, *args, **kwargs):
        self.base.physical_units(*args, **kwargs)

    def is_derived_array(self, v, fam=None):
        return self.base.is_derived_array(v)

    def unlink_array(self, name):
        self.base.unlink_array(name)

    def get_index_list(self, relative_to, of_particles=None):
        if of_particles is None:
            of_particles = np.arange(len(self))

        if relative_to is self:
            return of_particles

        return self.base.get_index_list(relative_to, util.concatenate_indexing(self._slice, of_particles))


class IndexedSubSnap(SubSnap):
    """Represents a subset of the simulation particles according
    to an index array."""
    def __init__(self, base, index_array):

        self._descriptor = "indexed"
        self.base = base
        self._inherit()

        self._unifamily = base._unifamily
        self._file_units_system = base._file_units_system

        if isinstance(index_array, filt.Filter):
            self._descriptor = index_array._descriptor
            index_array = index_array.where(base)[0]

        elif isinstance(index_array, tuple):
            if isinstance(index_array[0], np.ndarray):
                index_array = index_array[0]
            else:
                index_array = np.array(index_array)
        else:
            index_array = np.asarray(index_array)

        findex = base._family_index()[index_array]
        # Check the family index array is monotonically increasing
        # If not, the family slices cannot be implemented
        if not all(np.diff(findex) >= 0):
            raise ValueError(
                "Families must retain the same ordering in the SubSnap")

        self._slice = index_array
        self._family_slice = {}
        self._family_indices = {}
        self._num_particles = len(index_array)

        # Find the locations of the family slices
        for i, fam in enumerate(base.families()):
            ids = np.where(findex == i)[0]
            if len(ids) > 0:
                new_slice = slice(ids.min(), ids.max()+1)
                self._family_slice[fam] = new_slice
                self._family_indices[fam] = np.asarray(index_array[
                                                       new_slice])-base._get_family_slice(fam).start

        """
        # old code for monotonically increasing arrays only
        for fam in family._registry :
            base_slice = base._get_family_slice(fam)
            start = util.index_of_first(index_array,base_slice.start)
            stop = util.index_of_first(index_array, base_slice.stop)
            new_slice=slice(start,stop)
            self._family_slice[fam] = new_slice
            self._family_indices[fam] = np.asarray(index_array[new_slice])-base_slice.start
        """

    def _get_family_slice(self, fam):
        # A bit messy: jump out the SubSnap inheritance chain
        # and call SimSnap method directly...
        return SimSnap._get_family_slice(self, fam)

    def _get_family_array(self, name, fam, index=None, always_writable=False):
        sl = self._family_indices[fam]
        sl = util.concatenate_indexing(sl, index)

        return self.base._get_family_array(name, fam, sl, always_writable)

    def _set_family_array(self, name, family, value, index=None):
        self.base._set_family_array(name, family, value,
                                    util.concatenate_indexing(self._family_indices[family], index))

    def _create_array(self, *args, **kwargs):
        self.base._create_array(*args, **kwargs)


class FamilySubSnap(SubSnap):
    """Represents a one-family portion of a parent snap object"""
    def __init__(self, base, fam):
        self.base = base

        self._inherit()

        self._slice = base._get_family_slice(fam)
        self._unifamily = fam
        self._descriptor = ":"+fam.name
        # Use the slice attributes to find sub array length
        self._num_particles = self._slice.stop-self._slice.start
        self._file_units_system = base._file_units_system

    def __delitem__(self, name):
        if name in self.base.keys():
            raise ValueError(
                "Cannot delete global simulation property from sub-view")
        elif name in self.base.family_keys(self._unifamily):
            self.base._del_family_array(name, self._unifamily)

    def keys(self):
        global_keys = self.base.keys()
        family_keys = self.base.family_keys(self._unifamily)
        return list(set(global_keys).union(family_keys))

    def family_keys(self, fam=None):
        # We now define there to be no family-specific subproperties,
        # because all properties can be accessed through standard
        # __setitem__, __getitem__ methods
        return []

    def _get_family_slice(self, fam):
        if fam is self._unifamily:
            return slice(0, len(self))
        else:
            return slice(0, 0)

    def _get_array(self, name, index=None, always_writable=False):
        try:
            return SubSnap._get_array(self, name, index, always_writable)
        except KeyError:
            return self.base._get_family_array(name, self._unifamily, index, always_writable)

    def _create_array(self, array_name, ndim=1, dtype=None, zeros=True, derived=False, shared=None):
        # Array creation now maps into family-array creation in the parent
        self.base._create_family_array(
            array_name, self._unifamily, ndim, dtype, derived, shared)

    def _set_array(self, name, value, index=None):
        if name in self.base.keys():
            self.base._set_array(
                name, value, util.concatenate_indexing(self._slice, index))
        else:
            self.base._set_family_array(name, self._unifamily, value, index)

    def _create_family_array(self, array_name, family, ndim, dtype, derived, shared):
        self.base._create_family_array(array_name, family, ndim, dtype, derived, shared)

    def _promote_family_array(self, *args, **kwargs):
        pass

    def _load_array(self, array_name, fam=None, **kwargs):
        if fam is self._unifamily or fam is None:
            self.base._load_array(array_name, self._unifamily, **kwargs)

    def _derive_array(self, array_name, fam=None):
        if fam is self._unifamily or fam is None:
            self.base._derive_array(array_name, self._unifamily)


def _new(n_particles=0, **families):
    """Create a blank SimSnap, with the specified number of particles.

    Position, velocity and mass arrays are created and filled
    with zeros.

    By default all particles are taken to be dark matter.
    To specify otherwise, pass in keyword arguments specifying
    the number of particles for each family, e.g.

    f = new(dm=50, star=25, gas=25)
    """

    if len(families) == 0:
        families = {'dm': n_particles}

    t_fam = []
    tot_particles = 0

    for k, v in families.items():

        assert isinstance(v, int)
        t_fam.append((family.get_family(k), v))
        tot_particles += v

    x = SimSnap()
    x._num_particles = tot_particles
    x._filename = "<created>"

    x._create_arrays(["pos", "vel"], 3)
    x._create_arrays(["mass"], 1)

    rt = 0
    for k, v in t_fam:
        x._family_slice[k] = slice(rt, rt+v)
        rt += v

    x._decorate()
    return x
