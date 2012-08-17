"""

gadgethdf
=========


Implementation of backend reader for GadgetHDF files by Andrew Pontzen.

The gadget array names are mapped into pynbody array names according
to the mappings given by the config.ini section [gadgethdf-name-mapping].

The gadget particle groups are mapped into pynbody families according
to the mappings specified by the config.ini section [gadgethdf-type-mapping].
This can be many-to-one (many gadget particle types mapping into one
pynbody family), but only datasets which are common to all gadget types
will be available from pynbody.

Currently spanned files are not supported (i.e. only the contents of
one file will be loaded).
"""


from __future__ import with_statement # for py2.5

from . import snapshot, array, util
from . import family
from . import units
from . import config
from . import config_parser

import ConfigParser

import struct, os
import numpy as np
import functools
import warnings
import sys

try :
    import h5py
except ImportError :
    h5py=None

_type_map = {}
for x in family.family_names() :
    try :
        _type_map[family.get_family(x)] = \
                 [q for q in config_parser.get('gadgethdf-type-mapping',x).split(",")]
    except ConfigParser.NoOptionError :
        pass

_name_map = {}
_rev_name_map = {}
for a, b in config_parser.items("gadgethdf-name-mapping") :
    _rev_name_map[a] = b
    _name_map[b] = a

def _translate_array_name(name, reverse=False) :
    try :
        if reverse :
            return _rev_name_map[name]
        else :
            return _name_map[name]
    except KeyError :
        return name

def _append_if_array(to_list, name, obj) :
    if not hasattr(obj, 'keys') :
        to_list.append(name)
    
class GadgetHDFSnap(snapshot.SimSnap) :
    def __init__(self, filename) :

        global config
        super(GadgetHDFSnap,self).__init__()
        
        self._filename = filename

        if not h5py.is_hdf5(filename) :
            h1 = h5py.File(filename+".0.hdf5")
            numfiles = h1['Header'].attrs['NumFilesPerSnapshot']
            self._hdf = [h5py.File(filename+"."+str(i)+".hdf5", "r") for i in xrange(numfiles)]
        else :
            self._hdf = [h5py.File(filename,"r")]

        self._family_slice = {}

        self._loadable_keys = set([])
        self._family_arrays = {}
        self._arrays = {}
        self.properties = {}
        
        sl_start = 0
        for x in _type_map :
            l = 0
            for n in _type_map[x] :
                name = n
                for hdf in self._hdf :
                    l+=hdf[name]['Coordinates'].shape[0]
            self._family_slice[x] = slice(sl_start,sl_start+l)
            
            k = self._get_hdf_allarray_keys(self._hdf[0][name])
            self._loadable_keys = self._loadable_keys.union(set(k))
            sl_start+=l

        self._loadable_keys = [_translate_array_name(x, reverse=True) for x in self._loadable_keys]
        self._num_particles = sl_start
        
        self._decorate()
        

    def _family_has_loadable_array(self, fam, name) :
        """Returns True if the array can be loaded for the specified family.
        If fam is None, returns True if the array can be loaded for all families."""
        
        if fam is None :
            return all([self._family_has_loadable_array(fam_x, name) for fam_x in self._family_slice])
                
        else :        
            translated_name = _translate_array_name(name)
            for n in _type_map[fam] :
                if translated_name not in self._get_hdf_allarray_keys(self._hdf[0][n]) : return False
            return True

    def loadable_keys(self) :
        return self._loadable_keys

    @staticmethod
    def _write(self, filename=None) :
        raise RuntimeError, "Not implemented"

	global config
	
        with self.lazy_off : # prevent any lazy reading or evaluation
	    
            if filename is None :
		filename = self._filename
		
	    if config['verbose'] : print>>sys.stderr, "GadgetHDF: writing main file as",filename

            self._hdf_out = h5py.File(filename,"w")
        
    def _write_array(self, array_name, filename=None) :
        raise RuntimeError, "Not implemented"

    @staticmethod
    def _get_hdf_allarray_keys(group) :
        """Return all HDF array keys underneath group (includes nested groups)"""
        k = []
        group.visititems(functools.partial(_append_if_array, k))
        return k
    
    @staticmethod
    def _get_hdf_dataset(particle_group, hdf_name) :
        """Return the HDF dataset resolving /'s into nested groups"""
        ret = particle_group
        for tpart in hdf_name.split("/") :
            ret = ret[tpart]
        return ret
        
    def _load_array(self, array_name, fam=None) :
        if not self._family_has_loadable_array( fam, array_name) :
            raise IOError, "No such array on disk"
        else :
            if fam is not None :
                famx = fam
            else :
                famx = self._family_slice.keys()[0]

            translated_name = _translate_array_name(array_name)

            hdf0 = self._hdf[0]
            
            assert len(hdf0[_type_map[famx][0]][translated_name].shape)<=2
            dy = 1
            if len(hdf0[_type_map[famx][0]][translated_name].shape)>1 :
                dy = hdf0[_type_map[famx][0]][translated_name].shape[1]

            dtype = hdf0[_type_map[famx][0]][translated_name].dtype
            
            if fam is None :
                self._create_array(array_name, dy, dtype=dtype)
                self[array_name].set_default_units()
            else :
                self[fam]._create_array(array_name, dy, dtype=dtype)
                self[fam][array_name].set_default_units()

                
            if fam is not None :
                fams = [fam]
            else :
                fams = self._family_slice.keys()

            for f in fams :
                i0 = 0
                for t in _type_map[f] :
                    for hdf in self._hdf :
                        dataset = self._get_hdf_dataset(hdf[t], translated_name)
                        i1 = i0+len(dataset)
                        dataset.read_direct(self[f][array_name][i0:i1])
                        i0=i1
	
    @staticmethod
    def _can_load(f) :
        try :
            if h5py.is_hdf5(f) or h5py.is_hdf5(f+".0.hdf5") :
                return True
            else :
                return False
        except AttributeError :
            if "hdf5" in f :
                warnings.warn("It looks like you're trying to load HDF5 files, but python's HDF support (h5py module) is missing.", RuntimeWarning)
            return False
        

@GadgetHDFSnap.decorator
def do_properties(sim) :
    atr = sim._hdf[0]['Header'].attrs
    sim.properties['a'] = atr['ExpansionFactor']
    sim.properties['time'] = units.Gyr*atr['Time_GYR']
    sim.properties['omegaM0'] = atr['Omega0']
    sim.properties['omegaB0'] = atr['OmegaBaryon']
    sim.properties['omegaL0'] = atr['OmegaLambda']
    sim.properties['boxsize'] = atr['BoxSize']
    sim.properties['z'] = (1./sim.properties['a'])-1
    sim.properties['h'] = atr['HubbleParam']

@GadgetHDFSnap.decorator
def do_units(sim) :
    cosmo = (sim._hdf[0]['Parameters']['NumericalParameters'].attrs['ComovingIntegrationOn'])!=0
    atr = sim._hdf[0]['Units'].attrs
    vel_unit = atr['UnitVelocity_in_cm_per_s']*units.cm/units.s
    dist_unit = atr['UnitLength_in_cm']*units.cm
    if cosmo :
        dist_unit/=units.h
    mass_unit = atr['UnitMass_in_g']*units.g
    if cosmo:
        mass_unit/=units.h

    sim._file_units_system=[units.Unit(x) for x in [vel_unit,dist_unit,mass_unit,"K"]]
