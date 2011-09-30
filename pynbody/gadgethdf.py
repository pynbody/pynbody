"""gadgethdf.py

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
    
class GadgetHDFSnap(snapshot.SimSnap) :
    def __init__(self, filename) :

        global config
        super(GadgetHDFSnap,self).__init__()
        
        self._filename = filename
	try:
	    self._hdf = h5py.File(filename)
	except IOError :
	    # IOError here can be caused if no append access is available
	    # so explicitly open as read-only
	    self._hdf = h5py.File(filename,"r")
	    
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
                l+=self._hdf[name]['Coordinates'].shape[0]
            self._family_slice[x] = slice(sl_start,sl_start+l)
            self._loadable_keys = self._loadable_keys.union(set(self._hdf[name].keys()))
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
                if translated_name not in self._hdf[n].keys() : return False
            return True

    def loadable_keys(self) :
        return self._loadable_keys

    @staticmethod
    def _write(self, filename=None) :
        raise RuntimeError, "Not implemented"
        
    def _write_array(self, array_name, filename=None) :
        raise RuntimeError, "Not implemented"
        
    def _load_array(self, array_name, fam=None) :
        if not self._family_has_loadable_array( fam, array_name) :
            raise IOError, "No such array on disk"
        else :
            if fam is not None :
                famx = fam
            else :
                famx = self._family_slice.keys()[0]

            translated_name = _translate_array_name(array_name)
            
            assert len(self._hdf[_type_map[famx][0]][translated_name].shape)<=2
            dy = 1
            if len(self._hdf[_type_map[famx][0]][translated_name].shape)>1 :
                dy = self._hdf[_type_map[famx][0]][translated_name].shape[1]

            dtype = self._hdf[_type_map[famx][0]][translated_name].dtype
            
            if fam is None :
                self._create_array(array_name, dy, dtype=dtype)
            else :
                self[fam]._create_array(array_name, dy, dtype=dtype)

                
            if fam is not None :
                fams = [fam]
            else :
                fams = self._family_slice.keys()

            for f in fams :
                i0 = 0
                for t in _type_map[f] :
                    dataset = self._hdf[t][translated_name]
                    i1 = i0+len(dataset)
                    dataset.read_direct(self[f][array_name][i0:i1])
	
    @staticmethod
    def _can_load(f) :
        try :
            if h5py.is_hdf5(f) :
                return True
        except AttributeError :
            return False
        

@GadgetHDFSnap.decorator
def do_properties(sim) :
    atr = sim._hdf['Header'].attrs
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
    cosmo = (sim._hdf['Parameters']['NumericalParameters'].attrs['ComovingIntegrationOn'])!=0
    atr = sim._hdf['Units'].attrs
    vel_unit = atr['UnitVelocity_in_cm_per_s']*units.cm/units.s
    dist_unit = atr['UnitLength_in_cm']*units.cm
    if cosmo :
        dist_unit/=units.h
    mass_unit = atr['UnitMass_in_g']*units.g
    if cosmo:
        mass_unit/=units.h

    sim._file_units_system=[units.Unit(x) for x in [vel_unit,dist_unit,mass_unit,"K"]]
