import configparser
from .. import config_parser, family, units
from .gadgethdf import GadgetHDFSnap
import numpy as np

_default_type_map = {}
for x in family.family_names():
   try:
    _default_type_map[family.get_family(x)] = \
        [q.strip() for q in config_parser.get('firehdf-type-mapping', x).split(",")]
   except configparser.NoOptionError:
    pass

_all_hdf_particle_groups = []
for hdf_groups in _default_type_map.values():
    for hdf_group in hdf_groups:
        _all_hdf_particle_groups.append(hdf_group)


class FIREHDFSnap(GadgetHDFSnap):
    """Reads FIRE-like HDF snapshots"""
    _readable_hdf5_test_key = "PartType0/ParticleIDGenerationNumber"
    _namemapper_config_section = "firehdf-name-mapping"
    
    def _get_units_from_hdf_attr(self, hdfattrs) :
        # FIRE does not store units as attributes
        return units.NoUnit()

                
    def __init_family_map(self):
        type_map = {}
        for fam, g_types in _default_type_map.items():
            my_types = []
            for x in g_types:
                # Get all keys from all hdf files
                for hdf in self._hdf_files:
                    if x in list(hdf.keys()):
                        my_types.append(x)
                        break
            if len(my_types):
                type_map[fam] = my_types
        self._family_to_group_map = type_map
        
    def _init_unit_information(self):
        # Uses gadget default units, FIRE doesn't store units
        atr = self._hdf_files.get_parameter_attrs()
        if self._velocity_unit_key not in atr.keys():
            vel_unit = config_parser.get('firehdf-units', 'vel')
            dist_unit = config_parser.get('firehdf-units', 'pos')
            mass_unit = config_parser.get('firehdf-units', 'mass')
            self._file_units_system = [units.Unit(x) for x in [
                vel_unit, dist_unit, mass_unit, "K"]]
            return
        
        
@FIREHDFSnap.derived_array
def He(self) :
    He = self['metals_list'][:,1]
    return He
    
@FIREHDFSnap.derived_array
def H(self) :
    H = 1 - self['metals_list'][:,0] - self['He']
    return H
   
@FIREHDFSnap.derived_array
def C(self) :
    C = self['metals_list'][:,2]
    return C
    
@FIREHDFSnap.derived_array
def N(self) :
    N = self['metals_list'][:,3]
    return N
    
@FIREHDFSnap.derived_array
def O(self) :
    O = self['metals_list'][:,4]
    return O
    
@FIREHDFSnap.derived_array
def Ne(self) :
    Ne = self['metals_list'][:,5]
    return Ne
     
@FIREHDFSnap.derived_array
def Mg(self) :
    Mg = self['metals_list'][:,6]
    return Mg
    
@FIREHDFSnap.derived_array
def Si(self) :
    Si = self['metals_list'][:,7]
    return Si
    
@FIREHDFSnap.derived_array
def S(self) :
    S = self['metals_list'][:,8]
    return S
    
@FIREHDFSnap.derived_array
def Ca(self) :
    Ca = self['metals_list'][:,9]
    return Ca
    
@FIREHDFSnap.derived_array
def Fe(self) :
    Fe = self['metals_list'][:,10]
    return Fe
    
@FIREHDFSnap.derived_array
def metals(self) :
    metals = np.array(self['metals_list'][:,0], dtype = np.float64)
    # Specify dtype to fix clamping problems when interpolating magnitudes,
    # should we do this for all element mass fractions?
    # PENDING: check if position 0 is always the metal_fraction
    return metals
