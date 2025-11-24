import configparser
from .. import config_parser, family, units
from ..array import SimArray
from .gadgethdf import GadgetHDFSnap

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
    metals = self['metals_list'][:,0]
    # PENDING: there's some small discrepancy with np.sum(self['metals_list'][:,2:], axis = 1), 
    # but the FIRE-2 public release info is incorrect, as self['metals_list'][:,0] 
    # is clearly not equal to the H mass fraction
    return metals
    
# PENDING: "some" FIRE-2 simulations include additional metal_list fields
# for r-process calculations
