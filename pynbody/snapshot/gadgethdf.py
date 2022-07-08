"""

gadgethdf
=========


Implementation of backend reader for GadgetHDF files by Andrew Pontzen.

The gadget array names are mapped into pynbody array mes according
to the mappings given by the config.ini section [gadgethdf-name-mapping].

The gadget particle groups are mapped into pynbody families according
to the mappings specified by the config.ini section [gadgethdf-type-mapping].
This can be many-to-one (many gadget particle types mapping into one
pynbody family), but only datasets which are common to all gadget types
will be available from pynbody.

Spanned files are supported. To load a range of files snap.0, snap.1, ... snap.n,
pass the filename 'snap'. If you pass snap.0, only that particular file will
be loaded.
"""


  # for py2.5

import configparser
import functools
import itertools
import logging
import warnings

import numpy as np

from .. import config_parser, family, halo, units, util
from . import SimSnap, namemapper

logger = logging.getLogger('pynbody.snapshot.gadgethdf')

try:
    import h5py
except ImportError:
    h5py = None

_default_type_map = {}
for x in family.family_names():
    try:
        _default_type_map[family.get_family(x)] = \
                 [q.strip() for q in config_parser.get('gadgethdf-type-mapping', x).split(",")]
    except configparser.NoOptionError:
        pass

_all_hdf_particle_groups = []
for hdf_groups in _default_type_map.values():
    for hdf_group in hdf_groups:
        _all_hdf_particle_groups.append(hdf_group)




class DummyHDFData:

    """A stupid class to allow emulation of mass arrays for particles
    whose mass is in the header"""

    def __init__(self, value, length, dtype):
        self.value = value
        self.length = length
        self.size = length
        self.shape = (length, )
        self.dtype = np.dtype(dtype)

    def __len__(self):
        return self.length

    def read_direct(self, target):
        target[:] = self.value


class GadgetHdfMultiFileManager:
    _nfiles_groupname = "Header"
    _nfiles_attrname = "NumFilesPerSnapshot"
    _subgroup_name = None

    def __init__(self, filename, mode='r') :
        self._mode = mode
        if h5py.is_hdf5(filename):
            self._filenames = [filename]
            self._numfiles = 1
        else:
            h1 = h5py.File(filename + ".0.hdf5", mode)
            self._numfiles = h1[self._nfiles_groupname].attrs[self._nfiles_attrname]
            self._filenames = [filename+"."+str(i)+".hdf5" for i in range(self._numfiles)]

        self._open_files = {}

    def __iter__(self) :
        for i in range(self._numfiles) :
            if i not in self._open_files:
                self._open_files[i] = h5py.File(self._filenames[i], self._mode)
                if self._subgroup_name is not None:
                    self._open_files[i] = self._open_files[i][self._subgroup_name]

            yield self._open_files[i]

    def __getitem__(self, i) :
        try :
            return self._open_files[i]
        except KeyError :
            self._open_files[i] = next(itertools.islice(self,i,i+1))
            return self._open_files[i]

    def get_header_attrs(self):
        return self[0].parent['Header'].attrs

    def get_parameter_attrs(self):
        try:
            attrs = self[0].parent['Parameters'].attrs
        except KeyError:
            attrs = []

        if len(attrs) == 0:
            return self.get_header_attrs()
        else:
            return attrs


    def get_unit_attrs(self):
        return self[0].parent['Units'].attrs

    def get_file0_root(self):
        return self[0].parent

    def iterroot(self):
        for item in self:
            yield item.parent

    def reopen_in_mode(self, mode):
        if mode!=self._mode:
            self._open_files = {}
            self._mode = mode



class SubfindHdfMultiFileManager(GadgetHdfMultiFileManager):
    _nfiles_groupname = "FOF"
    _nfiles_attrname = "NTask"
    _subgroup_name = "FOF"


class GadgetHDFSnap(SimSnap):
    """
    Class that reads HDF Gadget data
    """

    _multifile_manager_class = GadgetHdfMultiFileManager
    _readable_hdf5_test_key = "PartType?"
    _size_from_hdf5_key = "ParticleIDs"

    def __init__(self, filename, ):
        super().__init__()

        self._filename = filename

        self._init_hdf_filemanager(filename)

        self._translate_array_name = namemapper.AdaptiveNameMapper('gadgethdf-name-mapping')
        self.__init_unit_information()
        self.__init_family_map()
        self.__init_file_map()
        self.__init_loadable_keys()
        self.__infer_mass_dtype()

        self._decorate()

    def _get_hdf_header_attrs(self):
        return self._hdf_files.get_header_attrs()

    def _get_hdf_parameter_attrs(self):
        return self._hdf_files.get_parameter_attrs()

    def _get_hdf_unit_attrs(self):
        return self._hdf_files.get_unit_attrs()

    def _init_hdf_filemanager(self, filename):
        self._hdf_files = self._multifile_manager_class(filename)

    def __init_loadable_keys(self):

        self._loadable_family_keys = {}
        all_fams = self.families()
        if len(all_fams)==0:
            return

        for fam in all_fams:
            self._loadable_family_keys[fam] = {"mass"}
            for hdf_group in self._all_hdf_groups_in_family(fam):
                for this_key in self._get_hdf_allarray_keys(hdf_group):
                    ar_name = self._translate_array_name(this_key, reverse=True)
                    self._loadable_family_keys[fam].add(ar_name)
            self._loadable_family_keys[fam] = list(self._loadable_family_keys[fam])

        self._loadable_keys = set(self._loadable_family_keys[all_fams[0]])
        for fam_keys in self._loadable_family_keys.values():
            self._loadable_keys.intersection_update(fam_keys)

        self._loadable_keys = list(self._loadable_keys)

    def _all_hdf_groups(self):
        for hdf in self._hdf_files:
            for hdf_family_name in _all_hdf_particle_groups:
                if hdf_family_name in hdf:
                    yield hdf[hdf_family_name]

    def _all_hdf_groups_in_family(self, fam):
        for hdf_family_name in self._family_to_group_map[fam]:
            for hdf in self._hdf_files:
                if hdf_family_name in hdf:
                    if self._size_from_hdf5_key in hdf[hdf_family_name]:
                        yield hdf[hdf_family_name]

    def __init_file_map(self):
        family_slice_start = 0

        all_families_sorted = self._families_ordered()

        for fam in all_families_sorted:
            family_length = 0
            for hdf_group in self._all_hdf_groups_in_family(fam):
                family_length += hdf_group[self._size_from_hdf5_key].size

            self._family_slice[fam] = slice(family_slice_start, family_slice_start + family_length)
            family_slice_start += family_length


        self._num_particles = family_slice_start

    def __infer_mass_dtype(self):
        """Some files have a mixture of header-based masses and, for other partile types, explicit mass
        arrays. This routine decides in advance the correct dtype to assign to the mass array, whichever
        particle type it is loaded for."""
        mass_dtype = np.float64
        for hdf in self._all_hdf_groups():
            if "Coordinates" in hdf:
                mass_dtype = hdf['Coordinates'].dtype
        self._mass_dtype = mass_dtype

    def _families_ordered(self):
        # order by the PartTypeN
        all_families = list(self._family_to_group_map.keys())
        all_families_sorted = sorted(all_families, key=lambda v: self._family_to_group_map[v][0])
        return all_families_sorted

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

    def _family_has_loadable_array(self, fam, name):
        """Returns True if the array can be loaded for the specified family.
        If fam is None, returns True if the array can be loaded for all families."""
        return name in self.loadable_keys(fam)


    def _get_all_particle_arrays(self, gtype):
        """Return all array names for a given gadget particle type"""

        # this is a hack to flatten a list of lists
        l = [item for sublist in [self._get_hdf_allarray_keys(x[gtype]) for x in self._hdf_files] for item in sublist]

        # now just return the unique items by converting to a set
        return list(set(l))

    def loadable_keys(self, fam=None):
        if fam is None:
            return self._loadable_keys
        else:
            return self._loadable_family_keys[fam]


    @staticmethod
    def _write(self, filename=None):
        raise RuntimeError("Not implemented")

    def write_array(self, array_name, fam=None, overwrite=False):
        translated_name = self._translate_array_name(array_name)

        self._hdf_files.reopen_in_mode('r+')

        if fam is None:
            target = self
            all_fams_to_write = self.families()
        else:
            target = self[fam]
            all_fams_to_write = [fam]

        for writing_fam in all_fams_to_write:
            i0 = 0
            target_array = self[writing_fam][array_name]
            for hdf in self._all_hdf_groups_in_family(writing_fam):
                npart = hdf['ParticleIDs'].size
                i1 = i0 + npart
                target_array_this = target_array[i0:i1]

                dataset = self._get_or_create_hdf_dataset(hdf, translated_name,
                                                          target_array_this.shape,
                                                          target_array_this.dtype)

                dataset.write_direct(target_array_this.reshape(dataset.shape))

                i0 = i1

    @staticmethod
    def _get_hdf_allarray_keys(group):
        """Return all HDF array keys underneath group (includes nested groups)"""
        keys = []

        def _append_if_array(to_list, name, obj):
            if not hasattr(obj, 'keys'):
                to_list.append(name)

        group.visititems(functools.partial(_append_if_array, keys))
        return keys

    def _get_or_create_hdf_dataset(self, particle_group, hdf_name, shape, dtype):
        if self._translate_array_name(hdf_name,reverse=True)=='mass':
            raise OSError("Unable to write the mass block due to Gadget header format")

        ret = particle_group
        for tpart in hdf_name.split("/")[:-1]:
            ret =ret[tpart]

        dataset_name = hdf_name.split("/")[-1]
        return ret.require_dataset(dataset_name, shape, dtype, exact=True)


    def _get_hdf_dataset(self, particle_group, hdf_name):
        """Return the HDF dataset resolving /'s into nested groups, and returning
        an apparent Mass array even if the mass is actually stored in the header"""

        if self._translate_array_name(hdf_name,reverse=True)=='mass':
            try:
                pgid = int(particle_group.name[-1])
                mtab = particle_group.parent['Header'].attrs['MassTable'][pgid]
                if mtab > 0:
                    return DummyHDFData(mtab, particle_group[self._size_from_hdf5_key].size,
                                        self._mass_dtype)
            except (IndexError, KeyError):
                pass

        ret = particle_group
        for tpart in hdf_name.split("/"):
            ret = ret[tpart]
        return ret

    @staticmethod
    def _get_cosmo_factors(hdf, arr_name) :
        """Return the cosmological factors for a given array"""
        match = [s for s in GadgetHDFSnap._get_hdf_allarray_keys(hdf) if ((s.endswith("/"+arr_name)) & ('PartType' in s))]
        if (arr_name == 'Mass' or arr_name == 'Masses') and len(match) == 0:
            # mass stored in header. We're out in the cold on our own.
            warnings.warn("Masses are either stored in the header or have another dataset name; assuming the cosmological factor %s" % units.h**-1)
            return units.Unit('1.0'), units.h**-1
        if len(match) > 0 :
            try:
                aexp = hdf[match[0]].attrs['aexp-scale-exponent']
            except KeyError:
                # gadget4 <sigh>
                aexp = hdf[match[0]].attrs['a_scaling']
            try:
                hexp = hdf[match[0]].attrs['h-scale-exponent']
            except KeyError:
                # gadget4 <sigh>
                hexp = hdf[match[0]].attrs['h_scaling']
            return units.a**util.fractions.Fraction.from_float(float(aexp)).limit_denominator(), units.h**util.fractions.Fraction.from_float(float(hexp)).limit_denominator()
        else :
            return units.Unit('1.0'), units.Unit('1.0')

    def _get_units_from_hdf_attr(self, hdfattrs) :
        """Return the units based on HDF attributes VarDescription"""
        VarDescription = str(hdfattrs['VarDescription'])
        CGSConversionFactor = float(hdfattrs['CGSConversionFactor'])
        aexp = hdfattrs['aexp-scale-exponent']
        hexp = hdfattrs['h-scale-exponent']

        arr_units = self._get_units_from_description(VarDescription, CGSConversionFactor)

        if not np.allclose(aexp, 0.0):
            arr_units *= (units.a)**util.fractions.Fraction.from_float(float(aexp)).limit_denominator()
        if not np.allclose(hexp, 0.0):
            arr_units *= (units.h)**util.fractions.Fraction.from_float(float(hexp)).limit_denominator()

        return arr_units

    def _get_units_from_description(self, description, expectedCgsConversionFactor=None):
        arr_units = units.Unit('1.0')
        conversion = 1.0
        for unitname in list(self._hdf_unitvar.keys()):
            power = 1.
            if unitname in description:
                sstart = description.find(unitname)
                if sstart > 0:
                    if description[sstart - 1] == "/":
                        power *= -1.
                if len(description) > sstart + len(unitname):
                    # Just check we're not at the end of the line
                    if description[sstart + len(unitname)] == '^':
                        ## Has an index, check if this is negative
                        if description[sstart + len(unitname) + 1] == "-":
                            power *= -1.
                            power *= float(
                                description[sstart + len(unitname) + 2:-1].split()[0])  ## Search for the power
                        else:
                            power *= float(
                                description[sstart + len(unitname) + 1:-1].split()[0])  ## Search for the power
                if not np.allclose(power, 0.0):
                    arr_units *= self._hdf_unitvar[unitname] ** util.fractions.Fraction.from_float(
                        float(power)).limit_denominator()
                if not np.allclose(power, 0.0):
                    conversion *= self._hdf_unitvar[unitname].in_units(
                        self._hdf_cgsvar[unitname]) ** util.fractions.Fraction.from_float(
                        float(power)).limit_denominator()

        if expectedCgsConversionFactor is not None:
            if not np.allclose(conversion, expectedCgsConversionFactor, rtol=1e-3):
                raise units.UnitsException(
                    "Error with unit read out from HDF. Inferred CGS conversion factor is {!r} but HDF requires {!r}".format(
                    conversion, expectedCgsConversionFactor))

        return arr_units

    def _load_array(self, array_name, fam=None):
        if not self._family_has_loadable_array(fam, array_name):
            raise OSError("No such array on disk")
        else:

            translated_name = self._translate_array_name(array_name)
            dtype, dy, units = self.__get_dtype_dims_and_units(fam, translated_name)

            if fam is None:
                target = self
                all_fams_to_load = self.families()
            else:
                target = self[fam]
                all_fams_to_load = [fam]

            target._create_array(array_name, dy, dtype=dtype)

            if units is not None:
                target[array_name].units = units
            else:
                target[array_name].set_default_units()

            for loading_fam in all_fams_to_load:
                i0 = 0
                for hdf in self._all_hdf_groups_in_family(loading_fam):
                    npart = hdf['ParticleIDs'].size
                    i1 = i0+npart

                    dataset = self._get_hdf_dataset(hdf, translated_name)

                    target_array = self[loading_fam][array_name][i0:i1]
                    assert target_array.size == dataset.size

                    dataset.read_direct(target_array.reshape(dataset.shape))

                    i0 = i1

    def __get_dtype_dims_and_units(self, fam, translated_name):
        if fam is None:
            fam = self.families()[0]

        inferred_units = units.NoUnit()
        representative_dset = None
        representative_hdf = None
        # not all arrays are present in all hdfs so need to loop
        # until we find one
        for hdf0 in self._hdf_files:
            try:
                representative_dset = self._get_hdf_dataset(hdf0[
                                                  self._family_to_group_map[fam][0]], translated_name)
                representative_hdf = hdf0
                if hasattr(representative_dset, "attrs"):
                    inferred_units = self._get_units_from_hdf_attr(representative_dset.attrs)

                if len(representative_dset)!=0:
                    # suitable for figuring out everything we need to know about this array
                    break
            except KeyError:
                continue
        if representative_dset is None:
            raise KeyError("Array is not present in HDF file")


        assert len(representative_dset.shape) <= 2

        if len(representative_dset.shape) > 1:
            dy = representative_dset.shape[1]
        else:
            dy = 1

        # Some versions of gadget fold the 3D arrays into 1D.
        # So check if the dimensions make sense -- if not, assume we're looking at an array that
        # is 3D and cross your fingers
        npart = len(representative_hdf[self._family_to_group_map[fam][0]]['ParticleIDs'])

        if len(representative_dset) != npart:
            dy = len(representative_dset) // npart

        dtype = representative_dset.dtype
        if translated_name=="Mass":
            dtype = self._mass_dtype
        return dtype, dy, inferred_units

    def __init_unit_information(self):
        try:
            atr = self._hdf_files.get_unit_attrs()
        except KeyError:
            # Gadget 4 stores unit information in Parameters attr <sigh>
            atr = self._hdf_files.get_parameter_attrs()
            if 'UnitVelocity_in_cm_per_s' not in atr.keys():
                warnings.warn("No unit information found in GadgetHDF file. Using gadget default units.", RuntimeWarning)
                vel_unit = config_parser.get('gadget-units', 'vel')
                dist_unit = config_parser.get('gadget-units', 'pos')
                mass_unit = config_parser.get('gadget-units', 'mass')
                self._file_units_system = [units.Unit(x) for x in [
                    vel_unit, dist_unit, mass_unit, "K"]]
                return

        # Define the SubFind units, we will parse the attribute VarDescriptions for these
        vel_unit = atr['UnitVelocity_in_cm_per_s'] * units.cm / units.s
        dist_unit = atr['UnitLength_in_cm'] * units.cm
        mass_unit = atr['UnitMass_in_g'] * units.g
        try:
            time_unit = atr['UnitTime_in_s'] * units.s
        except KeyError:
            # Gadget 4 seems not to store time units explicitly <sigh>
            time_unit = dist_unit/vel_unit

        # Create a dictionary for the units, this will come in handy later
        unitvar = {'U_V': vel_unit, 'U_L': dist_unit, 'U_M': mass_unit,
                   'U_T': time_unit, '[K]': units.K,
                   'SEC_PER_YEAR': units.yr, 'SOLAR_MASS': units.Msol,
                   'solar masses / yr': units.Msol/units.yr, 'BH smoothing': dist_unit}
        # Some arrays like StarFormationRate don't follow the patter of U_ units
        cgsvar = {'U_M': 'g', 'SOLAR_MASS': 'g', 'U_T': 's',
                  'SEC_PER_YEAR': 's', 'U_V': 'cm s**-1', 'U_L': 'cm', '[K]': 'K',
                  'solar masses / yr': 'g s**-1', 'BH smoothing': 'cm'}

        self._hdf_cgsvar = cgsvar
        self._hdf_unitvar = unitvar

        cosmo = 'HubbleParam' in list(self._get_hdf_parameter_attrs().keys())
        if cosmo:
            try:
                for fac in self._get_cosmo_factors(self._hdf_files[0], 'Coordinates'): dist_unit *= fac
            except KeyError:
                dist_unit *= units.a * units.h**-1
                warnings.warn("Unable to find cosmological factors in HDF file; assuming position is %s" % dist_unit)
            try:
                for fac in self._get_cosmo_factors(self._hdf_files[0], 'Velocities'): vel_unit *= fac
            except KeyError:
                vel_unit *= units.a**(1,2)
                warnings.warn("Unable to find cosmological factors in HDF file; assuming velocity is %s" % vel_unit)
            try:
                for fac in self._get_cosmo_factors(self._hdf_files[0], 'Mass'): mass_unit *= fac
            except KeyError:
                mass_unit *= units.h**-1
                warnings.warn("Unable to find cosmological factors in HDF file; assuming mass is %s" % mass_unit)

        self._file_units_system = [units.Unit(x) for x in [
            vel_unit, dist_unit, mass_unit, "K"]]

    @classmethod
    def _test_for_hdf5_key(cls, f):
        with h5py.File(f, "r") as h5test:
            test_key = cls._readable_hdf5_test_key
            if test_key[-1]=="?":
                # try all particle numbers in turn
                for p in range(6):
                    test_key = test_key[:-1]+str(p)
                    if test_key in h5test:
                        return True
                return False
            else:
                return test_key in h5test

    @classmethod
    def _can_load(cls, f):

        if hasattr(h5py, "is_hdf5"):
            if h5py.is_hdf5(f):
                return cls._test_for_hdf5_key(f)
            elif h5py.is_hdf5(f+".0.hdf5"):
                return cls._test_for_hdf5_key(f+".0.hdf5")
            else:
                return False
        else:
            if "hdf5" in f:
                warnings.warn(
                    "It looks like you're trying to load HDF5 files, but python's HDF support (h5py module) is missing.", RuntimeWarning)
            return False



@GadgetHDFSnap.decorator
def do_properties(sim):
    atr = sim._get_hdf_header_attrs()

    # expansion factor could be saved as redshift
    try:
        sim.properties['a'] = atr['ExpansionFactor']
    except KeyError:
        sim.properties['a'] = 1. / (1 + atr['Redshift'])

    # Gadget 4 stores parameters in a separate dictionary <sigh>. For older formats, this will point back to the same
    # as the header attributes.
    atr = sim._get_hdf_parameter_attrs()

    # not all omegas need to be specified in the attributes
    try:
        sim.properties['omegaB0'] = atr['OmegaBaryon']
    except KeyError:
        pass

    sim.properties['omegaM0'] = atr['Omega0']
    sim.properties['omegaL0'] = atr['OmegaLambda']
    sim.properties['boxsize'] = atr['BoxSize'] * sim.infer_original_units('cm')
    sim.properties['z'] = (1. / sim.properties['a']) - 1
    sim.properties['h'] = atr['HubbleParam']

    # time unit might not be set in the attributes
    if "Time_GYR" in atr:
        sim.properties['time'] = units.Gyr * atr['Time_GYR']
    else:
        from .. import analysis
        sim.properties['time'] = analysis.cosmology.age(sim) * units.Gyr

    for s,value in sim._get_hdf_header_attrs().items():
        if s not in ['ExpansionFactor', 'Time_GYR', 'Time', 'Omega0', 'OmegaBaryon', 'OmegaLambda', 'BoxSize', 'HubbleParam']:
            sim.properties[s] = value

###################
# SubFindHDF class
###################

class SubFindHDFSnap(GadgetHDFSnap) :
    """
    Class to read Gadget's SubFind HDF data
    """
    _multifile_manager_class = SubfindHdfMultiFileManager
    _readable_hdf5_test_key = "FOF"

    def __init__(self, filename) :
        super().__init__(filename)

    def halos(self) :
        return halo.SubFindHDFHaloCatalogue(self)


class EagleLikeHDFSnap(GadgetHDFSnap):
    """Reads Eagle-like HDF snapshots (download at http://data.cosma.dur.ac.uk:8080/eagle-snapshots/)"""
    _readable_hdf5_test_key = "PartType1/SubGroupNumber"

    def halos(self, subs=None):
        """Load the Eagle FOF halos, or if subs is specified the Subhalos of the given FOF halo number.

        *subs* should be an integer specifying the parent FoF number"""
        if subs:
            if not np.issubdtype(type(subs), np.integer):
                raise ValueError("The subs argument must specify the group number")
            parent_group = self[self['GroupNumber']==subs]
            if len(parent_group)==0:
                raise ValueError("No group found with id %d"%subs)

            cat = halo.GrpCatalogue(parent_group,
                                     array="SubGroupNumber", ignore=np.max(self['SubGroupNumber']))
            cat._keep_subsnap_alive = parent_group # by default, HaloCatalogue only keeps a weakref (should this be changed?)
            return cat
        else:
            return halo.GrpCatalogue(self, array="GroupNumber", ignore=np.max(self['GroupNumber']))

## Gadget has internal energy variable
@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def u(self) :
    """Gas internal energy derived from snapshot variable or temperature"""
    try:
        u = self['InternalEnergy']
    except KeyError:
        gamma = 5./3
        u = self['temp']*units.k/(self['mu']*units.m_p*(gamma-1))

    return u

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def p(sim) :
    """Calculate the pressure for gas particles, including polytropic equation of state gas"""

    critpres = 2300. * units.K * units.m_p / units.cm**3 ## m_p K cm^-3
    critdens = 0.1 * units.m_p / units.cm**3 ## m_p cm^-3
    gammaeff = 4./3.

    oneos = sim.g['OnEquationOfState'] == 1.

    p = sim.g['rho'].in_units('m_p cm**-3') * sim.g['temp'].in_units('K')
    p[oneos] = critpres * (sim.g['rho'][oneos].in_units('m_p cm**-3')/critdens)**gammaeff

    return p

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def HII(sim) :
    """Number of HII ions per proton mass"""

    return sim.g["hydrogen"] - sim.g["HI"]

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def HeIII(sim) :
    """Number of HeIII ions per proton mass"""

    return sim.g["hetot"] - sim.g["HeII"] - sim.g["HeI"]

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def ne(sim) :
    """Number of electrons per proton mass, ignoring the contribution from He!"""
    ne = sim.g["HII"]  #+ sim["HeII"] + 2*sim["HeIII"]
    ne.units = units.m_p**-1

    return ne

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def rho_ne(sim) :
    """Electron number density per SPH particle, currently ignoring the contribution from He!"""

    return sim.g["ne"].in_units("m_p**-1") * sim.g["rho"].in_units("m_p cm**-3")

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def dm(sim) :
    """Dispersion measure per SPH particle currently ignoring n_e contribution from He """

    return sim.g["rho_ne"]

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def cosmodm(sim) :
    """Cosmological Dispersion measure per SPH particle includes (1+z) factor, currently ignoring n_e contribution from He """

    return sim.g["rho_ne"] * (1. + sim.g["redshift"])
@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def redshift(sim) :
    """Redshift from LoS Velocity 'losvel' """

    return np.exp( sim['losvel'].in_units('c') ) - 1.

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def doppler_redshift(sim) :
    """Doppler Redshift from LoS Velocity 'losvel' using SR """

    return np.sqrt( (1. + sim['losvel'].in_units('c')) / (1. - sim['losvel'].in_units('c'))  ) - 1.

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def em(sim) :
    """Emission Measure (n_e^2) per particle to be integrated along LoS"""

    return sim.g["rho_ne"]*sim.g["rho_ne"]

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def halpha(sim) :
    """H alpha intensity (based on Emission Measure n_e^2) per particle to be integrated along LoS"""

    ## Rate at which recombining electrons and protons produce Halpha photons.
    ## Case B recombination assumed from Draine (2011)
    #alpha = 2.54e-13 * (sim.g['temp'].in_units('K') / 1e4)**(-0.8163-0.0208*np.log(sim.g['temp'].in_units('K') / 1e4))
    #alpha.units = units.cm**(3) * units.s**(-1)

    ## H alpha intensity = coeff * EM
    ## where coeff is h (c / Lambda_Halpha) / 4Pi) and EM is int rho_e * rho_p * alpha
    ## alpha = 7.864e-14 T_1e4K from http://astro.berkeley.edu/~ay216/08/NOTES/Lecture08-08.pdf
    coeff = (6.6260755e-27) * (299792458. / 656.281e-9) / (4.*np.pi) ## units are erg sr^-1
    alpha = coeff * 7.864e-14 * (1e4 / sim.g['temp'].in_units('K'))

    alpha.units = units.erg * units.cm**(3) * units.s**(-1) * units.sr**(-1) ## It's intensity in erg cm^3 s^-1 sr^-1

    return alpha * sim["em"] # Flux erg cm^-3 s^-1 sr^-1

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def c_n_sq(sim) :
    """Turbulent amplitude C_N^2 for use in SM calculations (e.g. Eqn 20 of Macquart & Koay 2013 ApJ 776 2) """

    ## Spectrum of turbulence below the SPH resolution, assume Kolmogorov
    beta = 11./3.
    L_min = 0.1*units.Mpc
    c_n_sq = ((beta - 3.)/((2.)*(2.*np.pi)**(4.-beta)))*L_min**(3.-beta)*sim["em"]
    c_n_sq.units = units.m**(-20,3)

    return c_n_sq

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def hetot(self) :
    return self["He"]

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def hydrogen(self) :
    return self["H"]

## Need to use the ionisation fraction calculation here which gives ionisation fraction
## based on the gas temperature, density and redshift for a CLOUDY table
@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def HI(sim) :
    """Fraction of Neutral Hydrogen HI use limited CLOUDY table"""

    import pynbody.analysis.hifrac

    return pynbody.analysis.hifrac.calculate(sim.g,ion='hi')

## Need to use the ionisation fraction calculation here which gives ionisation fraction
## based on the gas temperature, density and redshift for a CLOUDY table, then applying
## selfshielding for the dense, star forming gas on the equation of state
@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def HIeos(sim) :
    """Fraction of Neutral Hydrogen HI use limited CLOUDY table, assuming dense EoS gas is selfshielded"""

    import pynbody.analysis.hifrac

    return pynbody.analysis.hifrac.calculate(sim.g,ion='hi', selfshield='eos')

## Need to use the ionisation fraction calculation here which gives ionisation fraction
## based on the gas temperature, density and redshift for a CLOUDY table, then applying
## selfshielding for the dense, star forming gas on the equation of state AND a further
## pressure based limit for
@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def HID12(sim) :
    """Fraction of Neutral Hydrogen HI use limited CLOUDY table, using the Duffy +12a prescription for selfshielding"""

    import pynbody.analysis.hifrac

    return pynbody.analysis.hifrac.calculate(sim.g,ion='hi', selfshield='duffy12')

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def HeI(sim) :
    """Fraction of Helium HeI"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='hei')

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def HeII(sim) :
    """Fraction of Helium HeII"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='heii')

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def OI(sim) :
    """Fraction of Oxygen OI"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='oi')

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def OII(sim) :
    """Fraction of Oxygen OII"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='oii')

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def OVI(sim) :
    """Fraction of Oxygen OVI"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='ovi')

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def CIV(sim) :
    """Fraction of Carbon CIV"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='civ')

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def NV(sim) :
    """Fraction of Nitrogen NV"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='nv')

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def SIV(sim) :
    """Fraction of Silicon SiIV"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='siiv')

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def MGII(sim) :
    """Fraction of Magnesium MgII"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='mgii')

# The Solar Abundances used in Gadget-3 OWLS / Eagle / Smaug sims
XSOLH=0.70649785
XSOLHe=0.28055534
XSOLC=2.0665436E-3
XSOLN=8.3562563E-4
XSOLO=5.4926244E-3
XSOLNe=1.4144605E-3
XSOLMg=5.907064E-4
XSOLSi=6.825874E-4
XSOLS=4.0898522E-4
XSOLCa=6.4355E-5
XSOLFe=1.1032152E-3

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def feh(self) :
    minfe = np.amin(self['Fe'][np.where(self['Fe'] > 0)])
    self['Fe'][np.where(self['Fe'] == 0)]=minfe
    return np.log10(self['Fe']/self['H']) - np.log10(XSOLFe/XSOLH)

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def sixh(self) :
    minsi = np.amin(self['Si'][np.where(self['Si'] > 0)])
    self['Si'][np.where(self['Si'] == 0)]=minsi
    return np.log10(self['Si']/self['Si']) - np.log10(XSOLSi/XSOLH)

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def sxh(self) :
    minsx = np.amin(self['S'][np.where(self['S'] > 0)])
    self['S'][np.where(self['S'] == 0)]=minsx
    return np.log10(self['S']/self['S']) - np.log10(XSOLS/XSOLH)

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def mgxh(self) :
    minmg = np.amin(self['Mg'][np.where(self['Mg'] > 0)])
    self['Mg'][np.where(self['Mg'] == 0)]=minmg
    return np.log10(self['Mg']/self['Mg']) - np.log10(XSOLMg/XSOLH)

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def oxh(self) :
    minox = np.amin(self['O'][np.where(self['O'] > 0)])
    self['O'][np.where(self['O'] == 0)]=minox
    return np.log10(self['O']/self['H']) - np.log10(XSOLO/XSOLH)

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def nexh(self) :
    minne = np.amin(self['Ne'][np.where(self['Ne'] > 0)])
    self['Ne'][np.where(self['Ne'] == 0)]=minne
    return np.log10(self['Ne']/self['Ne']) - np.log10(XSOLNe/XSOLH)

@SubFindHDFSnap.derived_quantity
def hexh(self) :
    minhe = np.amin(self['He'][np.where(self['He'] > 0)])
    self['He'][np.where(self['He'] == 0)]=minhe
    return np.log10(self['He']/self['He']) - np.log10(XSOLHe/XSOLH)

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def cxh(self) :
    mincx = np.amin(self['C'][np.where(self['C'] > 0)])
    self['C'][np.where(self['C'] == 0)]=mincx
    return np.log10(self['C']/self['H']) - np.log10(XSOLC/XSOLH)

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def caxh(self) :
    mincax = np.amin(self['Ca'][np.where(self['Ca'] > 0)])
    self['Ca'][np.where(self['Ca'] == 0)]=mincax
    return np.log10(self['Ca']/self['H']) - np.log10(XSOLCa/XSOLH)

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def nxh(self) :
    minnx = np.amin(self['N'][np.where(self['N'] > 0)])
    self['N'][np.where(self['N'] == 0)]=minnx
    return np.log10(self['N']/self['H']) - np.log10(XSOLH/XSOLH)

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def ofe(self) :
    minox = np.amin(self['O'][np.where(self['O'] > 0)])
    self['O'][np.where(self['O'] == 0)]=minox
    minfe = np.amin(self['Fe'][np.where(self['Fe'] > 0)])
    self['Fe'][np.where(self['Fe'] == 0)]=minfe
    return np.log10(self['O']/self['Fe']) - np.log10(XSOLO/XSOLFe)

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def mgfe(sim) :
    minmg = np.amin(sim['Mg'][np.where(sim['Mg'] > 0)])
    sim['Mg'][np.where(sim['Mg'] == 0)]=minmg
    minfe = np.amin(sim['Fe'][np.where(sim['Fe'] > 0)])
    sim['Fe'][np.where(sim['Fe'] == 0)]=minfe
    return np.log10(sim['Mg']/sim['Fe']) - np.log10(XSOLMg/XSOLFe)

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def nefe(sim) :
    minne = np.amin(sim['Ne'][np.where(sim['Ne'] > 0)])
    sim['Ne'][np.where(sim['Ne'] == 0)]=minne
    minfe = np.amin(sim['Fe'][np.where(sim['Fe'] > 0)])
    sim['Fe'][np.where(sim['Fe'] == 0)]=minfe
    return np.log10(sim['Ne']/sim['Fe']) - np.log10(XSOLNe/XSOLFe)

@GadgetHDFSnap.derived_quantity
@SubFindHDFSnap.derived_quantity
def sife(sim) :
    minsi = np.amin(sim['Si'][np.where(sim['Si'] > 0)])
    sim['Si'][np.where(sim['Si'] == 0)]=minsi
    minfe = np.amin(sim['Fe'][np.where(sim['Fe'] > 0)])
    sim['Fe'][np.where(sim['Fe'] == 0)]=minfe
    return np.log10(sim['Si']/sim['Fe']) - np.log10(XSOLSi/XSOLFe)
