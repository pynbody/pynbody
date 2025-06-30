import configparser
import logging
import warnings

from .. import config_parser, family, units
from .gadgethdf import GadgetHDFSnap, _GadgetHdfMultiFileManager

logger = logging.getLogger('pynbody.snapshot.pkdgravhdf')

try:
    import h5py
except ImportError:
    h5py = None

_pkd_default_type_map = {}
for x in family.family_names():
    try:
        _pkd_default_type_map[family.get_family(x)] = \
            [q.strip() for q in config_parser.get('pkdgrav3hdf-type-mapping', x).split(",")]
    except configparser.NoOptionError:
        pass

_pkd_all_hdf_particle_groups = []
for hdf_groups in _pkd_default_type_map.values():
    for hdf_group in hdf_groups:
        _pkd_all_hdf_particle_groups.append(hdf_group)


class _PkdgravHdfMultiFileManager(_GadgetHdfMultiFileManager) :
    _nfiles_groupname = "Header"
    _nfiles_attrname = "NumFilesPerSnapshot"
    _namemapper_config_section = "pkdgrav3hdf-name-mapping"

    def _make_filename_for_cpu(self, filename, n):
        return filename + f".{n}"

    def get_cosmo_attrs(self):
        return self[0].parent['Cosmology'].attrs


class PkdgravHDFSnap(GadgetHDFSnap):
    # PKDGRAV3 creates files by default with the format
    # {name}.{step:05d}.{n}
    # where `n>=0` is the index of the file when doing parallel writting.
    # It does not contain any ".hdf5" extension in the file names.
    _multifile_manager_class = _PkdgravHdfMultiFileManager
    _readable_hdf5_test_group = "Header"
    _readable_hdf5_test_attr = "Header", "PKDGRAV version"

    def _get_units_from_hdf_attr(self, hdfattrs) :
        # pkdgrav does not store units as attributes
        return units.NoUnit()

    def _all_hdf_groups(self):
        for hdf_family_name in _pkd_all_hdf_particle_groups:
            yield from self._hdf_files.iter_particle_groups_with_name(hdf_family_name)

    @classmethod
    def _guess_file_ending(cls, f):
        return f.with_suffix(f.suffix + ".0")

    def _init_unit_information(self):
        try:
            atr = self._hdf_files.get_unit_attrs()
        except KeyError:
            warnings.warn("No unit information found in PkdgravHDF file",
                          RuntimeWarning)
            return {}, {}

        vel_unit = atr['KmPerSecUnit'] * 1e5 * units.cm / units.s
        dist_unit = atr['KpcUnit'] * units.kpc.in_units("cm") * units.cm
        mass_unit = atr['MsolUnit'] * units.Msol.in_units("g") * units.g
        time_unit = atr['SecUnit'] * units.s

        # Create a dictionary for the units, this will come in handy later
        unitvar = {'U_V': vel_unit, 'U_L': dist_unit, 'U_M': mass_unit,
                   'U_T': time_unit, '[K]': units.K,
                   'SEC_PER_YEAR': units.yr, 'SOLAR_MASS': units.Msol}
        # Last two units are to catch occasional arrays like StarFormationRate
        # which don't follow the patter of U_ units unfortunately
        cgsvar = {'U_M': 'g', 'SOLAR_MASS': 'g', 'U_T': 's',
                  'SEC_PER_YEAR': 's', 'U_V': 'cm s**-1', 'U_L': 'cm', '[K]': 'K'}

        self._hdf_cgsvar = cgsvar
        self._hdf_unitvar = unitvar

        cosmo = 'HubbleParam' in list(self._get_hdf_parameter_attrs().keys())
        if cosmo:
            dist_unit *= units.a
            vel_unit *= units.a

        self._file_units_system = [units.Unit(x) for x in [
            vel_unit, dist_unit, mass_unit, "K"]]

    def _get_hdf_cosmo_attrs(self):
        return self._hdf_files.get_cosmo_attrs()

    def _init_properties(self):
        atr = self._get_hdf_header_attrs()
        # Some attributes may be stored in the Cosmology header
        cosmo_atr = self._get_hdf_cosmo_attrs()

        cosmo_run = cosmo_atr['Cosmological run']
        if cosmo_run:
            self.properties['z'] = atr['Redshift']
            self.properties['a'] = 1. / (1 + atr['Redshift'])
            self.properties['eps'] = float(atr['Softening']) * self._hdf_unitvar['U_L']

            # Not all omegas need to be specified in the attributes
            try:
                self.properties['omegaB0'] = cosmo_atr['Omega_b']
            except KeyError:
                pass

            self.properties['omegaM0'] = cosmo_atr['Omega_m']
            self.properties['omegaL0'] = cosmo_atr['Omega_lambda']
            self.properties['boxsize'] = atr['BoxSize'] / \
                cosmo_atr['HubbleParam'] * units.Mpc * units.a
            self.properties['h'] = cosmo_atr['HubbleParam']
        else:
            self.properties['z'] = 0
            self.properties['a'] = 1
            self.properties['eps'] = 0

        self.properties['time'] = atr['Time'] * self._hdf_unitvar['U_T']

        for s, value in self._get_hdf_header_attrs().items():
            if s not in ['Time', 'Omega_m', 'Omega_b', 'Omega_lambda',
                         'BoxSize', 'HubbleParam']:
                self.properties[s] = value

        for s, value in self._get_hdf_cosmo_attrs().items():
            if s not in ['Time', 'Omega_m', 'Omega_b', 'Omega_lambda',
                         'BoxSize', 'HubbleParam']:
                self.properties[s] = value
