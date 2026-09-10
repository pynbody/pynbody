"""Implements reading gusteau snapshots.

Gusteau (https://gusteau-spec.readthedocs.io/) is a formalisation of the SWIFT
HDF5 snapshot layout, intended as a common target for transcoding snapshots
from different simulation codes. Its particle groups carry descriptive names
(``Gas``, ``ColdDarkMatter``, ``Stars``, ``BlackHoles``) rather than SWIFT's
``PartTypeN``, but the specification requires the traditional names to be
present as aliases, and the per-dataset unit metadata follows SWIFT's
convention with only cosmetic renaming. This module therefore builds on
:class:`~pynbody.snapshot.swift.SwiftSnap`, adjusting only the metadata
locations and attribute names that gusteau spells differently.

Support is currently minimal: snapshots can be opened, their properties and
arrays read, and units interpreted. Region selection (which for gusteau relies
on its "packing cubes" spatial index) and writing arrays back out are not yet
implemented.
"""

import h5py

from .gadgethdf import GadgetHDFSnap
from .swift import ExtractScalarWrapper, SwiftMultiFileManager, SwiftSnap


class GusteauMultiFileManager(SwiftMultiFileManager):
    """Manages access to the HDF5 files of a gusteau snapshot.

    Gusteau keeps its run metadata in different places to SWIFT: the internal
    code units live in a top-level ``Units`` group (as for plain GadgetHDF),
    while the parameter file contents are nested inside ``RunInfo``.
    """

    _nfiles_attrname = "Num_files_per_snapshot"

    def get_unit_attrs(self):
        return self[0].parent['Units'].attrs

    def get_header_attrs(self):
        return self[0].parent['Header'].attrs

    def get_parameter_attrs(self):
        try:
            attrs = self[0].parent['RunInfo/Parameters'].attrs
        except KeyError:
            return self.get_header_attrs()

        if len(attrs) == 0:
            return self.get_header_attrs()
        return attrs

    def _read_cell_metadata(self, h1):
        raise NotImplementedError(
            "pynbody cannot yet read the gusteau spatial index ('packing cubes'), so it is "
            "not possible to load only part of a gusteau snapshot by region"
        )


class GusteauSnap(SwiftSnap):
    """Reads gusteau snapshots.

    .. versionadded:: 2.8.0
    """

    _multifile_manager_class = GusteauMultiFileManager

    _gusteau_header_attrs = ('Particle_names', 'Part_type_mapping')
    """Header attributes which, taken together, identify a file as following the gusteau spec"""

    _length_unit_key = 'Unit_length_CGS'
    _mass_unit_key = 'Unit_mass_CGS'
    _time_unit_key = 'Unit_time_CGS'
    _temperature_unit_key = 'Unit_temperature_CGS'

    _cosmological_flag_key = 'Is_cosmological'
    _fof_group_id_default_key = 'FOF_group_id_default'
    _scalefactor_unitvar_name = 'a_scale'
    _hubble_unitvar_name = 'h_scale'

    _namemapper_config_section = 'gusteau-name-mapping'

    @classmethod
    def _test_for_hdf5_key(cls, f):
        """Return True if the given file follows the gusteau specification.

        Gusteau files are deliberately also readable as GadgetHDF files, because the spec
        requires the descriptively-named particle groups to be aliased onto the traditional
        ``PartTypeN`` names. Identification therefore has to rest on something only gusteau
        writes, and the ``Header`` attributes describing that aliasing (``Particle_names``
        and ``Part_type_mapping``) serve that purpose; no other GadgetHDF variant has them.
        """
        with h5py.File(f, "r") as h5test:
            if "Header" not in h5test:
                return False
            header_attrs = h5test["Header"].attrs
            return all(k in header_attrs for k in cls._gusteau_header_attrs)

    @classmethod
    def _unit_name_from_exponent_attr_name(cls, attr_name):
        """Return the unit variable named by an exponent attribute, e.g. 'U_L' for 'U_L_exponent'."""
        return attr_name[:-len("_exponent")]

    def _init_properties(self):
        header = ExtractScalarWrapper(self._hdf_files[0]['Header'].attrs)
        cosmo = ExtractScalarWrapper(self._hdf_files[0]['Cosmology'].attrs)

        assert header['Dimension'] == 3, \
            "Sorry, pynbody is only set up to deal with 3-dimensional gusteau simulations"

        if self._is_cosmological():
            # note that pynbody derives 'z' from 'a', so there is no need to set it explicitly
            self.properties['a'] = cosmo['Scale_factor']
            self.properties['h'] = cosmo['h']
            self.properties['omegaM0'] = cosmo['Omega_matter']
            self.properties['omegaL0'] = cosmo['Omega_lambda']
            for pynbody_name, gusteau_name in (('omegaB0', 'Omega_baryon'),
                                               ('omegaC0', 'Omega_darkmatter'),
                                               ('omegaNu0', 'Omega_nu_0')):
                if gusteau_name in cosmo.underlying:
                    self.properties[pynbody_name] = cosmo[gusteau_name]

            # The bounding box is stored as (x0, y0, z0, x1, y1, z1); pynbody only understands
            # cubic periodic boxes, so check that is what we have.
            bounding_box = header.underlying['Bounding_box']
            side_lengths = bounding_box[3:] - bounding_box[:3]
            assert (side_lengths == side_lengths[0]).all(), \
                "Sorry, pynbody is only set up to deal with cubic gusteau simulation volumes"
            self.properties['boxsize'] = side_lengths[0] * self._get_coordinate_units()

        # As for swift, this should NOT be infer_original_units('s'), which assumes a three-way
        # consistency between position, velocity and time units that gusteau does not respect
        # for cosmological simulations.
        self.properties['time'] = header['Time'] * self._hdf_unitvar['U_t']

    def _get_coordinate_units(self):
        """Return the units of the coordinate arrays, which are also those of the bounding box.

        These cannot be inferred from the file unit system alone, because a snapshot transcoded
        from a code which scales its lengths by the Hubble parameter retains that scaling.
        """
        for group_name in self._family_to_group_map[self._families_ordered()[0]]:
            for hdf_group in self._hdf_files.iter_particle_groups_with_name(group_name):
                if 'Coordinates' in hdf_group:
                    return self._get_units_from_hdf_attr(hdf_group['Coordinates'].attrs)
        return self.infer_original_units('m')

    # Gusteau particle groups nest their arrays into subgroups (e.g. Gas/Thermal/Temperatures),
    # and their Number_of_fields attribute counts the fields of the source snapshot rather than
    # the datasets present here. Use the generic GadgetHDF implementation, which walks the whole
    # group, rather than SWIFT's shortcut based on the field count.
    _get_hdf_allarray_keys = staticmethod(GadgetHDFSnap._get_hdf_allarray_keys)

    def write_array(self, *args, **kwargs):
        raise NotImplementedError(
            "pynbody cannot yet write arrays into a gusteau snapshot. Doing so would require "
            "updating the file's Metadata index, which describes every group, dataset and "
            "attribute it contains."
        )

    def halos(self, **kwargs):
        # SwiftSnap insists on the FOF parameter which records the 'no group' value, but a
        # transcoded snapshot need not carry the originating code's parameter file at all.
        if self._fof_group_id_default_key in self._hdf_files.get_parameter_attrs():
            return super().halos(**kwargs)
        return GadgetHDFSnap.halos(self, **kwargs)
