"""Implements reading gusteau snapshots.

GUSTEAU (the Grand Unified Snapshot That Everyone Agrees Upon,
https://gusteau-spec.readthedocs.io/) is a specification for a self-describing,
code-agnostic HDF5 snapshot format, intended both as an output option for
simulation codes and as a target for transcoding existing snapshots. It is
closely modelled on SWIFT's HDF5 output: particle groups carry descriptive
names (``Gas``, ``ColdDarkMatter``, ``Stars``, ...) with their arrays organised
into subgroups, and every dataset carries SWIFT-style unit exponents. This
module therefore builds on :class:`~pynbody.snapshot.swift.SwiftSnap`, adjusting
the metadata locations and attribute names that gusteau spells differently.

A gusteau file has five required top-level groups -- ``Header``, ``Code``,
``Cosmology``, ``RunInfo`` and ``Units`` -- and carries the Gadget-2 header
fields for backwards compatibility. Its particle groups are conventionally
aliased onto the Gadget-2 ``PartTypeN`` names, and pynbody currently reads the
arrays through those aliases exactly as it does for other GadgetHDF variants,
rather than following ``/Header.Particle_names`` and
``/Header.Part_type_mapping``. A snapshot with particle types that no
``PartTypeN`` name is mapped onto therefore exposes only the mapped ones.

Support is currently minimal: snapshots can be opened, their properties and
arrays read, and units interpreted. Reading the optional ``/Cubes`` spatial
index (and hence selecting a sub-region), and writing arrays back out, are not
yet implemented.

This module was written against version 0.3.0 of the specification.
"""

import warnings

import h5py
import numpy as np

from .. import units
from .gadgethdf import GadgetHDFSnap
from .swift import ExtractScalarWrapper, SwiftMultiFileManager, SwiftSnap


class GusteauMultiFileManager(SwiftMultiFileManager):
    """Manages access to the HDF5 file of a gusteau snapshot.

    Gusteau keeps its run metadata in different places to SWIFT: the internal
    code units live in a top-level ``Units`` group (as for plain GadgetHDF),
    while the run's parameter file contents are nested inside ``RunInfo``.
    """

    def _get_num_files(self, first_file):
        """Return the number of files in the snapshot, which for gusteau is always one.

        ``/Header.Num_files_per_snapshot`` records how many files the *source* snapshot was
        spread across, but a gusteau snapshot presents all its particles through virtual
        datasets in a single file regardless, so pynbody must not go looking for siblings.
        """
        return 1

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
            "pynbody cannot yet read gusteau's optional /Cubes spatial index, so it is not "
            "possible to load only part of a gusteau snapshot by region. Note that the index "
            "is written by the packingcubes package, whose on-disk format the gusteau "
            "specification describes as not yet finalised."
        )


class GusteauSnap(SwiftSnap):
    """Reads gusteau snapshots.

    .. versionadded:: 2.8.0
    """

    _multifile_manager_class = GusteauMultiFileManager

    _readable_hdf5_test_key = "Header"

    _gusteau_required_groups = ('Header', 'Code', 'Cosmology', 'RunInfo', 'Units')
    """The top-level groups which the gusteau specification requires to be present"""

    _length_unit_key = 'Unit_length_CGS'
    _mass_unit_key = 'Unit_mass_CGS'
    _time_unit_key = 'Unit_time_CGS'
    _temperature_unit_key = 'Unit_temperature_CGS'

    _cosmological_flag_key = 'Is_cosmological'
    _fof_group_id_default_key = 'FOF_group_id_default'
    _scalefactor_unitvar_name = 'a_scale'
    _hubble_unitvar_name = 'h_scale'

    _cosmology_attr_map = (('omegaM0', 'Omega_matter'),
                           ('omegaL0', 'Omega_lambda'),
                           ('omegaB0', 'Omega_baryon'),
                           ('omegaC0', 'Omega_darkmatter'),
                           ('omegaNu0', 'Omega_nu_0'))

    _unknown_cosmology_value = -1
    """The value gusteau's 'best effort' cosmology attributes take when the quantity is unknown"""

    _namemapper_config_section = 'gusteau-name-mapping'

    @classmethod
    def _test_for_hdf5_key(cls, f):
        """Return True if the given file follows the gusteau specification.

        Gusteau files are deliberately also readable as GadgetHDF files: their headers carry the
        Gadget-2 header fields, and their particle groups are aliased onto the traditional
        ``PartTypeN`` names. Identification therefore has to rest on something only gusteau
        writes.

        The spec mandates five top-level groups, and a ``/Header.Source`` attribute which names
        either the specification version the file was written to (``GUSTEAUvX.Y.Z``) or the
        translation used to produce it. That combination is what we look for here; ``Source`` is
        also where to look if version-dependent behaviour is ever needed.
        """
        with h5py.File(f, "r") as h5test:
            if not all(group in h5test for group in cls._gusteau_required_groups):
                return False
            return 'Source' in h5test['Header'].attrs

    @classmethod
    def _unit_name_from_exponent_attr_name(cls, attr_name):
        """Return the unit variable named by an exponent attribute, e.g. 'U_L' for 'U_L_exponent'."""
        return attr_name[:-len("_exponent")]

    def _init_properties(self):
        header = ExtractScalarWrapper(self._hdf_files[0]['Header'].attrs)
        cosmo = ExtractScalarWrapper(self._hdf_files[0]['Cosmology'].attrs)

        dimension = int(header['Dimension'])
        assert dimension == 3, \
            "Sorry, pynbody is only set up to deal with 3-dimensional gusteau simulations"

        cosmological = self._is_cosmological()

        if cosmological:
            # note that pynbody derives 'z' from 'a', so there is no need to set it explicitly
            self.properties['a'] = cosmo['Scale_factor']
            self.properties['h'] = cosmo['h']
            for pynbody_name, gusteau_name in self._cosmology_attr_map:
                # only Omega_matter and Omega_lambda are required; the others are 'best effort'
                # and take the value -1 when the writer did not know them
                if gusteau_name in cosmo.underlying:
                    value = cosmo[gusteau_name]
                    if value != self._unknown_cosmology_value:
                        self.properties[pynbody_name] = value

            self.properties['boxsize'] = self._get_boxsize(header, dimension)

        self.properties['time'] = self._get_time(header, cosmological)

    def _get_boxsize(self, header, dimension):
        """Return the side length of the simulation volume, from the gusteau bounding box.

        Gusteau stores ``/Header.Bounding_box`` as an origin followed by widths, i.e.
        ``[x, y, z, dx, dy, dz]`` in 3D, truncated to ``[x, dx]`` in 1D and so on.

        The spec fixes the bounding box's unit conversion (multiply by ``/Units.Unit_length_CGS``)
        but says nothing about its cosmological scalings, which for a dataset are recorded in the
        dataset's own ``a_scale_exponent`` and ``h_scale_exponent``. We therefore take the box to
        share the units of the coordinates, which is the reading the spec's own description of
        ``Coordinates`` -- a position "within the periodic simulation domain of BoxSize" --
        implies, and the only one under which pynbody can wrap positions into the box.
        """
        bounding_box = np.asarray(header.underlying['Bounding_box'])
        side_lengths = bounding_box[dimension:2 * dimension]
        assert np.all(side_lengths == side_lengths[0]), \
            "Sorry, pynbody is only set up to deal with cubic gusteau simulation volumes"
        return side_lengths[0] * self._get_coordinate_units()

    def _get_time(self, header, cosmological):
        """Return the time of this snapshot, from the gusteau header.

        ``/Header.Time`` is the snapshot time in internal units, except that a code with no
        notion of absolute time may instead store the negative of the scale factor. In that case
        recover the age from the cosmology, as pynbody does for other formats which omit the time.
        """
        time = header['Time']

        if time >= 0:
            # As for swift, this should NOT be infer_original_units('s'), which assumes a
            # three-way consistency between position, velocity and time units that gusteau does
            # not respect for cosmological simulations.
            return time * self._hdf_unitvar['U_t']

        if not cosmological:
            warnings.warn("The gusteau header gives a negative time, which the specification "
                          "reserves for the scale factor of a cosmological simulation, but this "
                          "snapshot declares itself non-cosmological. Taking the time at face "
                          "value.", RuntimeWarning)
            return time * self._hdf_unitvar['U_t']

        from .. import analysis
        return analysis.cosmology.age(self) * units.Gyr

    def _get_coordinate_units(self):
        """Return the units of the coordinate arrays.

        These cannot be inferred from the file unit system alone, because a snapshot transcoded
        from a code which scales its lengths by the Hubble parameter retains that scaling in the
        per-dataset exponents.
        """
        for group_name in self._family_to_group_map[self._families_ordered()[0]]:
            for hdf_group in self._hdf_files.iter_particle_groups_with_name(group_name):
                if 'Coordinates' in hdf_group:
                    return self._get_units_from_hdf_attr(hdf_group['Coordinates'].attrs)
        return self.infer_original_units('m')

    # Gusteau particle groups nest their arrays into subgroups (e.g. Gas/Thermal/Temperatures),
    # so use the generic GadgetHDF implementation, which walks the whole group, rather than
    # SWIFT's shortcut based on a count of the group's immediate children.
    _get_hdf_allarray_keys = staticmethod(GadgetHDFSnap._get_hdf_allarray_keys)

    def write_array(self, *args, **kwargs):
        raise NotImplementedError(
            "pynbody cannot yet write arrays into a gusteau snapshot. Doing so would require "
            "updating the spec-mandated /.Metadata attribute, which indexes every group, "
            "dataset and attribute in the file."
        )

    def halos(self, **kwargs):
        # SwiftSnap insists on the FOF parameter which records the 'no group' value, but the
        # RunInfo/Parameters group is only 'best effort' in gusteau, and in any case a snapshot
        # transcoded from another code will use that code's parameter names.
        if self._fof_group_id_default_key in self._hdf_files.get_parameter_attrs():
            return super().halos(**kwargs)
        return GadgetHDFSnap.halos(self, **kwargs)
