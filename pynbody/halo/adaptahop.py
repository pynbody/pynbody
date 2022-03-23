import os.path
import re
from typing import Sequence

import numpy as np

from .. import array, units, util
from ..extern.cython_fortran_utils import FortranFile
from . import DummyHalo, Halo, HaloCatalogue, logger

unit_length = units.Unit("Mpc")
unit_vel = units.Unit("km s**-1")
unit_mass = 1e11 * units.Unit("Msol")
unit_angular_momentum = unit_mass * unit_vel * unit_length
unit_energy = unit_mass * unit_vel ** 2
unit_temperature = units.Unit("K")
unit_density = unit_mass / unit_length ** 3

UNITS = {
    'pos_x': unit_length,
    'pos_y': unit_length,
    'pos_z': unit_length,
    'shape_a': unit_length,
    'shape_b': unit_length,
    'shape_c': unit_length,
    'nfw_R_c': unit_length,
    'max_distance': unit_length,
    'R200c': unit_length,
    'r_half_mass': unit_length,
    'r_90percent_mass': unit_length,
    'max_velocity_radius': unit_length,
    'radius_profile': unit_length,
    'virial_radius': unit_length,
    'velocity_dispersion': unit_vel,
    'vel_x': unit_vel,
    'vel_y': unit_vel,
    'vel_z': unit_vel,
    'max_velocity': unit_vel,
    'virial_velocity': unit_vel,
    'angular_momentum_x': unit_angular_momentum,
    'angular_momentum_y': unit_angular_momentum,
    'angular_momentum_z': unit_angular_momentum,
    'm': unit_mass,
    'M200c': unit_mass,
    'm_contam': unit_mass,
    'mtot': unit_mass,
    'mtot_contam': unit_mass,
    'virial_mass': unit_mass,
    'kinetic_energy': unit_energy,
    'potential_energy': unit_energy,
    'total_energy': unit_energy,
    'virial_temperature': unit_temperature,
    'nfw_rho0': unit_density,
    'density_profile': unit_density,
}


class BaseAdaptaHOPCatalogue(HaloCatalogue):
    """A AdaptaHOP Catalogue. AdaptaHOP output files must be in
    Halos/<simulation_number>/ directory or specified by fname"""

    _AdaptaHOP_fname = None
    _halos = None
    _headers = None
    _fname = ""
    _read_contamination = False

    _halo_attributes = tuple()
    _halo_attributes_contam = tuple()
    _header_attributes = tuple()

    def __init__(self, sim, fname=None, read_contamination=None, longint=None):
        if FortranFile is None:
            raise RuntimeError(
                "Support for AdaptaHOP requires the package `cython-fortran-file` to be installed."
            )

        if fname is None:
            for fname in AdaptaHOPCatalogue._enumerate_hop_tag_locations_from_sim(sim):
                if os.path.exists(fname):
                    break

            if not os.path.exists(fname):
                raise RuntimeError(
                    "Unable to find AdaptaHOP brick file in simulation directory"
                )
        if read_contamination is None or longint is None:
            read_contamination, longint = self._detect_file_format(fname)
        self._read_contamination = read_contamination
        self._longint = longint

        self._header_attributes = self.convert_i8b(self._header_attributes, longint)
        self._halo_attributes = self.convert_i8b(self._halo_attributes, longint)
        self._halo_attributes_contam = self.convert_i8b(self._halo_attributes_contam, longint)

        self._header_attributes = self.convert_i8b(self._header_attributes, longint)
        self._halo_attributes = self.convert_i8b(self._halo_attributes, longint)
        self._halo_attributes_contam = self.convert_i8b(self._halo_attributes_contam, longint)

        # Call parent class
        super().__init__(sim)

        # Initialize internal data
        self._base_dm = sim.dm

        self._halos = {}
        self._fname = fname
        self._AdaptaHOP_fname = fname

        logger.debug("AdaptaHOPCatalogue loading offsets")

        # Compute offsets
        self._ahop_compute_offset()
        logger.debug("AdaptaHOPCatalogue loaded")

    def _detect_file_format(self, fname):
        """
        Detect if the file is in the old format or the new format.
        """
        with FortranFile(fname) as fpu:
            longint_flag = self._detect_longint(fpu, (False, True))

            # Now attempts reading the first halo data
            attrs, attrs_contam = (self.convert_i8b(_, longint_flag) for _ in (self._halo_attributes, self._halo_attributes_contam))

            if len(attrs_contam) == 0:
                read_contamination = False
            else:
                fpu.skip(3) # number + ids of parts + halo_ID
                fpu.read_attrs(attrs)
                try:
                    fpu.read_attrs(attrs_contam)
                    read_contamination = True
                except (ValueError, OSError):
                    read_contamination = False

        return read_contamination, longint_flag

    @staticmethod
    def convert_i8b(original_headers, longint):
        headers = []
        for key, count, dtype in original_headers:
            if dtype == "i8b":
                if longint:
                    dtype = "l"
                else:
                    dtype = "i"
            headers.append((key, count, dtype))
        return tuple(headers)

    def precalculate(self):
        """Speed up future operations by precalculating the indices
        for all halos in one operation. This is slow compared to
        getting a single halo, however."""
        # Get the mapping from particle to halo
        self._base_dm._family_index()  # filling the cache
        self._group_array = self.get_group_array(group_to_indices=True)

    def _ahop_compute_offset(self):
        """
        Compute the offset in the brick file of each halo.
        """

        with FortranFile(self._fname) as fpu:
            self._headers = fpu.read_attrs(self._header_attributes)

            nhalos = self._headers["nhalos"]
            nsubs = self._headers["nsubs"]

            Nskip = len(self._halo_attributes)
            if self._read_contamination:
                Nskip += len(self._halo_attributes_contam)

            for _ in range(nhalos + nsubs):
                ipos = fpu.tell()
                fpu.skip(2)  # number + ids of parts
                halo_ID = fpu.read_int()
                fpu.skip(Nskip)

                # Fill-in data
                dummy = DummyHalo()
                dummy.properties["file_offset"] = ipos
                self._halos[halo_ID] = dummy

    def calc_item(self, halo_id):
        return self._get_halo(halo_id)

    def _get_halo(self, halo_id):
        if halo_id not in self._halos:
            raise KeyError(f"Halo with id '{halo_id}' does not seem to exist in the catalog.")

        halo = self._halos[halo_id]
        halo_dummy = self._halos[halo_id]
        if isinstance(halo, DummyHalo):
            halo = self._read_halo_data(halo_id, halo.properties["file_offset"])
            halo.dummy = halo_dummy
            self._halos[halo_id] = halo

        return halo

    def _read_member_helper(self, fpu, expected_size):
        """Read the member array from file, and return it *sorted*.

        The function automatically find whether the particle ids are stored in
        32 or 64 bits.
        """
        default_dtype = getattr(self.base, "_iord_dtype", "i")
        possible_dtypes = list({"i", "q"} - {default_dtype})

        dtypes = [default_dtype] + possible_dtypes
        ipos = fpu.tell()

        for dtype in dtypes:
            try:
                iord_array = fpu.read_vector(dtype)
                if iord_array.size == expected_size:
                    # Store dtype for next time
                    self.base._iord_dtype = dtype

                    if not util.is_sorted(iord_array) == 1:
                        return np.sort(iord_array)
                    else:
                        return iord_array

            except ValueError:
                pass
            # Rewind
            fpu.seek(ipos)

        # Could not read, throw an error
        raise RuntimeError("Could not read iord!")

    def _read_halo_data(self, halo_id, offset):
        """
        Reads the halo data from file -- AdaptaHOP specific.

        Parameters
        ----------
        halo_id : int
            The id of the halo
        offset : int
            The location in the file (in bytes)

        Returns
        -------
        halo : Halo object
            The halo object, filled with the data read from file.
        """
        with FortranFile(self._fname) as fpu:
            fpu.seek(offset)
            if self._longint:
                npart = fpu.read_int64()
            else:
                npart = fpu.read_int()
            iord_array = self._read_member_helper(fpu, npart)
            halo_id_read = fpu.read_int()
            assert halo_id == halo_id_read
            if self._read_contamination:
                attrs = self._halo_attributes + self._halo_attributes_contam
            else:
                attrs = self._halo_attributes
            props = fpu.read_attrs(attrs)

        # /!\: AdaptaHOP assumes that 1Mpc == 3.08e24 (exactly)
        boxsize = self.base.properties["boxsize"]
        Mpc2boxsize = boxsize.in_units("cm") / 3.08e24  # Hard-coded in AdaptaHOP...

        # Add units for known fields
        # NOTE: we need to list the items, as the dictionary is updated in place
        for k, v in list(props.items()):
            if k in ("pos_x", "pos_y", "pos_z"):
                # convert positions between [-Lbox/2, Lbox/2] to [0, Lbox].
                v = v / Mpc2boxsize + 0.5
                unit = boxsize
            elif k in UNITS:
                unit = UNITS[k]
            else:
                continue
            props[k] = array.SimArray(
                v,
                sim=self.base,
                units=unit,
                dtype=v.dtype,
            )

        props["file_offset"] = offset
        props["npart"] = npart
        props["members"] = iord_array
        # Create position and velocity
        props["pos"] = array.SimArray(
            [props["pos_x"], props["pos_y"], props["pos_z"]],
            units=props["pos_x"].units,
            sim=self.base,
        )
        props["vel"] = array.SimArray(
            [props["vel_x"], props["vel_y"], props["vel_z"]],
            units=props["vel_x"].units,
            sim=self.base,
        )

        # Create halo object and fill properties
        if hasattr(self, "_group_to_indices"):
            index_array = self._group_to_indices[halo_id]
            iord_array = None
        else:
            index_array = None
            iord_array = iord_array
        halo = Halo(
            halo_id, self, self._base_dm, index_array=index_array, iord_array=iord_array
        )
        for k, v in props.items():
            halo.properties[k] = v

        # Need to convert the units of the halo object as we
        # just updated them
        halo._autoconvert_properties()

        return halo

    def get_group_array(self, family="dm", group_to_indices=False):
        """Return an array with an integer for each particle in the simulation
        indicating which halo that particle is associated with. If there are multiple
        levels (i.e. subhalos), the number returned corresponds to the lowest level, i.e.
        the smallest subhalo.

        Arguments
        ---------
        family : optional, default : dm
            The family of the particles that make the group
        group_to_indices : optional, bool
            If True, store the mapping from groups to particle on-disk location.

        Returns
        -------
        igrp : int array
            An array that contains the index of the group that contains each particle.
        """
        logger.debug("Get_group_array")
        if family is None:
            family == self.base.families()[0]
        elif isinstance(family, str):
            families = self.base.families()
            matched_families = [f for f in families if f.name == family]
            if len(matched_families) != 1:
                raise Exception("Could not find family %s" % family)
            family = matched_families[0]
        try:
            data = self.base[family]
        except:
            logger.error(type(self.base))
            logger.error(type(family))
            raise

        iord = data["iord"]
        iord_argsort = data["iord_argsort"]

        igrp = np.zeros(len(data), dtype=int) - 1

        if group_to_indices:
            grp2indices = {}
        with FortranFile(self._fname) as fpu:
            for halo_id, halo in self._halos.items():
                fpu.seek(halo.properties["file_offset"])
                if self._longint:
                    npart = fpu.read_int64()
                else:
                    npart = fpu.read_int()
                particle_ids = self._read_member_helper(fpu, npart)

                indices = util.binary_search(particle_ids, iord, iord_argsort)
                assert all(indices < len(iord))

                igrp[indices] = halo_id

                if group_to_indices:
                    grp2indices[halo_id] = indices
        if group_to_indices:
            self._group_to_indices = grp2indices
        return igrp

    @classmethod
    def _detect_longint(cls, fpu: FortranFile, longint_flags: Sequence) -> bool:
        for longint_flag in longint_flags:
            try:
                fpu.seek(0)
                fpu.read_attrs(cls.convert_i8b(cls._header_attributes, longint_flag))
                return longint_flag
            except (ValueError, OSError):
                pass

        raise ValueError("Could not detect longint")

    @classmethod
    def _can_load(cls, sim, arr_name="grp", *args, **kwa):
        candidates = [
            fname
            for fname in cls._enumerate_hop_tag_locations_from_sim(sim)
        ]
        valid_candidates = [fname for fname in candidates if os.path.exists(fname)]
        if len(valid_candidates) == 0:
            return False

        # Logic is as follows:
        # - If `longint` is provided, try loading with it.
        # - Otherwise, try loading with and without it
        use_longint = kwa.pop("longint", None)
        if use_longint is None:
            longint_flags = [True, False]
        else:
            longint_flags = use_longint
        for fname in valid_candidates:
            with FortranFile(fname) as fpu:
                try:
                    cls._detect_longint(fpu, longint_flags)
                    return True
                except ValueError:
                    pass
        return False

    @staticmethod
    def _enumerate_hop_tag_locations_from_sim(sim):
        try:
            match = re.search("output_([0-9]{5})", sim.filename)
            if match is None:
                raise OSError(
                    "Cannot guess the HOP catalogue filename for %s" % sim.filename
                )
            isim = int(match.group(1))
            name = "tree_bricks%03d" % isim
        except (OSError, ValueError):
            return []

        s_filename = os.path.abspath(sim.filename)
        s_dir = os.path.dirname(s_filename)

        ret = [
            os.path.join(s_filename, name),
            os.path.join(s_filename, "Halos", name),
            os.path.join(s_dir, "Halos", name),
            os.path.join(s_dir, "Halos", "%d" % isim, name),
        ]
        return ret

    def _can_run(self, *args, **kwa):
        return False


class NewAdaptaHOPCatalogue(BaseAdaptaHOPCatalogue):
    _header_attributes = (
        ("npart", 1, "i8b"),
        ("massp", 1, "d"),
        ("aexp", 1, "d"),
        ("omega_t", 1, "d"),
        ("age", 1, "d"),
        (("nhalos", "nsubs"), 2, "i"),
    )

    # List of the attributes read from file. This does *not* include the number of particles,
    # the list of particles, the id of the halo and the "timestep" (first 4 records).
    _halo_attributes = (
        ("timestep", 1, "i"),
        (
            ("level", "host_id", "first_subhalo_id", "n_subhalos", "next_subhalo_id"),
            5,
            "i",
        ),
        ("m", 1, "d"),
        ("ntot", 1, "i8b"),
        ("mtot", 1, "d"),
        # Note: we use pos_z instead of z to prevent confusion with redshift
        (("pos_x", "pos_y", "pos_z"), 3, "d"),
        (("vel_x", "vel_y", "vel_z"), 3, "d"),
        (("angular_momentum_x", "angular_momentum_y", "angular_momentum_z"), 3, "d"),
        (("max_distance", "shape_a", "shape_b", "shape_c"), 4, "d"),
        (("kinetic_energy", "potential_energy", "total_energy"), 3, "d"),
        ("spin", 1, "d"),
        ("velocity_dispersion", 1, "d"),
        (("virial_radius", "virial_mass", "virial_temperature", "virial_velocity"), 4, "d"),
        (("max_velocity_radius", "max_velocity"), 2, "d"),
        ("nfw_concentration", 1, "d"),
        (("R200c", "M200c"), 2, "d"),
        (("r_half_mass", "r_90percent_mass"), 2, "d"),
        ("radius_profile", -1, "d"),
        ("density_profile", -1, "d"),
        (("nfw_rho0", "nfw_R_c"), 2, "d"),
    )

    _halo_attributes_contam = (
        ("contaminated", 1, "i"),
        (("m_contam", "mtot_contam"), 2, "d"),
        (("n_contam", "ntot_contam"), 2, "i8b"),
    )


class AdaptaHOPCatalogue(BaseAdaptaHOPCatalogue):
    _header_attributes = (
        ("npart", 1, "i"),
        ("massp", 1, "f"),
        ("aexp", 1, "f"),
        ("omega_t", 1, "f"),
        ("age", 1, "f"),
        (("nhalos", "nsubs"), 2, "i"),
    )
    # List of the attributes read from file. This does *not* include the number of particles,
    # the list of particles, the id of the halo and the "timestep" (first 4 records).
    _halo_attributes = (
        ("timestep", 1, "i"),
        (
            ("level", "host_id", "first_subhalo_id", "n_subhalos", "next_subhalo_id"),
            5,
            "i",
        ),
        ("m", 1, "f"),
        # Note: we use pos_z instead of z to prevent confusion with redshift
        (("pos_x", "pos_y", "pos_z"), 3, "f"),
        (("vel_x", "vel_y", "vel_z"), 3, "f"),
        (("angular_momentum_x", "angular_momentum_y", "angular_momentum_z"), 3, "f"),
        (("max_distance", "shape_a", "shape_b", "shape_c"), 4, "f"),
        (("kinetic_energy", "potential_energy", "total_energy"), 3, "f"),
        ("spin", 1, "f"),
        (("virial_radius", "virial_mass", "virial_temperature", "virial_velocity"), 4, "f"),
        (("nfw_rho0", "nfw_R_c"), 2, "f"),
    )

    _halo_attributes_contam = tuple()
