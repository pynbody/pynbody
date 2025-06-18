import os.path
import re
from typing import Sequence

import numpy as np

from .. import array, units
from ..extern.cython_fortran_utils import FortranFile
from . import HaloCatalogue, logger
from .details import iord_mapping, number_mapping
from .details.particle_indices import HaloParticleIndices

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
    """Handles catalogues produced by AdaptaHOP."""

    _AdaptaHOP_fname = None
    _halos = None
    _headers = None
    _fname = ""
    _read_contamination = False

    _halo_attributes = tuple()
    _halo_attributes_contam = tuple()
    _header_attributes = tuple()

    def __init__(self, sim, filename=None, read_contamination=None, longint=None):
        """Initialise a AdaptaHOP catalogue.

        Parameters
        ----------

        sim : ~pynbody.snapshot.simsnap.SimSnap
            The snapshot to which this catalogue is attached.

        filename : str, optional
            The filename of the AdaptaHOP catalogue (``path/to/tree_bricksXXX``). If not specified, the
            code will attempt to find the catalogue in the simulation directory.

        read_contamination : bool, optional
            Whether to read information about contamination of each halo. If not specified, the code will attempt to
            detect the format. Note that if specifying read_contamination, longint must also be specified.

        longint : bool, optional
            Whether to read 64-bit integers. If not specified, the code will attempt to detect the format. Note that if
            specifying longint, read_contamination must also be specified.
        """
        if FortranFile is None:
            raise RuntimeError(
                "Support for AdaptaHOP requires the package `cython-fortran-file` to be installed."
            )

        if filename is None:
            for filename in AdaptaHOPCatalogue._enumerate_hop_tag_locations_from_sim(sim):
                if os.path.exists(filename):
                    break

            if not os.path.exists(filename):
                raise RuntimeError(
                    "Unable to find AdaptaHOP brick file in simulation directory"
                )

        self._fname = filename

        if (read_contamination or longint) is not None and (read_contamination is None or longint is None):
            raise ValueError("If specifying read_contamination or longint, both must be specified")

        if read_contamination is None or longint is None:
            read_contamination, longint = self._detect_file_format(filename)
        self._read_contamination = read_contamination
        self._longint = longint

        if self._longint:
            self._length_type = np.int64
        else:
            self._length_type = np.int32


        self._header_attributes = self.convert_i8b(self._header_attributes, longint)
        self._halo_attributes = self.convert_i8b(self._halo_attributes, longint)
        self._halo_attributes_contam = self.convert_i8b(self._halo_attributes_contam, longint)

        self._header_attributes = self.convert_i8b(self._header_attributes, longint)
        self._halo_attributes = self.convert_i8b(self._halo_attributes, longint)
        self._halo_attributes_contam = self.convert_i8b(self._halo_attributes_contam, longint)

        logger.debug("AdaptaHOPCatalogue loading offsets")

        self._get_halo_numbers_and_file_offsets()

        super().__init__(sim, number_mapper=number_mapping.create_halo_number_mapper(self._halo_numbers))

        # Initialize internal data
        self._base_dm = sim.dm
        self._iord_to_fpos = iord_mapping.make_iord_to_offset_mapper(self._base_dm["iord"])

        # dm needs to be at start of family map -- technically we will assume all particles are in dm
        # but then the parent class will use the position offsets as though they refer to the whole file
        dm_offset = self.base._get_family_slice('dm').start
        if dm_offset>0:
            self._iord_to_fpos = iord_mapping.IordOffsetModifier(self._iord_to_fpos, dm_offset)

        self._halos = {}

        self._AdaptaHOP_fname = filename


        logger.debug("AdaptaHOPCatalogue loaded")

    @classmethod
    def _detect_file_format(cls, fname):
        """
        Detect if the file is in the old format or the new format.
        """
        with FortranFile(fname) as fpu:
            longint_flag = cls._detect_longint(fpu, (False, True))

            # Now attempts reading the first halo data
            attrs, attrs_contam = (cls.convert_i8b(_, longint_flag) for _ in (cls._halo_attributes, cls._halo_attributes_contam))

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
                    dtype = "q"
                else:
                    dtype = "i"
            headers.append((key, count, dtype))
        return tuple(headers)

    def _get_halo_numbers_and_file_offsets(self):
        """
        Compute the offset in the brick file of each halo.
        """

        with FortranFile(self._fname) as fpu:
            self._headers = fpu.read_attrs(self._header_attributes)

            nhalos = self._headers["nhalos"]
            nsubs = self._headers["nsubs"]

            self._halo_numbers = np.empty(nhalos + nsubs, dtype=int)
            self._file_offsets = np.empty(nhalos + nsubs, dtype=self._length_type)
            self._npart = np.empty(nhalos + nsubs, dtype=self._length_type)

            Nskip = len(self._halo_attributes)
            if self._read_contamination:
                Nskip += len(self._halo_attributes_contam)

            for i in range(nhalos + nsubs):
                self._file_offsets[i] = fpu.tell()
                npart = fpu.read_int32_or_64()
                self._npart[i] = npart
                fpu.skip(1)  # skip over fortran field with ids of parts
                self._halo_numbers[i] = fpu.read_int32_or_64()
                fpu.skip(Nskip)     # skip over attributes

    def _get_all_particle_indices(self):
        particle_ids = np.empty(self._npart.sum(), dtype=self._length_type)
        particle_slices = np.empty((len(self), 2), dtype=self._length_type)
        start = 0

        with FortranFile(self._fname) as fpu:
            for i in range(len(self)):
                fpu.seek(self._file_offsets[i])
                npart = fpu.read_int32_or_64()

                stop = start+npart
                particle_ids[start:stop] = self._iord_to_fpos.map_ignoring_order(self._read_member_helper(fpu, npart))
                particle_slices[i] = [start, stop]
                start = stop

        assert stop == len(particle_ids)
        assert (particle_ids < len(self.base)).all()

        return HaloParticleIndices(particle_ids, particle_slices)

    def _get_particle_indices_one_halo(self, halo_number):
        halo_index = self.number_mapper.number_to_index(halo_number)
        offset = self._file_offsets[halo_index]
        with FortranFile(self._fname) as fpu:
            fpu.seek(offset)
            npart = fpu.read_int32_or_64()

            assert npart == self._npart[halo_index]

            return self._iord_to_fpos.map_ignoring_order(self._read_member_helper(fpu, npart))


    def _read_member_helper(self, fpu, expected_size):
        """Read the member array from file

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

                    # we used to perform a sort here, but it's now done in the IordToFposIndex class
                    return iord_array

            except ValueError:
                pass
            # Rewind
            fpu.seek(ipos)

        # Could not read, throw an error
        raise RuntimeError("Could not read iord!")

    def get_properties_one_halo(self, i):
        index = self.number_mapper.number_to_index(i)
        offset = self._file_offsets[index]

        with FortranFile(self._fname) as fpu:
            fpu.seek(offset)
            npart = fpu.read_int32_or_64()
            fpu.skip(1) # iord array

            # After PR#821, AdaptaHOP catalogues can have short in headers but long int iords,
            # or be using long ints fully everywhere. Updates in FortranFile now deal with this.
            halo_id_read = fpu.read_int32_or_64()
            assert i == halo_id_read
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

        return props


    @classmethod
    def _detect_longint(cls, fpu: FortranFile, longint_flags: Sequence) -> bool:
        for longint_flag in longint_flags:
            try:
                fpu.seek(0)
                fpu.read_attrs(cls.convert_i8b(cls._header_attributes, longint_flag))
                return longint_flag
            except (ValueError, OSError):
                pass

        raise ValueError(
            f"{cls.__name__} could not detect longint. "
            "Most likely, this class is expecting the wrong header/data blocks "
            "compared to what is stored in the halo catalogue."
        )

    @classmethod
    def _can_load(cls, sim, filename=None, arr_name="grp", *args, **kwa):
        if cls is BaseAdaptaHOPCatalogue:
            return False # Must load a specialisation

        if filename is None:
            candidates = [
                fname
                for fname in cls._enumerate_hop_tag_locations_from_sim(sim)
            ]
        else:
            candidates = [filename]
        valid_candidates = [fname for fname in candidates if os.path.exists(fname)]
        if len(valid_candidates) == 0:
            return False

        for fname in valid_candidates:
            try:
                cls._detect_file_format(fname)
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


class NewAdaptaHOPCatalogueFullyLongInts(NewAdaptaHOPCatalogue):
    _header_attributes = (
        ("npart", 1, "i8b"),
        ("massp", 1, "d"),
        ("aexp", 1, "d"),
        ("omega_t", 1, "d"),
        ("age", 1, "d"),
        (("nhalos", "nsubs"), 2, "i8b"),
    )

    # List of the attributes read from file. This does *not* include the number of particles,
    # the list of particles, the id of the halo and the "timestep" (first 4 records).
    _halo_attributes = (
        ("timestep", 1, "i8b"),
        (
            ("level", "host_id", "first_subhalo_id", "n_subhalos", "next_subhalo_id"),
            5,
            "i8b",
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
        ("contaminated", 1, "i8b"),
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
