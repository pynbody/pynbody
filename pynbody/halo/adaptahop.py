import os.path
import re
import struct
from itertools import repeat
from scipy.io import FortranFile as FF
import weakref

import numpy as np

from . import HaloCatalogue, Halo, logger
from .. import util, units
from ..snapshot.ramses import RamsesSnap

from ..extern.cython_fortran_utils import FortranFile


class DummyHalo(object):
    def __init__(self):
        self.properties = {}


unit_length = units.Mpc
unit_vel = units.km / units.s
unit_mass = 1e11 * units.Msol
unit_angular_momentum = unit_mass * unit_vel * unit_length
unit_energy = unit_mass * unit_vel ** 2
unit_temperature = units.K
unit_density = unit_mass / unit_length ** 3

MAPPING = (
    ("x y z a b c R_c r rvir", unit_length),
    ("vx vy vz vvir", unit_vel),
    ("lx ly lz", unit_angular_momentum),
    ("m mvir", unit_mass),
    ("ek ep etot", unit_energy),
    ("Tvir", unit_temperature),
    ("rho0", unit_density),
)
UNITS = {}
for k, u in MAPPING:
    for key, unit in zip(k.split(), repeat(u)):
        UNITS[key] = unit


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

    def __init__(self, sim, fname=None, read_contamination=False, longint=False):

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

        self._read_contamination = read_contamination
        self._longint = longint

        self._header_attributes = self.convert_i8b(self._header_attributes, longint)
        self._halo_attributes = self.convert_i8b(self._halo_attributes, longint)
        self._halo_attributes_contam = self.convert_i8b(self._halo_attributes_contam, longint)

        # Call parent class
        super(BaseAdaptaHOPCatalogue, self).__init__(sim)

        # Initialize internal data
        self._base_dm = sim.dm

        self._halos = {}
        self._fname = fname
        self._AdaptaHOP_fname = fname

        logger.debug("AdaptaHOPCatalogue loading offsets")

        # Compute offsets
        self._ahop_compute_offset()
        logger.debug("AdaptaHOPCatalogue loaded")

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
            if self._read_halo_data:
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
            raise Exception()

        halo = self._halos[halo_id]
        halo_dummy = self._halos[halo_id]
        halo = self._read_halo_data(halo_id, halo.properties["file_offset"])
        halo.dummy = halo_dummy

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

        # Convert positions between [-Lbox/2, Lbox/2] to [0, Lbox].
        # /!\: AdaptaHOP assumes that 1Mpc == 3.08e24 (exactly)
        boxsize = self.base.properties["boxsize"]
        Mpc2boxsize = boxsize.in_units("cm") / 3.08e24  # Hard-coded in AdaptaHOP...
        for k in "xyz":
            props[k] = boxsize.in_units("Mpc") * (props[k] / Mpc2boxsize + 0.5)

        # Add units for known fields
        for k, v in list(props.items()):
            if k in UNITS:
                props[k] = v * UNITS[k]

        props["file_offset"] = offset
        props["npart"] = npart
        props["members"] = iord_array

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
        halo.properties.update(props)

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
            logger.error((type(self.base)))
            logger.error((type(family)))
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
    def _can_load(cls, sim, arr_name="grp", *args, **kwa):
        candidates = [
            fname
            for fname in cls._enumerate_hop_tag_locations_from_sim(sim)
        ]
        valid_candidates = [fname for fname in candidates if os.path.exists(fname)]
        if len(valid_candidates) == 0:
            return False

        longint = kwa.pop("longint", False)
        for fname in valid_candidates:
            with FortranFile(fname) as fpu:
                try:
                    fpu.read_attrs(cls.convert_i8b(cls._header_attributes, longint))
                    return True
                except (ValueError, IOError):
                    pass

        return False

    @staticmethod
    def _enumerate_hop_tag_locations_from_sim(sim):
        try:
            match = re.search("output_([0-9]{5})", sim.filename)
            if match is None:
                raise IOError(
                    "Cannot guess the HOP catalogue filename for %s" % sim.filename
                )
            isim = int(match.group(1))
            name = "tree_bricks%03d" % isim
        except (IOError, ValueError):
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
        (("x", "y", "z"), 3, "d"),
        (("vx", "vy", "vz"), 3, "d"),
        (("lx", "ly", "lz"), 3, "d"),
        (("r", "a", "b", "c"), 4, "d"),
        (("ek", "ep", "etot"), 3, "d"),
        ("spin", 1, "d"),
        ("sigma", 1, "d"),
        (("rvir", "mvir", "Tvir", "vvir"), 4, "d"),
        (("rmax", "vmax"), 2, "d"),
        ("c", 1, "d"),
        (("r200", "m200"), 2, "d"),
        (("r50", "r90"), 2, "d"),
        ("rr3D", -1, "d"),
        ("rho3d", -1, "d"),
        (("rho0", "R_c"), 2, "d"),
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
        (("x", "y", "z"), 3, "f"),
        (("vx", "vy", "vz"), 3, "f"),
        (("lx", "ly", "lz"), 3, "f"),
        (("r", "a", "b", "c"), 4, "f"),
        (("ek", "ep", "etot"), 3, "f"),
        ("spin", 1, "f"),
        (("rvir", "mvir", "Tvir", "vvir"), 4, "f"),
        (("rho0", "R_c"), 2, "f"),
    )

    _halo_attributes_contam = tuple()