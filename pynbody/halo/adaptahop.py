import os.path
import re
import struct
from itertools import repeat
from scipy.io import FortranFile as FF

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

    def __init__(self, sim, fname=None, read_contamination=False):

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

        # Call parent class
        super(BaseAdaptaHOPCatalogue, self).__init__(sim)

        # Initialize internal data
        self._halos = {}
        self._fname = fname
        self._AdaptaHOP_fname = fname
        self._read_contamination = read_contamination

        logger.debug("AdaptaHOPCatalogue loading offsets")

        # Compute offsets
        self._ahop_compute_offset()
        logger.debug("AdaptaHOPCatalogue loaded")

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
        if not isinstance(halo, Halo):
            halo = self._halos[halo_id] = self._read_halo_data(
                halo_id, halo.properties["file_offset"]
            )
        return halo

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

        default_dtype = getattr(self.base, "_iord_dtype", "i")
        possible_dtypes = list({"i", "q"} - {default_dtype})

        dtypes = [default_dtype] + possible_dtypes

        def _helper(fpu, expected_size):
            ipos = fpu.tell()

            for dtype in dtypes:
                try:
                    iord_array = fpu.read_vector(dtype)
                    if iord_array.size == expected_size:
                        # Store dtype for next time
                        self.base._iord_dtype = dtype
                        return iord_array

                except ValueError:
                    pass
                # Rewind
                fpu.seek(ipos)

            # Could not read, throw an error
            raise RuntimeError("Could not read iord for halo %s!", halo_id)

        with FortranFile(self._fname) as fpu:
            fpu.seek(offset)
            npart = fpu.read_int()
            iord_array = _helper(fpu, npart)
            halo_id_read = fpu.read_int()
            assert halo_id == halo_id_read
            if self._read_contamination:
                attrs = self._halo_attributes + self._halo_attributes_contam
            else:
                attrs = self._halo_attributes
            props = fpu.read_attrs(attrs)

        # Convert positions between [-Lbox/2, Lbox/2] to [0, Lbox].
        # /!\: AdaptaHOP assumes that 1Mpc == 3.08e24 (exactly)
        boxsize = self._base().properties["boxsize"]
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
        halo = Halo(
            halo_id, self, self.base.dm, index_array=None, iord_array=iord_array
        )
        halo.properties.update(props)

        return halo

    @staticmethod
    def _can_load(sim, arr_name="grp", *args, **kwa):
        exists = any(
            [
                os.path.exists(fname)
                for fname in AdaptaHOPCatalogue._enumerate_hop_tag_locations_from_sim(
                    sim
                )
            ]
        )
        return exists

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
        ("npart", 1, "i"),
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
        ("ntot", 1, "i"),
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
        (("n_contam", "ntot_contam"), 2, "i"),
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
