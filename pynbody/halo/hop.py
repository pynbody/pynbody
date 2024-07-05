import os.path
import re
import struct

import numpy as np

from .number_array import HaloNumberCatalogue


class HOPCatalogue(HaloNumberCatalogue):
    """Represents a HOP Catalogue as used by Ramses."""

    dtype = np.int32

    def __init__(self, sim, filename=None):
        """Create a new HOP catalogue object from the given simulation object.

        Parameters
        ----------

        sim : :class:`pynbody.snapshot.SimSnap`
            The simulation snapshot to which this catalogue applies.

        filename : str, optional
            The filename of the HOP catalogue. The filename should be either the ``grp*.tag`` file or the
            ``hop*.hop`` file that HOP produces. If no filename is provided, the code will attempt to find a
            suitable ``grp*.tag`` file in the Ramses output folder, in the folder in which the simulation is
            located, or in a ``hop`` subfolder.

        """

        if filename is None:
            for filename in HOPCatalogue._enumerate_hop_tag_locations_from_sim(sim):
                if os.path.exists(filename):
                    break

            if not os.path.exists(filename):
                raise RuntimeError("Unable to find HOP .tag file in simulation directory")

        sim._create_array('hop_grp', dtype=self.dtype)
        sim['hop_grp'] = -1
        with open(filename, "rb") as f:
            num_part = self._get_npart_from_file(f)

            if num_part != len(sim.dm):
                raise RuntimeError("Mismatching number of particles between "
                                   "snapshot %s and HOP file %s. Check your pynbody configuration for any missing"
                                   " particle fields or partial loading" % (sim.filename, filename))

            sim.dm['hop_grp'] = np.fromfile(f, self.dtype, len(sim.dm))
        super().__init__(sim, array="hop_grp", ignore=-1)

    @classmethod
    def _get_npart_from_file(cls, f):
        num_part, = struct.unpack('i', f.read(4))
        if num_part == 8:
            # fortran-formatted output
            num_part, num_grps, _, _ = struct.unpack('iiii', f.read(16))
        else:
            # plain binary output
            num_grps, = struct.unpack('i', f.read(4))
        return num_part

    @classmethod
    def _can_load(cls, sim, filename=None):
        # Hop output must be in output directory or in output_*/hop directory
        if filename is not None:
            if not os.path.exists(filename):
                return False
            with open(filename, "rb") as f:
                num_part = cls._get_npart_from_file(f)
            return num_part == len(sim.dm)
        else:
            exists = any([os.path.exists(fname) for fname in HOPCatalogue._enumerate_hop_tag_locations_from_sim(sim)])
            return exists

    @staticmethod
    def _extract_hop_name_from_sim(sim):
        match = re.search("output_([0-9]*)", sim.filename)
        if match is None:
            raise OSError("Cannot guess the HOP catalogue filename for %s" % sim.filename)
        return "grp%s.tag" % match.group(1)

    @staticmethod
    def _enumerate_hop_tag_locations_from_sim(sim):
        try:
            name = HOPCatalogue._extract_hop_name_from_sim(sim)
        except OSError:
            return []

        s_filename = os.path.abspath(sim.filename)

        return [os.path.join(os.path.dirname(s_filename),name),
                os.path.join(s_filename,name),
                os.path.join(s_filename,'hop',name)]
