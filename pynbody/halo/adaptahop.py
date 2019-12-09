import os.path
import re
import struct
from scipy.io import FortranFile as FF

import numpy as np

from . import HaloCatalogue, Halo
from .. import util, fortran_utils as fpu


class DummyHalo(object):

    def __init__(self):
        self.properties = {}


class AdaptaHOPCatalogue(HaloCatalogue):
    """A AdaptaHOP Catalogue. AdaptaHOP output files must be in
    Halos/<simulation_number>/ directory or specified by fname"""

    _AdaptaHOP_fname = None
    _halos = None
    _metadata = None
    _fname = ''
    _read_contamination = False

    # List of the attributes read from file. This does *not* include the number of particles,
    # the list of particles, the id of the halo and the "timestep" (first 4 records).
    _halo_attributes = (
        (('level', 'host_id', 'first_subhalo_id', 'n_subhalos', 'next_subhalo_id'), 5, 'i'),
        ('m', 1, 'f'),
        (('x', 'y', 'z'), 3, 'f'),
        (('vx', 'vy', 'vz'), 3, 'f'),
        (('lx', 'ly', 'lz'), 3, 'f'),
        (('r', 'a', 'b', 'c'), 4, 'f'),
        (('ek', 'ep', 'etot'), 3, 'f'),
        ('spin', 1 , 'f'),
        (('rvir', 'mvir', 'Tvir', 'vvir'), 4, 'f'),
        (('rho0', 'R_c'), 2, 'f'))

    def __init__(self, sim, fname=None, read_contamination=False):

        if fname is None:
            for fname in AdaptaHOPCatalogue._enumerate_hop_tag_locations_from_sim(sim):
                if os.path.exists(fname):
                    break

            if not os.path.exists(fname):
                raise RuntimeError("Unable to find AdaptaHOP brick file in simulation directory")

        # Initialize internal data
        self._halos = {}
        self._fname = fname
        self._AdaptaHOP_fname = fname
        self._metadata = {}
        if read_contamination:
            raise NotImplementedError

        # Call parent class
        super(AdaptaHOPCatalogue, self).__init__(sim)

        # Compute offsets
        self._ahop_compute_offset()

    def _ahop_compute_offset(self):
        """
        Compute the offset in the brick file of each halo.
        """
        with open(self._fname, 'rb') as f:
            npart = fpu.read_vector(f, 'i')[0]
            massp = fpu.read_vector(f, 'f')[0]
            aexp = fpu.read_vector(f, 'f')[0]
            omega_t = fpu.read_vector(f, 'f')[0]
            age = fpu.read_vector(f, 'f')[0]
            nhalos, nsubs = fpu.read_vector(f, 'i')

            self._metadata.update(
                npart=npart, massp=massp, aexp=aexp, omega_t=omega_t, age=age, nhalos=nhalos,
                nsubs=nsubs
            )

            for _ in range(nhalos + nsubs):
                ipos = f.tell()
                # Structure is as follows:
                #    1. number of particles
                #    2. list of particles IDs
                #    3. halo ID
                #    4. timestep
                #    5. parent/children infos
                #    6. total mass
                #    7. center
                #    8. velocity
                #    9. angular momentum
                #   10. shape (ellipticity)
                #   11. energies
                #   12. spin parameter
                #   13. virial parameters
                #   14. NFW parameters
                # (15). contamination
                fpu.skip(f, 2)
                halo_ID = fpu.read_vector(f, 'i')[0]
                if self._read_contamination:
                    fpu.skip(f, 12)
                else:
                    fpu.skip(f, 11)
                dummy = DummyHalo()
                dummy.properties['file_offset'] = ipos
                self._halos[halo_ID] = dummy

    def calc_item(self, halo_id):
        return self._get_halo(halo_id)

    def _get_halo(self, halo_id):
        if halo_id not in self._halos:
            raise Exception()

        halo = self._halos[halo_id]
        if not isinstance(halo, Halo):
            halo = self._halos[halo_id] = \
                self._read_halo_data(halo_id, halo.properties['file_offset'])
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
        with open(self._fname, 'rb') as f:
            f.seek(offset)
            npart = fpu.read_vector(f, 'i')[0]
            index_array = fpu.read_vector(f, 'i')
            halo_id_read = fpu.read_vector(f, 'i')[0]
            assert halo_id == halo_id_read
            fpu.skip(f, 1)  # skip timestep
            parameters = fpu.read_attrs(f, self._halo_attributes)

        parameters['file_offset'] = offset
        parameters['npart'] = npart

        # Create halo object and fill properties
        halo = Halo(halo_id, self, self.base, index_array)
        halo.parameters.update(parameters)

        return halo


    @staticmethod
    def _can_load(sim, arr_name='grp'):
        exists = any([os.path.exists(fname) for fname in AdaptaHOPCatalogue._enumerate_hop_tag_locations_from_sim(sim)])
        return exists

    @staticmethod
    def _enumerate_hop_tag_locations_from_sim(sim):
        try:
            match = re.search("output_([0-9]{5})", sim.filename)
            if match is None:
                raise IOError("Cannot guess the HOP catalogue filename for %s" % sim.filename)
            isim = int(match.group(1))
            name = 'tree_bricks%d' % isim
        except (IOError, ValueError):
            return []

        s_filename = os.path.abspath(sim.filename)
        s_dir = os.path.dirname(s_filename)
        s_dir_parent = os.path.dirname(s_dir)

        return [os.path.join(s_dir, name),
                os.path.join(s_dir_parent, 'Halos', '%d' % isim, name)]

    def _can_run(self):
        return False