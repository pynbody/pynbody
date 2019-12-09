import os.path
import re
import struct
from scipy.io import FortranFile as FF

import numpy as np

from . import HaloCatalogue, Halo
from .. import util, fortran_utils as fpu

from yt.utilities.cython_fortran_utils import FortranFile


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
        ('m', 1, 'd'),
        ('ntot', 1, 'i'),
        ('mtot', 1, 'd'),
        (('x', 'y', 'z'), 3, 'd'),
        (('vx', 'vy', 'vz'), 3, 'd'),
        (('lx', 'ly', 'lz'), 3, 'd'),
        (('r', 'a', 'b', 'c'), 4, 'd'),
        (('ek', 'ep', 'etot'), 3, 'd'),
        ('spin', 1 , 'd'),
        ('sigma', 1, 'd'),
        (('rvir', 'mvir', 'Tvir', 'vvir'), 4, 'd'),
        (('rmax', 'vmax'), 2, 'd'),
        ('c', 1, 'd'),
        (('r200', 'm200'), 2, 'd'),
        (('r50', 'r90'), 2, 'd'),
        ('rr3D', -1, 'd'),
        ('rho3d', -1, 'd'),
        (('rho0', 'R_c'), 2, 'd')
    )

    _halo_attributes_contam = (
        ('contaminated', 1, 'i'),
        (('m_contam', 'mtot_contam'), 2, 'd'),
        (('n_contam', 'ntot_contam'), 2, 'i')
    )

    def __init__(self, sim, fname=None, read_contamination=False):

        if fname is None:
            for fname in AdaptaHOPCatalogue._enumerate_hop_tag_locations_from_sim(sim):
                print('Trying {fname}'.format(fname=fname))
                if os.path.exists(fname):
                    break

            if not os.path.exists(fname):
                raise RuntimeError("Unable to find AdaptaHOP brick file in simulation directory")

        # Initialize internal data
        self._halos = {}
        self._fname = fname
        self._AdaptaHOP_fname = fname
        self._metadata = {}
        self._read_contamination = read_contamination

        # Call parent class
        super(AdaptaHOPCatalogue, self).__init__(sim)

        # Compute offsets
        self._ahop_compute_offset()

    def _ahop_compute_offset(self):
        """
        Compute the offset in the brick file of each halo.
        """
        with FortranFile(self._fname) as fpu:
            npart = fpu.read_vector('i')[0]
            massp = fpu.read_vector('d')[0]
            aexp = fpu.read_vector('d')[0]
            omega_t = fpu.read_vector('d')[0]
            age = fpu.read_vector('d')[0]
            nhalos, nsubs = fpu.read_vector('i')

            self._metadata.update(
                npart=npart, massp=massp, aexp=aexp, omega_t=omega_t, age=age, nhalos=nhalos,
                nsubs=nsubs
            )

            for _ in range(nhalos + nsubs):
                ipos = fpu.tell()
                fpu.skip(2)  # number + ids of parts
                halo_ID = fpu.read_vector('i')[0]
                fpu.skip(1)  # timesteps

                # Skip data from halo
                fpu.skip(len(self._halo_attributes))
                if self._read_contamination:
                    fpu.skip(len(self._halo_attributes_contam))

                # Fill-in dummy container
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
        with FortranFile(self._fname) as fpu:
            fpu.seek(offset)
            npart = fpu.read_int()
            index_array = fpu.read_vector('i')
            halo_id_read = fpu.read_int()
            assert halo_id == halo_id_read
            fpu.skip(1)  # skip timestep
            if self._read_contamination:
                attrs = self._halo_attributes + self._halo_attributes_contam
            else:
                attrs = self._halo_attributes
            parameters = fpu.read_attrs(attrs)

        parameters['file_offset'] = offset
        parameters['npart'] = npart

        # Create halo object and fill properties
        halo = Halo(halo_id, self, self.base, index_array)
        halo.parameters.update(parameters)

        return halo


    @staticmethod
    def _can_load(sim, arr_name='grp', *args, **kwa):
        exists = any([os.path.exists(fname) for fname in AdaptaHOPCatalogue._enumerate_hop_tag_locations_from_sim(sim)])
        return exists

    @staticmethod
    def _enumerate_hop_tag_locations_from_sim(sim):
        try:
            match = re.search("output_([0-9]{5})", sim.filename)
            if match is None:
                raise IOError("Cannot guess the HOP catalogue filename for %s" % sim.filename)
            isim = int(match.group(1))
            name = 'tree_bricks%03d' % isim
        except (IOError, ValueError):
            return []

        s_filename = os.path.abspath(sim.filename)
        s_dir = os.path.dirname(s_filename)
        s_dir_parent = os.path.dirname(s_dir)

        return [os.path.join(s_dir, name),
                os.path.join(s_dir, 'Halos', name),
                os.path.join(s_dir_parent, 'Halos', '%d' % isim, name)]

    def _can_run(self, *args, **kwa):
        return False