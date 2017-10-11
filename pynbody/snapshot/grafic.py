"""

grafic
======

Support for loading grafIC files
"""

from .. import util
from .. import array
from .. import chunk
from .. import family
from .. import analysis
from .. import units
from . import SimSnap

from ..util import read_fortran, read_fortran_series, grid_gen

import numpy as np
import os
import functools
import warnings
import scipy
import copy

_float_data_type = np.dtype('f4')
_int_data_type = np.dtype('i8')

genic_header = np.dtype([('nx', 'i4'), ('ny', 'i4'), ('nz', 'i4'),
                         ('dx', 'f4'), ('lx', 'f4'), ('ly', 'f4'),
                         ('lz', 'f4'), ('astart', 'f4'), ('omegam', 'f4'),
                         ('omegal', 'f4'), ('h0', 'f4')])

Tcmb = 2.72548


def _monitor(i):
    """Debug tool to monitor what's coming out of an iterable"""
    for q in i:
        print "_monitor:", q
        yield q


def _midway_fortran_skip(f, alen, pos):
    bits = np.fromfile(f, util._head_type, 2)
    assert (alen == bits[0] and alen == bits[1]
            ), "Incorrect FORTRAN block sizes"


_max_buflen = 1024 ** 2


class GrafICSnap(SimSnap):

    @staticmethod
    def _can_load(f):
        return os.path.isdir(f) and os.path.exists(os.path.join(f, "ic_velcx"))

    def __init__(self, f, take=None):
        super(GrafICSnap, self).__init__()
        f_cx = file(os.path.join(f, "ic_velcx"))
        self._header = read_fortran(f_cx, genic_header)[0]
        h = self._header
        self._dlen = int(h['nx'] * h['ny'])
        self.properties['a'] = float(h['astart'])
        self.properties['h'] = float(h['h0']) / 100.
        self.properties['omegaM0'] = float(h['omegam'])
        self.properties['omegaL0'] = float(h['omegal'])

        disk_family_slice = {family.dm: slice(0, self._dlen * int(h['nz']))}
        self._load_control = chunk.LoadControl(
            disk_family_slice, _max_buflen, take)
        self._family_slice = self._load_control.mem_family_slice
        self._num_particles = self._load_control.mem_num_particles
        self._filename = f

        boxsize = self._header['dx'] * self._header['nx']
        self.properties['boxsize'] = boxsize * units.Unit("Mpc a")

    def _derive_mass(self):
        boxsize = self._header['dx'] * self._header['nx']
        rho = analysis.cosmology.rho_M(self, unit='Msol Mpc^-3 a^-3')
        tot_mass = rho * boxsize ** 3  # in Msol
        part_mass = tot_mass / self._header['nx'] ** 3
        self._create_array('mass')
        self['mass'][:] = part_mass
        self['mass'].units = "Msol"

    def _derive_pos(self):
        self._create_array('pos', 3)
        self['pos'].units = "Mpc a"
        pos = self['pos']
        nx, ny, nz = [int(self._header[x]) for x in 'nx', 'ny', 'nz']

        # the following is equivalent to
        #
        # self['z'],self['y'],self['x'] = np.mgrid[0.0:self._header['nx'], 0.0:self._header['ny'], 0.0:self._header['nz']]
        #
        # but works on partial loading without having to generate the entire mgrid
        # (which might easily exceed the available memory for a big grid)

        pos_cache = np.empty((_max_buflen, 3))
        fp0 = 0
        for readlen, buf_index, mem_index in self._load_control.iterate(family.dm, family.dm):
            if mem_index is not None:
                pos[mem_index] = grid_gen(
                    slice(fp0, fp0 + readlen), nx, ny, nz, pos=pos_cache)[buf_index]
            fp0 += readlen

        self['pos'] *= self._header['dx'] * self._header['nx']
        a = self.properties['a']

        self['pos'] += self['zeldovich_offset']
        self['pos'] += (self._header['lx'], self._header['ly'], self._header['lz'])

    def _derive_vel(self):
        self._create_array('vel', 3)
        target_buffer = self['vel']
        target_buffer.units = 'km s^-1'
        h = self._header
        if self.properties['a'] != float(h['astart']):
            z0 = 1. / h['astart'] - 1
            a_bdot_original = (
                float(h['astart']) * analysis.cosmology.rate_linear_growth(self, z=z0))
            ratio = self.properties[
                'a'] * analysis.cosmology.rate_linear_growth(self) / a_bdot_original
            warnings.warn(
                "You have manually changed the redshift of these initial conditions before loading velocities; the velocities will be scaled as appropriate", RuntimeWarning)
        else:
            ratio = 1.0

        for vd in 'x', 'y', 'z':
            target_buffer = self['v' + vd]
            filename = os.path.join(self._filename, 'ic_velc' + vd)

            self._read_grafic_file(filename, target_buffer, _float_data_type)

            target_buffer*=ratio

    def _read_iord(self):
        # this is a proprietary extension to the grafic format used by genetIC
        filename = os.path.join(self._filename, 'ic_particle_ids')
        if not os.path.exists(filename):
            raise IOError, "No particle ID array"

        self._create_array('iord',dtype=_int_data_type)
        self._read_grafic_file(filename, self['iord'], _int_data_type)

    def _read_deltab(self):
        # this is a proprietary extension to the grafic format used by genetIC
        filename = os.path.join(self._filename, 'ic_deltab')
        if not os.path.exists(filename):
            raise IOError, "No deltab array"

        self._create_array('deltab',dtype=_float_data_type)
        self._read_grafic_file(filename, self['deltab'], _float_data_type)

    def _read_grafic_file(self, filename, target_buffer, data_type):
        with open(filename, 'rb') as f:
            h = read_fortran(f, genic_header)
            length = self._dlen * data_type.itemsize
            alen = np.fromfile(f, util._head_type, 1)
            if alen != length:
                raise IOError, "Unexpected FORTRAN block length %d!=%d" % (
                    alen, length)
            readpos = 0
            for readlen, buf_index, mem_index in (self._load_control.iterate_with_interrupts(family.dm, family.dm,
                                                                                             np.arange(
                                                                                                 1, h['nz']) * (
                                                                                                 h['nx'] * h['ny']),
                                                                                             functools.partial(
                                                                                                 _midway_fortran_skip, f,
                                                                                                 length))):

                if buf_index is not None:
                    re = np.fromfile(f, data_type, readlen)
                    target_buffer[mem_index] = re[buf_index]
                else:
                    f.seek(data_type.itemsize * readlen, 1)
            alen = np.fromfile(f, util._head_type, 1)
            if alen != length:
                raise IOError, "Unexpected FORTRAN block length (tail) %d!=%d" % (
                    alen, length)

    def _load_array(self, name, fam=None):

        if fam is not family.dm and fam is not None:
            raise IOError, "Only DM particles supported"

        if name == "mass":
            self._derive_mass()
        elif name == "pos":
            self._derive_pos()
        elif name == "vel":
            self._derive_vel()
        elif name=="iord":
            self._read_iord()
        elif name=="deltab":
            self._read_deltab()
        else:
            raise IOError, "No such array"
