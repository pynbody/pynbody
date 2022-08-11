"""

grafic
======

Support for loading grafIC files
"""

import glob
import os
import warnings

import numpy as np

from .. import analysis, chunk, family, units
from ..extern.cython_fortran_utils import FortranFile
from ..util import grid_gen
from . import SimSnap

_float_data_type = 'f'
_int_data_type = 'l'

genic_header = dict(
    keys=('nx', 'ny', 'nz', 'dx', 'lx', 'ly', 'lz', 'astart', 'omegam', 'omegal', 'h0'),
    dtype='i,i,i,f,f,f,f,f,f,f,f'
)

Tcmb = 2.72548


def _monitor(i):
    """Debug tool to monitor what's coming out of an iterable"""
    for q in i:
        print("_monitor:", q)
        yield q


_max_buflen = 1024 ** 2


class GrafICSnap(SimSnap):

    @staticmethod
    def _can_load(f):
        return os.path.isdir(f) and os.path.exists(os.path.join(f, "ic_velcx"))

    def __init__(self, f, take=None, use_pos_file=True):
        super().__init__()
        with FortranFile(os.path.join(f, "ic_velcx")) as f_cx:
            self._header = {
                k: v
                for k, v in zip(genic_header['keys'], f_cx.read_vector(genic_header['dtype'])[0])}
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
        self._use_pos_file = self._can_use_pos_file() and use_pos_file

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
        self._setup_pos_grid()
        if self._use_pos_file:
            self._displace_pos_from_file()
        else:
            self._displace_pos_zeldovich()

    def _can_use_pos_file(self):
        return os.path.exists(os.path.join(self._filename, 'ic_poscx'))

    def _displace_pos_from_file(self):
        for vd in 'x', 'y', 'z':
            target_buffer = self[vd]
            filename = os.path.join(self._filename, 'ic_posc' + vd)
            diff_buffer = np.empty_like(target_buffer)

            self._read_grafic_file(filename, diff_buffer, _float_data_type)

            # Do unit conversion. Caution: this assumes displacements are in Mpc a h^-1, which is slightly inconsistent
            # with the original GrafIC documentation -- but later implementations of the format seem to assume this.
            diff_buffer/=self.properties['h']
            target_buffer+=diff_buffer


    def _displace_pos_zeldovich(self):
        self['pos'] += self['zeldovich_offset']

    def _setup_pos_grid(self):
        self._create_array('pos', 3)
        self['pos'].units = "Mpc a"
        pos = self['pos']
        nx, ny, nz = (int(self._header[x]) for x in ('nx', 'ny', 'nz'))
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
            raise OSError("No particle ID array")

        self._create_array('iord',dtype=_int_data_type)
        self._read_grafic_file(filename, self['iord'], _int_data_type)

    def _read_deltab(self):
        # this is a proprietary extension to the grafic format used by genetIC
        filename = os.path.join(self._filename, 'ic_deltab')
        if not os.path.exists(filename):
            raise OSError("No deltab array")

        self._create_array('deltab',dtype=_float_data_type)
        self._read_grafic_file(filename, self['deltab'], _float_data_type)

    def _read_refmap(self):
        # refinement map as produced by MUSIC and genetIC
        filename = os.path.join(self._filename, 'ic_refmap')
        if not os.path.exists(filename):
            raise OSError("No refmap array")

        self._create_array('refmap',dtype=_float_data_type)
        self._read_grafic_file(filename, self['refmap'], _float_data_type)

    def _read_pvar(self):
        # passive variable map as produced by MUSIC and genetIC
        filename = os.path.join(glob.glob(os.path.join(self._filename, "ic_pvar*[0-9]"))[0])
        if not os.path.exists(filename):
            raise OSError("No pvar array")

        self._create_array('pvar',dtype=_float_data_type)
        self._read_grafic_file(filename, self['pvar'], _float_data_type)

    def _read_grafic_file(self, filename, target_buffer, data_type):
        with FortranFile(filename) as f:
            h = {k: v for k, v
                 in zip(genic_header['keys'], f.read_vector(genic_header['dtype'])[0])}

            def dummy_interrupt(pos):
                pass

            for readlen, buf_index, mem_index in \
                    self._load_control.iterate_with_interrupts(
                        family.dm, family.dm,
                        np.arange(1, h['nz']) * (h['nx'] * h['ny']),
                        dummy_interrupt):
                if buf_index is None:
                    f.skip(1)
                    continue
                sliced_data = f.read_vector(data_type)
                if len(sliced_data) != self._dlen:
                    raise OSError(
                        'Expected a slice of length {}, got {}'.format(
                            self._dlen, len(sliced_data)
                        ))
                target_buffer[mem_index] = sliced_data[buf_index]

    def _load_array(self, name, fam=None):

        if fam is not family.dm and fam is not None:
            raise OSError("Only DM particles supported")

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
        elif name == "refmap":
            self._read_refmap()
        elif name == "pvar":
            self._read_pvar()
        else:
            raise OSError("No such array")
