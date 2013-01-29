"""

genic
=====

Support for loading genIC files
"""

from . import util
from . import snapshot
from . import array
from . import chunk
from . import family
from . import analysis
from . import units

from util import read_fortran, read_fortran_series

import numpy as np
import os
import functools
import warnings
import scipy, scipy.weave

_data_type = np.dtype('f4')

genic_header = np.dtype([('nx', 'i4'), ('ny', 'i4'), ('nz', 'i4'),
                         ('dx', 'f4'), ('lx', 'f4'), ('ly', 'f4'),
                         ('lz', 'f4'), ('astart', 'f4'), ('omegam', 'f4'),
                         ('omegal', 'f4'), ('h0','f4')])


def _monitor(i) :
    """Debug tool to monitor what's coming out of an iterable"""
    for q in i :
        print "_monitor:",q
        yield q
        
def _midway_fortran_skip(f, alen, pos) :
    bits = np.fromfile(f, util._head_type, 2)
    assert (alen==bits[0] and alen==bits[1]), "Incorrect FORTRAN block sizes"

def _grid_gen(indices, nx, ny, nz, pos=None):
    if pos is None :
        pos = np.empty((util.indexing_length(indices),3))
    if isinstance(indices, slice) :
        start = indices.start
        stop = indices.stop
        step = indices.step
        if step is None : step = 1
        code = """
        float x,y,z;
        int n;
        int i=0;
        for(int n=start; n< stop; n+=step) {
        x = n%nx;
        y = (n/nx)%ny;
        z = (n/(nx*ny))%nz;

        POS2(i,0)=float(x);
        POS2(i,1)=float(y);
        POS2(i,2)=float(z);
        i++;
        }
        """
        assert type(nx) is int
        assert type(step) is int
        scipy.weave.inline(code, ['pos', 'nx','ny','nz', 'start' ,'stop', 'step'])
    else :
        code = """
        float x,y,z;
        int n;
        for(int i=0; i<Nindices[0]; i++) {
        n=INDICES1(i);
        x = n%nx;
        y = (n/nx)%ny;
        z = (n/(nx*ny))%nz;

        POS2(i,0)=float(x);
        POS2(i,1)=float(y);
        POS2(i,2)=float(z);
        }
        """
        
        scipy.weave.inline(code, ['pos', 'nx','ny','nz', 'indices'])
    return pos


_max_buflen = 1024**2

class GenICSnap(snapshot.SimSnap) :
    @staticmethod
    def _can_load(f) :
        return os.path.isdir(f) and os.path.exists(os.path.join(f, "ic_velcx"))

    def __init__(self, f, take=None) :
        super(GenICSnap,self).__init__()
        f_cx = file(os.path.join(f, "ic_velcx"))
        self._header = read_fortran(f_cx, genic_header)
        h = self._header
        self._dlen = int(h['nx']*h['ny'])
        self.properties['a'] = float(h['astart'])
        self.properties['h'] = float(h['h0'])/100.
        self.properties['omegaM0'] = float(h['omegam'])
        self.properties['omegaL0'] = float(h['omegal'])
        
        disk_family_slice = { family.dm: slice(0, self._dlen*int(h['nz'])) }
        self._load_control = chunk.LoadControl(disk_family_slice, _max_buflen, take)
        self._family_slice = self._load_control.mem_family_slice
        self._num_particles = self._load_control.mem_num_particles
        self._filename = f

    def _load_array(self, name, fam=None) :
        
        if fam is not family.dm and fam is not None :
            raise IOError, "Only DM particles supported"

        if name=="mass" :
            boxsize = self._header['dx']*self._header['nx']
            rho = analysis.cosmology.rho_M(self, unit='Msol Mpc^-3 a^-3')
            tot_mass = rho*boxsize**3 # in Msol
            part_mass = tot_mass/self._header['nx']**3
            self._create_array('mass')
            self['mass'][:] = part_mass
            self['mass'].units="Msol"
            
        elif name=="pos" :
            self._create_array('pos', 3)
            self['pos'].units="Mpc a"
            pos=self['pos']
            nx,ny,nz = [int(self._header[x]) for x in 'nx','ny','nz']

            # the following is equivalent to
            #
            # self['z'],self['y'],self['x'] = np.mgrid[0.0:self._header['nx'], 0.0:self._header['ny'], 0.0:self._header['nz']]
            #
            # but works on partial loading without having to generate the entire mgrid
            # (which might easily exceed the available memory for a big grid)

            pos_cache = np.empty((_max_buflen,3))
            fp0 = 0
            for readlen, buf_index, mem_index in self._load_control.iterate(family.dm, family.dm) :
                if mem_index is not None :
                    pos[mem_index] = _grid_gen(slice(fp0, fp0+readlen), nx, ny, nz, pos=pos_cache)[buf_index]
                fp0+=readlen
            
            self['pos']*=self._header['dx']
            a = self.properties['a']
            bdot_by_b = analysis.cosmology.rate_linear_growth(self, unit='km Mpc^-1 s^-1')/analysis.cosmology.linear_growth_factor(self)

            # offset position according to zeldovich approximation
            self['offset'] = self['vel']/(a*bdot_by_b)
            self['offset'].units=self['vel'].units/units.Unit('km Mpc^-1 s^-1 a^-1')
            self['pos']+=self['offset']
            
        elif name=="vel" :
            self._create_array('vel', 3)
            vel = self['vel']
            vel.units='km s^-1'
            h = self._header
            if self.properties['a']!=float(h['astart']) :
                z0 = 1./h['astart']-1
                a_bdot_original =  (float(h['astart']) * analysis.cosmology.rate_linear_growth(self, z=z0))
                ratio = self.properties['a'] * analysis.cosmology.rate_linear_growth(self) / a_bdot_original
                warnings.warn("You have manually changed the redshift of these initial conditions before loading velocities; the velocities will be scaled as appropriate", RuntimeWarning)
            else :
                ratio = 1.0
                    
            for vd in 'x','y','z' :
                vel = self['v'+vd]
                f = file(os.path.join(self._filename, 'ic_velc'+vd))
                h = read_fortran(f, genic_header)
                
                length = self._dlen * _data_type.itemsize
                
                alen = np.fromfile(f, util._head_type, 1)
                if alen!=length :
                    raise IOError, "Unexpected FORTRAN block length %d!=%d"%(alen,length)

                readpos = 0
                
                for readlen, buf_index, mem_index in (self._load_control.iterate_with_interrupts(family.dm, family.dm,
                                                                                                 np.arange(1,h['nz'])*(h['nx']*h['ny']),
                                                                                                 functools.partial(_midway_fortran_skip, f, length))) :

                    if buf_index is not None :
                        re = np.fromfile(f, _data_type ,readlen)
                        vel[mem_index] = re[buf_index]*ratio
                    else :
                        f.seek(_data_type.itemsize*readlen, 1)

                alen = np.fromfile(f, util._head_type, 1)
                if alen!=length :
                    raise IOError, "Unexpected FORTRAN block length (tail) %d!=%d"%(alen,length)
                
        else :
            raise IOError, "No such array"
                
