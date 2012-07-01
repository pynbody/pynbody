"""

ramses
======

Implements classes and functions for handling RAMSES files. AMR cells
are loaded as particles. You rarely need to access this module
directly as it will be invoked automatically via pynbody.load.

"""

from __future__ import with_statement # for py2.5
from __future__ import division

from . import snapshot, array, util
from . import family
from . import units
from . import config, config_parser
from . import chunk

import struct, os
import numpy as np
import gzip
import sys
import warnings
import copy
import types
import math
import re

_head_type = np.dtype('i4')
_float_type = np.dtype('f8')

def _read_fortran(f, dtype, n=1) :
    if not isinstance(dtype, np.dtype) :
        dtype = np.dtype(dtype)
        
    length = n * dtype.itemsize
    alen = np.fromfile(f, _head_type, 1)
    if alen!=length :
        raise IOError, "Unexpected FORTRAN block length %d!=%d"%(alen,length)
    data = np.fromfile(f, dtype, n)
    alen = np.fromfile(f, _head_type, 1)
    if alen!=length :
        raise IOError, "Unexpected FORTRAN block length (tail) %d!=%d"%(alen,length)

    return data

def _read_fortran_series(f, dtype) :
    q = np.zeros(1,dtype=dtype)
    for i in xrange(len(dtype.fields)) :
        q[0][i] = _read_fortran(f, dtype[i], 1)
    return q[0]

def _timestep_id(basename) :
    try:
        return re.findall("output_([0-9]*)/*$", basename)[0]
    except IndexError :
        return None
    
def _cpu_id(i) :
    return str(i).rjust(5,"0")

ramses_particle_header = np.dtype([('ncpu', 'i4'), ('ndim', 'i4'), ('npart', 'i4'),
                                   ('randseed', 'i4', (4,)), ('nstar', 'i4'), ('mstar', 'f8'),
                                   ('mstar_lost', 'f8'), ('nsink', 'i4')])

particle_blocks = map(str.strip,config_parser.get('ramses',"particle-blocks").split(","))

class RamsesSnap(snapshot.SimSnap) :
    def __init__(self, dirname, **kwargs) :

        warnings.warn("RamsesSnap is in development and can currently only handle particle data (assumed dark matter particles)", RuntimeWarning)
        
        global config
        super(RamsesSnap,self).__init__()

        self._timestep_id = _timestep_id(dirname)
        self._filename = dirname

        ndm = self._count_dm_particles()
        
        self._num_particles = ndm
        self._family_slice[family.dm] = slice(0, ndm)
        
        self._decorate()


    def _particle_filename(self, cpu_id) :
        return self._filename+"/part_"+self._timestep_id+".out"+_cpu_id(cpu_id)

    def _count_dm_particles(self) :
        
        fn = self._filename
        
        f = file(self._particle_filename(1))
    
        head = _read_fortran_series(f, ramses_particle_header)
        assert head['ndim']==3

        npart = head['npart']
        self.ncpu = head['ncpu']

        for i in xrange(2, self.ncpu+1) :
            f = file(self._particle_filename(i))
            npart+=_read_fortran_series(f, ramses_particle_header)['npart']

        return npart

    def _load_dm_block(self, blockname) :
        offset = particle_blocks.index(blockname)
        ind0 = 0

        for i in xrange(1, self.ncpu+1) :
            f = file(self._particle_filename(i))
            header = _read_fortran_series(f, ramses_particle_header)
            skip_size = header['npart']*_float_type.itemsize+2*_head_type.itemsize
            
            f.seek(skip_size*offset,1)
            ind1 = ind0+header['npart']
  
            self.dm[blockname][ind0:ind1]=_read_fortran(f, _float_type, header['npart'])
            f.close()
            ind0 = ind1
            
    def _load_array(self, array_name, fam=None) :

        # Framework always calls with 3D name. Ramses blocks are
        # stored as 1D slices.
        if array_name in self._split_arrays :
            for array_1D in self._array_name_ND_to_1D(array_name) :
                self._load_array(array_1D, fam)
         
        elif array_name in particle_blocks :
            if array_name not in self :
                self._create_array(array_name)
            self._load_dm_block(array_name)

        
        if array_name in self and hasattr(self[array_name].units, "_no_unit") :
            self[array_name].units = self._default_units_for(array_name)
                            
    @staticmethod
    def _can_load(f) :
        tsid = _timestep_id(f)
        if tsid :
            return os.path.isdir(f) and os.path.exists(f+"/info_"+tsid+".txt")
        return False


@RamsesSnap.decorator
def load_infofile(sim) :
    sim._info = {}
    f = file(sim._filename+"/info_"+_timestep_id(sim._filename)+".txt")
    for l in f :
        if '=' in l :
            name, val = map(str.strip,l.split('='))
            try:
                if '.' in val :
                    sim._info[name] = float(val)
                else :
                    sim._info[name] = int(val)
            except ValueError :
                sim._info[name] = val


@RamsesSnap.decorator
def translate_info(sim) :

    cosmo = 'aexp' in sim._info
    
    sim.properties['a'] = sim._info['aexp']
    sim.properties['omegaM0'] = sim._info['omega_m']
    sim.properties['omegaL0'] = sim._info['omega_l']
    

    # N.B. these conversion factors provided by ramses already have the
    # correction from comoving to physical units
    d_unit = sim._info['unit_d']*units.Unit("g cm^-3")
    t_unit = sim._info['unit_t']*units.Unit("s")
    l_unit = sim._info['unit_l']*units.Unit("cm")

    sim.properties['boxsize'] = sim._info['boxlen'] * l_unit

    sim._file_units_system = [d_unit, t_unit, l_unit]
