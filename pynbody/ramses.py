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
_int_type = np.dtype('i4')

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

def _skip_fortran(f, n=1) :
    for i in xrange(n) :
        alen = np.fromfile(f, _head_type, 1)
        f.seek(alen,1)
        alen2 = np.fromfile(f, _head_type, 1)
        assert alen==alen2
        
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

ramses_amr_header = np.dtype([('ncpu', 'i4'), ('ndim', 'i4'), ('ng', 'i4', (3,)), 
                              ('nlevelmax', 'i4'), ('ngridmax', 'i4'), ('nboundary', 'i4'),
                              ('ngrid', 'i4'), ('boxlen', 'f8')])

ramses_hydro_header = np.dtype([('ncpu', 'i4'), ('nvarh', 'i4'), ('ndim', 'i4'), ('nlevelmax', 'i4'),
                                ('nboundary', 'i4'), ('gamma', 'f8')])

particle_blocks = map(str.strip,config_parser.get('ramses',"particle-blocks").split(","))

hydro_blocks = map(str.strip,config_parser.get('ramses',"hydro-blocks").split(","))

class RamsesSnap(snapshot.SimSnap) :
    def __init__(self, dirname, **kwargs) :
        """Initialize a RamsesSnap. Extra kwargs supported:

         *cpus* : a list of the CPU IDs to load in. If not set, load all CPU's data.
         """

        warnings.warn("RamsesSnap is in development and may not behave well", RuntimeWarning)
        
        global config
        super(RamsesSnap,self).__init__()

        self._timestep_id = _timestep_id(dirname)
        self._filename = dirname
        self._load_infofile()

        assert self._info['ndim']==3
        self._ndim = 3 # in future could support lower dimensions
        
        self.ncpu = self._info['ncpu']

        if 'cpus' in kwargs :
            self._cpus = kwargs['cpus']
        else :
            self._cpus = range(1, self.ncpu+1)

        self._maxlevel = kwargs.get('maxlevel',None)
        
        ndm = self._count_dm_particles()
        ngas = self._count_gas_cells()
        
        self._num_particles = ndm+ngas
        self._family_slice[family.dm] = slice(0, ndm)
        self._family_slice[family.gas] = slice(ndm, ndm+ngas)
        
        self._decorate()



    def _load_infofile(self) :
        self._info = {}
        f = file(self._filename+"/info_"+_timestep_id(self._filename)+".txt")
        for l in f :
            if '=' in l :
                name, val = map(str.strip,l.split('='))
                try:
                    if '.' in val :
                        self._info[name] = float(val)
                    else :
                        self._info[name] = int(val)
                except ValueError :
                    self._info[name] = val

    def _particle_filename(self, cpu_id) :
        return self._filename+"/part_"+self._timestep_id+".out"+_cpu_id(cpu_id)

    def _amr_filename(self, cpu_id) :
        return self._filename+"/amr_"+self._timestep_id+".out"+_cpu_id(cpu_id)

    def _hydro_filename(self, cpu_id) :
        return self._filename+"/hydro_"+self._timestep_id+".out"+_cpu_id(cpu_id)

    def _count_dm_particles(self) :

        npart = 0

        for i in self._cpus :
            f = file(self._particle_filename(i))
            npart+=_read_fortran_series(f, ramses_particle_header)['npart']

        return npart

    def _count_gas_cells(self) :
        ncell = 0

        for coords, refine, cpu, level in self._level_iterator() :
            ncell+=(refine==0).sum()
            

        """Old code only works when reading all CPUs :-(

        for i in self._cpus :
            f = file(self._amr_filename(i))
            header = _read_fortran_series(f, ramses_amr_header)
            _skip_fortran(f,13)
            n_per_level = _read_fortran(f, _int_type, header['nlevelmax']*header['ncpu']).reshape(( header['nlevelmax'], header['ncpu']))
            assert n_per_level.sum()==header['ngrid']
            ncell+=n_per_level[:self._maxlevel,i-1].sum()*7 # 8 cells minus 1 parent cell which won't get read
        """    
         
        return ncell

    def _level_iterator(self) :
        """Walks the AMR grid levels on disk, yielding a tuplet of coordinates and
        refinement maps and levels working through the available CPUs and levels."""
        
        for cpu in self._cpus :
            f = file(self._amr_filename(cpu))
            header = _read_fortran_series(f, ramses_amr_header)
            _skip_fortran(f, 13)
            n_per_level = _read_fortran(f, _int_type, header['nlevelmax']*header['ncpu']).reshape(( header['nlevelmax'], header['ncpu']))
            _skip_fortran(f,1)
            if header['nboundary']>0 :
                _skip_fortran(f,3)
            _skip_fortran(f,2)
            if self._info['ordering type']=='bisection' :
                _skip_fortran(f, 5)
            else :
                _skip_fortran(f, 1)
            _skip_fortran(f, 3)

            for level in xrange(self._maxlevel or header['nlevelmax']) :
    
                # loop through those CPUs with grid data (includes ghost regions)
                for cpuf in 1+np.where(n_per_level[level,:]!=0)[0] :
                    #print "CPU=",cpu,"CPU on disk=",cpuf,"npl=",n_per_level[level,cpuf-1]
                    if cpuf==cpu :
                        # this is the data we want
                        _skip_fortran(f,3) # grid, next, prev index

                        # store the coordinates in temporary arrays. We only want
                        # to copy it if the cell is not refined
                        x0,y0,z0 = [_read_fortran(f, _float_type, n_per_level[level,cpu-1]) for ar in range(self._ndim)]

                        _skip_fortran(f,1 # father index
                                      + 2*self._ndim # nbor index
                                      + 2*(2**self._ndim) # son index,cpumap,refinement map
                                      )
   
                        refine = np.array([_read_fortran(f,_int_type,n_per_level[level,cpu-1]) for i in xrange(2**self._ndim)])

                        if level==self._maxlevel :
                            refine[:] = 0

                        yield (x0,y0,z0),refine,cpuf,level

                            
                    else :
                        # skip ghost regions from other CPUs
                        _skip_fortran(f,3+self._ndim+1+2*self._ndim+3*2**self._ndim)

    


        
    def _load_gas_pos(self) :
        i0 = 0
        dims = [self.gas[i] for i in 'x','y','z']
        self.gas['pos'].set_default_units()
        smooth = self.gas['smooth']
        smooth.set_default_units()
        
        subgrid_index = np.arange(2**self._ndim)[:,np.newaxis]
        subgrid_z = np.floor((subgrid_index)/4)
        subgrid_y = np.floor((subgrid_index-4*subgrid_z)/2)
        subgrid_x = np.floor(subgrid_index-2*subgrid_y-4*subgrid_z)
        subgrid_x-=0.5
        subgrid_y-=0.5
        subgrid_z-=0.5
        
        for (x0,y0,z0), refine, cpu, level in self._level_iterator() :
            dx = self._info['boxlen']*0.5**(level+1)
             
            x0 = x0+dx*subgrid_x
            y0 = y0+dx*subgrid_y
            z0 = z0+dx*subgrid_z

            mark = np.where(refine==0)
    
            i1 = i0+len(mark[0])
            for q,d in zip(dims,[x0,y0,z0]) :
                q[i0:i1]=d[mark]

            smooth[i0:i1]=dx
            
            i0=i1

            

    def _load_gas_vars(self) :
        i1 = 0
        
        dims = []
        for i in hydro_blocks :
            if i not in self.gas :
                self.gas._create_array(i)
            dims.append(self.gas[i])
            self.gas[i].set_default_units()
            
        nvar = len(dims)

        grid_info_iter = self._level_iterator()

        if config['verbose'] :
            print>>sys.stderr, "RamsesSnap: loading hydro files",
        
        for cpu in self._cpus :

            if config['verbose'] :
                print>>sys.stderr, cpu,
                sys.stderr.flush()
            

            f = file(self._hydro_filename(cpu))
            header = _read_fortran_series(f, ramses_hydro_header)
 
            if header['nvarh']!=nvar :
                # This should probably be an IOError, but then it would be obscured by
                # silent lazy-loading failure...?
                raise RuntimeError, "Number of hydro variables does not correspond to config.ini specification (expected %d, got %d in file)"%(nvar, header['nvarh'])
        
            for level in xrange(self._maxlevel or header['nlevelmax']) :
                
                for cpuf in xrange(1,header['ncpu']+1) :
                    flevel = _read_fortran(f, 'i4')[0]
                    assert flevel-1==level
                    ncache = _read_fortran(f, 'i4')[0]
               
                    if ncache>0 :
                        if cpuf==cpu :
                            coords, refine, gi_cpu, gi_level =  grid_info_iter.next()
                            mark = np.where(refine==0)

                            assert gi_level==level
                            assert gi_cpu==cpu

                        if cpuf==cpu and len(mark[0])>0 :
                            for icel in xrange(2**self._ndim) :
                                i0 = i1
                                i1 = i0+len(refine[icel])-refine[icel].sum()
                                for ar in dims :
                                    ar[i0:i1] = _read_fortran(f, _float_type, ncache)[(refine[icel]==0)]

                        else :
                            _skip_fortran(f, 2**self._ndim*len(dims))

        if config['verbose'] :
            print>>sys.stderr, "done"

                              
    def _load_dm_block(self, blockname) :
        offset = particle_blocks.index(blockname)
        ind0 = 0

        for i in self._cpus :
            f = file(self._particle_filename(i))
            header = _read_fortran_series(f, ramses_particle_header)
            skip_size = header['npart']*_float_type.itemsize+2*_head_type.itemsize
            
            f.seek(skip_size*offset,1)
            ind1 = ind0+header['npart']
  
            self.dm[blockname][ind0:ind1]=_read_fortran(f, _float_type, header['npart'])
            f.close()
            ind0 = ind1
            
    def _load_array(self, array_name, fam=None) :
       
        if fam is family.dm :
            # Framework always calls with 3D name. Ramses particle blocks are
            # stored as 1D slices.
            if array_name in self._split_arrays :
                for array_1D in self._array_name_ND_to_1D(array_name) :
                    self._load_array(array_1D, fam)

            elif array_name in particle_blocks and fam is family.dm :
                if array_name not in self.dm :
                    self.dm._create_array(array_name)
                self._load_dm_block(array_name)
            else :
                raise IOError, "No such array on disk"
        elif fam is family.gas :
            
            if array_name=='pos' or array_name=='smooth' :
                if 'pos' not in self.gas :
                    self.gas._create_array('pos',3)
                if 'smooth' not in self.gas :
                    self.gas._create_array('smooth')
                self._load_gas_pos()
            elif array_name=='vel' or array_name in hydro_blocks :
                self._load_gas_vars()
            else :
                raise IOError, "No such array on disk"
            

        self_fam = self[fam] if fam else self
        
        if array_name in self_fam and hasattr(self_fam[array_name].units, "_no_unit") :
            self_fam[array_name].units = self._default_units_for(array_name)
                            
    @staticmethod
    def _can_load(f) :
        tsid = _timestep_id(f)
        if tsid :
            return os.path.isdir(f) and os.path.exists(f+"/info_"+tsid+".txt")
        return False






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


@RamsesSnap.derived_quantity
def mass(sim) :
    return sim['rho']*sim['smooth']**3
