"""

ramses
======

Implements classes and functions for handling RAMSES files. AMR cells
are loaded as particles. You rarely need to access this module
directly as it will be invoked automatically via pynbody.load.


For a complete demo on how to use RAMSES outputs with pynbody, look at
the `ipython notebook demo
<http://nbviewer.ipython.org/github/pynbody/pynbody/blob/master/examples/notebooks/pynbody_demo-ramses.ipynb>`_

"""

from __future__ import with_statement # for py2.5
from __future__ import division

from . import snapshot, array, util
from . import family
from . import units
from . import config, config_parser
from . import chunk
from . import util
from . import analysis

from util import read_fortran, read_fortran_series, skip_fortran

import struct, os
import numpy as np
import gzip
import sys
import warnings
import copy
import types
import math
import re
import functools
import time

multiprocess_num = int(config_parser.get('ramses',"parallel-read"))
multiprocess = (multiprocess_num>1)

if multiprocess :
    try:
        import multiprocessing, posix_ipc
        remote_exec = array.shared_array_remote
        remote_map = array.remote_map
    except ImportError:
        warnings.warn("RamsesSnap is configured to use multiple processes, but the posix_ipc module is missing. Reverting to single thread.", RuntimeWarning)
        multiprocess = False
        
if not multiprocess:
    def remote_exec(fn) :
        def q(*args) :
            t0 = time.time()
            r = fn(*args)
            return r
        return q
    
    def remote_map(*args, **kwargs) :
        return map(*args[1:], **kwargs)





_float_type = np.dtype('f8')
_int_type = np.dtype('i4')



def _timestep_id(basename) :
    try:
        return re.findall("output_([0-9]*)/*$", basename)[0]
    except IndexError :
        return None
    
def _cpu_id(i) :
    return str(i).rjust(5,"0")



@remote_exec
def _cpui_count_particles(filename) :
    distinguisher_field = int(particle_distinguisher[0])
    distinguisher_type = np.dtype(particle_distinguisher[1])

    f = file(filename)
    header = read_fortran_series(f, ramses_particle_header)
    npart_this = header['npart']
    try:
        skip_fortran(f,distinguisher_field)
        data = read_fortran(f,distinguisher_type,header['npart'])
    except TypeError:
        data = np.zeros(npart_this)
    
    my_mask = (data!=0)
    nstar_this = (data!=0).sum()
    return npart_this, nstar_this, my_mask

@remote_exec
def _cpui_load_particle_block(filename, dm_ar, star_ar, offset, ind0_dm, ind0_star, _type, star_mask, nstar) :
    f = file(filename)
    header = read_fortran_series(f, ramses_particle_header)

    skip_fortran(f, offset)

    ind1_dm = ind0_dm+header['npart']-nstar
    ind1_star = ind0_star+nstar

    data = read_fortran(f, _type, header['npart'])
    try:
        if len(star_mask)>0 :
            dm_ar[ind0_dm:ind1_dm]=data[~star_mask]
            star_ar[ind0_star:ind1_star]=data[star_mask]
        else :
            dm_ar[ind0_dm:ind1_dm]=data
    except ValueError:
        # this translates into the data block loaded from disk not being
        # long enough
        raise IOError, "Could not load particle block"

    f.close()


def _cpui_level_iterator(cpu, amr_filename, bisection_order, maxlevel, ndim) :
    f = file(amr_filename, 'rb')
    header = read_fortran_series(f, ramses_amr_header)
    skip_fortran(f, 13)

    n_per_level = read_fortran(f, _int_type, header['nlevelmax']*header['ncpu']).reshape(( header['nlevelmax'], header['ncpu']))
    skip_fortran(f,1)
    if header['nboundary']>0 :
        skip_fortran(f,2)
        n_per_level_boundary = read_fortran(f, _int_type, header['nlevelmax']*header['nboundary']).reshape(( header['nlevelmax'], header['nboundary']))

    skip_fortran(f,2)
    if bisection_order :
        skip_fortran(f, 5)
    else :
        skip_fortran(f, 1)
    skip_fortran(f, 3)

    offset = np.array(header['ng'],dtype='f8')/2
    offset-=0.5

    coords = np.zeros(3,dtype=_float_type)

    for level in xrange(maxlevel or header['nlevelmax']) :

        # loop through those CPUs with grid data (includes ghost regions)
        for cpuf in 1+np.where(n_per_level[level,:]!=0)[0] :
            #print "CPU=",cpu,"CPU on disk=",cpuf,"npl=",n_per_level[level,cpuf-1]

            if cpuf==cpu :

                # this is the data we want
                skip_fortran(f,3) # grid, next, prev index

                # store the coordinates in temporary arrays. We only want
                # to copy it if the cell is not refined
                coords = [read_fortran(f, _float_type, n_per_level[level,cpu-1]) for ar in range(ndim)]

                
                # stick on zeros if we're in less than 3D
                coords+=[np.zeros_like(coords[0]) for ar in range(3-ndim)]

                
                skip_fortran(f,1 # father index
                              + 2*ndim # nbor index
                              + 2*(2**ndim) # son index,cpumap,refinement map
                              )

                refine = np.array([read_fortran(f,_int_type,n_per_level[level,cpu-1]) for i in xrange(2**ndim)])

                if level==maxlevel :
                    refine[:] = 0

                
                coords[0]-=offset[0]; coords[1]-=offset[1]; coords[2]-=offset[2]
                # x0-=offset[0]; y0-=offset[1]; z0-=offset[2]

                yield coords,refine,cpuf,level


            else :

                # skip ghost regions from other CPUs
                skip_fortran(f,3+ndim+1+2*ndim+3*2**ndim)

        if header['nboundary']>0 :
            for boundaryf in np.where(n_per_level_boundary[level, :]!=0)[0] :

                        skip_fortran(f,3+ndim+1+2*ndim+3*2**ndim)

@remote_exec
def _cpui_count_gas_cells(level_iterator_args) :
    ncell = 0
    for coords, refine, cpu, level in _cpui_level_iterator(*level_iterator_args) :
        ncell+=(refine==0).sum()
    return ncell

@remote_exec
def _cpui_load_gas_pos(pos_array, smooth_array, ndim, boxlen, i0, level_iterator_args) :
    dims = [pos_array[:,i] for i in range(ndim)]
    subgrid_index = np.arange(2**ndim)[:,np.newaxis]
    subgrid_z = np.floor((subgrid_index)/4)
    subgrid_y = np.floor((subgrid_index-4*subgrid_z)/2)
    subgrid_x = np.floor(subgrid_index-2*subgrid_y-4*subgrid_z)
    subgrid_x-=0.5
    subgrid_y-=0.5
    subgrid_z-=0.5

    for (x0,y0,z0), refine, cpu, level in _cpui_level_iterator(*level_iterator_args) :
        dx = boxlen*0.5**(level+1)
            
        x0 = boxlen*x0+dx*subgrid_x
        y0 = boxlen*y0+dx*subgrid_y
        z0 = boxlen*z0+dx*subgrid_z

        mark = np.where(refine==0)

        i1 = i0+len(mark[0])
        for q,d in zip(dims,[x0,y0,z0][:ndim]) :
            q[i0:i1]=d[mark]

        smooth_array[i0:i1]=dx
        i0=i1

_gv_load_hydro = 0
_gv_load_gravity = 1

@remote_exec
def _cpui_load_gas_vars(dims, maxlevel, ndim, filename, cpu, lia, i1,
                        mode = _gv_load_hydro ) :
    
    if config['verbose'] :
        print>>sys.stderr, cpu,
        sys.stderr.flush()
        
    nvar = len(dims)
    grid_info_iter = _cpui_level_iterator(*lia)

    f = file(filename)

    check_nvar_file = False

    if mode is _gv_load_hydro :
        header = read_fortran_series(f, ramses_hydro_header)
        
        nvar_file = header['nvarh']
    else :
        header = read_fortran_series(f, ramses_grav_header)
        nvar_file=4

    if nvar_file<nvar :
        warnings.warn("Fewer hydro variables are in this RAMSES dump than are defined in config.ini (expected %d, got %d in file)"%(nvar, nvar_file), RuntimeWarning)
        nvar = nvar_file
        dims = dims[:nvar]
    elif nvar_file>nvar :
        warnings.warn("More hydro variables (%d) are in this RAMSES dump than are defined in config.ini (%d)"%(nvar_file,nvar), RuntimeWarning)

    for level in xrange(maxlevel or header['nlevelmax']) :

        for cpuf in xrange(1,header['ncpu']+1) :
            flevel = read_fortran(f, 'i4')[0]
            ncache = read_fortran(f, 'i4')[0]
            assert flevel-1==level

            if ncache>0 :
                if cpuf==cpu :

                    coords, refine, gi_cpu, gi_level =  grid_info_iter.next()
                    mark = np.where(refine==0)

                    assert gi_level==level
                    assert gi_cpu==cpu

                if cpuf==cpu and len(mark[0])>0 :
                    for icel in xrange(2**ndim) :
                        i0 = i1
                        i1 = i0+(refine[icel]==0).sum()
                        for ar in dims :
                            ar[i0:i1] = read_fortran(f, _float_type, ncache)[(refine[icel]==0)]


                        skip_fortran(f, (nvar_file-nvar))

                else :
                    skip_fortran(f, (2**ndim)*nvar_file)

        for boundary in xrange(header['nboundary']) :
            flevel = read_fortran(f, 'i4')[0]
            ncache = read_fortran(f, 'i4')[0]
            if ncache>0 :
                skip_fortran(f, (2**ndim)*nvar_file)
                
                        
ramses_particle_header = np.dtype([('ncpu', 'i4'), ('ndim', 'i4'), ('npart', 'i4'),
                                   ('randseed', 'i4', (4,)), ('nstar', 'i4'), ('mstar', 'f8'),
                                   ('mstar_lost', 'f8'), ('nsink', 'i4')])

ramses_amr_header = np.dtype([('ncpu', 'i4'), ('ndim', 'i4'), ('ng', 'i4', (3,)), 
                              ('nlevelmax', 'i4'), ('ngridmax', 'i4'), ('nboundary', 'i4'),
                              ('ngrid', 'i4'), ('boxlen', 'f8')])

ramses_hydro_header = np.dtype([('ncpu', 'i4'), ('nvarh', 'i4'), ('ndim', 'i4'), ('nlevelmax', 'i4'),
                                ('nboundary', 'i4'), ('gamma', 'f8')])

ramses_grav_header = np.dtype([('ncpu', 'i4'), ('ndim', 'i4'), ('nlevelmax', 'i4'),
                                ('nboundary', 'i4')])

particle_blocks = map(str.strip,config_parser.get('ramses',"particle-blocks").split(","))
particle_format = map(str.strip,config_parser.get('ramses',"particle-format").split(","))

hydro_blocks = map(str.strip,config_parser.get('ramses',"hydro-blocks").split(","))
grav_blocks = map(str.strip,config_parser.get('ramses',"gravity-blocks").split(","))

particle_distinguisher = map(str.strip, config_parser.get('ramses', 'particle-distinguisher').split(","))

class RamsesSnap(snapshot.SimSnap) :
    reader_pool = None
    
    def __init__(self, dirname, **kwargs) :
        """Initialize a RamsesSnap. Extra kwargs supported:

         *cpus* : a list of the CPU IDs to load. If not set, load all CPU's data.
         *maxlevel* : the maximum refinement level to load. If not set, the deepest level is loaded.
         """
    
        
        
        global config
        super(RamsesSnap,self).__init__()

        if multiprocess :
            self._shared_arrays = True
            if (RamsesSnap.reader_pool is None) :
                RamsesSnap.reader_pool = multiprocessing.Pool(multiprocess_num)

        
        self._timestep_id = _timestep_id(dirname)
        self._filename = dirname
        self._load_infofile()

        # determine whether we have explicit information about
        # what particle blocks are present
        self._new_format = False
        self._particle_blocks = self._info.get('particle-blocks',particle_blocks)
        if 'particle-blocks' in self._info :
            self._new_format = True
            self._particle_blocks = ['x','y','z','vx','vy','vz']+self._particle_blocks[2:]

        assert self._info['ndim']<=3
        if self._info['ndim']<3 :
            warnings.warn("Snapshots with less than three dimensions are supported only experimentally", RuntimeWarning)
            
        self._ndim = self._info['ndim']
        
        self.ncpu = self._info['ncpu']

        if 'cpus' in kwargs :
            self._cpus = kwargs['cpus']
        else :
            self._cpus = range(1, self.ncpu+1)

        self._maxlevel = kwargs.get('maxlevel',None)

        ndm, nstar = self._count_particles()

        has_gas = os.path.exists(self._hydro_filename(1)) or kwargs.get('force_gas',False)
        
        ngas = self._count_gas_cells() if has_gas else 0
        
        self._num_particles = ndm+ngas+nstar
        self._family_slice[family.dm] = slice(0, ndm)
        self._family_slice[family.star] = slice(ndm, ndm+nstar)
        self._family_slice[family.gas] = slice(ndm+nstar, ndm+nstar+ngas)
        
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
        try : 
            f = file(self._filename+"/header_"+_timestep_id(self._filename)+".txt")
            # most of this file is unhelpful, but in the latest ramses
            # version, there is information on the particle fields present
            for l in f :
                if "level" in l :
                    self._info['particle-blocks']=l.split()
        except IOError : 
            warnings.warn("No header file found -- no particle block information available")

    def _particle_filename(self, cpu_id) :
        return self._filename+"/part_"+self._timestep_id+".out"+_cpu_id(cpu_id)

    def _amr_filename(self, cpu_id) :
        return self._filename+"/amr_"+self._timestep_id+".out"+_cpu_id(cpu_id)

    def _hydro_filename(self, cpu_id) :
        return self._filename+"/hydro_"+self._timestep_id+".out"+_cpu_id(cpu_id)

    def _grav_filename(self, cpu_id) :
        return self._filename+"/grav_"+self._timestep_id+".out"+_cpu_id(cpu_id)
    
    def _count_particles(self) :
        """Returns ndm, nstar where ndm is the number of dark matter particles
        and nstar is the number of star particles."""

        npart = 0
        nstar = 0

        dm_i0 = 0
        star_i0 = 0
        
        self._star_mask = []
        self._nstar = []
        self._dm_i0 = []
        self._star_i0 = []

        if not os.path.exists(self._particle_filename(1)) :
            return 0,0
        
        results = remote_map(self.reader_pool,
                             _cpui_count_particles,
                             [self._particle_filename(i) for i in self._cpus])
        
        for npart_this, nstar_this, my_mask in results :
            self._dm_i0.append(dm_i0)
            self._star_i0.append(star_i0)
            dm_i0+=(npart_this-nstar_this)
            star_i0+=nstar_this
            npart+=npart_this
            nstar+=nstar_this
            
            self._nstar.append(nstar_this)
            self._star_mask.append(my_mask)
            

        return npart-nstar,nstar

    
    
    def _count_gas_cells(self) :
        ncells = remote_map(self.reader_pool, _cpui_count_gas_cells,
                            [self._cpui_level_iterator_args(xcpu) for xcpu in self._cpus])
        self._gas_i0 = np.cumsum([0]+ncells)[:-1]
        return np.sum(ncells)


    def _cpui_level_iterator_args(self, cpu=None) :
        if cpu :
            return cpu, self._amr_filename(cpu), self._info['ordering type']=='bisection', self._maxlevel, self._ndim
        else :
            return [self._cpui_level_iterator_args(x) for x in self._cpus]
        
    def _level_iterator(self) :
        """Walks the AMR grid levels on disk, yielding a tuplet of coordinates and
        refinement maps and levels working through the available CPUs and levels."""
        
        for cpu in self._cpus :
            for x in _cpui_level_iterator(*self._cpui_level_iterator_args(cpu)) :
                yield x

        
    def _load_gas_pos(self) :
        i0 = 0
        self.gas['pos'].set_default_units()
        smooth = self.gas['smooth']
        smooth.set_default_units()
        
        
        boxlen = self._info['boxlen']

        
        remote_map(self.reader_pool,
                   _cpui_load_gas_pos,
                    [self.gas['pos']]*len(self._cpus),
                    [self.gas['smooth']]*len(self._cpus),
                    [self._ndim]*len(self._cpus),
                    [boxlen]*len(self._cpus),
                    self._gas_i0,
                    self._cpui_level_iterator_args())
        
        

    def _load_gas_vars(self, mode=_gv_load_hydro) :
        i1 = 0
        
        dims = []
        
        for i in [hydro_blocks,grav_blocks][mode] :
            if i not in self.gas :
                self.gas._create_array(i)
            if self._ndim<3 and i[-1]=='z' :
                continue
            if self._ndim<2 and i[-1]=='y' :
                continue
            dims.append(self.gas[i])
            self.gas[i].set_default_units()
            
        nvar = len(dims)

        grid_info_iter = self._level_iterator()

        if config['verbose'] :
            print>>sys.stderr, "RamsesSnap: loading %s files"%(['hydro','grav'][mode]),

        filenamer = [self._hydro_filename, self._grav_filename][mode]
        
        remote_map(self.reader_pool,
                   _cpui_load_gas_vars,
                   [dims]*len(self._cpus),
                   [self._maxlevel]*len(self._cpus),
                   [self._ndim]*len(self._cpus),
                   [filenamer(i) for i in self._cpus],
                   self._cpus,
                   self._cpui_level_iterator_args(),
                   self._gas_i0,
                   [mode]*len(self._cpus))

        if mode is _gv_load_gravity :
            # potential is awkwardly in expected units divided by box size
            self.gas['phi']*=self._info['boxlen']
 
        if config['verbose'] :
            print>>sys.stderr, "done"

                              
    def _load_particle_block(self, blockname) :
        offset = self._particle_blocks.index(blockname)
        _type = np.dtype(particle_format[offset])
        ind0_dm = 0
        ind0_star = 0

        if blockname not in self.dm :
            self.dm._create_array(blockname, dtype=_type)
        if blockname not in self.star :
            self.star._create_array(blockname, dtype=_type)

        dm_ar = self.dm[blockname]
        star_ar = self.star[blockname]
        if len(star_ar)==0 :
            star_ar = np.array(star_ar)
        if len(dm_ar)==0 :
            dm_ar=np.array(dm_ar)
            
        remote_map(self.reader_pool,
                   _cpui_load_particle_block,
                   [self._particle_filename(i) for i in self._cpus],
                   [dm_ar]*len(self._cpus),
                   [star_ar]*len(self._cpus),
                   [offset]*len(self._cpus),
                   self._dm_i0,
                   self._star_i0,
                   [_type]*len(self._cpus),
                   self._star_mask,
                   self._nstar)
        
        # The potential is awkwardly not in physical units, but in 
        # physical units divided by the box size. This was different
        # in an intermediate version, but then later made consistent
        # with the AMR phi output. So, we need to make a correction here
        # IF we are dealing with the latest ramses format.

        if self._new_format and blockname is 'phi' :
            self.dm['phi']*=self._info['boxlen']
            self.star['phi']*=self._info['boxlen']

    

    def _load_particle_cpuid(self) :
        ind0_dm = 0
        ind0_star = 0
        for i, star_mask, nstar in zip(self._cpus, self._star_mask, self._nstar) :
            f = file(self._particle_filename(i))
            header = read_fortran_series(f, ramses_particle_header)
            f.close()
            ind1_dm = ind0_dm+header['npart']-nstar
            ind1_star = ind0_star+nstar
            self.dm['cpu'][ind0_dm:ind1_dm] = i
            self.star['cpu'][ind0_star:ind1_star] = i
            ind0_dm, ind0_star = ind1_dm, ind1_star

    def _load_gas_cpuid(self) :
        gas_cpu_ar = self.gas['cpu']
        i1 = 0
        for coords, refine, cpu, level in self._level_iterator() :
            for cell in xrange(2**self._ndim) :
                i0 = i1
                i1 = i0+(refine[cell]==0).sum()
                gas_cpu_ar[i0:i1] = cpu
            

    def loadable_keys(self, fam=None) :
        if fam is None :
            x = set()
            for f0 in self.families() :
                x = x.union(self.loadable_keys(f0))
            return list(x)
        else :
            if fam is family.dm or fam is family.star :
                return self._particle_blocks
            elif fam is family.gas :
                return ['x','y','z','smooth']+hydro_blocks
        
    def _load_array(self, array_name, fam=None) :
        if array_name=='cpu' :
            self['cpu'] = np.zeros(len(self), dtype=int)
            self._load_particle_cpuid()
            self._load_gas_cpuid()
            
        elif fam is family.dm or fam is family.star :
            # Framework always calls with 3D name. Ramses particle blocks are
            # stored as 1D slices.
            if array_name in self._split_arrays :
                for array_1D in self._array_name_ND_to_1D(array_name) :
                    self._load_array(array_1D, fam)

            elif array_name in self._particle_blocks :
                self._load_particle_block(array_name)
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
            elif array_name in grav_blocks :
                self._load_gas_vars(1)
            else :
                raise IOError, "No such array on disk"
        elif fam is None and array_name in ['pos','vel'] :
            # synchronized loading of pos/vel information
            if 'pos' not in self :
                self._create_array('pos',3)
            if 'vel' not in self :
                self._create_array('vel',3)
            if 'smooth' not in self.gas :
                self.gas._create_array('smooth')

            if len(self.gas) > 0 :
                self._load_gas_pos()
                self._load_gas_vars()
                
            self._load_array('vel', family.dm)
            self._load_array('pos', family.dm)
        elif fam is None and array_name is 'mass' :
            self._create_array('mass')
            self._load_particle_block('mass')
            self['mass'].set_default_units()
            if len(self.gas) > 0 :
                gasmass = mass(self.gas)
                gasmass.convert_units(self['mass'].units)
                self.gas['mass'] = gasmass
        else :
            raise IOError, "No such array on disk"
            

        self_fam = self[fam] if fam else self

        # The following is now done by SnapShot (and done better):
        #if array_name in self_fam and hasattr(self_fam[array_name].units, "_no_unit") :
        #    self_fam[array_name].units = self._default_units_for(array_name)
                            
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
    sim.properties['h'] = sim._info['H0']/100.

    # N.B. these conversion factors provided by ramses already have the
    # correction from comoving to physical units
    d_unit = sim._info['unit_d']*units.Unit("g cm^-3")
    t_unit = sim._info['unit_t']*units.Unit("s")
    l_unit = sim._info['unit_l']*units.Unit("cm")

    sim.properties['boxsize'] = sim._info['boxlen'] * l_unit

    if sim._info['omega_k'] == sim._info['omega_l'] == sim._info['omega_b'] == 0.0 :
        sim.properties['time'] = sim._info['time'] * t_unit
    else :
        sim.properties['time'] = analysis.cosmology.age(sim)*units.Unit('Gyr')

    sim._file_units_system = [d_unit, t_unit, l_unit]


@RamsesSnap.derived_quantity
def mass(sim) :
    return sim['rho']*sim['smooth']**3

@RamsesSnap.derived_quantity
def tform(sim) :
    return sim.properties['time']-sim['age']

@RamsesSnap.derived_quantity
def temp(sim) :
    return ((sim['p']/sim['rho'])*(1.22*units.m_p/units.k)).in_units("K")

