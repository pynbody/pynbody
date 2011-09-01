from . import snapshot
from snapshot import SimSnap
from . import array
from . import analysis
from . import sph
from . import config
import numpy as np
import sys

@SimSnap.derived_quantity
def r(self):
    return ((self['pos']**2).sum(axis = 1))**(1,2)

@SimSnap.derived_quantity
def rxy(self):
    return ((self['pos'][:,0:2]**2).sum(axis = 1))**(1,2)

@SimSnap.derived_quantity
def vr(self):
    """Radial velocity"""
    return (self['pos']*self['vel']).sum(axis=1)/self['r']


@SimSnap.derived_quantity
def v2(self) :
    """Squared velocity"""
    return (self['vel']**2).sum(axis=1)

@SimSnap.derived_quantity
def vt(self) :
    """Tangential velocity"""
    return np.sqrt(self['v2']-self['vr']**2)


@SimSnap.derived_quantity
def ke(self) :
    """Specific kinetic energy"""
    return 0.5*(self['vel']**2).sum(axis=1)

@SimSnap.derived_quantity
def te(self) :
    """Specific total energy"""
    return self['ke']+self['phi']

@SimSnap.derived_quantity
def j(self) :
    """Specific angular momentum"""
    angmom = np.cross(self['pos'], self['vel']).view(array.SimArray)
    angmom.units = self['pos'].units*self['vel'].units
    return angmom

@SimSnap.derived_quantity
def vrxy(self):
    return (self['pos'][:,0:2]*self['vel'][:,0:2]).sum(axis=1)/self['rxy']

@SimSnap.derived_quantity
def vcxy(self) :
    f = (self['x']*self['vy']-self['y']*self['vx'])/self['rxy']
    f[np.where(f!=f)]=0
    return f


#@SimSnap.derived_quantity
#def smooth(self) :
#    return (self['mass']/self['rho'])**(1,3)

@SimSnap.derived_quantity
def smooth(self):
    
    if hasattr(self,'kdtree') is False : 
        import kdtree
        if config['verbose']: print>>sys.stderr, 'Building tree with leafsize=16'
        if config['tracktime']:
            import time
            start = time.clock()
        self.kdtree = kdtree.KDTree(self['pos'], self['vel'], self['mass'], leafsize=16)
        if config['tracktime'] : 
            end = time.clock()
            print>>sys.stderr, 'Tree build done in %5.3g s'%(end-start)
        elif config['verbose'] : print>>sys.stderr, 'Tree build done.'
        
    if config['verbose']: print>>sys.stderr, 'Smoothing with 32 nearest neighbours'
    sm = array.SimArray(np.empty(len(self['pos'])), self['pos'].units)
    if config['tracktime']:
        import time
        start = time.clock()

    self.kdtree.populate(sm, 'hsm', nn=32) 
    
    if config['tracktime'] : 
        end = time.clock()
        print>>sys.stderr, 'Smoothing done in %5.3g s'%(end-start)
    elif config['verbose']: print>>sys.stderr, 'Smoothing done.'

    return sm 

@SimSnap.derived_quantity
def rho(self):
    #return self['mass']/self['smooth']**3
    if hasattr(self,'kdtree') is False: 
        import kdtree
        if config['verbose']: print>>sys.stderr, 'Building tree with leafsize=16'
        kdt = kdtree.KDTree(self['pos'], self['vel'], self['mass'], leafsize=16) 
        if config['verbose']: print>>sys.stderr, 'Tree build done.'
        self.kdtree = kdt
    
    if config['verbose']: print>>sys.stderr, 'Calculating density with 32 nearest neighbours'
    sm = array.SimArray(np.empty(len(self['pos'])), self['mass'].units/self['pos'].units**3)
    self.kdtree.populate(sm, 'rho', nn=32, smooth=self['smooth'])
    
    if config['verbose']: print>>sys.stderr, 'Density done.'
    return sm 

@SimSnap.derived_quantity
def v_mean(self):
    import kdtree 

    if config['verbose']: print 'Building tree with leafsize=16'
    kdt = kdtree.KDTree(self['pos'], self['vel'], self['mass'], leafsize=16)
    if config['verbose']: print 'Tree build done.'

    
    
    if config['verbose']: print 'Calculating mean velocity with 32 nearest neighbours'
    sm = array.SimArray(np.empty((len(self['pos']),3)), self['vel'])
    kdt.populate(sm, 'v_mean', nn=32, smooth=self['smooth'], rho=self['rho'])
    if config['verbose']: print 'Mean velocity done.'

    return sm 

@SimSnap.derived_quantity
def v_disp(self):
    import kdtree 

    if config['verbose']: print 'Building tree with leafsize=16'
    kdt = kdtree.KDTree(self['pos'], self['vel'], self['mass'], leafsize=16)
    if config['verbose']: print 'Tree build done.'

    self['rho']
    
    if config['verbose']: print 'Calculating velocity dispersion with 32 nearest neighbours'
    sm = array.SimArray(np.empty(len(self['pos'])), self['vel'].units)
    kdt.populate(sm, 'v_disp', nn=32, smooth=self['smooth'], rho=self['rho']) 
    if config['verbose']: print 'Velocity dispersion done.'

    return sm 

@SimSnap.derived_quantity
def age(self) :
    return self.properties['time'].in_units(self['tform'].units, **self.conversion_context()) - self['tform']

@SimSnap.derived_quantity
def u_mag(self) :
    return analysis.luminosity.calc_mags(self,band='u')

@SimSnap.derived_quantity
def b_mag(self) :
    return analysis.luminosity.calc_mags(self,band='b')

@SimSnap.derived_quantity
def v_mag(self) :
    return analysis.luminosity.calc_mags(self,band='v')

@SimSnap.derived_quantity
def r_mag(self) :
    return analysis.luminosity.calc_mags(self,band='r')

@SimSnap.derived_quantity
def i_mag(self) :
    return analysis.luminosity.calc_mags(self,band='i')

@SimSnap.derived_quantity
def j_mag(self) :
    return analysis.luminosity.calc_mags(self,band='j')

@SimSnap.derived_quantity
def h_mag(self) :
    return analysis.luminosity.calc_mags(self,band='h')

@SimSnap.derived_quantity
def k_mag(self) :
    return analysis.luminosity.calc_mags(self,band='k')

@SimSnap.derived_quantity
def B_mag(self) :
    return analysis.luminosity.calc_mags(self,band='b')

@SimSnap.derived_quantity
def V_mag(self) :
    return analysis.luminosity.calc_mags(self,band='v')

@SimSnap.derived_quantity
def R_mag(self) :
    return analysis.luminosity.calc_mags(self,band='r')

@SimSnap.derived_quantity
def I_mag(self) :
    return analysis.luminosity.calc_mags(self,band='i')

@SimSnap.derived_quantity
def J_mag(self) :
    return analysis.luminosity.calc_mags(self,band='j')

@SimSnap.derived_quantity
def H_mag(self) :
    return analysis.luminosity.calc_mags(self,band='h')

@SimSnap.derived_quantity
def K_mag(self) :
    return analysis.luminosity.calc_mags(self,band='k')

@SimSnap.derived_quantity
def theta(self) :
	"""Angle from the z axis, from [0:2pi]"""
	return np.arccos(self['z']/self['r'])

@SimSnap.derived_quantity
def alt(self) :
	"""Angle from the horizon, from [-pi/2:pi/2]"""
	return np.pi/2 - self['theta']

@SimSnap.derived_quantity
def az(self) :
	"""Angle in the xy plane from the x axis, from [-pi:pi]"""
	return np.arctan2(self['y'],self['x'])
