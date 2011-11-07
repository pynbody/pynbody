"""

derived
=======

Holds procedures for creating new arrays from existing ones, e.g. for
getting the radial position.
<http://code.google.com/p/pynbody/wiki/AutomagicCalculation>

"""

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

bands_available = ['u','b','v','r','i','j','h','k','b']

for band in bands_available :
    X = lambda s, b=str(band): analysis.luminosity.calc_mags(s,band=b)
    X.__name__ = band+"_mag"
    SimSnap.derived_quantity(X)

    X = lambda s, b=str(band): (10**(-0.4*s[b+"_mag"]))*s['rho']/s['mass']
    X.__name__ = band+"_lum_den"
    SimSnap.derived_quantity(X)

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
