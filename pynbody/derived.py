"""
derived
=======

Holds procedures for creating new arrays from existing ones, e.g. for
getting the radial position. For more information see :ref:`derived`.

"""

from . import snapshot
from snapshot import SimSnap
from . import array
from . import analysis
from . import sph
from . import config
from . import units
import numpy as np
import sys
    
@SimSnap.derived_quantity
def r(self):
    """Radial position"""
    return ((self['pos']**2).sum(axis = 1))**(1,2)

@SimSnap.derived_quantity
def rxy(self):
    """Cylindrical radius in the x-y plane"""
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
def j2(self) :
    """Square of the specific angular momentum"""
    return (self['j']**2).sum(axis=1)

@SimSnap.derived_quantity
def jz(self):
    """z-component of the angular momentum"""
    return self['j'][:,2]

@SimSnap.derived_quantity
def vrxy(self):
    """Cylindrical radial velocity in the x-y plane"""
    return (self['pos'][:,0:2]*self['vel'][:,0:2]).sum(axis=1)/self['rxy']

@SimSnap.derived_quantity
def vcxy(self) :
    """Cylindrical tangential velocity in the x-y plane"""
    f = (self['x']*self['vy']-self['y']*self['vx'])/self['rxy']
    f[np.where(f!=f)]=0
    return f

@SimSnap.derived_quantity
def vphi(self):
    """Azimuthal velocity (synonym for vcxy)"""
    return self['vcxy']

@SimSnap.derived_quantity
def vtheta(self):
    """Velocity projected to polar direction"""
    return (np.cos(self['az'])*np.cos(self['theta'])*self['vx'] +
        np.sin(self['az'])*np.cos(self['theta'])*self['vy'] -
        np.sin(self['theta'])*self['vy'])

@SimSnap.derived_quantity
def v_mean(self):
    """SPH-smoothed mean velocity"""
    import sph
    
    sph.build_tree(self)
    
    nsmooth = config['sph']['smooth-particles']
    
    if config['verbose']: print 'Calculating mean velocity with %d nearest neighbours' % nsmooth

    sm = array.SimArray(np.empty((len(self['pos']),3)), self['vel'])
    self.kdtree.populate(sm, 'v_mean', nn=nsmooth, smooth=self['smooth'], rho=self['rho'])
    if config['verbose']: print 'Mean velocity done.'

    return sm 

@SimSnap.derived_quantity
def v_disp(self):
    """SPH-smoothed local velocity dispersion"""
    import sph

    sph.build_tree(self)
    nsmooth = config['sph']['smooth-particles']
    self['rho']
    
    if config['verbose']: print 'Calculating velocity dispersion with %d nearest neighbours' % nsmooth

    sm = array.SimArray(np.empty(len(self['pos'])), self['vel'].units)
    self.kdtree.populate(sm, 'v_disp', nn=nsmooth, smooth=self['smooth'], rho=self['rho']) 
    if config['verbose']: print 'Velocity dispersion done.'

    return sm 

@SimSnap.derived_quantity
def age(self) :
    """Stellar age determined from formation time and current snapshot time"""
    return self.properties['time'].in_units(self['tform'].units, **self.conversion_context()) - self['tform']

bands_available = ['u','b','v','r','i','j','h','k','U','B','V','R','I',
                   'J','H','K']

for band in bands_available :
    X = lambda s, b=str(band): analysis.luminosity.calc_mags(s,band=b)
    X.__name__ = band+"_mag"
    X.__doc__ = band+" magnitude from analysis.luminosity.calc_mags"""
    SimSnap.derived_quantity(X)

    X = lambda s, b=str(band): (10**(-0.4*s[b+"_mag"]))*s['rho']/s['mass']
    X.__name__ = band+"_lum_den"
    X.__doc__ = band+" luminosity density from analysis.luminosity.calc_mags"""
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

@SimSnap.derived_quantity
def cs(self):
    """Sound speed"""
    return np.sqrt(5.0*units.k*self['temp'] / self['mu']/units.m_p)

@SimSnap.derived_quantity
def mu(self,t0=None) :
    """Estimate of mean molecular mass"""
    
@SimSnap.derived_quantity
def mu(sim,t0=None) :
    """Relative atomic mass, i.e. number of particles per
    proton mass, ignoring metals (since we generally only know the
    mass fraction of metals, not their specific atomic numbers)"""
    try:
        x =  sim["HI"]+2*sim["HII"]+sim["HeI"]+2*sim["HeII"]+3*sim["HeIII"]
    except KeyError :
        x = np.empty(len(sim)).view(array.SimArray)
        if t0 is None :
            t0 = sim['temp']
        x[np.where(t0>=1e4)[0]] = 0.59
        x[np.where(t0<1e4)[0]] = 1.3

    x.units = units.Unit("1")
    #x.units = units.m_p**-1
    return x
    
@SimSnap.derived_quantity
def p(sim) :
    """Pressure"""
    p = sim["u"]*sim["rho"]*(2./3)
    p.convert_units("Pa")
    return p

@SimSnap.derived_quantity
def u(self) :
    """Gas internal energy derived from temperature"""
    gamma = 5./3
    return self['temp']*units.k/(self['mu']*units.m_p*(gamma-1))

@SimSnap.derived_quantity
def temp(self) :
    """Gas temperature derived from internal energy"""
    gamma = 5./3
    mu_est = np.ones(len(self))
    for i in range(5) :
        temp=(self['u']*units.m_p/units.k)*(mu_est*(gamma-1))
        temp.convert_units("K")
        mu_est = mu(self, temp)
    return temp

@SimSnap.derived_quantity
def zeldovich_offset(self) :
    """The position offset in the current snapshot according to
    the Zel'dovich approximation applied to the current velocities.
    (Only useful in the generation or analysis of initial conditions.)"""
    from . import analysis
    bdot_by_b = analysis.cosmology.rate_linear_growth(self, unit='km Mpc^-1 s^-1')/analysis.cosmology.linear_growth_factor(self)

    a = self.properties['a']
    
    offset = self['vel']/(a*bdot_by_b)
    offset.units=self['vel'].units/units.Unit('km Mpc^-1 s^-1 a^-1')
    return offset


@SimSnap.derived_quantity
def aform(self) :
    """The expansion factor at the time specified by the tform array."""

    from . import analysis
    z = analysis.cosmology.redshift(self, self['tform'])
    a = 1./(1.+z)
    return a
