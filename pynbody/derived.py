from . import snapshot
from snapshot import SimSnap
from . import array
from . import analysis
import numpy as np

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
def rho(self):
    return self['mass']/self['smooth']**3

@SimSnap.derived_quantity
def smooth(self) :
    return (self['mass']/self['rho'])**(1,3)

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
