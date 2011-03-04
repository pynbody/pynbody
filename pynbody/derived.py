from . import snapshot
from snapshot import SimSnap
from tipsy import TipsySnap
from . import array
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

XSOLFe=0.117E-2
XSOLO=0.96E-2
XSOLH=0.706
XSOLC=3.03e-3
XSOLN=1.105e-3
XSOLNe=2.08e-4
XSOLMg=5.13e-4
XSOLSi=6.53e-4

@TipsySnap.derived_quantity
def hetot(self) :
    return 0.236+(2.1*self['metals'])

@TipsySnap.derived_quantity
def hydrogen(self) :
    return 1.0-self['metals']-self['hetot']

@TipsySnap.derived_quantity
def feh(self) :
    minfe = np.amin(self['FeMassFrac'][np.where(self['FeMassFrac'] > 0)])
    self['FeMassFrac'][np.where(self['FeMassFrac'] == 0)]=minfe
    return np.log10(self['FeMassFrac']/self['hydrogen']) - np.log10(XSOLFe/XSOLH)

@TipsySnap.derived_quantity
def ofe(self) :
    minox = np.amin(self['OxMassFrac'][np.where(self['OxMassFrac'] > 0)])
    self['OxMassFrac'][np.where(self['OxMassFrac'] == 0)]=minox
    minfe = np.amin(self['FeMassFrac'][np.where(self['FeMassFrac'] > 0)])
    self['FeMassFrac'][np.where(self['FeMassFrac'] == 0)]=minfe
    return np.log10(self['OxMassFrac']/self['FeMassFrac']) - np.log10(XSOLO/XSOLFe)

@TipsySnap.derived_quantity
def mgfe(self) :
    minmg = np.amin(self['MgMassFrac'][np.where(self['MgMassFrac'] > 0)])
    self['MgMassFrac'][np.where(self['MgMassFrac'] == 0)]=minmg
    minfe = np.amin(self['FeMassFrac'][np.where(self['FeMassFrac'] > 0)])
    self['FeMassFrac'][np.where(self['FeMassFrac'] == 0)]=minfe
    return np.log10(self['MgMassFrac']/self['FeMassFrac']) - np.log10(XSOLMg/XSOLFe)

@TipsySnap.derived_quantity
def nefe(self) :
    minne = np.amin(self['NeMassFrac'][np.where(self['NeMassFrac'] > 0)])
    self['NeMassFrac'][np.where(self['NeMassFrac'] == 0)]=minne
    minfe = np.amin(self['FeMassFrac'][np.where(self['FeMassFrac'] > 0)])
    self['FeMassFrac'][np.where(self['FeMassFrac'] == 0)]=minfe
    return np.log10(self['NeMassFrac']/self['FeMassFrac']) - np.log10(XSOLNe/XSOLFe)

@TipsySnap.derived_quantity
def sife(self) :
    minsi = np.amin(self['SiMassFrac'][np.where(self['SiMassFrac'] > 0)])
    self['SiMassFrac'][np.where(self['SiMassFrac'] == 0)]=minsi
    minfe = np.amin(self['FeMassFrac'][np.where(self['FeMassFrac'] > 0)])
    self['FeMassFrac'][np.where(self['FeMassFrac'] == 0)]=minfe
    return np.log10(self['SiMassFrac']/self['FeMassFrac']) - np.log10(XSOLSi/XSOLFe)

