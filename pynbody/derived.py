from . import snapshot
from snapshot import SimSnap

@SimSnap.derived_quantity
def r(self):
    return ((self['pos']**2).sum(axis = 1))**(1,2)

@SimSnap.derived_quantity
def rxy(self):
    return ((self['pos'][:,0:2]**2).sum(axis = 1))**(1,2)
           
@SimSnap.derived_quantity
def vr(self):
    return (self['pos']*self['vel']).sum(axis=1)/self['r']

@SimSnap.derived_quantity
def v2(self) :
    return (self['vel']**2).sum(axis=1)

@SimSnap.derived_quantity
def ke(self) :
    return 0.5*(self['vel']**2).sum(axis=1)


@SimSnap.derived_quantity
def vrxy(self):
    return (self['pos'][:,0:2]*self['vel'][:,0:2]).sum(axis=1)/self['r']
           
@SimSnap.derived_quantity
def rho(self):
    return self['mass']/self['smooth']**3

@SimSnap.derived_quantity
def smooth(self) :
    return (self['mass']/self['rho'])**(1,3)
