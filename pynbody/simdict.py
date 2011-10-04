"""
simdict module

Defines an augmented dictionary class for SimSnap properties where
entries need to be managed e.g.  for defining default entries, or
for ensuring consistency between equivalent properties like
redshift and scalefactor.
"""

import warnings
from . import config



class SimDict(dict) :
    _getters = {}
    _setters = {}    

    @staticmethod
    def getter(f) :
        SimDict._getters[f.__name__] = f

    @staticmethod
    def setter(f) :
        SimDict._setters[f.__name__] = f
        
    def __getitem__(self, k) :
        if k in self :
            return dict.__getitem__(self, k)
        elif k in SimDict._getters :
            return SimDict._getters[k](self)
        else :
            raise KeyError, k

    def __setitem__(self, k, v) :
        if k in SimDict._setters :
            SimDict._setters[k](self, v)
        else :
            dict.__setitem__(self, k, v)


@SimDict.getter
def z(d) :
    if d["a"] is None :
        return None
    return 1.0/d["a"] -1.0

@SimDict.setter
def z(d,z) :
    if z is None :
        d["a"] = None
    else :
        d["a"] = 1.0/(1.0+z)
        


def default_fn(name, value) :
    """Return a getter function for the default name/value pair"""
    def f(d) :
        warnings.warn("Assuming default value for property '%s'=%.2e"%(name,value), RuntimeWarning)
        d[name]=value
        return value
    f.__name__ = name
    
    return f

for k in config['default-cosmology'] :
    SimDict.getter(default_fn(k, config['default-cosmology'][k]))
