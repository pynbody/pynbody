""" 

ionfrac
=======

calculates ionization fractions - NEEDS DOCUMENTATION

"""

import numpy as np
try :
    import scipy, scipy.weave
    from scipy.weave import inline
except ImportError :
    pass

import os
from ..array import SimArray
from pynbody import config

def calculate(sim,ion='ovi') :
    """

    calculate -- documentation placeholder

    """

    global config
    # ionization fractions calculated for optically thin case with
    # CLOUDY v 10.0.  J. Xavier Prochaska + Joe Hennawi have many 
    # helper idl routines for running CLOUDY
    iffile = os.path.join(os.path.dirname(__file__),"ionfracs.npz")
    if os.path.exists(iffile) :
        # import data
        if config['verbose']: print "Loading "+iffile
        ifs=np.load(iffile)
    else :
        raise IOError, "ionfracs.npz (Ion Fraction table) not found"

    # allocate temporary metals that we can play with
    # before inlining, the views on the arrays must be standard np.ndarray
    # otherwise the normal numpy macros are not generated
    x_vals = ifs['redshiftvals'].view(np.ndarray)
    y_vals = ifs['tempvals'].view(np.ndarray)
    z_vals = ifs['denvals'].view(np.ndarray)
    vals = ifs[ion+'if'].view(np.ndarray)
    x = np.zeros(len(sim.gas))
    x[:] = sim.properties['z']
    y = np.log10(sim.gas['temp']).view(np.ndarray)
    z = np.log10(sim.gas['rho'].in_units('m_p cm^-3')).view(np.ndarray)
    n = len(sim.gas)
    n_x_vals = len(x_vals)
    n_y_vals = len(y_vals)
    n_z_vals = len(z_vals)
    result_array = np.zeros(n)

    # get values off grid to minmax
    x[np.where(x < np.min(x_vals))] = np.min(x_vals)
    x[np.where(x > np.max(x_vals))] = np.max(x_vals)
    y[np.where(y < np.min(y_vals))] = np.min(y_vals)
    y[np.where(y > np.max(y_vals))] = np.max(y_vals)
    z[np.where(z < np.min(z_vals))] = np.min(z_vals)
    z[np.where(z > np.max(z_vals))] = np.max(z_vals)

    #interpolate
    if config['verbose']: print "Interpolating "+ion+" values"
    code = file(os.path.join(os.path.dirname(__file__),'interpolate3d.c')).read()
    inline(code,['n','n_x_vals','x_vals','n_y_vals','y_vals','n_z_vals',
                 'z_vals','x','y','z','vals','result_array'])
    
    return 10**result_array



