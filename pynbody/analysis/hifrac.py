"""

hifrac
=======

calculates Hydrogen ionization fraction - limited version of ionfrac to make use of CLOUDY HI table

"""

import numpy as np

from .interpolate import interpolate3d

import os
from ..array import SimArray
import logging
logger = logging.getLogger('pynbody.analysis.hifrac')

from pynbody import config


def calculate(sim,ion='hi',selfshield=False) :
    """

    calculate -- documentation placeholder

    """
    import h5py
    global config
    # ionization fractions calculated for optically thin case with
    # CLOUDY for Duffy et al. (2012) overall h1frac.py based on
    # ionfrac.py routine, ultimately should be merged when other
    # elements are added to the hdf5 file
    iffile = os.path.join(os.path.dirname(__file__),"h1.hdf5")
    if os.path.exists(iffile) :
        # import data
        logger.info("Loading %s" % iffile)
        ifs=h5py.File(iffile,'r')
    else :
        raise IOError("h1.hdf5 (HI Fraction table) not found")

    # allocate temporary metals that we can play with
    # before inlining, the views on the arrays must be standard np.ndarray
    # otherwise the normal numpy macros are not generated
    x_vals = ifs['logd'][:].view(np.ndarray)
    y_vals = ifs['logt'][:].view(np.ndarray)
    z_vals = ifs['redshift'][:].view(np.ndarray)
    vals = np.log10(ifs['ionbal'][:]).view(np.ndarray)
    ifs.close()

    z = np.zeros(len(sim.gas)).view(np.ndarray)
    z[:] = sim.properties['z']
    y = np.log10(sim.gas['temp']).view(np.ndarray)
    x = np.log10(sim.gas['rho'].in_units('m_p cm^-3')).view(np.ndarray)
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
    logger.info("Interpolation %s values" % ion)
    result_array = interpolate3d(x, y, z, x_vals, y_vals, z_vals, vals)

    ## Selfshield criteria assume all EoS gas
    if selfshield != False:
        result_array[sim.gas['OnEquationOfState'] == 1.] = 0.
    ## Selfshield criteria from Duffy et al 2012a (in addition to EoS gas)
        if selfshield == 'duffy12':
            result_array[(sim.gas['p'].in_units('m_p K cm**-3') > 150.) & (sim.gas['temp'].in_units('K') < 10.**(4.5))] = 0.

    ## Get as HI per proton mass (essentially multiplying the HI fraction by the Hydrogen mass fraction)
    result_array += np.log10(sim.gas['hydrogen'])

    result_array = (10.**result_array).view(SimArray)
#    result_array.units = units.m_p**-1

    return result_array
