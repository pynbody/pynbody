"""

ionfrac
=======

calculates ionization fractions - NEEDS DOCUMENTATION

"""

import logging
import os

import numpy as np

from pynbody import config

logger = logging.getLogger('pynbody.analysis.ionfrac')

from .interpolate import interpolate3d


def calculate(sim, ion='ovi', mode='old'):
    """

    calculate -- documentation placeholder

    """

    global config
    # ionization fractions calculated for optically thin case with
    # CLOUDY v 10.0.  J. Xavier Prochaska + Joe Hennawi have many
    # helper idl routines for running CLOUDY
    iffile = os.path.join(os.path.dirname(__file__), "ionfracs.npz")
    if os.path.exists(iffile):
        # import data
        logger.info("Loading %s" % iffile)
        ifs = np.load(iffile)
    else:
        raise OSError("ionfracs.npz (Ion Fraction table) not found")

    # allocate temporary metals that we can play with
    # before inlining, the views on the arrays must be standard np.ndarray
    # otherwise the normal numpy macros are not generated
    x_vals = ifs['redshiftvals'].view(np.ndarray)
    y_vals = ifs['tempvals'].view(np.ndarray)
    z_vals = ifs['denvals'].view(np.ndarray)
    vals = ifs[ion + 'if'].view(np.ndarray)
    x = np.zeros(len(sim.gas))
    x[:] = sim.properties['z']
    y = np.log10(sim.gas['temp']).view(np.ndarray)
    z = np.log10(sim.gas['rho'].in_units('m_p cm^-3')).view(np.ndarray)
    n = len(sim.gas)
    n_x_vals = len(x_vals)
    n_y_vals = len(y_vals)
    n_z_vals = len(z_vals)

    # get values off grid to minmax
    x[np.where(x < np.min(x_vals))] = np.min(x_vals)
    x[np.where(x > np.max(x_vals))] = np.max(x_vals)
    y[np.where(y < np.min(y_vals))] = np.min(y_vals)
    y[np.where(y > np.max(y_vals))] = np.max(y_vals)
    z[np.where(z < np.min(z_vals))] = np.min(z_vals)
    z[np.where(z > np.max(z_vals))] = np.max(z_vals)

    # interpolate
    logger.info("Interpolation %s values" % ion)
    result_array = interpolate3d(x, y, z, x_vals, y_vals, z_vals, vals)

    return 10 ** result_array
