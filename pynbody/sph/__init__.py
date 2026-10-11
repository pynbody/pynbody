"""
Low-level functionality for SPH interpolation

For most users, SPH functionality is accessed through the derived arrays `rho` and `smooth` in a snapshot object.
These arrays are calculated on-the-fly when accessed. The SPH kernels are defined in :mod:`pynbody.sph.kernels`.

.. versionchanged:: 2.9.0

  Rendering of images, grids and maps (previously :func:`pynbody.sph.render_image`, :func:`pynbody.sph.render_3d_grid`,
  :func:`pynbody.sph.render_spherical_image` and the :mod:`pynbody.sph.renderers` module) has moved to
  :mod:`pynbody.imaging`, since it now also renders AMR simulations.

Underlying all of the above is the :class:`~pynbody.kdtree.KDTree`, which finds the neighbours and provides the
lower-level SPH operations themselves, such as :meth:`~pynbody.kdtree.KDTree.sph_mean` and
:meth:`~pynbody.kdtree.KDTree.sph_divergence`. It also supports user-defined SPH operations, which are not
otherwise available in pynbody: :meth:`~pynbody.kdtree.KDTree.pair_reduce` accumulates an arbitrary pairwise
function over neighbour pairs, so that operations such as artificial viscosity or conduction can be expressed
without a python-level loop over particles. For finer control, :meth:`~pynbody.kdtree.KDTree.pair_blocks` exposes
the neighbour pairs themselves.

"""

import copy
import logging
import math
import os
import sys
import threading
import time
import warnings
from time import process_time

import numpy as np
import scipy
import scipy.ndimage

logger = logging.getLogger('pynbody.sph')

from .. import array, config, config_parser, kdtree, snapshot, units, util
from . import kernels


@snapshot.simsnap.SimSnap.stable_derived_array
def smooth(sim):
    """Return the smoothing length array for the simulation, using the configured number of neighbours"""
    sim.build_tree()

    logger.info('Smoothing with %d nearest neighbours' %
                config['sph']['smooth-particles'])

    sm = array.SimArray(np.empty(len(sim['pos']), dtype=sim['pos'].dtype), sim['pos'].units)

    start = time.time()
    sim.kdtree.set_array_ref('smooth', sm)
    sim.kdtree.populate('hsm', config['sph']['smooth-particles'])
    end = time.time()

    logger.info('Smoothing done in %5.3gs' % (end - start))
    sim._kdtree_derived_smoothing = True
    return sm

def _get_smooth_array_ensuring_compatibility(sim):
    # On-disk smoothing information may conflict; KDTree assumes the number of nearest neighbours
    # is rigidly adhered to. Thus we must use our own self-consistent smoothing.
    if 'smooth' in sim:
        if not getattr(sim, '_kdtree_derived_smoothing', False):
            smooth_ar = smooth(sim)
        else:
            smooth_ar = sim['smooth']
    else:
        sim['smooth'] = smooth_ar = smooth(sim)
    return smooth_ar

@snapshot.simsnap.SimSnap.stable_derived_array
def rho(sim):
    """Return the SPH density array for the simulation, using the configured number of neighbours"""
    sim.build_tree()

    logger.info('Calculating SPH density')
    rho = array.SimArray(
        np.empty(len(sim['pos'])), sim['mass'].units / sim['pos'].units ** 3,
        dtype=sim['pos'].dtype)


    start = time.time()


    sim.kdtree.set_array_ref('smooth', _get_smooth_array_ensuring_compatibility(sim))
    sim.kdtree.set_array_ref('mass', sim['mass'])
    sim.kdtree.set_array_ref('rho', rho)

    sim.kdtree.populate('rho', config['sph']['smooth-particles'])

    end = time.time()
    logger.info('Density calculation done in %5.3g s' % (end - start))

    return rho
