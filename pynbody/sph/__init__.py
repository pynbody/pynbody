"""
Low-level functionality for SPH interpolation and rendering

For most users, SPH functionality is accessed through the derived arrays `rho` and `smooth` in a snapshot object.
These arrays are calculated on-the-fly when accessed.

SPH-interpolated images and grids can also be generated. For most users, the :func:`pynbody.plot.sph.image` function
is the most convenient interface for rendering images of SPH simulations.

A slightly lower-level approach is to call :func:`render_image` directly; or to get closer still to the underlying
machinery, use :func:`~pynbody.sph.renderers.make_render_pipeline` to create a renderer object, which then produces
an image when the :meth:`~pynbody.sph.renderers.ImageRendererBase.render` method is called.

The :func:`to_3d_grid` function can be used to create a 3D grid of interpolated values from an SPH simulation. This
functionality is also a thin wrapper around the :func:`~pynbody.sph.renderers.make_render_pipeline` function and
its associated classes in the :mod:`~pynbody.sph.renderers` module.

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
from . import kernels, renderers


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

def render_spherical_image(snap, quantity='rho', nside=None, kernel=None, denoise=None, out_units=None, threaded=None,
                           weight=None, qty=None):
    """Render an SPH image projected onto the sky around the origin.

    At present, only projection is supported (i.e., there is no implementation for rendering on a spherical
    shell). For example, if rendering density, the results are in units of mass per solid angle. The image is
    returned in healpix format, with the specified nside.

    Weighted projections are supported, e.g. one may look at the projected temperature, weighted by density, by
    passing 'temp' as the qty and 'rho' as the weight.

    Parameters
    ----------

    snap : snapshot.simsnap.SimSnap
        The snapshot to render

    quantity : str | np.ndarray
        The name of the array within the simulation to render, or an actual array. Default 'rho'

    weight : str, bool, optional
        The name of the array within the simulation to use as a weight for averaging down the line of sight; or
        True to use volume weighting.

    nside : int
        The healpix nside resolution to use (must be power of 2)

    kernel : str, kernels.KernelBase, optional
        The Kernel object to use (defaults to 3D spline kernel)

    denoise : bool, optional
        if True, divide through by an estimate of the discreteness noise. The returned image is then not strictly an
        SPH estimate, but this option can be useful to reduce noise.

    out_units : str, optional
        The units to convert the output image into

    threaded : bool, optional
        Whether to render the image across multiple threads. Yes if true; no if false. The number of threads to be
        used is determined by the configuration file. If None, the use of threading is also determined by the
        configuration file.

    qty : str, optional
        Deprecated - use 'quantity' instead


    """

    if qty is not None:
        warnings.warn("The 'qty' parameter is deprecated; use 'quantity' instead", DeprecationWarning)
        quantity = qty

    renderer = renderers.make_render_pipeline(snap, quantity, nside=nside, target='healpix', kernel=kernel,
                                              out_units=out_units, threaded=threaded,
                                              approximate_fast=False, weight=weight)

    return renderer.render()


def render_3d_grid(snap, quantity='rho', nx=None, ny=None, nz=None, width="10 kpc",
                   x2=None, out_units=None, kernel=None, approximate_fast=None,
                   threaded=None,  denoise=None, qty=None):
    """Create a 3d grid via SPH interpolation

    Parameters
    ----------

    snap : snapshot.simsnap.SimSnap
        The snapshot to render

    quantity : str | np.ndarray
        The name of the array within the simulation to render, or an actual array. Default 'rho'

    nx : int, optional
        The number of pixels wide to make the grid. If not specified, the default is to use the resolution
        from the configuration file.

    ny : int, optional
        The number of pixels tall to make the grid. If not specified, the default is to use the same as nx.

    nz : int, optional
        The number of pixels deep to make the grid. If not specified, the default is to use the same as nx.

    width : float, str, optional
        The width of the grid.

    x2 : float, optional
        Deprecated - use width instead. If provided, x2 overrides the width parameter and specifies half
        the width.

    out_units : str, optional
        The units to convert the output grid into

    kernel : str, kernels.KernelBase, optional
        The kernel to be used for the image rendering. If None, the default kernel is assigned. For more information
        see :func:`kernels.create_kernel`.

    approximate_fast : bool, optional
        Whether to render the image using a lower-resolution approximation for large smoothing lengths. The default
        is None, in which case the use of approximation is determined by the configuration file.

    threaded : bool, optional
        Whether to render the image across multiple threads. Yes if true; no if false. The number of threads to be
        used is determined by the configuration file. If None, the use of threading is also determined by the
        configuration file.

    denoise : bool, optional
        Whether to include denoising in the rendering process. If None, denoising is applied only if the image
        is likely to benefit from it. If True, denoising is to be forced on the image; if that is actually
        impossible, this routine raises an exception. If False, denoising is never applied.

    qty : str, optional
        Deprecated - use 'quantity' instead

    """

    if x2 is not None:
        width = x2*2

    if qty is not None:
        warnings.warn("The 'qty' parameter is deprecated; use 'quantity' instead", DeprecationWarning)
        quantity = qty

    renderer = renderers.make_render_pipeline(snap, quantity=quantity, resolution=nx, width=width,
                                              out_units = out_units, kernel = kernel,
                                              approximate_fast=approximate_fast, threaded=threaded,
                                              denoise=denoise, target='volume', nx=nx, ny=ny, nz=nz)
    return renderer.render()

@util.deprecated("to_3d_grid is deprecated; use render_3d_grid instead")
def to_3d_grid(*args, **kwargs):
    """Deprecated alias for :func:`render_3d_grid`"""
    return render_3d_grid(*args, **kwargs)

def render_image(*args, **kwargs):
    """Render an SPH image. This is a wrapper around the :mod:`renderers` module for convenience.

    All arguments and keyword arguments are forwarded to :func:`renderers.make_render_pipeline`,
    and the result is immediately rendered.
    """

    return renderers.make_render_pipeline(*args, **kwargs).render()
