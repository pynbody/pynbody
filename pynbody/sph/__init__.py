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

def render_spherical_image(snap, qty='rho', nside=8, kernel=None, denoise=None, out_units=None, threaded=False):
    """Render an SPH image on a spherical surface. Requires healpy libraries to be installed.

    Parameters
    ----------

    snap : snapshot.simsnap.SimSnap
        The snapshot to render

    qty : str
        The name of the array within the simulation to render

    nside : int
        The healpix nside resolution to use (must be power of 2)

    distance : float
        The distance of the shell (for 3D kernels) or maximum distance of the skewers (2D kernels)

    kernel : str, kernels.KernelBase, optional
        The Kernel object to use (defaults to 3D spline kernel)

    denoise : bool, optional
        if True, divide through by an estimate of the discreteness noise. The returned image is then not strictly an
        SPH estimate, but this option can be useful to reduce noise.

    out_units : str, optional
        The units to convert the output image into

    threaded : bool, optional
        if False, render on a single core. *Currently threading is not supported for spherical images, because
        healpy does not release the gil*.

    """

    kernel = kernels.create_kernel(kernel)

    if kernel.h_power == 3:
        kernel = kernel.projection()

    if denoise is None:
        denoise = renderers._auto_denoise(snap, kernel)

    if denoise and not _kernel_suitable_for_denoise(kernel):
        raise ValueError("Denoising not supported with this kernel type. Re-run with denoise=False")

    renderer = _render_spherical_image

    if threaded is None:
        threaded = config_parser.getboolean('sph', 'threaded-image')

    if threaded:
        raise RuntimeError("Threading is not supported for spherical images, because healpy does not release the gil")

    im = renderer(snap, qty, nside, kernel, denoise, out_units)
    return im


def _render_spherical_image(snap, qty='rho', nside=8, kernel=None, denoise=None, out_units=None, __threaded=False, snap_slice=None):

    kernel = kernels.create_kernel(kernel)

    if denoise is None:
        denoise = _auto_denoise(snap, kernel)

    if denoise and not _kernel_suitable_for_denoise(kernel):
        raise ValueError("Denoising not supported with this kernel type. Re-run with denoise=False")

    if out_units is not None:
        conv_ratio = (snap[qty].units * snap['mass'].units / (snap['rho'].units * units.sr)).ratio(out_units, **snap.conversion_context())

    if snap_slice is None:
        snap_slice = slice(len(snap))
    with snap.immediate_mode:
        h, pos, mass, rho, qtyar = (snap[x].view(
            np.ndarray)[snap_slice] for x in ('smooth', 'pos', 'mass', 'rho', qty))

    im = _render.render_spherical_image_core(rho, mass, qtyar, pos,  h, nside, kernel)

    im = im.view(array.SimArray)

    im.units = snap[qty].units * snap["mass"].units / \
        snap["rho"].units / units.sr
    im.sim = snap

    if out_units is not None:
        im.convert_units(out_units)

    return im

def to_3d_grid(snap, qty='rho', nx=None, ny=None, nz=None, width="10 kpc",
               x2=None, out_units=None, kernel=None, approximate_fast=None,
               threaded=None,  denoise=None):
    """Create a 3d grid via SPH interpolation

    Parameters
    ----------

    snap : snapshot.simsnap.SimSnap
        The snapshot to render

    qty : str
        The name of the array within the simulation to render

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

    """

    if x2 is not None:
        width = x2*2

    renderer = renderers.make_render_pipeline(snap, quantity=qty, resolution=nx, width=width,
                                              out_units = out_units, kernel = kernel,
                                              approximate_fast=approximate_fast, threaded=threaded,
                                              denoise=denoise, grid_3d=True, nx=nx, ny=ny, nz=nz)
    return renderer.render()

def render_image(*args, **kwargs):
    """Render an SPH image. This is a wrapper around the :mod:`renderers` module for convenience.

    All arguments and keyword arguments are forwarded to :func:`renderers.make_render_pipeline`,
    and the result is immediately rendered.
    """

    return renderers.make_render_pipeline(*args, **kwargs).render()
