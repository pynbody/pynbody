"""
Rendering of images, grids and maps from simulations.

.. versionadded:: 2.9.0

  This package was split out of :mod:`pynbody.sph`, since it renders AMR simulations (by default, exactly, as cubic
  cells; see :class:`~pynbody.imaging.renderers.AMRImageRenderer`) as well as SPH simulations. The previous locations
  remain available but are deprecated.

For most users, the :func:`pynbody.plot.image` function is the most convenient interface for rendering images of
simulations.

A slightly lower-level approach is to call :func:`render_image` directly; or to get closer still to the underlying
machinery, use :func:`~pynbody.imaging.renderers.make_render_pipeline` to create a renderer object, which then
produces an image when the :meth:`~pynbody.imaging.renderers.ImageRendererBase.render` method is called.

The :func:`render_3d_grid` function can be used to create a 3D grid of interpolated values, and
:func:`render_spherical_image` a healpix map. These are also thin wrappers around
:func:`~pynbody.imaging.renderers.make_render_pipeline` and its associated classes in the
:mod:`~pynbody.imaging.renderers` module.

"""

import warnings

from . import renderers


def render_spherical_image(snap, quantity='rho', nside=None, kernel=None, denoise=None, out_units=None, threaded=None,
                           weight=None, distance=None, qty=None):
    """Render an SPH image onto the sky around the origin, in healpix format with the specified nside.

    By default the image is projected onto the sky, i.e. integrated along the line of sight. For example, if
    rendering density, the results are then in units of mass per solid angle.

    Alternatively, if ``distance`` is provided, a thin spherical shell is sampled at that radius using the full 3D
    kernel. The result then carries volumetric units (e.g. a density, if rendering density) rather than a
    per-solid-angle column.

    Weighted projections are supported, e.g. one may look at the projected temperature, weighted by density, by
    passing 'temp' as the qty and 'rho' as the weight.

    .. versionchanged :: 2.5.0

      The parameter *distance* has been added. If provided, the quantity is sampled on a thin spherical shell at
      this radius (in position units).

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

    distance : float, str, units.UnitBase, optional
        If provided, render a thin spherical shell at this radius (in position units) using the full 3D kernel,
        instead of the default line-of-sight projection. The result then carries volumetric units.

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
                                              shell_distance=distance, out_units=out_units, threaded=threaded,
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
        see :func:`pynbody.sph.kernels.create_kernel`.

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

def render_image(*args, **kwargs):
    """Render an SPH image. This is a wrapper around the :mod:`renderers` module for convenience.

    All arguments and keyword arguments are forwarded to :func:`renderers.make_render_pipeline`,
    and the result is immediately rendered.
    """

    return renderers.make_render_pipeline(*args, **kwargs).render()
