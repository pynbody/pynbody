"""
Routines for plotting smoothed quantities

"""

import warnings

import matplotlib
import numpy as np
import pylab as p

from .. import config, sph, units as _units
from ..sph import renderers


def _width_in_sim_units(sim, width):
	if isinstance(width, str) or issubclass(width.__class__, _units.UnitBase):
		if isinstance(width, str):
			width = _units.Unit(width)
		width = width.in_units(sim['pos'].units, **sim.conversion_context())
	return width

def sideon_image(sim, *args, **kwargs):
    """Create an image of the given simulation, side-on to the disc of the passed halo.

    This routine is a small wrapper around :func:`~pynbody.plot.sph.image` that rotates the simulation so that the disc
    of the passed halo is side-on, makes the SPH image, then rotates the simulation back to its original orientation.

    More flexible control over the orientation of the simulation can be achieved by using
    :func:`~pynbody.analysis.angmom.sideon` in combination with :func:`~pynbody.plot.sph.image`.

    """

    from ..analysis import angmom

    with angmom.sideon(sim):
        return image(sim, *args, **kwargs)


def faceon_image(sim, *args, **kwargs):
    """Create an image of the given simulation, face-on to the disc of the passed halo.

    This routine is a small wrapper around :func:`~pynbody.plot.sph.image` that rotates the simulation so that the disc
    of the passed halo is face-on, makes the SPH image, then rotates the simulation back to its original orientation.

    More flexible control over the orientation of the simulation can be achieved by using
    :func:`~pynbody.analysis.angmom.faceon` in combination with :func:`~pynbody.plot.sph.image`.

    """

    from ..analysis import angmom

    with angmom.faceon(sim):
        return image(sim, *args, **kwargs)


def contour(sim, qty, width="10 kpc", resolution=None, units=None, axes=None, label=True, log=True, weight=None,
			contour_kwargs=None, smooth_floor=0.0, _transform=None):
    """Create an image of the given quantity then turn it into contours.

    Parameters
    ----------

    sim : pynbody.snapshot.simsnap.SimSnap
        The simulation snapshot to plot. The image is generated in the plane z=0, or
        projected along the z axis.

    qty : str | pynbody.array.SimArray
        The name of the array to interpolate. Default is 'rho', which gives a density
        image. Alternatively, an array can be passed in.

    width : str or float, optional
        The overall width and height of the plot. If a float, it is assumed
        to be in units of sim['pos']. It can also be passed in as a string indicating
        the units, e.g. '10 kpc'. (Default is '10 kpc')

    resolution : int, optional
        The number of pixels wide and tall. (Default is determined by the
        :ref:`configuration file <configuration>`.)

    units : str or pynbody.units.Unit, optional
        The units of the output. Default is None, in which case the units of the input
        quantity are used. If the units correspond to integrating the quantity along
        a spatial dimension, the output is a projected image. For example, if the units
        are 'Msol kpc^-2', and the quantity is 'rho', the output is a projected image of
        the surface density.

    axes : matplotlib.axes.Axes, optional
        Axes instance on which the image will be shown; if None, the current pyplot figure is
        used. (Default is False)

    label : bool, optional
        Whether to label the contours. (Default is True)

    log : bool, optional
        If True, the image is log-scaled before being contoured. (Default is True)

    weight : str, optional
        If set, the requested quantity is volume-averaged down the line of sight, weighted either
        by volume (if weight is True) or by a specified quantity. (Default is None)

    contour_kwargs : dict, optional
        Additional keyword arguments to pass to the matplotlib contour function. (Default is None)

    smooth_floor : float, optional
        The minimum size of the smoothing kernel, either as a float or a unit string.
		Setting this to a non-zero value makes smoother, clearer contours but loses fine detail.
		Default is 0.0.

    _transform : function, optional
        A function to apply to the image before contouring. (Default is None)

    """

    if resolution is None:
        resolution = config['image-default-resolution']

    if axes is None:
        axes = p.gca()

    if contour_kwargs is None:
        contour_kwargs = {}

    width = _width_in_sim_units(sim, width)
    pixel_size = width / resolution

    # width of image must be a pixel wider than the width of the final contour field, since the contours
    # are based on the centres of the pixels not their edges

    pipeline = renderers.make_render_pipeline(sim.s, quantity=qty, width=width + pixel_size,
                                              weight = weight, out_units = units, resolution = resolution,
                                              smooth_floor = smooth_floor)

    im = pipeline.render()

    if log:
        im = np.log10(im)

    if _transform:
        im = _transform(im)

    # width of image was expanded above, so this is now the positions of the centres
    X = np.linspace(-width/2, width/2, resolution)

    CS = axes.contour(X, X, im, **contour_kwargs)
    if label:
        axes.clabel(CS, fontsize=12, inline=True)


def velocity_image(sim, qty='rho', vector_qty='vel', width="10 kpc", mode='quiver',
                   vector_color='black', vector_edgecolor='black', vector_resolution=40,
                   vector_scale=None, key=True, key_x=0.5, key_y=0.88,
                   key_color='k', key_edge_color='k', key_bg_color='w',
                   key_length="100 km s**-1",
                   stream_density=1.0, stream_linewidth = 1.0,
                   weight=None, restrict_depth=False,
                   **kwargs):
    """
    Make an SPH image of the given simulation with velocity vectors overlaid on top.

    Any keyword argument that can be passed to :func:`~pynbody.plot.sph.image` can also be passed
    to this function. See that function for a full list of options.

    Parameters
    ----------

    sim : pynbody.snapshot.simsnap.SimSnap
        The simulation snapshot to plot. The image is generated in the plane z=0, or
        projected along the z axis.

    qty : str | pynbody.array.SimArray
        The name of the array to interpolate. Default is 'rho', which gives a density
        image. Alternatively, an array can be passed in.

    vector_qty : str
        The name of the array to use for the vectors. Default is 'vel'.

    width : str or float, optional
        The overall width and height of the plot. If a float, it is assumed
        to be in units of sim['pos']. It can also be passed in as a string indicating
        the units, e.g. '10 kpc'. (Default is '10 kpc')

    mode : str, optional
        The type of plot to make. Options are 'quiver' or 'stream'. (Default is 'quiver')

    vector_color : str, optional
        The color of the velocity vectors. (Default is 'black')

    vector_edgecolor : str, optional
        The color for the edges of the velocity vectors. (Default is 'black')

    vector_resolution : int, optional
        The number of velocity vectors to generate in each dimension. (Default is 40)

    vector_scale : str or float, optional
        The length of a vector that would result in a displayed length of the
        figure width/height. This can be provided as a unit string or a float, in which
        case it is interpreted as having the same units as the vector array being plotted.
        Default is None, in which case the vectors are scaled by matplotlib's quiver function.
        This option is only used in 'quiver' mode.

    key : bool, optional
        Whether or not to inset a key showing the scale of the vectors.
        Only used if in 'quiver' mode. (Default is True)

    key_x : float, optional
        The x position of the key in 'quiver' mode, if key is True. (Default is 0.5)

    key_y : float, optional
        The y position of the key in 'quiver' mode, if key is True. (Default is 0.88)

    key_color : str, optional
        The color of the key arrow/text in 'quiver' mode, if key is True. (Default is 'white')

    key_edge_color : str, optional
        The color of the border around the key in 'quiver' mode, if key is True. (Default is 'black')

    key_bg_color : str, optional
        The color of the background of the key in 'quiver' mode, if key is True. (Default is 'white')

    key_length : str, optional
        The velocity to use for the key in 'quiver' mode, if key is True. (Default is '100 km s**-1')

    stream_density : float, optional
        The density of stream lines in 'stream' mode. (Default is 1.0)

    stream_linewidth : float, optional
        The width of stream lines in 'stream' mode. (Default is 1.0)

    **kwargs :
        Any additional keyword arguments to pass to :func:`~pynbody.plot.sph.image`.

    """



    if 'av_z' in kwargs:
        weight = kwargs.pop('av_z')
        warnings.warn("av_z is deprecated; use weight instead", DeprecationWarning)



    vx_name, vy_name, _ = sim._array_name_ND_to_1D(vector_qty)

    weight_for_vector = weight

    if 'units' in kwargs and _units_imply_projection(sim, qty, _units.Unit(kwargs['units'])) and weight is None:
        weight_for_vector = 'rho'

    vel_pipeline = renderers.make_render_pipeline(sim, quantity=vx_name, width=width,
                                                  resolution=vector_resolution, weight=weight_for_vector,
                                                  restrict_depth=restrict_depth)

    vx = vel_pipeline.render()
    vel_pipeline.set_quantity(vy_name)
    vy = vel_pipeline.render()

    key_unit = _units.Unit(key_length)

    if isinstance(width, str) or issubclass(width.__class__, _units.UnitBase):
        if isinstance(width, str):
            width = _units.Unit(width)
        width = width.in_units(sim['pos'].units, **sim.conversion_context())

    width = float(width)

    pixel_size = width / float(vector_resolution)
    X, Y = np.meshgrid(np.linspace(-width / 2 + pixel_size/2, width / 2 - pixel_size/2, vector_resolution),
                       np.linspace(-width / 2 + pixel_size/2, width / 2 - pixel_size/2, vector_resolution))

    im = image(sim, qty=qty, width=width, weight=weight, restrict_depth=restrict_depth,
               **kwargs)

    axes = kwargs.get('axes', None)
    if axes is None:
        axes = p.gca()

    if mode == 'quiver':
        if vector_scale is None:
            Q = axes.quiver(X, Y, vx, vy, color=vector_color, edgecolor=vector_edgecolor)
        else:
            if isinstance(vector_scale, str):
                vector_scale = _units.Unit(vector_scale)
            if _units.is_unit(vector_scale):
                vector_scale = vector_scale.in_units(sim['vel'].units)
            Q = axes.quiver(X, Y, vx, vy, scale=vector_scale, color=vector_color,
                         edgecolor=vector_edgecolor)
        if key:
            from . import util
            qk = util.PynbodyQuiverKey(Q, key_x, key_y,
                                      key_unit.in_units(sim['vel'].units, **sim.conversion_context()),
                                            "$" + key_unit.latex() + "$",
                                      color=key_color, labelcolor=key_color,
                                      boxedgecolor=key_edge_color, boxfacecolor=key_bg_color)
            qk.set_zorder(6)
            p.gca().add_artist(qk)
    elif mode == 'stream' :
        Q = axes.streamplot(X, Y, vx, vy, color=vector_color, density=stream_density, linewidth=stream_linewidth)

    axes.set_xlim(-width/2, width/2)
    axes.set_ylim(-width/2, width/2)


    return im


def volume(sim, qty='rho', width=None, resolution=200,
           color=(1.0,1.0,1.0),vmin=None,vmax=None,
           dynamic_range=4.0,log=True,
           create_figure=True):
    """Create a volume rendering of the given simulation using mayavi.

    .. warning ::
        This function requires mayavi to be installed. However, mayavi does not seem to be under
        active development and is not compatible with the latest versions of python.
        As a result, this function will probably be removed in future versions of pynbody.
        For a more modern alternative, consider using `topsy <https://github.com/pynbody/topsy/>`_.

    Parameters
    ----------
    sim : pynbody.snapshot.simsnap.SimSnap
        The simulation snapshot to visualize

    qty : str, optional
        The name of the array to interpolate. Default is 'rho', which gives a density image.

    width : str or float, optional
        The width of the cube to generate, centered on the origin. If None, the width is determined
        by the extent of the simulation snapshot.

    resolution : int, optional
        The number of elements along each side of the cube. (Default is 200)

    color : tuple, optional
        The color of the volume rendering. The value of each voxel is used to set the opacity.

    vmin : float, optional
        The value for zero opacity. If None, this is inferred from vmax and dynamic_range.

    vmax : float, optional
        The value for full opacity. If None, the maximum value of the image is used.

    dynamic_range : float, optional
        The dynamic range in dex to use if vmin and vmax are not specified.
        Default is 4.0

    log : bool, optional
        If True, the image is log-scaled before passing to mayavi. (Default is True)

    create_figure : bool, optional
        If True, create a new mayavi figure before rendering. (Default is True)

    """

    import mayavi
    from mayavi import mlab
    from tvtk.util.ctf import ColorTransferFunction, PiecewiseFunction



    if create_figure:
        fig = mlab.figure(size=(500,500),bgcolor=(0,0,0))

    grid_data = sph.to_3d_grid(sim,qty=qty,nx=resolution,
                               x2=None if width is None else width/2)



    if log:
        grid_data = np.log10(grid_data)
        if vmin is None:
            vmin = grid_data.max()-dynamic_range
        if vmax is None:
            vmax = grid_data.max()
    else:
        if vmin is None:
            vmin = np.min(grid_data)
        if vmax is None:
            vmax = np.max(grid_data)

    grid_data[grid_data<vmin]=vmin
    grid_data[grid_data>vmax]=vmax

    otf = PiecewiseFunction()
    otf.add_point(vmin,0.0)
    otf.add_point(vmax,1.0)

    sf = mayavi.tools.pipeline.scalar_field(grid_data)
    V = mlab.pipeline.volume(sf,color=color,vmin=vmin,vmax=vmax)




    V.trait_get('volume_mapper')['volume_mapper'].blend_mode = 'maximum_intensity'

    if color is None:
        ctf = ColorTransferFunction()
        ctf.add_rgb_point(vmin,107./255,124./255,132./255)
        ctf.add_rgb_point(vmin+(vmax-vmin)*0.8,200./255,178./255,164./255)
        ctf.add_rgb_point(vmin+(vmax-vmin)*0.9,1.0,210./255,149./255)
        ctf.add_rgb_point(vmax,1.0,222./255,141./255)

        V._volume_property.set_color(ctf)
        V._ctf = ctf
        V.update_ctf = True

    V._otf = otf
    V._volume_property.set_scalar_opacity(otf)


    return V


def _units_imply_projection(sim, qty, units):
    if isinstance(qty, str):
        qty = sim[qty]

    try:
        qty.units.ratio(units, **sim.conversion_context())
        # if this fails, perhaps we're requesting a projected image?
        return False
    except _units.UnitsException:
        # if the following fails, there's no interpretation this routine
        # can cope with. The error will be allowed to propagate.
        qty.units.ratio(
            units / (sim['x'].units), **sim.conversion_context())
        return True

def spherical_image(sim, qty='rho', nside=None, kernel=None, threaded=None, units=None,
                    weight=False, log=True, vmin=None, vmax=None, cmap=None, xsize=1600):
    """Make an SPH image on the sky around the origin.

    .. note::

      While pynbody does not require the healpy module to be installed, this function requires healpy to render
      the healpix image onto a Mollweide projection.

      To install healpy, use ``pip install healpy``.

    Parameters
    ----------

    sim : pynbody.snapshot.simsnap.SimSnap
        The simulation snapshot to plot. The output will be a projected spherical image centred on the origin of the
        particles in the snapshot.

    qty : str | pynbody.array.SimArray
        The name of the array to interpolate. Default is 'rho', which gives a density image. Alternatively, an array
        can be passed in.

    nside : int
        The healpix nside resolution to use (must be power of 2)

    kernel : str, optional
        SPH kernel to use for smoothing; see :func:`~pynbody.sph.kernels.create_kernel` for options.

    units : str or pynbody.units.Unit, optional
        The units of the output. Default is None, in which case the units of the input quantity are used. Note that
        unless using *weight*, the output is a projected angular image. For example, for density, the output unit
        is mass per solid angle.

    threaded : bool, optional
        If True, use threads to parallelise the rendering. (Default is set in the config file).

    weight : bool or str, optional
        If True, the requested quantity is volume-averaged down the line of sight. If a string, the requested quantity
        is averaged down the line of sight weighted by the named array (e.g. use 'rho' for density-weighted quantity).

    log : bool, optional
        If True, the image is log-scaled. (Default is True)

    vmin : float, optional
        The minimum value for the color scale. If None, the minimum value of the image.

    vmax : float, optional
        The maximum value for the color scale. If None, the maximum value of the image.

    cmap : matplotlib.colors.Colormap or str, optional
        Colormap to use. If None, the default colormap is used.

    xsize : int, optional
        The *xsize* parameter for healpy.mollview, which determines the resolution of the projection (i.e. does not
        affect the resolution of the actual image, but the presentation.) Default is 1600.

    """
    import healpy as hp
    image = sph.render_spherical_image(sim, quantity=qty, nside=nside, kernel=kernel, out_units=units,
                                       weight=weight, threaded=threaded)

    unit = image.units
    if unit is not None:
        unit = f'${unit.latex()}$'
    else:
        unit = ''
    hp.mollview(image, title=None, hold=True, norm='log' if log else None, unit=unit, cmap=cmap, min=vmin, max=vmax,
                xsize=xsize)

def image(sim, qty='rho', width="10 kpc", resolution=None, units=None, log=True,
          vmin=None, vmax=None, weight=False, z_camera=None, clear=True, cmap=None,
          title=None, colorbar_label=None, qtytitle=None, show_cbar=True, axes=None,
          noplot=False, return_image=False, return_array=False,
          fill_nan=True, fill_val=0.0, linthresh=None,
          restrict_depth = False, threaded=True, approximate_fast=None, denoise=None,
          kernel=None,
          **kwargs):
    """
    Make an image of the given simulation, using SPH or denoised-SPH interpolation.

    Parameters
    ----------
    sim : pynbody.snapshot.simsnap.SimSnap
        The simulation snapshot to plot. The image is generated in the plane z=0, or
        projected along the z axis.

    qty : str | pynbody.array.SimArray
        The name of the array to interpolate. Default is 'rho', which gives a density
        image. Alternatively, an array can be passed in.

    width : str or float, optional
        The overall width and height of the plot. If a float, it is assumed
        to be in units of sim['pos']. It can also be passed in as a string indicating
        the units, e.g. '10 kpc'. (Default is '10 kpc')

    resolution : int, optional
        The number of pixels wide and tall. (Default is determined by the
        :ref:`configuration file <configuration>`.)

    units : str or pynbody.units.Unit, optional
        The units of the output. Default is None, in which case the units of the input
        quantity are used. If the units correspond to integrating the quantity along
        a spatial dimension, the output is a projected image. For example, if the units
        are 'Msol kpc^-2', and the quantity is 'rho', the output is a projected image of
        the surface density.

    log : bool, optional
        If True, the image is log-scaled. (Default is True)

    vmin : float, optional
        The minimum value for the color scale. If None, the minimum value of the image.

    vmax : float, optional
        The maximum value for the color scale. If None, the maximum value of the image.

    weight : bool or str, optional
        If True, the requested quantity is volume-averaged down the line of sight. If a string, the
        requested quantity is averaged down the line of sight weighted by the named array
        (e.g. use 'rho' for density-weighted quantity).

    restrict_depth : bool, optional
        If True, restrict the depth of the image to the width of the image. (Default is False)

    z_camera : float, optional
        If set, a perspective image is rendered, as though the camera is a pinhole camera
        at (0,0,z_camera). The frustrum is defined by the width of the image in the plane z=0.

    clear : bool, optional
        Whether to clear the axes before plotting. (Default is True)

    cmap : matplotlib.colors.Colormap or str, optional
        Colormap to use. If None, the default colormap is used.

    title : str, optional
        Plot title.

    colorbar_label : str, optional
        Colorbar label. If not provided, one will be generated from the quantity name and units.

    show_cbar : bool, optional
        Whether to automatically plot the colorbar. (Default is True)

    axes : matplotlib.axes.Axes, optional
        Axes instance on which the image will be shown; if None, the current pyplot figure is
        used. (Default is False)

    noplot : bool, optional
        If True, the image is not displayed, only the image array is returned. This option therefore
        implies return_array = True.

    return_image : bool, optional
        If True, the image instance returned by imshow is returned. (Default is False)

    return_array : bool, optional
        If True, the numpy array of the image is returned. (Default is False)

    fill_nan : bool, optional
        If any of the image values are NaN, replace with fill_val. (Default is True)

    fill_val : float, optional
        The fill value to use when replacing NaNs. (Default is 0.0)

    linthresh : float, optional
        If the image has negative and positive values and a log scaling is requested, the part
        between -linthresh and linthresh is shown on a linear scale to avoid divergence at 0.

    kernel : str, optional
        SPH kernel to use for smoothing; see :func:`~pynbody.sph.kernels.create_kernel` for options.

    approximate_fast : bool, optional
        If True, speed up the image-making by rendering large kernels onto a lower-resolution image
        first, which is then interpolated to the final resolution. (Default is set in the config file).

    threaded : bool, optional
        If True, use threads to parallelise the rendering. (Default is set in the config file).

    qtytitle : str, optional
        Deprecated alias for colorbar_label.

    av_z : bool or str, optional
        Deprecated alias for weight.

    Returns
    -------

    matplotlib.image.AxesImage
        The image instance returned by imshow, if return_image is True.

    numpy.ndarray
        The image array, if return_array is True or noplot is True.

    """

    global config

    if not noplot:
        import matplotlib.pylab as plt
        if axes:
            p = axes
        else:
            p = plt

    if qtytitle is not None:
        warnings.warn("qtytitle is deprecated; use colorbar_label instead", DeprecationWarning)
        colorbar_label = qtytitle

    if kwargs.get('av_z', None) is not None:
        weight = kwargs.pop('av_z')
        warnings.warn("av_z is deprecated; use weight instead", DeprecationWarning)

    if kwargs.get('ret_im', None) is not None:
        return_image = kwargs.pop('ret_im')
        warnings.warn("ret_im is deprecated; use return_image instead", DeprecationWarning)

    if colorbar_label is None and isinstance(qty, str):
        qtytitle = qty
    else:
        qtytitle = None

    if weight and qtytitle:
        qtytitle = f"$\\langle${qtytitle}$\\rangle$"

    renderer = renderers.make_render_pipeline(sim, quantity=qty, width=width, resolution=resolution,
                                              out_units=units, weight=weight, restrict_depth=restrict_depth,
                                              kernel=kernel, z_camera=z_camera, threaded=threaded,
                                              approximate_fast=approximate_fast, denoise=denoise)

    # if width was provided e.g. as string, we'll need it as a float
    width = renderer.geometry.width

    if renderer.is_projected and qtytitle is not None:
        qtytitle = f"$\\int\\,${qtytitle}$\\,\\mathrm{{d}}z$"

    im = renderer.render()

    if fill_nan:
        im[np.isnan(im)] = fill_val

    if not noplot:

        # set the log or linear normalizations
        if log:
            try:
                im[np.where(im == 0)] = abs(im[np.where(abs(im != 0))]).min()
            except ValueError:
                raise ValueError("Failed to make a sensible logarithmic image. This probably means there are no particles in the view.")

            # check if there are negative values -- if so, use the symmetric
            # log normalization
            if (vmin is None and (im < 0).any() ) or ((vmin is not None) and vmin<0):

                # need to set the linear regime around zero -- set to by
                # default start at 1/1000 of the log range
                if linthresh is None:
                    linthresh = np.nanmax(abs(im)) / 1000.
                norm = matplotlib.colors.SymLogNorm(
                    linthresh, vmin=vmin, vmax=vmax)
            else:
                norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

        else:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        #
        # do the actual plotting
        #
        if clear and not axes:
            p.clf()

        ims = p.imshow(im[::-1, :].view(np.ndarray), extent=(-width / 2, width / 2, -width / 2, width / 2),
                       cmap=cmap, norm = norm)

        u_st = sim['pos'].units.latex()
        if not axes:
            plt.xlabel("$x/%s$" % u_st)
            plt.ylabel("$y/%s$" % u_st)
        else:
            p.set_xlabel("$x/%s$" % u_st)
            p.set_ylabel("$y/%s$" % u_st)

        if units is None:
            units = im.units

        if not isinstance(units, _units.UnitBase):
            units = _units.Unit(units)

        if units.latex() == "":
            units=""
        else:
            units = "$"+units.latex()+"$"

        if show_cbar:
            colorbar = plt.colorbar(ims)
            if colorbar_label is not None:
                colorbar.set_label(colorbar_label)
            elif qtytitle is not None:
                colorbar.set_label(qtytitle+"/"+units)


        if title is not None:
            if not axes:
                p.title(title)
            else:
                p.set_title(title)



    if return_image and return_array:
        return ims, im
    elif return_image:
        return ims
    elif return_array or noplot:
        return im
