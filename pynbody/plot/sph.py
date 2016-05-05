"""

sph
===

routines for plotting smoothed quantities

"""

import pylab as p
import matplotlib
import numpy as np
from .. import sph, config
from .. import units as _units
from matplotlib.ticker import ScalarFormatter # RS 
from matplotlib.ticker import FuncFormatter # RS

def fmt(x,pos):
    """
    Custom formatter to handle log of values for color bar
    Not sure why, but the LogFormatterExponenet sometimes
    adds and "e" to the value... ?? This simply returns
    log10 of the value.
    """
    return format(np.log10(x), '.0f')

def sideon_image(sim, *args, **kwargs):
    """

    Rotate the simulation so that the disc of the passed halo is
    side-on, then make an SPH image by passing the parameters into
    the function image

    For a description of keyword arguments see :func:`~pynbody.plot.sph.image`.

    """

    from ..analysis import angmom

    with angmom.sideon(sim):
        return image(sim, *args, **kwargs)


def faceon_image(sim, *args, **kwargs):
    """

    Rotate the simulation so that the disc of the passed halo is
    face-on, then make an SPH image by passing the parameters into
    the function image

    For a description of keyword arguments see :func:`~pynbody.plot.sph.image`.

    """

    from ..analysis import angmom

    with angmom.faceon(sim):
        return image(sim, *args, **kwargs)


def velocity_image(sim, width="10 kpc", vector_color='black', edgecolor='black', quiverkey_bg_color=None,
                   vector_resolution=40, scale=None, mode='quiver', key_x=0.3, key_y=0.9,
                   key_color='white', key_length="100 km s**-1", quiverkey=True, density=1.0,
                   vector_qty='vel', **kwargs):
    """

    Make an SPH image of the given simulation with velocity vectors overlaid on top.

    For a description of additional keyword arguments see :func:`~pynbody.plot.sph.image`,
    or see the `tutorial <http://pynbody.github.io/pynbody/tutorials/pictures.html#velocity-vectors>`_.

    **Keyword arguments:**

    *vector_color* (black): The color for the velocity vectors

    *edgecolor* (black): edge color used for the lines - using a color
     other than black for the *vector_color* and a black *edgecolor*
     can result in poor readability in pdfs

    *vector_resolution* (40): How many vectors in each dimension (default is 40x40)

    *quiverkey_bg_color* (none): The color for the legend (scale) background

    *scale* (None): The length of a vector that would result in a displayed length of the
    figure width/height.

    *mode* ('quiver'): make a 'quiver' or 'stream' plot

    *key_x* (0.3): Display x (width) position for the vector key (quiver mode only)

    *key_y* (0.9): Display y (height) position for the vector key (quiver mode only)

    *key_color* (white): Color for the vector key (quiver mode only)

    *key_length* (100 km/s): Velocity to use for the vector key (quiver mode only)

    *density* (1.0): Density of stream lines (stream mode only)

    *quiverkey* (True): Whether or not to inset the key

    *vector_qty* ('vel'): The name of the vector field to plot
    """

    subplot = kwargs.get('subplot', False)
    av_z = kwargs.get('av_z',None)
    if subplot:
        p = subplot
    else:
        import matplotlib.pylab as p

    vx_name, vy_name, _ = sim._array_name_ND_to_1D(vector_qty)

    vx = image(sim, qty=vx_name, width=width, log=False,
               resolution=vector_resolution, noplot=True,av_z=av_z)
    vy = image(sim, qty=vy_name, width=width, log=False,
               resolution=vector_resolution, noplot=True,av_z=av_z)
    key_unit = _units.Unit(key_length)

    if isinstance(width, str) or issubclass(width.__class__, _units.UnitBase):
        if isinstance(width, str):
            width = _units.Unit(width)
        width = width.in_units(sim['pos'].units, **sim.conversion_context())

    width = float(width)

    X, Y = np.meshgrid(np.arange(-width / 2, width / 2, width / vector_resolution),
                       np.arange(-width / 2, width / 2, width / vector_resolution))

    im = image(sim, width=width, **kwargs)

    if mode == 'quiver':
        if scale is None:
            Q = p.quiver(X, Y, vx, vy, color=vector_color, edgecolor=edgecolor)
        else:
            Q = p.quiver(X, Y, vx, vy, scale=_units.Unit(scale).in_units(
                sim['vel'].units), color=vector_color, edgecolor=edgecolor)
        if quiverkey:
        	qk = p.quiverkey(Q, key_x, key_y, key_unit.in_units(sim['vel'].units, **sim.conversion_context()),
                    r"$\mathbf{" + key_unit.latex() + "}$", labelcolor=key_color, color=key_color, fontproperties={'size': 24})
        if  quiverkey_bg_color is not None:
            qk.text.set_backgroundcolor(quiverkey_bg_color)
    elif mode == 'stream' :
        Q = p.streamplot(X, Y, vx, vy, color=vector_color, density=density)

    p.xlim(-width/2, width/2)
    p.ylim(-width/2, width/2)

    return im


def volume(sim, qty='rho', width=None, resolution=200,
           color=(1.0,1.0,1.0),vmin=None,vmax=None,
           dynamic_range=4.0,log=True,
           create_figure=True):
    """Create a volume rendering of the given simulation using mayavi.

    **Keyword arguments:**

    *qty* (rho): The name of the array to interpolate

    *width* (None): The width of the cube to generate, centered on the origin

    *resolution* (200): The number of elements along each side of the cube

    *color* (white): The color of the volume rendering. The value of each voxel
       is used to set the opacity.

    *vmin* (None): The value for zero opacity (calculated using dynamic_range if None)

    *vmax* (None): The value for full opacity (calculated from the maximum
       value in the region if None)

    *dynamic_range*: The dynamic range to use if vmin and vmax are not specified

    *log* (True): log-scale the image before passing to mayavi

    *create_figure* (True): create a new mayavi figure before rendering
    """

    import mayavi
    from mayavi import mlab
    from tvtk.util.ctf import PiecewiseFunction, ColorTransferFunction



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
        print vmin,vmax
        V._volume_property.set_color(ctf)
        V._ctf = ctf
        V.update_ctf = True

    V._otf = otf
    V._volume_property.set_scalar_opacity(otf)


    return V

def contour(*args, **kwargs):
    """
    Make an SPH image of the given simulation and render it as contours.
    nlevels and levels are passed to pyplot's contour command.

    Other arguments are as for *image*.
    """

    import copy
    kwargs_image = copy.copy(kwargs)
    nlevels = kwargs_image.pop('nlevels',None)
    levels = kwargs_image.pop('levels',None)
    width = kwargs_image.get('width','10 kpc')
    kwargs_image['noplot']=True
    im = image(*args, **kwargs_image)
    res = im.shape

    units = kwargs_image.get('units',None)

    if isinstance(width, str) or issubclass(width.__class__, _units.UnitBase):
        if isinstance(width, str):
            width = _units.Unit(width)
        sim = args[0]
        width = width.in_units(sim['pos'].units, **sim.conversion_context())

    width = float(width)
    x,y = np.meshgrid(np.linspace(-width/2,width/2,res[0]),np.linspace(-width/2,width/2,res[0]))

    p.contour(x,y,im,nlevels=nlevels,levels=levels)



def image(sim, qty='rho', width="10 kpc", resolution=500, units=None, log=True,
          vmin=None, vmax=None, av_z=False, filename=None,
          z_camera=None, clear=True, cmap=None,
          title=None, qtytitle=None, show_cbar=True, subplot=False,
          noplot=False, ret_im=False, fill_nan=True, fill_val=0.0, linthresh=None,
          **kwargs):
    """

    Make an SPH image of the given simulation.

    **Keyword arguments:**

    *qty* (rho): The name of the array to interpolate

    *width* (10 kpc): The overall width and height of the plot. If
     ``width`` is a float or an int, then it is assumed to be in units
     of ``sim['pos']``. It can also be passed in as a string
     indicating the units, i.e. '10 kpc', in which case it is
     converted to units of ``sim['pos']``.

    *resolution* (500): The number of pixels wide and tall

    *units* (None): The units of the output

    *av_z* (False): If True, the requested quantity is averaged down
            the line of sight (default False: image is generated in
            the thin plane z=0, unless output units imply an integral
            down the line of sight). If a string, the requested quantity
            is averaged down the line of sight weighted by the av_z
            array (e.g. use 'rho' for density-weighted quantity;
            the default results when av_z=True are volume-weighted).

    *z_camera* (None): If set, a perspective image is rendered. See
                :func:`pynbody.sph.image` for more details.

    *filename* (None): if set, the image will be saved in a file

    *clear* (True): whether to call clf() on the axes first

    *cmap* (None): user-supplied colormap instance

    *title* (None): plot title

    *qtytitle* (None): colorbar quantity title

    *show_cbar* (True): whether to plot the colorbar

    *subplot* (False): the user can supply a AxesSubPlot instance on
    which the image will be shown

    *noplot* (False): do not display the image, just return the image array

    *ret_im* (False): return the image instance returned by imshow

    *num_threads* (None) : if set, specify the number of threads for
    the multi-threaded routines; otherwise the pynbody.config default is used

    *fill_nan* (True): if any of the image values are NaN, replace with fill_val

    *fill_val* (0.0): the fill value to use when replacing NaNs

    *linthresh* (None): if the image has negative and positive values
     and a log scaling is requested, the part between `-linthresh` and
     `linthresh` is shown on a linear scale to avoid divergence at 0
    """

    if not noplot:
        import matplotlib.pylab as plt

    global config
    if not noplot:
        if subplot:
            p = subplot
        else:
            p = plt

    if isinstance(units, str):
        units = _units.Unit(units)

    if isinstance(width, str) or issubclass(width.__class__, _units.UnitBase):
        if isinstance(width, str):
            width = _units.Unit(width)
        width = width.in_units(sim['pos'].units, **sim.conversion_context())

    width = float(width)

    kernel = sph.Kernel()

    perspective = z_camera is not None
    if perspective and not av_z:
        kernel = sph.Kernel2D()

    if units is not None:
        try:
            sim[qty].units.ratio(units, **sim[qty].conversion_context())
            # if this fails, perhaps we're requesting a projected image?

        except _units.UnitsException:
            # if the following fails, there's no interpretation this routine
            # can cope with
            sim[qty].units.ratio(
                units / (sim['x'].units), **sim[qty].conversion_context())

            # if we get to this point, we want a projected image
            kernel = sph.Kernel2D()

    if av_z:
        if isinstance(kernel, sph.Kernel2D):
            raise _units.UnitsException(
                "Units already imply projected image; can't also average over line-of-sight!")
        else:
            kernel = sph.Kernel2D()
            if units is not None:
                aunits = units * sim['z'].units
            else:
                aunits = None

            if isinstance(av_z, str):
                if units is not None:
                    aunits = units * sim[av_z].units * sim['z'].units
                sim["__prod"] = sim[av_z] * sim[qty]
                qty = "__prod"

            else:
                av_z = "__one"
                sim["__one"] = np.ones_like(sim[qty])
                sim["__one"].units = "1"

            im = sph.render_image(sim, qty, width / 2, resolution, out_units=aunits, kernel=kernel,
                                  z_camera=z_camera, **kwargs)
            im2 = sph.render_image(sim, av_z, width / 2, resolution, kernel=kernel,
                                   z_camera=z_camera, **kwargs)

            top = sim.ancestor

            try:
                del top["__one"]
            except KeyError:
                pass

            try:
                del top["__prod"]
            except KeyError:
                pass

            im = im / im2

    else:
        im = sph.render_image(sim, qty, width / 2, resolution, out_units=units,
                              kernel=kernel,  z_camera=z_camera, **kwargs)

    if fill_nan:
        im[np.isnan(im)] = fill_val

    if not noplot:

        # set the log or linear normalizations
        if log:
            try:
                im[np.where(im == 0)] = abs(im[np.where(abs(im != 0))]).min()
            except ValueError:
                raise ValueError, "Failed to make a sensible logarithmic image. This probably means there are no particles in the view."

            # check if there are negative values -- if so, use the symmetric
            # log normalization
            if (vmin is None and (im < 0).any() ) or vmin<0:

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
        if clear and not subplot:
            p.clf()

        if ret_im:
            return p.imshow(im[::-1, :].view(np.ndarray), extent=(-width / 2, width / 2, -width / 2, width / 2),
                            vmin=vmin, vmax=vmax, cmap=cmap, norm = norm)

        ims = p.imshow(im[::-1, :].view(np.ndarray), extent=(-width / 2, width / 2, -width / 2, width / 2),
                       vmin=vmin, vmax=vmax, cmap=cmap, norm = norm)

        u_st = sim['pos'].units.latex()
        if not subplot:
            plt.xlabel("$x/%s$" % u_st)
            plt.ylabel("$y/%s$" % u_st)
        else:
            p.set_xlabel("$x/%s$" % u_st)
            p.set_ylabel("$y/%s$" % u_st)

        if units is None:
            units = im.units

        if log :
            units = r"$\log_{10}\,"+units.latex()+"$"
        else :
            if units.latex() is "":
                units=""
            else:
                units = "$"+units.latex()+"$"

        if show_cbar:
            if log:
                custom_formatter = FuncFormatter(fmt)
                ## l_f = LogFormatterExponent() # sometimes tacks 'e' on value...???
                l_f = custom_formatter
            else:
                l_f = ScalarFormatter()

            if qtytitle is not None:
                plt.colorbar(ims,format=l_f).set_label(qtytitle)
            else:
                plt.colorbar(ims,format=l_f).set_label(units)
        # colorbar doesn't work wtih subplot:  mappable is NoneType
        # elif show_cbar:
        #    import matplotlib.pyplot as mpl
        #    if qtytitle: mpl.colorbar().set_label(qtytitle)
        #    else:        mpl.colorbar().set_label(units)

        if title is not None:
            if not subplot:
                p.title(title)
            else:
                p.set_title(title)

        if filename is not None:
            p.savefig(filename)

        plt.draw()
        # plt.show() - removed by AP on 30/01/2013 - this should not be here as
        # for some systems you don't get back to the command prompt

    return im


def image_radial_profile(im, bins=100):

    xsize, ysize = np.shape(im)
    x = np.arange(-xsize / 2, xsize / 2)
    y = np.arange(-ysize / 2, ysize / 2)
    xs, ys = np.meshgrid(x, y)
    rs = np.sqrt(xs ** 2 + ys ** 2)
    hist, bin_edges = np.histogram(rs, bins=bins)
    inds = np.digitize(rs.flatten(), bin_edges)
    ave_vals = np.zeros(bin_edges.size)
    max_vals = np.zeros(bin_edges.size)
    min_vals = np.zeros(bin_edges.size)
    for i in np.arange(bin_edges.size):
        try:
            min_vals[i] = np.min(10 ** (im.flatten()[np.where(inds == i)]))
        except ValueError:
            min_vals[i] = float('nan')
        ave_vals[i] = np.mean(10 ** (im.flatten()[np.where(inds == i)]))
        try:
            max_vals[i] = np.max(10 ** (im.flatten()[np.where(inds == i)]))
        except ValueError:
            max_vals[i] = float('nan')

    return ave_vals, min_vals, max_vals, bin_edges
