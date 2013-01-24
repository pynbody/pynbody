"""

sph
===

routines for plotting smoothed quantities 

"""

import pylab as p
import numpy as np
from .. import sph, config
from .. import units as _units

def sideon_image(sim, *args, **kwargs) :
    """

    Rotate the simulation so that the disc of the passed halo is
    side-on, then make an SPH image by passing the parameters into
    the function image

    For a description of keyword arguments see :func:`~pynbody.plot.sph.image`.

    """

    from ..analysis import angmom
    if 'center' in kwargs:
        if kwargs['center']:
            angmom.sideon(sim)
    else :
        angmom.sideon(sim)
    return image(sim, *args, **kwargs)

def faceon_image(sim, *args, **kwargs) :
    """

    Rotate the simulation so that the disc of the passed halo is
    face-on, then make an SPH image by passing the parameters into
    the function image

    For a description of keyword arguments see :func:`~pynbody.plot.sph.image`.

    """

    from ..analysis import angmom
    angmom.faceon(sim)
    return image(sim, *args, **kwargs)


def image(sim, qty='rho', width=10, resolution=500, units=None, log=True, 
          vmin=None, vmax=None, av_z = False, filename=None, 
          z_camera=None, clear = True, cmap=None, center=False,
          title=None, qtytitle=None, show_cbar=True, subplot=False,
          noplot = False, ret_im=False, fill_nan = True, fill_val=0.0,
          **kwargs) :
    """

    Make an SPH image of the given simulation.

    **Keyword arguments:**

    *qty* (rho): The name of the array to interpolate

    *width* (10): The overall width and height of the plot in sim['pos'] units

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
    
    
    """
    import matplotlib.pylab as plt

    global config
    if subplot:
        p = subplot
    else :
        p = plt

    if isinstance(units, str) :
        units = _units.Unit(units)

    width = float(width)

    kernel = sph.Kernel()
 
    perspective = z_camera is not None
    if perspective and not av_z: kernel = sph.Kernel2D()

    
    if units is not None :
        try :
            sim[qty].units.ratio(units, **sim[qty].conversion_context())
            # if this fails, perhaps we're requesting a projected image?

        except _units.UnitsException :
            # if the following fails, there's no interpretation this routine can cope with
            sim[qty].units.ratio(units/(sim['x'].units), **sim[qty].conversion_context())

            kernel = sph.Kernel2D() # if we get to this point, we want a projected image
    
    if av_z :
        if isinstance(kernel, sph.Kernel2D) :
            raise _units.UnitsException("Units already imply projected image; can't also average over line-of-sight!")
        else :
            kernel = sph.Kernel2D()
            if units is not None :
                aunits = units*sim['z'].units
            else :
                aunits = None

            if isinstance(av_z, str) :
                if units is not None: 
                    aunits = units*sim[av_z].units*sim['z'].units
                sim["__prod"] = sim[av_z]*sim[qty]
                qty = "__prod"
                
            else :
                av_z = "__one"
                sim["__one"]=np.ones_like(sim[qty])
                sim["__one"].units="1"
                
            im = sph.render_image(sim,qty,width/2,resolution,out_units=aunits, kernel = kernel, 
                                          z_camera=z_camera, **kwargs)
            im2 = sph.render_image(sim, av_z, width/2, resolution, kernel=kernel, 
                                           z_camera=z_camera, **kwargs)
            
            top = sim.ancestor

            try:
                del top["__one"]
            except KeyError :
                pass

            try:
                del top["__prod"]
            except KeyError :
                pass
    
            im = im/im2
         
    else :

        im = sph.render_image(sim,qty,width/2,resolution,out_units=units, 
                                      kernel = kernel,  z_camera = z_camera, **kwargs)

    if fill_nan : 
        im[np.isnan(im)] = fill_val

    if log :
        im[np.where(im==0)] = abs(im[np.where(im!=0)]).min()
        im = np.log10(im)

    if clear and not subplot : p.clf()

    if ret_im:
        return plt.imshow(im[::-1,:],extent=(-width/2,width/2,-width/2,width/2), 
                 vmin=vmin, vmax=vmax, cmap=cmap)

    ims = p.imshow(im[::-1,:],extent=(-width/2,width/2,-width/2,width/2), 
                   vmin=vmin, vmax=vmax, cmap=cmap)

    u_st = sim['pos'].units.latex()
    if not subplot:
        plt.xlabel("$x/%s$"%u_st)
        plt.ylabel("$y/%s$"%u_st)
    else :
        p.set_xlabel("$x/%s$"%u_st)
        p.set_ylabel("$y/%s$"%u_st)

    if units is None :
        units = im.units
   

    if log :
        units = r"$\log_{10}\,"+units.latex()+"$"
    else :
        units = "$"+units.latex()+"$"

    if show_cbar:
        if qtytitle is not None: plt.colorbar(ims).set_label(qtytitle)
        else:                    plt.colorbar(ims).set_label(units)
    # colorbar doesn't work wtih subplot:  mappable is NoneType
    #elif show_cbar:
    #    import matplotlib.pyplot as mpl
    #    if qtytitle: mpl.colorbar().set_label(qtytitle)
    #    else:        mpl.colorbar().set_label(units)

    if title is not None:
        p.set_title(title)
        
    if filename is not None:
        p.savefig(filename)
        
    
    plt.draw()
    plt.show()

    return im

def image_radial_profile(im, bins=100):

    xsize, ysize = np.shape(im)
    x = np.arange(-xsize/2, xsize/2)
    y = np.arange(-ysize/2, ysize/2)
    xs, ys = np.meshgrid(x,y)
    rs = np.sqrt(xs**2 + ys**2)
    hist, bin_edges = np.histogram(rs,bins=bins)
    inds = np.digitize(rs.flatten(), bin_edges)
    ave_vals = np.zeros(bin_edges.size)
    max_vals = np.zeros(bin_edges.size)
    min_vals = np.zeros(bin_edges.size)
    for i in np.arange(bin_edges.size):
        try:
            min_vals[i] = np.min(10**(im.flatten()[np.where(inds == i)]))
        except ValueError:
            min_vals[i] = float('nan')
        ave_vals[i] = np.mean(10**(im.flatten()[np.where(inds == i)]))
        try:
            max_vals[i] = np.max(10**(im.flatten()[np.where(inds == i)]))
        except ValueError:
            max_vals[i] = float('nan')

    return ave_vals, min_vals, max_vals, bin_edges
