import pylab as p
import numpy as np
from .. import sph
from .. import units as _units

def sideon_image(sim, *args, **kwargs) :
    """Rotate the simulation so that the disc of the passed halo is
    side-on, then make an SPH image by passing the parameters into
    the function image"""

    from ..analysis import angmom
    angmom.sideon(sim)
    image(sim, *args, **kwargs)

def faceon_image(sim, *args, **kwargs) :
    """Rotate the simulation so that the disc of the passed halo is
    side-on, then make an SPH image by passing the parameters into
    the function image"""

    from ..analysis import angmom
    angmom.faceon(sim)
    image(sim, *args, **kwargs)


def image(sim, qty='rho', width=10, resolution=500, units=None, log=True, vmin=None, vmax=None, av_z = False,
          z_camera=None, clear = True, cmap=None) :
    """Make an SPH image of the given simulation.

    Keyword arguments

    qty -- The name of the array to interpolate (default 'rho')
    width -- The overall width and height of the plot in sim['pos'] units (default 10)
    resolution -- The number of pixels wide and tall
    units -- The units of the output
    av_z -- If True, the requested quantity is averaged down the line of sight
            (default False: image is generated in the thin plane z=0)
    z_camera -- If set, a perspective image is rendered. See pynbody.sph.image
                for more details.
    """

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

            sim["__one"]=np.ones_like(sim[qty])
            sim["__one"].units="1"
            im = sph.render_image(sim,qty,width/2,resolution,out_units=aunits, kernel = kernel, 
                                  z_camera=z_camera)
            im2 = sph.render_image(sim, "__one", width/2, resolution, kernel=kernel, 
                                   z_camera=z_camera)
            # del sim["__one"]
            
            im = im/im2
    else :
        
        im = sph.render_image(sim,qty,width/2,resolution,out_units=units, kernel = kernel,  z_camera = z_camera)

    if log :
        im[np.where(im==0)] = abs(im[np.where(im!=0)]).min()
        im = np.log10(im)

    if clear : p.clf()
    
    p.imshow(im[::-1,:],extent=(-width/2,width/2,-width/2,width/2), vmin=vmin, vmax=vmax, cmap=cmap)

    u_st = sim['pos'].units.latex()
    p.xlabel("$x/%s$"%u_st)
    p.ylabel("$y/%s$"%u_st)

    if units is None :
        units = im.units
   

    if log :
        units = r"$\log_{10}\,"+units.latex()+"$"
    else :
        units = "$"+units.latex()+"$"

    p.colorbar().set_label(units)
