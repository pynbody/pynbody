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

    
	
def image(sim, qty='rho', width=10, resolution=500, units=None, log=True, vmin=None, vmax=None) :
    """Make an SPH image of the given simulation.

    Keyword arguments

    qty -- The name of the array to interpolate (default 'rho')
    width -- The overall width and height of the plot in sim['pos'] units (default 10)
    resolution -- The number of pixels wide and tall
    units -- The units of the output
    """

    if isinstance(units, str) :
	units = _units.Unit(units)

    width = float(width)

    kernel = sph.Kernel()
    
    if units is not None :
	try :
	    sim[qty].units.ratio(units, **sim[qty].conversion_context())
	    # if this fails, perhaps we're requesting a projected image?
	    
	except _units.UnitsException :
	    # if the following fails, there's no interpretation this routine can cope with
	    sim[qty].units.ratio(units/(sim['x'].units), **sim[qty].conversion_context())
	    
	    kernel = sph.Kernel2D() # if we get to this point, we want a projected image

   
    im = sph.render_image(sim,qty,width/2,resolution,out_units=units, kernel = kernel)

    if log :
	im = np.log10(im)
    
    p.imshow(im[::-1,:],extent=(-width/2,width/2,-width/2,width/2), vmin=vmin, vmax=vmax)

    u_st = sim['pos'].units.latex()
    p.xlabel("$x/%s$"%u_st)
    p.ylabel("$y/%s$"%u_st)

    if units is None :
	units = sim[qty].units

   
    if log :
	units = r"$\log_{10}\,"+units.latex()+"$"
    else :
	units = "$"+units.latex()+"$"

    p.colorbar().set_label(units)
