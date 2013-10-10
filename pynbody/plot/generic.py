"""

generic
=======

Flexible and general plotting functions

"""

import numpy as np, pylab as plt
import pynbody
from ..analysis import profile, angmom, halo
from .. import config
from ..array import SimArray
from ..units import NoUnit

def qprof(sim,qty='metals',weights=None,q=(0.16,0.5,0.84),
          ax=False,ylabel=None,xlog=False,ylog=False,xlabel=None,
          facecolor='#BBBBBB',color='#BBBBBB',medcolor='black',filename=False):
    if ax: 
        p=ax
        f = plt.gca()
    else: f,p=plt.subplots(1)

    qp = pynbody.analysis.profile.QuantileProfile(sim,q=q,weights=weights)
    p.fill_between(qp['rbins'].in_units('kpc'),qp[qty][:,0],y2=qp[qty][:,2],
                   facecolor=facecolor,color=color,
                   where=np.isfinite(qp[qty][:,0]))
    p.plot(qp['rbins'].in_units('kpc'),qp[qty][:,1],color=medcolor)
    if xlabel is None: p.set_xlabel('r [kpc]')
    else: p.set_xlabel(xlabel)
    if ylabel is not None: p.set_ylabel(ylabel)
    if xlog: p.semilogx()
    if ylog: p.semilogy()

    if filename: f.savefig(filename)

def hist2d(xo, yo, weights=None, mass=None, gridsize=(100,100), nbins = None, make_plot = True, **kwargs):
    """
    Plot 2D histogram for arbitrary arrays that get passed in.

    **Input:**

       *x*: array

       *y*: array

    **Optional keywords:**

       *x_range*: list, array, or tuple
         size(x_range) must be 2. Specifies the X range.

       *y_range*: tuple
         size(y_range) must be 2. Specifies the Y range.

       *gridsize*: (int, int) (default (100,100)) 
         number of bins to use for the 2D histogram

       *nbins*: int
         number of bins for the histogram - if specified, gridsize is set to (nbins,nbins)

       *nlevels*: int
         number of levels to use for the contours

       *logscale*: boolean
         whether to use log or linear spaced contours

       *weights*: numpy array of same length as x and y
         if weights is passed, color corresponds to
         the mean value of weights in each cell

       *mass*: numpy array of masses same length as x andy
         must also have weights passed in.  If you just
         want to weight by mass, pass the masses to weights

       *colorbar*: boolean
         draw a colorbar
         
       *scalemin*: float
         minimum value to use for the color scale

       *scalemax*: float
         maximum value to use for the color scale
    """
    global config
    
    # process keywords
    x_range = kwargs.get('x_range',None)
    y_range = kwargs.get('y_range',None)
    xlogrange = kwargs.get('xlogrange', False)
    ylogrange = kwargs.get('ylogrange', False)
    ret_im = kwargs.get('ret_im',False)

    if y_range is not None :
        if len(y_range) != 2 : 
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        if ylogrange:
            y_range = [np.log10(np.min(yo)),np.log10(np.max(yo))]
        else:
            y_range = [np.min(yo),np.max(yo)]
        kwargs['y_range'] = y_range
            
    if x_range is not None:
        if len(x_range) != 2 :
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        if xlogrange:
            x_range = [np.log10(np.min(xo)), np.log10(np.max(xo))]
        else:
            x_range = [np.min(xo),np.max(xo)]
        kwargs['x_range'] = x_range

    if (xlogrange):
        x = np.log10(xo)
    else :
        x = xo
        
    if (ylogrange):
        y = np.log10(yo)
    else :
        y = yo
    
    if nbins is not None: 
        gridsize = (nbins,nbins)

    ind = np.where((x > x_range[0]) & (x < x_range[1]) &
                   (y > y_range[0]) & (y < y_range[1]))

    x = x[ind[0]]
    y = y[ind[0]]
    
    draw_contours = False
    if weights is not None and mass is not None: 
        draw_contours = True
        weights = weights[ind[0]]
        mass = mass[ind[0]]

        # produce a mass-weighted histogram of average weight values at each bin
        hist, ys, xs = np.histogram2d(y, x, weights=weights*mass, bins=gridsize,range=[y_range,x_range])
        hist_mass, ys, xs = np.histogram2d(y, x, weights=mass,bins=gridsize,range=[y_range,x_range])
        good = np.where(hist_mass > 0)
        hist[good] = hist[good]/hist_mass[good]
            
    else:
        if weights is not None : 
            # produce a weighted histogram
            weights = weights[ind[0]]
        elif mass is not None: 
            # produce a mass histogram
            weights = mass[ind[0]]
      
        hist, ys, xs = np.histogram2d(y, x, weights=weights, bins=gridsize,range=[y_range,x_range])
        
    try: 
        hist = SimArray(hist,weights.units)
    except AttributeError: 
        hist = SimArray(hist)

        
    try: 
        xs = SimArray(.5*(xs[:-1]+xs[1:]), x.units)
        ys = SimArray(.5*(ys[:-1]+ys[1:]), y.units)
    except AttributeError: 
        xs = .5*(xs[:-1]+xs[1:])
        ys = .5*(ys[:-1]+ys[1:])


    if ret_im :
        return make_contour_plot(hist, xs, ys, **kwargs)

    if make_plot : 
        make_contour_plot(hist, xs, ys, **kwargs)
        if draw_contours:
            make_contour_plot(SimArray(density_mass, mass.units),xs,ys,filled=False,clear=False,colorbar=False,colors='black',scalemin=nmin,nlevels=10)

    return hist, xs, ys
    


def gauss_kde(xo, yo, weights=None, mass = None, gridsize = (100,100), nbins = None,
              make_plot = True, nmin = None, nmax = None, **kwargs) :

    """
    Plot 2D gaussian kernel density estimate (KDE) given values at points (*x*, *y*). 

    Behavior changes depending on which keywords are passed: 

    If a *weights* array is supplied, produce a weighted KDE.
    
    If a *mass* array is supplied, a mass density is computed. 

    If both *weights* and *mass* are supplied, a mass-averaged KDE of the weights is 
    computed. 

    By default, norm=False is passed to :func:`~pynbody.plot.util.fast_kde` meaning
    that the result returned *is not* normalized such that the integral over the area
    equals one. 

Since this function produces a density estimate, the units of the
    output grid are different than for the output of
    :func:`~pynbody.plot.generic.hist2d`. To get to the same units,
    you must multiply by the size of the cells.

    
    **Input:**

       *xo*: array

       *yo*: array

    **Optional keywords:**

       *mass*: numpy array of same length as x and y 
         particle masses to be used for weighting
    
       *weights*: numpy array of same length as x and y
         if weights is passed, color corresponds to
         the mean value of weights in each cell

       *nmin*: float (default None)
         if *weights* and *mass* are both specified, the mass-weighted
         contours are only drawn where the mass exceeds *nmin*. 
          
       *gridsize*: (int, int) (default: 100,100)
         size of grid for computing the density estimate

       *nbins*: int
         number of bins for the histogram - if specified, gridsize is set to (nbins,nbins)

       *make_plot*: boolean (default: True)
         whether or not to produce a plot
         
    **Keywords passed to** :func:`~pynbody.plot.util.fast_kde`:
       
       *norm*: boolean (default: False) 
         If False, the output is only corrected for the kernel. If True,
         the result is normalized such that the integral over the area 
         yields 1. 
    
       *nocorrelation*: (default: False) If True, the correlation
         between the x and y coords will be ignored when preforming
         the KDE.
         
    **Keywords passed to** :func:`~pynbody.plot.generic.make_contour_plot`:

       *x_range*: list, array, or tuple
         size(x_range) must be 2. Specifies the X range.

       *y_range*: tuple
         size(y_range) must be 2. Specifies the Y range.
       
       *nlevels*: int
         number of levels to use for the contours

       *logscale*: boolean
         whether to use log or linear spaced contours

       *colorbar*: boolean
         draw a colorbar
         
       *scalemin*: float
         minimum value to use for the color scale

       *scalemax*: float
         maximum value to use for the color scale
    """
    from util import fast_kde
    from scipy.stats.kde import gaussian_kde

    global config
    
    # process keywords
    x_range = kwargs.get('x_range',None)
    y_range = kwargs.get('y_range',None)
    xlogrange = kwargs.get('xlogrange', False)
    ylogrange = kwargs.get('ylogrange', False)
    ret_im = kwargs.get('ret_im',False)

    
    if y_range is not None :
        if len(y_range) != 2 : 
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        if ylogrange:
            y_range = [np.log10(np.min(yo)),np.log10(np.max(yo))]
        else:
            y_range = [np.min(yo),np.max(yo)]
            
    if x_range is not None:
        if len(x_range) != 2 :
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        if xlogrange:
            x_range = [np.log10(np.min(xo)), np.log10(np.max(xo))]
        else:
            x_range = [np.min(xo),np.max(xo)]

    if (xlogrange):
        x = np.log10(xo)
    else :
        x = xo
    if (ylogrange):
        y = np.log10(yo)
    else :
        y = yo
        
    if nbins is not None:
        gridsize = (nbins,nbins)

    ind = np.where((x > x_range[0]) & (x < x_range[1]) &
                   (y > y_range[0]) & (y < y_range[1]))

    x = x[ind[0]]
    y = y[ind[0]]

    try:
        xs = SimArray(np.linspace(x_range[0], x_range[1], gridsize[0]+1),x.units)
    except AttributeError:
        xs = np.linspace(x_range[0], x_range[1], gridsize[0]+1)
    xs = .5*(xs[:-1]+xs[1:])
    try:
        ys = SimArray(np.linspace(y_range[0], y_range[1], gridsize[1]+1),y.units)
    except AttributeError:
        ys = np.linspace(y_range[0], y_range[1], gridsize[1]+1)
    ys = .5*(ys[:-1]+ys[1:])
    
    extents = [x_range[0],x_range[1],y_range[0],y_range[1]]
    
    draw_contours = False
    if weights is not None and mass is not None: 
        draw_contours = True
        weights = weights[ind[0]]
        mass = mass[ind[0]]
        
        # produce a mass-weighted gaussian KDE of average weight values at each bin
        density = fast_kde(x, y, weights=weights*mass, gridsize=gridsize,extents=extents, **kwargs)
        density_mass = fast_kde(x, y, weights=mass, gridsize=gridsize,extents=extents, **kwargs)
        good = np.where(density_mass > 0)
        density[good] = density[good]/density_mass[good]

        if nmin is not None : 
            density *= density_mass > nmin

    else:
        # produce a weighted gaussian KDE
        if weights is not None : 
            weights = weights[ind[0]]
        elif mass is not None : 
            weights = mass[ind[0]]

        density = fast_kde(x, y, weights=weights, gridsize=gridsize,
                           extents=extents, **kwargs)

    try: 
        density = SimArray(density,weights.units)
    except AttributeError: 
        density = SimArray(density)

    if ret_im: 
        return make_contour_plot(density,xs,ys,**kwargs)

    if make_plot : 
        make_contour_plot(density,xs,ys,**kwargs)
        if draw_contours:
            make_contour_plot(SimArray(density_mass, mass.units),xs,ys,filled=False,clear=False,colorbar=False,colors='black',scalemin=nmin,nlevels=10)

    return density, xs, ys


def make_contour_plot(arr, xs, ys, x_range=None, y_range=None, nlevels = 20, 
                      logscale=True, xlogrange=False, ylogrange=False, 
                      subplot=False, colorbar=False, ret_im=False, cmap=None,
                      clear=True,legend=False, scalemin = None, 
                      scalemax = None, filename = None, **kwargs) : 
    """
    Plot a contour plot of grid *arr* corresponding to bin centers
    specified by *xs* and *ys*.  Labels the axes and colobar with
    proper units taken from x

    Called by :func:`~pynbody.plot.generic.hist2d` and
    :func:`~pynbody.plot.generic.gauss_density`.
    
    **Input**: 
    
       *arr*: 2D array to plot

       *xs*: x-coordinates of bins
       
       *ys*: y-coordinates of bins

    **Optional Keywords**:
          
       *x_range*: list, array, or tuple (default = None)
         size(x_range) must be 2. Specifies the X range.

       *y_range*: tuple (default = None)
         size(y_range) must be 2. Specifies the Y range.

       *xlogrange*: boolean (default = False)
         whether the x-axis should have a log scale

       *ylogrange*: boolean (default = False)
         whether the y-axis should have a log scale

       *nlevels*: int (default = 20)
         number of levels to use for the contours

       *logscale*: boolean (default = True)
         whether to use log or linear spaced contours

       *colorbar*: boolean (default = False)
         draw a colorbar
         
       *scalemin*: float (default = arr.min())
         minimum value to use for the color scale

       *scalemax*: float (default = arr.max())
         maximum value to use for the color scale
    """


    from matplotlib import ticker, colors
    
    if not subplot :
        import matplotlib.pyplot as plt
    else :
        plt = subplot

    if scalemin is None: scalemin = np.min(arr[arr>0])
    if scalemax is None: scalemax = np.max(arr[arr>0])
    arr[arr<scalemin]=scalemin
    arr[arr>scalemax]=scalemax

    if 'norm' in kwargs: del(kwargs['norm'])
    
    if logscale:
        try:
            levels = np.logspace(np.log10(scalemin),       
                                 np.log10(scalemax),nlevels)
            cont_color=colors.LogNorm()
        except ValueError:
            raise ValueError('crazy contour levels -- try specifying the *levels* keyword or use a linear scale')
            
            return

        if arr.units != NoUnit() and arr.units != 1 :
            cb_label = '$log_{10}('+arr.units.latex()+')$'
        else :
            cb_label = '$log_{10}(N)$'
    else:
        levels = np.linspace(scalemin,
                             scalemax,nlevels)
        cont_color=None
        
        if arr.units != NoUnit() and arr.units != 1 :
            cb_label = '$'+arr.units.latex()+'$'
        else :
            cb_label = '$N$'
    
    if not subplot and clear : plt.clf()

    if ret_im :
        if logscale: arr = np.log10(arr)
        scalemin, scalemax = np.log10((scalemin,scalemax))
        return plt.imshow(arr, origin='down', vmin=scalemin, vmax=scalemax,
                          aspect = 'auto',cmap=cmap,
                          #aspect = np.diff(x_range)/np.diff(y_range),cmap=cmap,
                          extent=[x_range[0],x_range[1],y_range[0],y_range[1]])
    cs = plt.contourf(xs,ys,arr, levels, norm=cont_color,cmap=cmap,**kwargs)

    
    if kwargs.has_key('xlabel'):
        xlabel = kwargs['xlabel']
    else :
        try:
            if xlogrange: xlabel=r''+'$log_{10}('+xs.units.latex()+')$'
            else : xlabel = r''+'$x/' + xs.units.latex() +'$'
        except AttributeError:
            xlabel = None

    if xlabel :
        try:
            if subplot :
                plt.set_xlabel(xlabel)
            else:
                plt.xlabel(xlabel)
        except:
            pass

    if kwargs.has_key('ylabel'):
        ylabel = kwargs['ylabel']
    else :
        try:
            if ylogrange: ylabel='$log_{10}('+ys.units.latex()+')$'
            else : ylabel = r''+'$y/' + ys.units.latex() +'$'
        except AttributeError:
            ylabel=None

    if ylabel :
        try:
            if subplot :
                plt.set_ylabel(ylabel)
            else:
                plt.ylabel(ylabel)
        except:
            pass

    
#    if not subplot:
#        plt.xlim((x_range[0],x_range[1]))
#        plt.ylim((y_range[0],y_range[1]))

    if colorbar :
        cb = plt.colorbar(cs, format = "%.2e").set_label(r''+cb_label)
        
    if legend : plt.legend(loc=2)

    if (filename): 
        if config['verbose']: print "Saving "+filename
        plt.savefig(filename)



    

                  
def fourier_map(sim, nbins = 100, nmin = 1000, nphi=100, mmin=1, mmax=7, rmax=10, 
                levels = [.01,.05,.1,.2], subplot = None, ret = False, **kwargs) : 
    """

    Plot an overdensity map generated from a Fourier expansion of the
    particle distribution. A :func:`~pynbody.analysis.profile.Profile`
    is made and passed to :func:`~pynbody.plot.util.inv_fourier` to
    obtain an overdensity map. The map is plotted using the usual
    matplotlib.contour. 
    
    **Input**:

    *sim* :  a :func:`~pynbody.snapshot.SimSnap` object

    **Optional Keywords**:
    
    *nbins* (100) : number of radial bins to use for the profile

    *nmin* (1000) : minimum number of particles required per bin 

    *nphi* (100)  : number of azimuthal bins to use for the map

    *mmin* (1)    : lowest multiplicity Fourier component

    *mmax* (7)    : highest multiplicity Fourier component

    *rmax* (10)   : maximum radius to use when generating the profile

    *levels* [0.01,0.05,0.1,0.2] : tuple of levels for plotting contours
    
    *subplot* (None) : Axes object on which to plot the contours
    
    """
    from . import util

    if subplot is None : 
        import matplotlib.pylab as plt

    else : 
        plt = subplot

    p = pynbody.analysis.profile.Profile(sim,max=rmax,nbins=nbins)
    phi,phi_inv = util.inv_fourier(p,nmin,mmin,mmax,nphi)

    rr,pp = np.meshgrid(p['rbins'],phi)

    xx = (rr*np.cos(pp)).T
    yy = (rr*np.sin(pp)).T

    plt.contour(xx,yy,phi_inv,levels,**kwargs)
    
    if ret: 
        return xx,yy,phi_inv


def prob_plot(x,y,weight,nbins=(100,100),extent=None,axes=None,**kwargs) : 
    """ 

    Make a plot of the probability of y given x, i.e. p(y|x). The
    values are normalized such that the integral along each column is
    one.

    **Input**: 

    *x*: primary binning axis

    *y*: secondary binning axis

    *weight*: weights array

    *nbins*: tuple of length 2 specifying the number of bins in each direction

    *extent*: tuple of length 4 speciphysical extent of the axes
     (xmin,xmax,ymin,ymax)

    **Optional Keywords**:

    all optional keywords are passed on to the imshow() command

    """

    import matplotlib.pylab as plt

    assert(len(nbins)==2)
    grid = np.zeros(nbins)

    if extent is None : 
        extent = (min(x),max(x),min(y),max(y))

    xbinedges = np.linspace(extent[0],extent[1],nbins[0]+1)
    ybinedges = np.linspace(extent[2],extent[3],nbins[1]+1)
    

    for i in xrange(nbins[0]) : 
        
        ind = np.where((x > xbinedges[i])&(x < xbinedges[i+1]))[0]
        h, bins = np.histogram(y[ind],weights=weight[ind], bins = ybinedges, density = True)
        grid[:,i] = h

    if axes is None: 
        im = plt.imshow(grid,extent=extent,origin='lower',**kwargs)

    else : 
        im = axes.imshow(grid,extent=extent,origin='lower',**kwargs)

    cb = plt.colorbar(im,format='%.2f')
    cb.set_label(r'$P(y|x)$')
    
    return grid, xbinedges, ybinedges
