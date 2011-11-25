"""

generic
=======

Flexible and general plotting functions

"""

import numpy as np
from ..analysis import profile, angmom, halo
from .. import config
from ..array import SimArray
from ..units import NoUnit

def hist2d(xo, yo, weights=None, mass=None, gridsize=(100,100), make_plot = True, **kwargs):
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
    
    ind = np.where((x > x_range[0]) & (x < x_range[1]) &
                   (y > y_range[0]) & (y < y_range[1]))

    x = x[ind[0]]
    y = y[ind[0]]
    
    
    if weights is not None and mass is not None: 
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

        
    xs = SimArray(.5*(xs[:-1]+xs[1:]), x.units)
    ys = SimArray(.5*(ys[:-1]+ys[1:]), y.units)


    if make_plot : 
        make_contour_plot(hist, xs, ys, **kwargs)

                
    return hist, xs, ys
    


def gauss_kde(xo, yo, weights=None, mass = None, gridsize = (100,100), make_plot = True, **kwargs) :

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
    
       *weights*: numpy array of same length as x and y
         if weights is passed, color corresponds to
         the mean value of weights in each cell
          
       *gridsize*: (int, int) (default: 100,100)
         size of grid for computing the density estimate

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
        
    ind = np.where((x > x_range[0]) & (x < x_range[1]) &
                   (y > y_range[0]) & (y < y_range[1]))

    x = x[ind[0]]
    y = y[ind[0]]

    xs = SimArray(np.linspace(x_range[0], x_range[1], gridsize[0]+1),x.units)
    xs = .5*(xs[:-1]+xs[1:])
    ys = SimArray(np.linspace(y_range[0], y_range[1], gridsize[1]+1),y.units)
    ys = .5*(ys[:-1]+ys[1:])
    
    extents = [x_range[0],x_range[1],y_range[0],y_range[1]]

    if weights is not None and mass is not None: 
        weights = weights[ind[0]]
        mass = mass[ind[0]]
        
        # produce a mass-weighted gaussian KDE of average weight values at each bin
        density = fast_kde(x, y, weights=weights*mass, gridsize=gridsize,extents=extents, **kwargs)
        density_mass = fast_kde(x, y, weights=mass, gridsize=gridsize,extents=extents, **kwargs)
        good = np.where(density_mass > 0)
        density[good] = density[good]/density_mass[good]

    else:
        # produce a weighted gaussian KDE
        if weights is not None : 
            weights = weights[ind[0]]
        elif mass is not None : 
            weights = mass[ind[0]]
    
        density = fast_kde(x, y, weights=weights, gridsize=gridsize,extents=extents, **kwargs)

    try: 
        density = SimArray(density,weights.units)
    except AttributeError: 
        density = SimArray(density)

    if make_plot : 
        make_contour_plot(density,xs,ys,**kwargs)

    return density, xs, ys


def make_contour_plot(arr, xs, ys, x_range=None, y_range=None, nlevels = 20, 
                      logscale=True, xlogrange=False, ylogrange=False, subplot=False, colorbar=False,
                      clear=True,legend=False, scalemin = None, scalemax = None, filename = None, **kwargs) : 
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
    
    if clear : plt.clf()
    
#    plt.imshow(density, extent = [x_range[0],x_range[1],y_range[0],y_range[1]], origin='down', aspect = np.diff(x_range)/np.diff(y_range))
    cs = plt.contourf(xs,ys,arr, levels, norm=cont_color,**kwargs)

    
    if kwargs.has_key('xlabel'):
        xlabel = kwargs['xlabel']
    else :
        if xlogrange: xlabel=r''+'$log_{10}('+xs.units.latex()+')$'
        else : xlabel = r''+'$x/' + xs.units.latex() +'$'
    
    if subplot:
        plt.set_xlabel(xlabel)
    else:
        plt.xlabel(xlabel)

    if kwargs.has_key('ylabel'):
        ylabel = kwargs['ylabel']
    else :
        if ylogrange: ylabel='$log_{10}('+ys.units.latex()+')$'
        else : ylabel = r''+'$y/' + ys.units.latex() +'$'
    
    if subplot:
        plt.set_ylabel(ylabel)
    else:
        plt.ylabel(ylabel)

#    if not subplot:
#        plt.xlim((x_range[0],x_range[1]))
#        plt.ylim((y_range[0],y_range[1]))

    if colorbar :
        cb = plt.colorbar(cs, format = "%.2e").set_label(r''+cb_label)
        
    if legend : plt.legend(loc=2)

    if (filename): 
        if config['verbose']: print "Saving "+filename
        plt.savefig(filename)



    

                  
