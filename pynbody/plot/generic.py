"""

generic
=======

Flexible and general plotting functions

"""

import numpy as np
from ..analysis import profile, angmom, halo
from .. import config

def hist2d(xo, yo, weights=None, mass=None, nbins=100, nlevels = 20, logscale=True, 
           xlogrange=False, ylogrange=False,filename=None, subplot=False, 
           colorbar=False,clear=True,legend=False,scalemin=None,scalemax=None,**kwargs):
    """
    Plot 2D histogram for arbitrary arrays that get passed in.

    ** Input: **

       *x*: array

       *y*: array

    **Optional keywords:**

       *x_range*: list, array, or tuple
         size(x_range) must be 2. Specifies the X range.

       *y_range*: tuple
         size(y_range) must be 2. Specifies the Y range.

       *nbins*: int
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
    
    if not subplot:
        import matplotlib.pyplot as plt
    else :
        plt = subplot
    from matplotlib import ticker, colors


    if kwargs.has_key('y_range'):
        y_range = kwargs['y_range']
        assert len(y_range) == 2
    elif kwargs.has_key('yrange'):
        y_range = kwargs['yrange']
        assert len(y_range) == 2
    else:
        if ylogrange:
            y_range = (np.log10(np.min(yo)),np.log10(np.max(yo)))
        else:
            y_range = (np.min(yo),np.max(yo))

    if kwargs.has_key('x_range'):
        x_range = kwargs['x_range']
        assert len(x_range) == 2
    elif kwargs.has_key('xrange'):
        x_range = kwargs['xrange']
        assert len(x_range) == 2
    else:
        if xlogrange:
            x_range = (np.log10(np.min(xo)), np.log10(np.max(xo)))
        else:
            x_range = (np.min(xo),np.max(xo))

    if (xlogrange):
        x = np.log10(xo)
    else :
        x = xo
    if (ylogrange):
        y = np.log10(yo)
    else :
        y = yo

    if mass is not None:
        hist, xs, ys = np.histogram2d(y, x, weights=weights*mass, bins=nbins,range=[y_range,x_range])
        hist_weight, xs, ys = np.histogram2d(y, x, weights=mass,bins=nbins,range=[y_range,x_range])
        good = np.where(hist_weight > 0)
        hist[good] = hist[good]/hist_weight[good]
    else:
        hist, xs, ys = np.histogram2d(y, x, weights=weights, bins=nbins,range=[y_range,x_range])
        
    if logscale:
        try:
            if scalemin is None: scalemin = np.min(hist[hist>0])
            if scalemax is None: scalemax = np.max(hist[hist>0])

            levels = np.logspace(np.log10(scalemin),       # there must be an
                                 np.log10(scalemax),nlevels)      # easier way to do this...
            cont_color=colors.LogNorm()
        except ValueError:
            print 'crazy x or y range - please specify ranges'
            print 'y_range = ' + str(y_range)
            print 'x_range = ' + str(x_range)
            return

        if weights != None :
            cb_label = '$log_{10}('+weights.units.latex()+')$'
        else :
            cb_label = '$log_{10}(N)$'
    else:
        levels = np.linspace(np.min(hist),
                             np.max(hist), nlevels)
        cont_color=None
        
        if weights != None :
            cb_label = '$'+weights.units.latex()+'$'
        else :
            cb_label = '$N$'
    
    if clear : plt.clf()
    #
    # note that hist is strange and x/y values
    # are swapped
    #
    cs = plt.contourf(.5*(ys[:-1]+ys[1:]),.5*(xs[:-1]+xs[1:]), 
                     hist, levels, norm=cont_color,**kwargs)


    if kwargs.has_key('xlabel'):
        xlabel = kwargs['xlabel']
    else :
        if xlogrange: xlabel=r''+'$log_{10}('+xo.units.latex()+')$'
        else : xlabel = r''+'$x/' + xo.units.latex() +'$'
    
    if subplot:
        plt.set_xlabel(xlabel)
    else:
        plt.xlabel(xlabel)

    if kwargs.has_key('ylabel'):
        ylabel = kwargs['ylabel']
    else :
        if ylogrange: ylabel='$log_{10}('+yo.units.latex()+')$'
        else : ylabel = r''+'$y/' + yo.units.latex() +'$'
    
    if subplot:
        plt.set_ylabel(ylabel)
    else:
        plt.ylabel(ylabel)

    if not subplot:
        plt.xlim((x_range[0],x_range[1]))
        plt.ylim((y_range[0],y_range[1]))

    if colorbar :
        cb = plt.colorbar(cs, format = "%.2f").set_label(r''+cb_label)
        
    if legend : plt.legend(loc=2)

    if (filename): 
        if config['verbose']: print "Saving "+filename
        plt.savefig(filename)

    return hist, xs, ys
    


def gauss_density(xo, yo, weights=None, x_range = None, y_range = None, gridsize = (100,100), 
                  nlevels = 20, logscale = True, xlogrange = False, ylogrange = False, 
                  subplot = False, colorbar = False, clear = True, legend = False, 
                  scalemin = None, scalemax = None, filename = None, **kwargs) :

    """
    Plot 2D average gaussian density estimate given values *z* at points (*x*, *y*). 

    
    ** Input: **

       *x*: array

       *y*: array

    **Optional keywords:**
    
       *weights*: string 
         the quantity to use for the density
         calculation. Defaults to *None* so that just the 
         number density is calculated.
          
       *x_range*: list, array, or tuple
         size(x_range) must be 2. Specifies the X range.

       *y_range*: tuple
         size(y_range) must be 2. Specifies the Y range.

       *nbins*: int
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
    from fast_kde import fast_kde
    global config
    
    if not subplot:
        import matplotlib.pyplot as plt
    else :
        plt = subplot
    from matplotlib import ticker, colors
    
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

    xs = np.linspace(x_range[0], x_range[1], gridsize[0]+1)
    xs = .5*(xs[:-1]+xs[1:])
    ys = np.linspace(y_range[0], y_range[1], gridsize[1]+1)
    ys = .5*(ys[:-1]+ys[1:])
    
    if weights is not None: weights = weights[ind[0]]

    density = fast_kde(x,y,weights=weights,extents=(x_range[0],x_range[1],y_range[0],y_range[1]),gridsize=gridsize)
        
    if logscale:
        try:
            if scalemin is None: scalemin = np.min(density[density>0])
            if scalemax is None: scalemax = np.max(density[density>0])

            levels = np.logspace(np.log10(scalemin),       # there must be an
                                 np.log10(scalemax),nlevels)      # easier way to do this...
            cont_color=colors.LogNorm()
        except ValueError:
            print 'crazy x or y range - please specify ranges'
            print 'y_range = ' + str(y_range)
            print 'x_range = ' + str(x_range)
            return

        if weights != None :
            cb_label = '$log_{10}('+weights.units.latex()+')$'
        else :
            cb_label = '$log_{10}(N)$'
    else:
        levels = np.linspace(np.min(density),
                             np.max(density), nlevels)
        cont_color=None
        
        if weights != None :
            cb_label = '$'+weights.units.latex()+'$'
        else :
            cb_label = '$N$'
    
    if clear : plt.clf()
    #
    # note that hist is strange and x/y values
    # are swapped
    #

#    plt.imshow(density, extent = [x_range[0],x_range[1],y_range[0],y_range[1]], origin='down', aspect = np.diff(x_range)/np.diff(y_range))
    cs = plt.contourf(xs,ys,density, levels, norm=cont_color,**kwargs)


    if kwargs.has_key('xlabel'):
        xlabel = kwargs['xlabel']
    else :
        if xlogrange: xlabel=r''+'$log_{10}('+xo.units.latex()+')$'
        else : xlabel = r''+'$x/' + xo.units.latex() +'$'
    
    if subplot:
        plt.set_xlabel(xlabel)
    else:
        plt.xlabel(xlabel)

    if kwargs.has_key('ylabel'):
        ylabel = kwargs['ylabel']
    else :
        if ylogrange: ylabel='$log_{10}('+yo.units.latex()+')$'
        else : ylabel = r''+'$y/' + yo.units.latex() +'$'
    
    if subplot:
        plt.set_ylabel(ylabel)
    else:
        plt.ylabel(ylabel)

#    if not subplot:
#        plt.xlim((x_range[0],x_range[1]))
#        plt.ylim((y_range[0],y_range[1]))

    if colorbar :
        cb = plt.colorbar(cs, format = "%.2f").set_label(r''+cb_label)
        
    if legend : plt.legend(loc=2)

    if (filename): 
        if config['verbose']: print "Saving "+filename
        plt.savefig(filename)


    return density
    

                  
