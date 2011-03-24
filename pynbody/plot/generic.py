import numpy as np

def hist2d(xo, yo, weights=None, nbins=100, nlevels = 20, logscale=True, xlogrange=False,
           ylogrange=False,filename=None,colorbar=False,**kwargs):
    """
    Plot 2D histogram for arbitrary arrays that get passed in.

    Optional keyword arguments:

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

       *colorbar*: boolean
         draw a colorbar
    """
    import matplotlib.pyplot as plt
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
        
    hist, xs, ys = np.histogram2d(y, x, bins=nbins,range=[y_range,x_range])
    if weights != None :
        hist_weight, xs, ys = np.histogram2d(y, x, weights = weights,bins=nbins,range=[y_range,x_range])
        good = np.where(hist_weight > 0)
        hist[good] = hist_weight[good]/hist[good]

    if logscale:
        try:
            levels = np.logspace(np.log10(np.min(hist[hist>0])),       # there must be an
                                 np.log10(np.max(hist)),nlevels)      # easier way to do this...
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
        levels = np.linspace(np.min(hist[hist>0]),
                             np.max(hist), nlevels)
        cont_color=None

        if weights != None :
            cb_label = '$'+weights.units.latex()+'$'
        else :
            cb_label = '$N$'
    
    cs = plt.contourf(.5*(ys[:-1]+ys[1:]),.5*(xs[:-1]+xs[1:]), # note that hist is strange and x/y values
                                                          # are swapped
                     hist, levels, norm=cont_color)


    if kwargs.has_key('xlabel'):
        xlabel = kwargs['xlabel']
        plt.xlabel(xlabel)
    else :
        if xlogrange: label='$log_{10}('+xo.units.latex()+')$'
        plt.xlabel(r''+label)
    if kwargs.has_key('ylabel'):
        ylabel = kwargs['ylabel']
        plt.ylabel(ylabel)
    else :
        if ylogrange: label='$log_{10}('+yo.units.latex()+')$'
        plt.ylabel(r''+label)
    plt.xlim((x_range[0],x_range[1]))
    plt.ylim((y_range[0],y_range[1]))
    if (filename): plt.savefig(filename)

    if colorbar :
        cb = plt.colorbar(cs, format = "%.2f").set_label(r''+cb_label)
        

    
