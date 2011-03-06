import numpy as np

def hist2d(x, y, nbins=100, nlevels = 20, logscale=True, xlogrange=False,
           ylogrange=False,**kwargs):
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
    """
    import matplotlib.pyplot as plt
    from matplotlib import ticker, colors


    if kwargs.has_key('y_range'):
        y_range = kwargs['y_range']
        assert len(y_range) == 2
    else:
        if ylogrange:
            y_range = (np.log10(np.min(y)),np.log10(np.max(y)))
        else:
            y_range = (np.min(y),np.max(y))

    if kwargs.has_key('x_range'):
        x_range = kwargs['x_range']
        assert len(x_range) == 2
    else:
        if xlogrange:
            x_range = (np.log10(np.min(x)), np.log10(np.max(x)))
        else:
            x_range = (np.min(x),np.max(x))

    if (xlogrange and ylogrange) :
        hist, xs, ys = np.histogram2d(np.log10(x), np.log10(y),bins=nbins,range=[y_range,x_range])
    elif (xlogrange):
        hist, xs, ys = np.histogram2d(np.log10(x), y,bins=nbins,range=[y_range,x_range])
    elif (ylogrange):
        hist, xs, ys = np.histogram2d(x, np.log10(y),bins=nbins,range=[y_range,x_range])
    else :
        hist, xs, ys = np.histogram2d(x, y,bins=nbins,range=[y_range,x_range])


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
    else:
        levels = np.linspace(np.min(hist[hist>0]),
                             np.max(hist), nlevels)
        cont_color=None

    cs = plt.contourf(.5*(ys[:-1]+ys[1:]),.5*(xs[:-1]+xs[1:]), # note that hist is strange and x/y values
                                                          # are swapped
                     hist, levels, norm=cont_color)


    if kwargs.has_key('xlabel'):
        plt.xlabel(xlabel)
    if kwargs.has_key('ylabel'):
        plt.ylabel(ylabel)
    plt.xlim((x_range[0],x_range[1]))
    plt.ylim((y_range[0],y_range[1]))
