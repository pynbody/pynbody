import numpy as np

def rho_T(sim, nbins=100, nlevels = 20, log=True, **kwargs):
    """
    Plot Temperature vs. Density for the gas particles in the snapshot.

    Optional keyword arguments:

       *t_range*: list, array, or tuple
         size(t_range) must be 2. Specifies the temperature range.

       *rho_range*: tuple
         size(rho_range) must be 2. Specifies the density range.

       *nbins*: int
         number of bins to use for the 2D histogram

       *nlevels*: int
         number of levels to use for the contours

       *log*: boolean
         whether to use log or linear spaced contours
    """
    import matplotlib.pyplot as plt
    from matplotlib import ticker, colors


    if kwargs.has_key('t_range'):
        t_range = kwargs['t_range']
        assert len(t_range) == 2
    else:
        t_range = (np.min(sim.gas['temp']),np.max(sim.gas['temp']))
    if kwargs.has_key('rho_range'):
        rho_range = kwargs['rho_range']
        assert len(rho_range) == 2
    else:
        rho_range = (np.min(sim.gas['rho']), np.max(sim.gas['rho']))

    hist, x, y = np.histogram2d(np.log10(sim.gas['temp']), np.log10(sim.gas['rho']),bins=nbins,range=[t_range,rho_range])


    if log:
        try:
            levels = np.logspace(np.log10(np.min(hist[hist>0])),       # there must be an
                                 np.log10(np.max(hist)),nlevels)      # easier way to do this...
            cont_color=colors.LogNorm()
        except ValueError:
            print 'crazy temperature or density range - please specify ranges'
            return
    else:
        levels = np.linspace(np.min(hist[hist>0]),
                             np.max(hist), nlevels)
        cont_color=None

    cs = plt.contourf(.5*(y[:-1]+y[1:]),.5*(x[:-1]+x[1:]), # note that hist is strange and x/y values
                                                          # are swapped
                     hist, levels, norm=cont_color)


    plt.xlabel(r'$log_{10}(\rho/'+sim.gas['rho'].units.latex()+')$')
    plt.ylabel(r'$log_{10}(T/'+sim.gas['temp'].units.latex()+')$')
