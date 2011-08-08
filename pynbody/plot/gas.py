import numpy as np
import matplotlib.pyplot as plt
from ..analysis import profile, angmom, halo
from .generic import hist2d
from ..units import Unit

def rho_T(sim, rho_units="m_p cm**-3", **kwargs):
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

       *logscale*: boolean
         whether to use log or linear spaced contours
    """
    from matplotlib import ticker, colors

    if kwargs.has_key('t_range'):
        kwargs['y_range'] = kwargs['t_range']
        assert len(t_range) == 2

    if kwargs.has_key('rho_range'):
        kwargs['x_range'] = kwargs['rho_range']
        assert len(rho_range) == 2
    else:
        rho_range=False

    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
        del kwargs['xlabel']
    else:
        xlabel=r'log$_{10}$($n$/$'+Unit(rho_units).latex()+'$)'

    if 'ylabel' in kwargs:
        ylabel = kwargs['ylabel']
        del kwargs['ylabel']
    else:
        ylabel=r'log$_{10}$(T/$'+sim.gas['temp'].units.latex()+'$)'

    hist2d(sim.gas['rho'].in_units(rho_units),sim.gas['temp'],xlogrange=True,
           ylogrange=True, xlabel=xlabel, ylabel=ylabel, **kwargs)


def temp_profile(sim, center=True, r_units='kpc', bin_spacing = 'equaln', 
                 clear = True, filename=None,**kwargs) :
    """Centre on potential minimum, align so that the disk is in the
    x-y plane, then plot the amplitude of the 2nd fourier mode as a 
    function of radius."""

    if center :
        angmom.sideon(sim)

    if 'min' in kwargs :
        min_r = kwargs['min']
    else:
        min_r = sim['r'].min()
    if 'max' in kwargs :
        max_r = kwargs['max']
    else:
        max_r = sim['r'].max()

    pro = profile.Profile(sim.gas, type=bin_spacing, min = min_r, max = max_r)

    r = pro['rbins'].in_units(r_units)
    tempprof = pro['temp']

    if clear : plt.clf()

    plt.semilogy(r, tempprof)

    plt.xlabel("r / $"+r.units.latex()+"$")
    plt.ylabel("Temperature [K]")
    if (filename): 
        print "Saving "+filename
        plt.savefig(filename)
    
