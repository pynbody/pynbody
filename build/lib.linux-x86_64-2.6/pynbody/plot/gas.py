"""

gas
===

Functions for plotting gas quantities

"""

import numpy as np
import matplotlib.pyplot as plt
from ..analysis import profile, angmom, halo
from .generic import hist2d
from ..units import Unit

def rho_T(sim, rho_units=None, rho_range = None, t_range = None, **kwargs):
    """
    Plot Temperature vs. Density for the gas particles in the snapshot.

    **Optional keywords:**

       *rho_units*: specify the density units (default is the same units as the current 'rho' array)

       *t_range*: list, array, or tuple 
          ``size(t_range)`` must be 2. Specifies the temperature range.

       *rho_range:* tuple 
          ``size(rho_range)`` must be 2. Specifies the density range.

    See :func:`~pynbody.plot.generic.hist2d` for other plotting keyword options


    """
    from matplotlib import ticker, colors

    if rho_units is None: 
        rho_units = sim.gas['rho'].units

    if t_range is not None:
        kwargs['y_range'] = t_range
        assert len(kwargs['y_range']) == 2

    if rho_range is not None:
        kwargs['x_range'] = rho_range
        assert len(kwargs['x_range']) == 2
    else:
        rho_range=False

    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
        del kwargs['xlabel']
    else:
        xlabel=r'log$_{10}$($\rho$/$'+Unit(rho_units).latex()+'$)'

    if 'ylabel' in kwargs:
        ylabel = kwargs['ylabel']
        del kwargs['ylabel']
    else:
        ylabel=r'log$_{10}$(T/$'+sim.gas['temp'].units.latex()+'$)'

    hist2d(sim.gas['rho'].in_units(rho_units),sim.gas['temp'],xlogrange=True,
           ylogrange=True, xlabel=xlabel, ylabel=ylabel, **kwargs)


def temp_profile(sim, center=True, r_units='kpc', bin_spacing = 'equaln', 
                 clear = True, filename=None,**kwargs) :
    """

    Centre on potential minimum, align so that the disk is in the
    x-y plane, then plot the temperature profile as a 
    function of radius.

    """

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
    
