"""

phasediagram
============

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
import pynbody
import pynbody.units as units

def rho_T(sim, nbins=100, nlevels = 20, log=True, clear=True, **kwargs):
    """
    Plot Temperature vs. Density for the gas particles in the snapshot.

    **Optional keywords:**

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
    

    if clear: plt.clf()

    if kwargs.has_key('t_range'):
        t_range = kwargs['t_range']
        assert len(t_range) == 2
    else:
        t_range = (np.log10(np.min(sim.gas['temp'])),np.log10(np.max(sim.gas['temp'])))
    if kwargs.has_key('rho_range'):
        rho_range = kwargs['rho_range']
        assert len(rho_range) == 2
    else:
        rho_range = (np.log10(np.min(sim.gas['rho'])), np.log10(np.max(sim.gas['rho'])))

    hist, x, y = np.histogram2d(np.log10(sim.gas['temp']), np.log10(sim.gas['rho']),bins=nbins,range=[t_range,rho_range])


    if log:
        try:
            levels = np.logspace(np.log10(np.min(hist[hist>0])),       # there must be an
                                 np.log10(np.max(hist)),nlevels)      # easier way to do this...
            cont_color=colors.LogNorm()
        except ValueError:
            print 'crazy temperature or density range - please specify ranges'
            print 't_range = ' + str(t_range)
            print 'rho_range = ' + str(rho_range)
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
    plt.xlim((rho_range[0],rho_range[1]))
    plt.ylim((t_range[0],t_range[1]))


def overplot_mjeans(xrange,yrange) :
    def rho_mj(mj,temp) :
        return (1/mj**2*np.pi**5/36*(5./2*units.k*temp/
                                     (units.G*0.5*units.m_p))**3).in_units('m_p cm^-3')

    temp = pynbody.array.SimArray(np.logspace(1,6),units='K')

    for mj in np.logspace(0,6,7) :
        label = (r'$M_{j} = ' + '{0:.0e}'.format(mj)+' M_{\odot}$') 
        rho_mj_arr = np.log10(rho_mj(units.Unit(str(mj) + ' Msol'),temp))
        plt.plot(rho_mj_arr,np.log10(temp),label=label)

        angle = 2*np.arctan((np.log10(temp[1])-np.log10(temp[0]))/(rho_mj_arr[1]-rho_mj_arr[0]))*180/np.pi

        plt.text(rho_mj_arr[5],np.log10(temp[5]),label,rotation=angle, clip_on=True)

    plt.xlim(xrange)
    plt.ylim(yrange)

    
