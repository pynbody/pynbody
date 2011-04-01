import numpy as np
import matplotlib.pyplot as plt
from ..analysis import profile
from .generic import hist2d

def mdf(sim,filename=None,clear=True,**kwargs):
    '''Metallicity Distribution Function
    Usage:
    import pynbody.plot as pp
    pp.mdf(s,linestyle='dashed',color='k')
    '''
    nbins=100
    if clear : plt.clf()
    metpdf, bins, patches = plt.hist(sim.star['feh'],weights=sim.star['mass'],
                                     bins=nbins,histtype='step',normed=True,
                                     **kwargs)
    plt.xlabel('[Fe / H]')
    plt.ylabel('PDF')
    if (filename): plt.savefig(filename)


def ofefeh(sim,filename=None,**kwargs):
    '''
    Use hist2d module to make [O/Fe] vs. [Fe/H] plot
    Some common arguments
    x_range=[-2,0.5],y_range=[-0.2,1.0]
    '''
    hist2d(sim.star['feh'],sim.star['ofe'],filename=filename,
           xlabel="[Fe/H]",ylabel="[O/Fe]",**kwargs)
