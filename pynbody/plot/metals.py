import numpy as np
import matplotlib.pyplot as plt
from ..analysis import profile
from .generic import hist2d

def mdf(sim,filename=None,clear=True,range=[-5,0.3],**kwargs):
    '''Metallicity Distribution Function
    Usage:
    import pynbody.plot as pp
    pp.mdf(s,linestyle='dashed',color='k')

    Plots a metallicity distribution function to the best of matplotlib's
    abilities.  Unfortunately, the "normed" keyword is buggy and does not
    return a PDF.  The "density" keyword should, but it not yet supported
    in many versions of numpy.
    '''
    nbins=100
    if clear : plt.clf()
    metpdf, bins = np.histogram(sim.star['feh'],weights=sim.star['mass'],
                                bins=nbins,normed=True,range=range,**kwargs)#density=True,
    midpoints = 0.5*(bins[:-1] + bins[1:])
    import pdb; pdb.set_trace()
    plt.plot(midpoints,metpdf)
    plt.xlabel('[Fe / H]')
    plt.ylabel('PDF')
    if (filename): 
        print "Saving "+filename
        plt.savefig(filename)


def ofefeh(sim,filename=None,**kwargs):
    '''
    Use hist2d module to make [O/Fe] vs. [Fe/H] plot
    Some common arguments
    x_range=[-2,0.5],y_range=[-0.2,1.0]
    '''
    hist2d(sim.star['feh'],sim.star['ofe'],filename=filename,
           xlabel="[Fe/H]",ylabel="[O/Fe]",**kwargs)

