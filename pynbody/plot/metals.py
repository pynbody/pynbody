"""

metals
======

"""


import numpy as np
from ..analysis import profile
from .generic import hist2d, gauss_kde

def mdf(sim,filename=None,clear=True,range=[-5,0.3],axes=False,**kwargs):
    '''

    Metallicity Distribution Function
    
    Plots a metallicity distribution function to the best of
    matplotlib's abilities.  Unfortunately, the "normed" keyword is
    buggy and does not return a PDF.  The "density" keyword should,
    but it not yet supported in many versions of numpy.

    **Usage:**

    >>> import pynbody.plot as pp
    >>> pp.mdf(s,linestyle='dashed',color='k')

    
    '''
    nbins=100
    if axes:
        plt=axes
    else:
        import matplotlib.pyplot as plt
        if clear : plt.clf()
        plt.xlabel('[Fe / H]')
        plt.ylabel('PDF')

    metpdf, bins = np.histogram(sim['feh'],weights=sim['mass'],
                                bins=nbins,normed=True,range=range,**kwargs)#density=True,
    midpoints = 0.5*(bins[:-1] + bins[1:])

    plt.plot(midpoints,metpdf)
    if (filename): 
        print "Saving "+filename
        plt.savefig(filename)


def ofefeh(sim,fxn=gauss_kde,filename=None,**kwargs):
    '''

    Use :func:`~pynbody.plot.generic.hist2d` to make a [O/Fe] vs. [Fe/H] plot

    **Input:** 

    *sim*: snapshot to pull data from 

    **Optional Keywords:**

    *fxn*: a function with the same signature as functions
       :func:`~pynbody.plot.generic.hist2d` and
       :func:`~pynbody.plot.generic.gauss_kde` in
       :mod:`pynbody.plot.generic` default:
       :func:`pynbody.plot.generic.hist2d`


    see :func:`~pynbody.plot.generic.make_contour_plot` for other
    plot-related keywords.

    '''
    
    if 'subplot' in kwargs:
        fxn(sim['feh'],sim['ofe'],filename=filename,**kwargs)
    else:
        fxn(sim['feh'],sim['ofe'],filename=filename,
            xlabel="[Fe/H]",ylabel="[O/Fe]",**kwargs)

