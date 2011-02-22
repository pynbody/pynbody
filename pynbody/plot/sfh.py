import numpy as np
import matplotlib.pyplot as plt

def sfh(t,filename=None,**kwargs):
    nbins=100
    sfhist, bins, patches = plt.hist(t.star['tform'].in_units("Gyr"),
                                     weights=t.star['mass'].in_units('Msol')*1e-9*nbins / (t.star['tform'].in_units("Gyr").max() - t.star['tform'].in_units("Gyr").min()),
                                     bins=nbins,histtype='step',color='k')
    plt.xlabel('Time [Gyr]')
    plt.ylabel('SFR [M$_\odot$ yr$^{-1}$]')
    if (filename): plt.savefig(filename,**kwargs)
