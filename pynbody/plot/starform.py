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


def schmidtlaw(t,filename=None,pretime=50,**kwargs):
    starmass, bins = np.histogram(t.star['r'].[np.where(t.star['tform'].in_units("Myr") > t.properties['time'].in_units("Myr") - pretime)])
