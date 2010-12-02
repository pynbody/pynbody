import numpy as np
import matplotlib.pyplot as plt

def sfh(t,filename=None,**kwargs):
    sfhist, bins, patches = plt.hist(t.star['tform'].in_units("Gyr"),
                                     weights=t.star['mass'].in_units('Msol')*1e-9,
                                     bins=500,histtype='step',color='k')
    plt.xlabel('Time [Gyr]')
    plt.ylabel('SFR [M$_\odot$ yr$^{-1}$]')
    if (filename): plt.savefig(filename,**kwargs)
    else: plt.show()
    return plt
