import numpy as np


#
# A module for making profiles of particle properties
#


class Profile:
    """

    A basic profile class for arbitrary profiles. Stores information about bins etc.

    Stores several attributes that are useful for the end-user:

    ninbin - number in each bin
    midbins - midpoint of each bin
    partbin - for each data value (i.e. particle), which bin does it belong to
    binind - particle indices belonging to each bin.

    Usage:

    >>> prof = Profile(sqrt(x**2 + y**2), nbins = 100, max = 20)
    >>> prof.massprof(m)
    >>> plot(prof.midbins, prof.mass)

    """

    def __init__(self, x, **keylist):

        self.x = x

        if keylist.has_key('max'):
            self.max = keylist['max']
        else:
            self.max = np.max(x)
        if keylist.has_key('nbins'):
            self.nbins = keylist['nbins']
        else:
            self.nbins = 100
            
        if keylist.has_key('min'):
            self.min = keylist['min']
        else:
            self.min = float(self.max)/self.nbins


        self.ninbin, self.bins = np.histogram(x, range = [self.min,self.max],
                                              bins = self.nbins)

        self.midbins = 0.5*(self.bins[1:]+self.bins[:-1])

        self.binind = []

        self.partbin = np.digitize(x, self.bins)
        
        for i,bin in enumerate(self.bins[:-1]):
            ind = np.where(self.partbin == i)
            
            self.binind.append(ind)
            self.ninbin[i] = ind[0].size


    def massprof(self, pm):
        
        self.mass = np.zeros(self.nbins)

        for i, bin in enumerate(self.bins[:-1]):
            
            self.mass[i] = np.sum(pm[self.binind[i]])


    
