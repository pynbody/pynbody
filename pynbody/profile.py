import numpy as np
from . import family, snapshot


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


    This class might seem redundant, given that the __init__ function is
    basically just a wrapper for the numpy histogram function... but, it
    adds the very handy list of indices belonging to each bin and will be
    expanded to include a variety of other functions. It also eliminates the need
    to constantly create separate variables that histogram returns, by
    packaging them into a single class together with access functions.

    """

    def __init__(self, x, **kwargs):

        # The profile object is initialized given some array of values
        # and optional keyword parameters

        self.x = x

        if kwargs.has_key('max'):
            self.max = kwargs['max']
        else:
            self.max = np.max(x)
        if kwargs.has_key('nbins'):
            self.nbins = kwargs['nbins']
        else:
            self.nbins = 100
            
        if kwargs.has_key('min'):
            self.min = kwargs['min']
        else:
            self.min = float(self.max)/self.nbins


        self.ninbin, self.bins = np.histogram(x, range = [self.min,self.max],
                                              bins = self.nbins)

        # middle of the bins for convenience
        
        self.midbins = 0.5*(self.bins[1:]+self.bins[:-1])

        self.binind = []

        self.partbin = np.digitize(x, self.bins)
        
        for i,bin in enumerate(self.bins[:-1]):
            ind = np.where(self.partbin == i)
            
            self.binind.append(ind)
            self.ninbin[i] = ind[0].size

        # set up the empty profile dictionary
        self._profiles = {}


    def __len__(self):
        """Returns the number of bins used in this profile object"""
        return self.nbins

    def keys(self):
        """Returns a listing of available profile types"""
        return self._profiles.keys()

  ##   def massprof(self, pm):
        
##         self.mass = np.zeros(self.nbins)

##         for i, bin in enumerate(self.bins[:-1]):
            
##             self.mass[i] = np.sum(pm[self.binind[i]])


def radial_xy_profile(sim, fam, **kwargs):
    """Returns a Profile instance"""

    assert isinstance(sim, snapshot.SimSnap)
    assert fam in family.family_names()

    for i in family.family_names():
        if i == fam:
            return Profile(np.sqrt(sim[family.get_family(fam)]['x']**2 + sim[family.get_family(fam)]['y']**2))
