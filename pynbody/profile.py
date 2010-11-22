import numpy as np
from . import family, snapshot


#
# A module for making profiles of particle properties
#


class Profile:
    """

    A basic profile class for arbitrary profiles. Stores information about bins etc.

    Made to work with the pynbody SimSnap instances. The constructor only
    generates the bins and figures out which particles belong in which bin. All
    other profiles are generated with other functions.

    Input:

    sim : a simulation snapshot - this can be any subclass of SimSnap

    Optional Keywords:

    dim (default = 2): specifies whether it's a 2D or 3D profile - in the
                       2D case, the bins are generated in the xy plane

    log (default = False): log spaced or not

    min (default = min(x)): minimum value to consider
    max (default = max(x)): maximum value to consider
    nbins (default = 100): number of bins


    Implemented profile functions:

    density_profile

    Additional functions should call a _gen_x_profile function in the Profile
    class and define a "x_profile" function that can be called to return
    a Profile object. 
    

    Examples:

    >>> s = pynbody.load('mysim')
    >>> import pynbody.profile as profile
    >>> p = profile.Profile(s) # 2D profile of the whole simulation - note
                               # that this only makes the bins etc. but
                               # doesn't generate the density
    >>> p.density_profile(s, prof=p) # now we have a density profile
    >>> p.keys()
    ['mass', 'ninbin', 'rho']
    >>> p.families()
    [<Family dm>, <Family star>, <Family gas>]
    
    >>> ps = profile.density_profile(s.s) # xy profile of the stars
    >>> ps = profile.density_profile(s.s, log=True # same, but with log bins
    >>> ps.families()
    [<Family star>]
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(ps.midbins, ps.rho, 'o')
    >>> plt.show()
    >>> plt.semilogy()


    """

    def __init__(self, sim, dim = 2, log = False, **kwargs):

        assert isinstance(sim, snapshot.SimSnap)
        self._sim = sim

        x = np.sqrt(np.sum(sim['pos'][:,0:dim]**2, axis = 1))
        self._x = x

        # The profile object is initialized given some array of values
        # and optional keyword parameters

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
            self.min = np.min(x[x>0])

        if log:
            self.bins = np.power(10,np.linspace(np.log10(self.min), np.log10(self.max), num = self.nbins+1))
        else:
            self.bins = np.linspace(self.min, self.max, num = self.nbins+1)

        self.ninbin, bins = np.histogram(self._x, self.bins)

        # middle of the bins for convenience
        
        self.midbins = 0.5*(self.bins[:-1]+self.bins[1:])

        self.binind = []

        self.partbin = np.digitize(self._x, self.bins)
        
        

        if dim == 2:
            self._binarea = np.pi*(self.bins[1:]**2 - self.bins[:-1]**2)
        else:
            self._binarea  = 4./3.*np.pi*(self.bins[1:]**3 - self.bins[:-1]**3)
            
        for i in np.arange(self.nbins)+1:
            ind = np.where(self.partbin == i)
            self.binind.append(ind)
            
        # set up the empty list of profiles
        self._profiles = {'ninbin':self.ninbin}


    def __len__(self):
        """Returns the number of bins used in this profile object"""
        return self.nbins
    

    def __getitem__(self, name):
        """Return the profile of a given kind"""

        if name in self._profiles:
            return self._profiles[name]
        else:
            return object.__getattribute__(self,name)

        
    def __getattribute__(self, name):
        """Return the profile of a given kind"""

        if name in self._profiles:
            return self._profiles[name]
        else:
            return object.__getattribute__(self,name)


    def keys(self):
        """Returns a listing of available profile types"""
        return self._profiles.keys()


    def families(self):
        """Returns the family of particles used"""
        return self._sim.families()


    def _gen_density_profile(self):
        
        self.mass = np.zeros(self.nbins)

        for i in range(self.nbins):
            self.mass[i] = np.sum(self._sim['mass'][self.binind[i]])

        self.rho = self.mass/self._binarea
        
        self._profiles['mass'] = self.mass
        self._profiles['rho'] = self.rho

    
def density_profile(sim, prof=None, dim = 2, **kwargs):
    """Returns a Profile instance with a mass density profile.
    If the profile object already exists, append the mass profile
    to the existing profiles, otherwise create a new Profile instance.

    """

    if prof == None:
        prof = Profile(sim, dim, **kwargs)
        prof._gen_density_profile()
        return prof

    else:
        prof._gen_density_profile()

        
        
    

