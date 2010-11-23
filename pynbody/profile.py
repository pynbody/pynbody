import numpy as np
from . import family, snapshot, fourier_decomp

#
# A module for making profiles of particle properties
#


class Profile:
    """

    A basic profile class for arbitrary profiles. Stores information about bins etc.

    Made to work with the pynbody SimSnap instances. The constructor only
    generates the bins and figures out which particles belong in which bin. 
    Profiles are generated lazy-loaded when a given property is requested.

    Input:

    sim : a simulation snapshot - this can be any subclass of SimSnap

    Optional Keywords:

    dim (default = 2): specifies whether it's a 2D or 3D profile - in the
                       2D case, the bins are generated in the xy plane

    log (default = False): log spaced or not

    min (default = min(x)): minimum value to consider
    max (default = max(x)): maximum value to consider
    nbins (default = 100): number of bins


    Output:

    a Profile object. To find out which profiles are available, use keys().
    The class defines  __get__ and __getitem__ methods so that
    these are equivalent:

    p.mass == p['mass']

    Implemented profile functions:

    rho

    Additional functions should use the profile_property to
    yield the desired profile. For example, to generate the density
    profile, all that is required is

    >>> p = profile(sim)
    >>> p.rho
    

    Examples:

    >>> s = pynbody.load('mysim')
    >>> import pynbody.profile as profile
    >>> p = profile.Profile(s) # 2D profile of the whole simulation - note
                               # that this only makes the bins etc. but
                               # doesn't generate the density
    >>> p.rho # now we have a density profile
    >>> p.keys()
    ['mass', 'ninbin', 'rho']
    >>> p.families()
    [<Family dm>, <Family star>, <Family gas>]
    
    >>> ps = profile.Profile(s.s) # xy profile of the stars
    >>> ps = profile.Profile(s.s, log=True) # same, but with log bins
    >>> ps.families()
    [<Family star>]
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(ps.midbins, ps.rho, 'o')
    >>> plt.semilogy()


    """

    
    def __init__(self, sim, dim = 2, type = 'lin', **kwargs):


        self._dim = dim
        self._type = type

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

        if type == 'log':
            self.bins = np.power(10,np.linspace(np.log10(self.min), np.log10(self.max), num = self.nbins+1))
        elif type == 'lin':
            self.bins = np.linspace(self.min, self.max, num = self.nbins+1)
        elif type == 'equaln':
            raise RuntimeError, "Equal-N bins not implemented yet"
        else:
            raise RuntimeError, "Bin type must be one of: lin, log, equaln"
            

        self.ninbin, bins = np.histogram(self._x, self.bins)

        # middle of the bins for convenience
        
        self.midbins = 0.5*(self.bins[:-1]+self.bins[1:])

        self.binind = []

        self.partbin = np.digitize(self._x, self.bins)
        
        
        assert dim in [2,3]
        if dim == 2:
            self._binarea = np.pi*(self.bins[1:]**2 - self.bins[:-1]**2)
        else:
            self._binarea  = 4./3.*np.pi*(self.bins[1:]**3 - self.bins[:-1]**3)
            
        for i in np.arange(self.nbins)+1:
            ind = np.where(self.partbin == i)
            self.binind.append(ind)
            
        # set up the empty list of profiles
        self._profiles = {'ninbin':self.ninbin}

        # set up a list of possible profiles
        #self._available_profiles = {'rho'}


    def __len__(self):
        """Returns the number of bins used in this profile object"""
        return self.nbins
    

    def __getitem__(self, name):
        """Return the profile of a given kind"""

        if name in self._profiles:
            return self._profiles[name]
        else:
            raise KeyError, name + " is not a valid profile"

        
    def __get__(self, name):
        """Return the profile of a given kind"""

        print 'in the get'

        if name in self._profiles:
            return self._profiles[name]
        else:
            return object.__get__(self,name)

    def __repr__(self):
        return ("<Profile: " +
                str(self.families()) + " ; " +
                str(self._dim) + "D ; " + 
                self._type) + " ; " + str(self.keys())+ ">"

    def keys(self):
        """Returns a listing of available profile types"""
        return self._profiles.keys()


    def families(self):
        """Returns the family of particles used"""
        return self._sim.families()


    class profile_property(object):
        """
        Lazy-load the required profiles.
        """

        def __init__(self, func):

            self._func = func
            self.__name__ = func.__name__
            self.__doc__ = func.__doc__

        def __get__(self, obj, obj_class):
            if obj is None:
                return obj

            # potentially bad style here? adding to the object's dictionary
            # so that it's calculated only once... but also want it in the
            # _profiles dictionary so that we can get a listing of available
            # profiles with p.keys()

            # Also - using the cached property like this seems to bypass the
            # __get__ method defined above - is there a way to chenge this?


            obj.__dict__[self.__name__] = obj._profiles[self.__name__] = self._func(obj)
            return obj.__dict__[self.__name__]

        def __delete__(self, obj):
            ###############################
            #THIS DOESN'T WORK!!!
            #
            # >>> p.mass
            # >>> del(p.mass)
            # >>> p.keys()
            # ['mass', 'ninbin']
            #
            # should've deleted the 'mass'
            ###############################
            
            print 'deleting'
            if self.__name__ in obj.__dict__:
                del obj.__dict__[self.__name__]
            if self.__name__ in self._profiles:
                del obj._profiles[self.__name__]


    @profile_property
    def mass(self):
        """
        Calculate mass in each bin
        """

        print '[calculating mass]'
        mass = np.zeros(self.nbins)
        for i in range(self.nbins):
            mass[i] = np.sum(self._sim['mass'][self.binind[i]])
        return mass
           
    @profile_property
    def rho(self):
        """
        Generate a radial density profile for the current type of profile
        """
        
        print '[calculating rho]'
        return self.mass/self._binarea

    @profile_property
    def fourier(self):
        """
        Generate a profile of fourier coefficients, amplitudes and phases
        """
        print '[calculating fourier decomposition]'

        f = {'c': np.zeros((7, self.nbins),dtype=complex),
             'amp': np.zeros((7, self.nbins)),
             'phi': np.zeros((7, self.nbins))}

        for i in range(self.nbins):
            if self.ninbin[i] > 100:
                f['c'][:,i] = fourier_decomp.fourier(self._sim['x'][self.binind[i]],
                                                     self._sim['y'][self.binind[i]],
                                                     self._sim['mass'][self.binind[i]])


        f['c'][:,self.mass>0] /= self.mass[self.mass>0]
        f['amp'] = np.sqrt(np.imag(f['c'])**2 + np.real(f['c'])**2)
        f['phi'] = np.arctan2(np.imag(f['c']), np.real(f['c']))

        return f
