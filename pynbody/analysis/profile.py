import numpy as np
from .. import family, snapshot, units, array, util
import math

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

    ndim (default = 2): specifies whether it's a 2D or 3D profile - in the
                       2D case, the bins are generated in the xy plane

    type (default = 'lin'): specifies whether bins should be spaced linearly ('lin'),
                            logarithmically ('log') or contain equal numbers of
                            particles ('equaln')

    min (default = min(x)): minimum value to consider
    max (default = max(x)): maximum value to consider
    nbins (default = 100): number of bins


    Output:

    a Profile object. To find out which profiles are available, use keys().
   
    Implemented profile functions:

    den: density
    fourier: provides fourier coefficients, amplitude and phase for m=0 to m=6.
             To access the amplitude profile of m=2 mode, do
             >>> p.fourier['amp'][2,:]



    Additional functions should use the profile_property to
    yield the desired profile. For example, to generate the density
    profile, all that is required is

    >>> p = profile(sim)
    >>> p.den


    Examples:

    >>> s = pynbody.load('mysim')
    >>> import pynbody.profile as profile
    >>> p = profile.Profile(s) # 2D profile of the whole simulation - note
                               # that this only makes the bins etc. but
                               # doesn't generate the density
    >>> p.den # now we have a density profile
    >>> p.keys()
    ['mass', 'n', 'den']
    >>> p.families()
    [<Family dm>, <Family star>, <Family gas>]

    >>> ps = profile.Profile(s.s) # xy profile of the stars
    >>> ps = profile.Profile(s.s, type='log') # same, but with log bins
    >>> ps.families()
    [<Family star>]
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(ps.r, ps.den, 'o')
    >>> plt.semilogy()


    """

    _profile_registry = {}

    def _calculate_x(self, sim) :
        return ((sim['pos'][:,0:self._ndim]**2).sum(axis = 1))**(1,2)

    def __init__(self, sim, ndim = 2, type = 'lin', **kwargs):


        self._ndim = ndim
        self._type = type
        self.sim = sim
        self._x = self._calculate_x(sim)
        x = self._x

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
            self.bin_edges = np.logspace(np.log10(self.min), np.log10(self.max), num = self.nbins+1)
        elif type == 'lin':
            self.bin_edges = np.linspace(self.min, self.max, num = self.nbins+1)
        elif type == 'equaln':
            self.bin_edges = util.equipartition(x, self.nbins, self.min, self.max)
        else:
            raise RuntimeError, "Bin type must be one of: lin, log, equaln"


        self.bin_edges = array.SimArray(self.bin_edges, x.units)
        self.bin_edges.sim = self.sim

        self.n, bins = np.histogram(self._x, self.bin_edges)
        self._setup_bins()

        # set up the empty list of profiles
        self._profiles = {'n':self.n}


    def _setup_bins(self) :
        # middle of the bins for convenience

        self.rbins = 0.5*(self.bin_edges[:-1]+self.bin_edges[1:])
        self.rbins.sim = self.sim

        # Width of the bins
        self.dr = np.diff(self.rbins)

        self.binind = []
        if len(self._x) > 0:
            self.partbin = np.digitize(self._x, self.bin_edges)
        else :
            self.partbin = np.array([])

        assert self._ndim in [2,3]
        if self._ndim == 2:
            self._binsize = np.pi*(self.bin_edges[1:]**2 - self.bin_edges[:-1]**2)
        else:
            self._binsize  = 4./3.*np.pi*(self.bin_edges[1:]**3 - self.bin_edges[:-1]**3)

        for i in np.arange(self.nbins)+1:
            ind = np.where(self.partbin == i)
            self.binind.append(ind)



    def D(self) :
        """Return a new profile object which can return derivatives of
        the profiles in this object.

        For example, p.D()["phi"] gives the first derivative of the "phi" p["phi"]. For an example
        use see module analysis.decomp"""

        return DerivativeProfile(self)


    def __len__(self):
        """Returns the number of bins used in this profile object"""
        return self.nbins


    def _get_profile(self, name) :
        """Return the profile of a given kind"""
        if name in self._profiles :
            return self._profiles[name]
        elif name in Profile._profile_registry :
            self._profiles[name] = Profile._profile_registry[name](self)
            return self._profiles[name]
        elif name in self.sim.keys() or name in self.sim.all_keys() :
            self._profiles[name] = self._auto_profile(name)
            return self._profiles[name]
        else :
            raise KeyError, name+" is not a valid profile"

    def _auto_profile(self, name) :
        result = np.zeros(self.nbins).view(array.SimArray)
        for i in range(self.nbins):
            result[i] = (self.sim[name][self.binind[i]]*self.sim['mass'][self.binind[i]]).sum()
        result/= self['mass']
        result.units = self.sim[name].units
        result.sim = self.sim
        return result

    def __getitem__(self, name):
        """Return the profile of a given kind"""
        return self._get_profile(name)

    def __delitem__(self, name) :
        del self._profiles[name]

    def __repr__(self):
        return ("<Profile: " +
                str(self.families()) + " ; " +
                str(self._ndim) + "D ; " +
                self._type) + " ; " + str(self.keys())+ ">"

    def keys(self):
        """Returns a listing of available profile types"""
        return self._profiles.keys()


    def families(self):
        """Returns the family of particles used"""
        return self.sim.families()

    def create_particle_array(self, profile_name, particle_name = None, out_sim = None) :
        """Create a particle array with the results of the profile
        calculation.

        After calling this function, sim[particle_name][i] ==
        profile[profile_name][bin_in_which_particle_i_sits]

        If particle_name is not specified, it defaults to the same as
        profile_name.
        """
        import scipy, scipy.interpolate

        if particle_name is None :
            particle_name = profile_name

        if out_sim is None :
            out_sim = self.sim
            out_x = self._x
        else :
            out_x = self._calculate_x(out_sim)

        # nearest-neighbour version if interpolation unavailable
        #px = np.digitize(out_x, self.bins)-1
        #ok = np.where((px>=0) * (px<len(self)))
        #out_sim[particle_name, ok[0]] = self[profile_name][px[ok]]

        in_y = self[profile_name]

        if in_y.min()>0 :
            use_log = True
            in_y = np.log(in_y)
        else :
            use_log = False

        interp = scipy.interpolate.interp1d(np.log10(self.r), in_y, 'linear', copy=False,
                                            bounds_error = False)
        rep = interp(np.log10(out_x))

        if use_log :
            out_sim[particle_name] = np.exp(rep)
        else :
            out_sim[particle_name] = rep

        rep[np.where(out_x>math.log10(self.r.max()))] = self[profile_name][-1]
        rep[np.where(out_x<math.log10(self.r.min()))] = self[profile_name][0]

        out_sim[particle_name].units = self[profile_name].units

    @staticmethod
    def profile_property(fn) :
        Profile._profile_registry[fn.__name__]=fn
        return fn


class DerivativeProfile(Profile) :
    def __init__(self, base) :
        self.base = base
        self.sim = base.sim
        self._ndim = base._ndim
        self._x = base._x
        self._type = base._type
        self.max = base.bin_edges.max()
        self.min = base.bin_edges.min()
        self.nbins = base.nbins-1
        self.rbins = base.rbins
        self.bin_edges = base.bin_edges
        self.n, bins = np.histogram(self._x, self.bin_edges)
        self._setup_bins()
        self._profiles = {'n': self.n}

    def _get_profile(self, name) :
        if name[-1]=="'" :
            # Calculate derivative. This simple differencing algorithm
            # could be made more sophisticated.
            return np.diff(self.base[name[:-1]])/self.base.dr
        else :
            return Profile._get_profile(self, name)





@Profile.profile_property
def mass(self):
    """
    Calculate mass in each bin
    """

    mass = np.zeros(self.nbins)
    for i in range(self.nbins):
        mass[i] = (self.sim['mass'][self.binind[i]]).sum()
    mass = array.SimArray(mass, self.sim['mass'].units)
    mass.sim = self.sim
    return mass

@Profile.profile_property
def den(self):
    """
    Generate a radial density profile for the current type of profile
    """

    return self.mass/self._binsize

@Profile.profile_property
def fourier(self):
    """
    Generate a profile of fourier coefficients, amplitudes and phases
    """
    from . import fourier_decomp

    f = {'c': np.zeros((7, self.nbins),dtype=complex),
         'amp': np.zeros((7, self.nbins)),
         'phi': np.zeros((7, self.nbins))}

    for i in range(self.nbins):
        if self.n[i] > 100:
            f['c'][:,i] = fourier_decomp.fourier(self.sim['x'][self.binind[i]],
                                                 self.sim['y'][self.binind[i]],
                                                 self.sim['mass'][self.binind[i]])


    f['c'][:,self.mass>0] /= self.mass[self.mass>0]
    f['amp'] = np.sqrt(np.imag(f['c'])**2 + np.real(f['c'])**2)
    f['phi'] = np.arctan2(np.imag(f['c']), np.real(f['c']))

    return f

@Profile.profile_property
def mass_enc(self):
    """
    Generate the enclosed mass profile
    """
    m_enc = array.SimArray(np.zeros(self.nbins), 'Msol')
    m_enc.sim = self.sim
    for i in range(self.nbins):
        m_enc[i] = self.mass[:i].sum()
    return m_enc

@Profile.profile_property
def g_spherical(self) :
    """The naive gravitational acceleration assuming spherical
    symmetry = GM_enc/r^2"""

    return (units.G*self.mass_enc/self.r**2)

@Profile.profile_property
def rotation_curve_spherical(self):
    """
    The naive rotation curve assuming spherical symmetry: vc = sqrt(G M_enc/r)
    """

    return ((units.G*self.mass_enc/self.r)**(1,2)).in_units('km s**-1')

@Profile.profile_property
def j_circ(p) :
    return p["v_circ"] * p.r

@Profile.profile_property
def v_circ(p) :
    return np.sqrt(p["phi'"]*p.r)

@Profile.profile_property
def E_circ(p) :
    return p["phi"] + 0.5*(p["phi'"]*p.r)



"""

@Profile.profile_property
def vr(self):

    Generate a mean radial velocity profile, where the vr vector
    is taken to be in three dimensions - for in-plane radial velocity
    use the vr_xy array.

    vr = np.zeros(self.nbins)
    for i in range(self.nbins):
        vr[i] = (self.sim['vr'][self.binind[i]]*self.sim['mass'][self.binind[i]]).sum()
    vr /= self['mass']
    return vr

@Profile.profile_property
def v_c_xy(self) :

    Generate a circular velocity profile assuming the disk is aligned in the
    x-y plane (which can be achieved in a single line using the faceon
    function in module angmom)

    v = np.zeros(self.nbins).view(array.SimArray)
    for i in range(self.nbins) :
        bi = self.binind[i]
        v[i] = (self.sim['mass'][bi] * np.sqrt(self.sim['vx'][bi]**2+self.sim['vy'][bi]**2)).sum()
    v/=self['mass']
    v.units = self.sim['vx'].units
    return v

@Profile.profile_property
def phi(self) :
    v = np.zeros(self.nbins).view(array.SimArray)
    for i in range(self.nbins) :
        bi = self.binind[i]
        v[i] = (self.sim['mass'][bi]*self.sim['phi'][bi]).sum()
    v/=self['mass']
    v.units = self.sim['phi'].units
    return v

@Profile.profile_property
def vrxy(self):

    Generate a mean radial velocity profile, where the vr vector
    is taken to be in three dimensions - for in-plane radial velocity
    use the vr_xy array.

    vrxy = np.zeros(self.nbins)
    for i in range(self.nbins):
        vrxy[i] = (self.sim['vrxy'][self.binind[i]]*self.sim['mass'][self.binind[i]]).sum()
    vrxy /= self['mass']
    return vrxy

"""
