"""

profile
=======

A set of classes and functions for making profiles of simulation
properties.

"""

import numpy as np
import pynbody
from .. import family, snapshot, units, array, util
import math

#
# A module for making profiles of particle properties
#


class Profile:
    """

    A basic profile class for arbitrary profiles. Stores information
    about bins etc.

    Made to work with the pynbody SimSnap instances. The constructor
    only generates the bins and figures out which particles belong in
    which bin.  Profiles are generated lazy-loaded when a given
    property is requested.

    **Input**:

    *sim* : a simulation snapshot - this can be any subclass of SimSnap

    **Optional Keywords**:

    *ndim* (default = 2): specifies whether it's a 2D or 3D profile - in the
                       2D case, the bins are generated in the xy plane

    *type* (default = 'lin'): specifies whether bins should be spaced linearly ('lin'),
                            logarithmically ('log') or contain equal numbers of
                            particles ('equaln')

    *min* (default = min(x)): minimum value to consider

    *max* (default = max(x)): maximum value to consider

    *nbins* (default = 100): number of bins

    *bins* : array like - predefined bin edges in units of binning quantity. If this
             keyword is set, the values of the keywords *type*, *nbins*, *min* and *max*
             will be ignored

    *calc_x* (default = None): function to use to calculate the value
     for binning. If None it defaults to the radial distance from
     origin (in either 2 or 3 dimensions), ut you can specify this
     function to return any value you want for making profiles along
     arbitrary axes. Depening on your function, the units of certain
     profiles (such as density) might not make sense.

    **Output**:

    a Profile object. To find out which profiles are available, use keys().
   
    **Implemented profile functions**:

    *density*    : density

    *mass*       : mass in each bin

    *mass_enc*   : enclosed mass 

    *fourier* : provides fourier coefficients, amplitude and phase for
     m=0 to m=6.  To access the amplitude profile of m=2 mode, do
     ``p['fourier']['amp'][2,:]``

    *dyntime*    : dynamical time

    *g_spherical*: GM_enc/r^2

    *rotation_curve_spherical*: rotation curve from vc = sqrt(GM/R) -
     can be very wrong!

    *j_circ*     : angular momentum of particles on circular orbits

    *v_circ* : circular velocity, aka rotation curve - calculated from
     the midplane gravity, so this can be expensive

    *E_circ*     : energy of particles on circular orbits in the midplane

    *omega*      : circular orbital frequency

    *kappa*      : radial orbital frequency

    *beta*       : 3-D velocity anisotropy parameter

    *magnitudes* : magnitudes in each bin - default band = 'v' 

    *sb*         : surface brightness - default band = 'v'
    

    Additional functions should use the profile_property to yield the
    desired profile.

    **Lazy-loading arrays:** 
    
    The Profile class will automatically compute a mass-weighted
    profile for any lazy-loadable array of its parent SimSnap object.

    **Dispersions:**
    
    To obtain a dispersion profile, attach a ``_disp`` after the desired
    quantity name.

    **RMS:**
    
    The root-mean-square of a quantity can be obtained by using a ``_rms`` suffix

    **Derivatives:**
    
    To compute a derivative of a profile, prepend a ``d_`` to the
    profile string, as in ``p['d_temp']`` to get a temperature gradient.

    **Saving and loading previously generated profiles:**
    
    Use the :func:`~pynbody.analysis.profile.Profile.write` function
    to write the current profiles with all the necessary information
    to a file. Initialize a profile with the load_from_file=True
    keyword to automatically load a previously saved profile. The
    filename is chosen automatically and corresponds to a hash
    generated from the positions of the particles used in the
    profile. This is to ensure that you are always looking at the same
    set of particles, centered in the same way. It also means you
    *must* use the same centering method if you want to reuse a saved
    profile.


    **Examples:**

    Density profile of the entire simulation: 

    >>> s = pynbody.load('mysim')
    >>> import pynbody.profile as profile
    >>> p = profile.Profile(s) # 2D profile of the whole simulation - note
                               # that this only makes the bins etc. but
                               # doesn't generate the density
    >>> p['density'] # now we have a density profile
    >>> p.keys()
    ['mass', 'n', 'density']
    >>> p.families()
    [<Family dm>, <Family star>, <Family gas>]

    
    Density profile of the stars:

    >>> ps = profile.Profile(s.s) # xy profile of the stars
    >>> ps = profile.Profile(s.s, type='log') # same, but with log bins
    >>> ps.families()
    [<Family star>]
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(ps['rbins'], ps['density'], 'o')
    >>> plt.semilogy()

    Metallicity profile of the gas in spherical shells (requires appropriate auxilliary files):

    >>> pg = profile.Profile(s.g, ndim=3)
    >>> pg['feh']
    SimSnap: deriving array feh
    TipsySnap: attempting to load auxiliary array 10/12M_hr.01000.FeMassFrac
    SimSnap: deriving array hydrogen
    SimSnap: deriving array hetot
    SimArray([  1.83251940e-01,   1.48439968e-02,  -4.09390892e-01,
    -1.82734736e+01])

    Radial velocity dispersion profile and its gradient:

    >>> ps = profile.Profile(s.s, max=15)
    >>> ps['vr_disp']
    SimSnap: deriving array vr
    SimSnap: deriving array r
    SimArray([ 118.80420996,  122.06102431,  131.13872886,  144.74447697,
    35.89904165,   37.59565128,   35.21608633,   35.03373379], '1.01e+00 km s**-1')

    >>> p['d_vr_disp']
    SimArray([  21.71365764,   41.11802081,   75.61694836,  105.28110255,
    2.664999  ,   -2.27668148,   -8.54033931,   -1.21577105], '1.01e+00 km s**-1 kpc**-1')
    

    Using another quantity for binning:

    >>> ps = profile.Profile(s.s, calc_x = lambda x: x.s['rform'])
    
    """

    _profile_registry = {}


    def _calculate_x(self, sim) :
        return ((sim['pos'][:,0:self.ndim]**2).sum(axis = 1))**(1,2)

    def __init__(self, sim, load_from_file = False, ndim = 2, type = 'lin', calc_x = None, **kwargs):

        generate_new = True
        if calc_x is None : 
            calc_x = self._calculate_x
        self.sim = sim
        self.type = type
        self.ndim = ndim
        self._x = calc_x(sim)
        x = self._x

        if load_from_file :
            import pickle
            
            filename = self._generate_hash_filename()
            
            try: 
                data = pickle.load(open(filename,'r'))
                self._properties = data['properties']
                self.max         = data['max']
                self.min         = data['min']
                self.nbins       = data['nbins']
                self._profiles   = data['profiles']
                self.binind      = data['binind']

                if pynbody.config['verbose'] : print 'Profile: loaded profile from ' + filename

                generate_new = False

            except IOError:
                print 'Profile: existing profile not found -- generating one from scratch'
                                                          
        if generate_new :
            self._properties = {}
            # The profile object is initialized given some array of values
            # and optional keyword parameters

            if kwargs.has_key('max'):
                if isinstance(kwargs['max'],str):
                    self.max = units.Unit(kwargs['max']).ratio(x.units, 
                                                               **sim.conversion_context())
                else: self.max = kwargs['max']
            else:
                self.max = np.max(x)
            if kwargs.has_key('bins'):
                self.nbins = len(kwargs['bins']) - 1
            elif kwargs.has_key('nbins'):
                self.nbins = kwargs['nbins']
            else:
                self.nbins = 100

            if kwargs.has_key('min'):
                if isinstance(kwargs['min'],str):
                    self.min = units.Unit(kwargs['min']).ratio(x.units, 
                                                               **sim.conversion_context())
                else : self.min = kwargs['min']
            else:
                self.min = np.min(x[x>0])

            if kwargs.has_key('bins'):
                self._properties['bin_edges'] = kwargs['bins']
                self.min = kwargs['bins'].min()
                self.max = kwargs['bins'].max()
            elif type == 'log':
                self._properties['bin_edges'] = np.logspace(np.log10(self.min), np.log10(self.max), num = self.nbins+1)
            elif type == 'lin':
                self._properties['bin_edges'] = np.linspace(self.min, self.max, num = self.nbins+1)
            elif type == 'equaln':
                self._properties['bin_edges'] = util.equipartition(x, self.nbins, self.min, self.max)
            else:
                raise RuntimeError, "Bin type must be one of: lin, log, equaln"


            self['bin_edges'] = array.SimArray(self['bin_edges'], x.units)
            self['bin_edges'].sim = self.sim

            n, bins = np.histogram(self._x, self['bin_edges'])
            self._setup_bins()

            # set up the empty list of profiles
            self._profiles = {'n':n}
        

    def _setup_bins(self) :
        # middle of the bins for convenience
        
        self._properties['rbins'] = 0.5*(self['bin_edges'][:-1]+
                                         self['bin_edges'][1:])
        
        # no idea why this next line was there... for conversion_context
        # self['rbins'].sim = self.sim

        # Width of the bins
        self._properties['dr'] = np.gradient(self['rbins']).view(array.SimArray)
        # be extra cautious carrying over stuff because sometimes fails
        self._properties['dr'].units = self['rbins'].units
        self._properties['dr'].sim = self.sim

        self.binind = []
        if len(self._x) > 0:
            self.partbin = np.digitize(self._x, self['bin_edges'])
        else :
            self.partbin = np.array([])

        assert self.ndim in [2,3]
        if self.ndim == 2:
            self._binsize = np.pi*(self['bin_edges'][1:]**2 - 
                                   self['bin_edges'][:-1]**2)
        else:
            self._binsize  = 4./3.*np.pi*(self['bin_edges'][1:]**3 - 
                                          self['bin_edges'][:-1]**3)

        # sort the partbin array
        from bisect import bisect
        sortind = self.partbin.argsort()
        sort_pind = self.partbin[sortind]

        # create the bin index arrays
        prev_index = bisect(sort_pind,0)
        for i in range(self.nbins):
            new_index = bisect(sort_pind,i+1)
            self.binind.append(np.sort(sortind[prev_index:new_index]))
            prev_index = new_index

    def __len__(self):
        """Returns the number of bins used in this profile object"""
        return self.nbins


    def _get_profile(self, name) :
        """Return the profile of a given kind"""
        x = name.split(",")
        if name in self._profiles :
            return self._profiles[name]

        elif x[0] in Profile._profile_registry :
            args = x[1:]
            self._profiles[name] = Profile._profile_registry[x[0]](self, *args)

            try :
                self._profiles[name].sim = self.sim
            except AttributeError :
                pass
            return self._profiles[name]

        elif name in self.sim.keys() or name in self.sim.all_keys() :
            self._profiles[name] = self._auto_profile(name)
            self._profiles[name].sim = self.sim
            return self._profiles[name]

        elif name[-5:]=="_disp" and name[:-5] in self.sim.keys() or name[:-5] in self.sim.all_keys() :
            if pynbody.config['verbose'] : print 'Profile: auto-deriving '+name
            self._profiles[name] = self._auto_profile(name[:-5], dispersion=True)
            self._profiles[name].sim = self.sim
            return self._profiles[name]

        elif name[-4:]=="_rms" and name[:-4] in self.sim.keys() or name[:-4] in self.sim.all_keys() :
            if pynbody.config['verbose'] : print 'Profile: auto-deriving '+name
            self._profiles[name] = self._auto_profile(name[:-4], rms=True)
            self._profiles[name].sim = self.sim
            return self._profiles[name]
        
        elif name[-4:]=="_med" and name[:-4] in self.sim.keys() or name[:-4] in self.sim.all_keys() :
            if pynbody.config['verbose'] : print 'Profile: auto-deriving '+name
            self._profiles[name] = self._auto_profile(name[:-4], median=True)
            self._profiles[name].sim = self.sim
            return self._profiles[name]
        
        elif name[0:2]=="d_" and name[2:] in self.keys() or name[2:] in self.derivable_keys() or name[2:] in self.sim.all_keys() :
#            if np.diff(self['dr']).all() < 1e-13 : 
            if pynbody.config['verbose'] : print 'Profile: '+name+'/dR'
            self._profiles[name] = np.gradient(self[name[2:]], self['dr'][0])
            self._profiles[name] = self._profiles[name] / self['dr'].units
            return self._profiles[name]
            #else :
            #    raise RuntimeError, "Derivatives only possible for profiles of fixed bin width."

        else :
            raise KeyError, name+" is not a valid profile"

    def _auto_profile(self, name, dispersion=False, rms=False, median=False) :
        result = np.zeros(self.nbins)
        
        # force derivation of array if necessary:
        self.sim[name]
        
        for i in range(self.nbins):
            subs = self.sim[self.binind[i]]
            name_array = subs[name].view(np.ndarray)
            mass_array = subs['mass'].view(np.ndarray)

            if dispersion :
                sq_mean = (name_array**2*mass_array).sum()/self['mass'][i]
                mean_sq = ((name_array*mass_array).sum()/self['mass'][i])**2
                try:
                    result[i] = math.sqrt(sq_mean - mean_sq)
                except ValueError :
                    result[i] = 0  # sq_mean<mean_sq occasionally from numerical roundoff

            elif rms : 
                result[i] = np.sqrt((name_array**2*mass_array).sum()/self['mass'][i])
            elif median :
                sorted_name = sorted(name_array)
                result[i] = sorted_name[int(np.floor(0.5*len(subs)))]
            else :
                result[i] = (name_array*mass_array).sum()/self['mass'][i]

        result = result.view(array.SimArray)
        result.units = self.sim[name].units
        result.sim = self.sim
        return result

    def __getitem__(self, name):
        """Return the profile of a given kind"""
        if name in self._properties:
            return self._properties[name]
        else:
            return self._get_profile(name)

    def __setitem__(self, name, item):
        """Set the profile or property by hand"""
        if name in self._properties:
            self._properties[name] = item
        elif name in self._profiles:
            self._profiles[name] = item
        else :
            raise KeyError, name + " is not a valid profile or property"

    def __delitem__(self, name) :
        del self._profiles[name]

    def __repr__(self):
        return ("<Profile: " +
                str(self.families()) + " ; " +
                str(self.ndim) + "D ; " +
                self.type) + " ; " + str(self.keys())+ ">"

    def keys(self):
        """Returns a listing of available profile types"""
        return self._profiles.keys()

    def derivable_keys(self):
        """Returns a list of possible profiles"""
        return self._profile_registry.keys()

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

    def _generate_hash_filename(self):
        """Create a filename for the saved profile from a hash using the binning data"""
        import hashlib
        
        try:
            filename = self.sim.base.filename+'.profile.'+(hashlib.md5(self._x,usedforsecurity=False)).hexdigest()
        except AttributeError:
            filename = self.sim.filename+'.profile.'+(hashlib.md5(self._x,usedforsecurity=False)).hexdigest()
            
        return filename


    def write(self):
        """
        Writes all the vital information of the profile to a file.

        To recover the profile, initialize a profile with the
        load_from_file=True keyword to automatically load a previously
        saved profile. The filename is chosen automatically and
        corresponds to a hash generated from the positions of the
        particles used in the profile. This is to ensure that you are
        always looking at the same set of particles, centered in the
        same way. It also means you *must* use the same centering
        method if you want to reuse a saved profile.

        
        """

        import pickle
        
        # record all the important data except for the snapshot itself
        # use the hash generated from the particle list for the file name suffix

        filename = self._generate_hash_filename()

        if pynbody.config['verbose']: print 'Profile: writing profile to ' + filename
            
        pickle.dump({'properties': self._properties, 
                     'max': self.max,
                     'min': self.min,
                     'nbins': self.nbins,
                     'profiles': self._profiles,
                     'binind': self.binind}, 
                    open(filename,'w'))
                     

    @staticmethod
    def profile_property(fn) :
        Profile._profile_registry[fn.__name__]=fn
        return fn


@Profile.profile_property
def mass(self):
    """
    Calculate mass in each bin
    """
    
    if pynbody.config['verbose'] : print 'Profile: mass()'
    mass = array.SimArray(np.zeros(self.nbins), self.sim['mass'].units)

    with self.sim.immediate_mode : 
        pmass = self.sim['mass'].view(np.ndarray)

    for i in range(self.nbins):
        mass[i] = (pmass[self.binind[i]]).sum()

    mass.sim = self.sim

    return mass

@Profile.profile_property
def density(self):
    """
    Generate a radial density profile for the current type of profile
    """
    if pynbody.config['verbose'] : print 'Profile: density()'
    return self['mass']/self._binsize

@Profile.profile_property
def fourier(self):
    """
    Generate a profile of fourier coefficients, amplitudes and phases
    """
    if pynbody.config['verbose'] : print 'Profile: fourier()'

    f = {'c': np.zeros((7, self.nbins),dtype=complex),
         'amp': np.zeros((7, self.nbins)),
         'phi': np.zeros((7, self.nbins))}

    for i in range(self.nbins):
        if self._profiles['n'][i] > 100:
            phi = np.arctan2(self.sim['y'][self.binind[i]], self.sim['x'][self.binind[i]])
            mass = self.sim['mass'][self.binind[i]]
            
            hist, binphi = np.histogram(phi,weights=mass,bins=100)
            binphi = .5*(binphi[1:]+binphi[:-1])
            for m in range(7) : 
                f['c'][m,i] = np.sum(hist*np.exp(-1j*m*binphi))


    f['c'][:,self['mass']>0] /= self['mass'][self['mass']>0]
    f['amp'] = np.sqrt(np.imag(f['c'])**2 + np.real(f['c'])**2)
    f['phi'] = np.arctan2(np.imag(f['c']), np.real(f['c']))

    return f

@Profile.profile_property
def mass_enc(self):
    """
    Generate the enclosed mass profile
    """
    if pynbody.config['verbose'] : print 'Profile: mass_enc()'
    m_enc = array.SimArray(np.zeros(self.nbins), self.sim['mass'].units)
    m_enc.sim = self.sim
    for i in range(self.nbins):
        m_enc[i] = self['mass'].in_units(self.sim['mass'].units)[:i].sum()
    return m_enc

@Profile.profile_property
def dyntime(self) :
    """The dynamical time of the bin, sqrt(R^3/2GM)."""
    if pynbody.config['verbose'] : print 'Profile: dyntime()'
    dyntime = (self['rbins']**3/(2*units.G*self['mass_enc']))**(1,2)
    return dyntime


@Profile.profile_property
def g_spherical(self) :
    """The naive gravitational acceleration assuming spherical
    symmetry = GM_enc/r^2"""

    return (units.G*self['mass_enc']/self['rbins']**2)

@Profile.profile_property
def rotation_curve_spherical(self):
    """
    The naive rotation curve assuming spherical symmetry: vc = sqrt(G M_enc/r)
    """
    
    return ((units.G*self['mass_enc']/self['rbins'])**(1,2))#.in_units('km s**-1')

@Profile.profile_property
def j_circ(p) :
    """Angular momentum of particles on circular orbits."""
    if pynbody.config['verbose'] : print 'Profile: j_circ()'
    return p['v_circ'] * p['rbins']


@Profile.profile_property
def v_circ(p, grav_sim=None) :
    """Circular velocity, i.e. rotation curve. Calculated by computing the gravity 
    in the midplane - can be expensive"""

    import pynbody.gravity.calc as gravity
    from .. import config

    global config

    grav_sim = grav_sim or p.sim

    if pynbody.config['verbose'] : 
        print 'Profile: v_circ() -- warning, disk must be in the x-y plane'

    
    # If this is a cosmological run, go up to the halo level
    #if hasattr(grav_sim,'base') and grav_sim.base.properties.has_key("halo_id") :
    #    while hasattr(grav_sim,'base') and grav_sim.base.properties.has_key("halo_id") :
    #        grav_sim = grav_sim.base
    
    #elif hasattr(grav_sim,'base') : 
    #    grav_sim = grav_sim.base
    
    if config['tracktime']:
        import time
        start = time.time()
        rc = gravity.midplane_rot_curve(grav_sim, p['rbins']).in_units(p.sim['vel'].units)
        end = time.time()
        if config['verbose']: print 'Rotation curve calculated in %5.3g s'%(end-start)
        return rc
    else:
        return gravity.midplane_rot_curve(grav_sim, p['rbins']).in_units(p.sim['vel'].units)

@Profile.profile_property
def E_circ(p) :
    """Energy of particles on circular orbits."""
    if pynbody.config['verbose'] : print 'Profile: E_circ()'
    return 0.5*(p['v_circ']**2) + p['pot']

@Profile.profile_property
def pot(p) :
    """Calculates the potential in the midplane - can be expensive"""
    #from . import gravity
    import pynbody.gravity.calc as gravity

    if pynbody.config['verbose'] : 
        print 'Profile: pot() -- warning, disk must be in the x-y plane'

    grav_sim = p.sim
    # Go up to the halo level
    while hasattr(grav_sim,'base') and grav_sim.base.properties.has_key("halo_id") :
        grav_sim = grav_sim.base
        
    if pynbody.config['tracktime']:
        import time
        start = time.clock()
        pot = gravity.midplane_potential(grav_sim, p['rbins']).in_units(p.sim['vel'].units**2)
        end = time.clock()
        if pynbody.config['verbose']: print 'Potential calculated in %5.3g s'%(end-start)
        return pot
    else:
        return gravity.midplane_potential(grav_sim, p['rbins']).in_units(p.sim['vel'].units**2)
    

@Profile.profile_property
def omega(p) :
    """Circular frequency Omega = v_circ/radius (see Binney & Tremaine Sect. 3.2)"""
    if pynbody.config['verbose'] : print 'Profile: omega()'
    prof = p['v_circ']/p['rbins']
    prof.set_units_like('km s**-1 kpc**-1')
    return prof

@Profile.profile_property
def kappa(p) :
    """Radial frequency kappa = sqrt(R dOmega^2/dR + 4 Omega^2) (see Binney & Tremaine Sect. 3.2)"""
    if pynbody.config['verbose'] : print 'Profile: kappa()'
    dOmegadR = np.gradient(p['omega']**2, p['dr'][0])
    dOmegadR.set_units_like('km**2 s**-2 kpc**-3')
    return np.sqrt(p['rbins']*dOmegadR + 4*p['omega']**2)

@Profile.profile_property
def beta(p) :
    """3D Anisotropy parameter as defined in Binney and Tremiane"""
    if pynbody.config['verbose'] : print 'Profile: beta()'
    assert p.ndim is 3
    return  1.5-(p['vx_disp']**2+p['vy_disp']**2+p['vz_disp']**2)/p['vr_disp']**2/2.

@Profile.profile_property
def magnitudes(self,band='v'):
    """
    Calculate magnitudes in each bin
    """
    from . import luminosity

    magnitudes = np.zeros(self.nbins)
    print "Calculating magnitudes"
    for i in range(self.nbins):
        magnitudes[i] = luminosity.halo_mag(self.sim[self.binind[i]],band=band)
    magnitudes = array.SimArray(magnitudes, units.Unit('1'))
    magnitudes.sim = self.sim
    return magnitudes

@Profile.profile_property
def sb(self,band='v') :
    # At 10 pc (distance for absolute magnitudes), 1 arcsec is 10 AU=1/2.06e4 pc
    # In [5]: (np.tan(np.pi/180/3600)*10.0)**2
    # Out[5]: 2.3504430539466191e-09
    # 1 square arcsecond is thus 2.35e-9 pc^2
    sqarcsec_in_bin = self._binsize.in_units('pc^2') / 2.3504430539466191e-09
    bin_luminosity = 10.0**(-0.4*self['magnitudes,'+band])
    #import pdb; pdb.set_trace()
    surfb = -2.5*np.log10(bin_luminosity / sqarcsec_in_bin)
    surfb = array.SimArray(surfb, units.Unit('1'))
    surfb.sim = self.sim
    return surfb

@Profile.profile_property
def Q(self) : 
    """Toomre Q parameter"""
    return (self['vr_disp']*self['kappa']/(3.36*self['density']*units.G)).in_units("")

@Profile.profile_property
def X(self) : 
    """X parameter defined as kappa^2*R/(2*pi*G*sigma*m) 
    See Binney & Tremaine 2008, eq. 6.77"""

    lambda_crit = 4.*np.pi**2 * units.G * self['density']/(self['kappa']**2)
    kcrit = 2.*np.pi/lambda_crit
    return (kcrit*self['rbins']/2).in_units("")


@Profile.profile_property
def jtot(self) : 
    """
    Magnitude of the total angular momentum 
    """
    jtot = np.zeros(self.nbins)
    
    for i in range(self.nbins) : 
        subs = self.sim[self.binind[i]]
        jx = (subs['j'][:,0]*subs['mass']).sum()/self['mass'][i]
        jy = (subs['j'][:,1]*subs['mass']).sum()/self['mass'][i]
        jz = (subs['j'][:,2]*subs['mass']).sum()/self['mass'][i]
        
        jtot[i] = np.sqrt(jx**2+jy**2+jz**2)

    return jtot

@Profile.profile_property
def j_theta(self): 
    """
    Angle that the angular momentum vector of the bin makes with respect to the xy-plane.
    """

    return np.arccos(self['jz']/self['jtot'])

@Profile.profile_property
def j_phi(self): 
    """
    Angle that the angular momentum vector of the bin makes with the x-axis in the xy plane.
    """
    j_phi = np.zeros(self.nbins)

    for i in range(self.nbins) : 
        subs = self.sim[self.binind[i]]
        jx = (subs['j'][:,0]*subs['mass']).sum()/self['mass'][i]
        jy = (subs['j'][:,1]*subs['mass']).sum()/self['mass'][i]
        j_phi[i] = np.arctan(jy,jx)
    
    return j_phi
    

class InclinedProfile(Profile) :
    """
    
    Creates a profile object to be used with a snapshot inclined by
    some known angle to the xy plane.

    In addition to the SimSnap object, it also requires the angle to
    initialize.

    **Example:**

    >>> s = pynbody.load('sim')
    >>> pynbody.analysis.angmom.faceon(s)
    >>> s.rotate_x(60)
    >>> p = pynbody.profile.InclinedProfile(s, 60)

    """

    def _calculate_x(self, sim) :
        # calculate an ellipsoidal radius
        return (sim['x']**2 + (sim['y']/np.cos(np.radians(self.angle)))**2)**(1,2)

    def __init__(self, sim, angle, load_from_file = False, ndim = 2, type = 'lin', **kwargs):
        self.angle = angle
        
        Profile.__init__(self,sim,load_from_file=load_from_file,ndim=ndim,type=type,**kwargs)

        # define the minor axis
        self._properties['rbins_min'] = np.cos(np.radians(self.angle))*self['rbins']


class VerticalProfile(Profile) : 
    """

    Creates a profile object that uses the absolute value of the z-coordinate for binning. 

    **Input**: 
    
    *sim*: snapshot to make a profile from

    *rmin*: minimum radius for particle selection in kpc
     
    *rmax*: maximum radius for particle selection in kpc

    *zmax*: maximum height to consider in kpc

    **Optional Keywords**: 

    *ndim*: if ndim=2, an edge-on projected profile is produced,
     i.e. density is in units of mass/pc^2. If ndim=3 a volume
     profile is made, i.e. density is in units of mass/pc^3.

    """

    def _calculate_x(self, sim) : 
        return array.SimArray(np.abs(sim['z']),sim['z'].units)

    def __init__(self,sim,rmin,rmax,zmax,load_from_file = False, ndim = 3, type = 'lin', **kwargs): 
        
        if isinstance(rmin, str): rmin = units.Unit(rmin)
        if isinstance(rmax, str): rmax = units.Unit(rmax)
        if isinstance(zmax, str): zmax = units.Unit(zmax)
        self.rmin = rmin
        self.rmax = rmax
        self.zmax = zmax

        # create a snapshot that only includes the section of disk we're interested in
        assert ndim in [2,3]
        if ndim == 3: 
            sub_sim = sim[pynbody.filt.Disc(rmax,zmax) & ~pynbody.filt.Disc(rmin,zmax)]
        else: 
            sub_sim = sim[(pynbody.filt.BandPass('x',rmin,rmax) | 
                          pynbody.filt.BandPass('x',-rmax,-rmin)) &
                          pynbody.filt.BandPass('z', -zmax,zmax)]

        Profile.__init__(self,sub_sim,load_from_file=load_from_file,ndim=ndim,type=type,**kwargs)


    def _setup_bins(self) : 
        Profile._setup_bins(self)

        dr = self.rmax - self.rmin

        if self.ndim == 2: 
            self._binsize = (self['bin_edges'][1:]-self['bin_edges'][:-1])*dr
        else : 
            area = array.SimArray(np.pi*(self.rmax**2-self.rmin**2),"kpc^2")
            self._binsize = (self['bin_edges'][1:]-self['bin_edges'][:-1])*area
        

class QuantileProfile(Profile) : 
    """

    Creates a profile object that returns the requested quantiles
    for a given array in a given bin.  The quantiles may be mass weighted.

    **Input**: 
    
    *sim*: snapshot to make a profile from

    *q (default: (0.16,0.5,0.84))*: 
             The quantiles that will be returned.
             Default is median with 1-sigma on either side.
             q can be of arbitrary length allowing the user to select
             any quantiles they desire.

    *weights (default:None)*:  
             What should be used to weight the quantile.  You will usually
             want to use particle mass: sim['mass'].  
             The default is to not weight by anything, weights=None.

    **Optional Keywords**: 

    *ndim*: if ndim=2, an edge-on projected profile is produced,
     i.e. density is in units of mass/pc^2. If ndim=3 a volume
     profile is made, i.e. density is in units of mass/pc^3.

    """

    def __init__(self,sim,q=(0.16,0.50,0.84),weights=None,load_from_file = False, ndim = 3, type = 'lin', **kwargs): 
        
        # create a snapshot that only includes the section of disk we're interested in
        self.quantiles=q
        self.qweights = weights

        Profile.__init__(self,sim,load_from_file=load_from_file,ndim=ndim,type=type,**kwargs)


    def _get_profile(self, name) :
        """Return the profile of a given kind"""
        x = name.split(",")
        if name in self._profiles :
            return self._profiles[name]

        elif name in self.sim.keys() or name in self.sim.all_keys() :
            self._profiles[name] = self._auto_profile(name)
            self._profiles[name].sim = self.sim
            return self._profiles[name]

        else :
            raise KeyError, name+" is not a valid QuantileProfile"


    def _auto_profile(self, name, dispersion=False, rms=False, median=False) :
        result = np.zeros((self.nbins,len(self.quantiles)))
        for i in range(self.nbins):
            subs = self.sim[self.binind[i]]
            with self.sim.immediate_mode : 
                name_array = subs[name].view(np.ndarray)
                sorted_array = sorted(name_array)
                topind = len(name_array)-1
                if self.qweights is not None:
                    sorted_weights = self.qweights[np.argsort(name_array)]
            
            for iq,q in enumerate(self.quantiles):
                #import pdb; pdb.set_trace()
                if len(name_array) > 0:
                    if self.qweights is None: 
                        ilow = int(np.floor(q*topind))
                        inc = q*topind - ilow
                        lowval = sorted_array[ilow]
                        hival = sorted_array[ilow+1]
                        result[i,iq] = lowval+inc*(hival-lowval)
                    else:
                        cumw= np.cumsum(sorted_weights)/np.sum(sorted_weights)
                        imin = min(np.arange(len(sorted_array)),key=lambda x:abs(cumw[x]-q))
                        inc = q-cumw[imin]
                        lowval = sorted_array[imin]
                        if inc > 0: nextval = sorted_array[imin+1]
                        else: 
                            if imin == 0: nextval = lowval
                            else: nextval = sorted_array[imin-1]
                        
                        result[i,iq] = lowval+inc*(nextval-lowval)
                        #if (result[i,iq] < 1e-6) : import pdb; pdb.set_trace()
                else:
                    result[i,iq] = np.nan
                    self['rbins'][i] = np.nan


        result = result.view(array.SimArray)
        result.units = self.sim[name].units
        result.sim = self.sim
        return result

