"""
Support for creating profiles of various quantities, normally as a function of 2d or 3d radius.

The functions defined in the module represent profiles that can be access from a
:class:`~pynbody.analysis.profile.Profile` object.

For more information and example usage, see the :ref:`profile` tutorial and the documentation for the
:class:`~pynbody.analysis.profile.Profile` class.
"""

import logging
import math
import pickle
import warnings
from time import process_time

import numpy as np

import pynbody

from .. import array, units, util

logger = logging.getLogger('pynbody.analysis.profile')


class Profile:

    """Generates profiles of specified quantities as a function of radius or other binning quantity.

    Any quantity known in the SimSnap can be profiled, meaning that the mean value of that
    quantity in each bin is calculated.

    .. seealso::
      For more information and example usage, see the :ref:`profile` tutorial.

      To define profiles of new quantities, see :meth:`profile_property`.

    **Implicit averaging**: If an array ``ar`` is defined in the underlying ``SimSnap``, then a profile of ``ar``
    can be accessed as ``p['ar']`` where ``p`` is a ``Profile`` object. For example, ``p['vr']`` gives
    the radial velocity profile. Implicitly, this is averaged over all particles in each bin,
    weighted by mass (unless an alternate weighting scheme is passed to the ``weight_by`` keyword argument of the
    constructor).

    **Dispersions**: One may append ``_rms`` or ``_disp`` to the name of a
    defined array to get the root-mean-square or dispersion profile, respectively. For example,
    ``p['vr_rms']`` gives the root-mean-square radial velocity profile, while ``p['vr_disp']`` gives
    the radial velocity dispersion profile. By definition, ``p['vr_disp']**2`` is the same as
    ``p['vr_rms']**2 - p['vr']**2``.

    **Derivatives**: One may also prepend ``d_`` to the name of a defined array to get the derivative, e.g.
    ``p['d_temp']`` gives the radial temperature gradient.

    **Non-array profiles**: Profiles can be defined that do not directly correspond to the average over an
    array in the snapshot. Examples include ``density``, ``mass`` and ``mass_enc``. These are implemented as
    functions in the :mod:`pynbody.analysis.profile` module; you can therefore find a list of available
    profiles by looking at the functions there. These profiles can be accessed in the same way as array profiles,
    e.g. ``p['density']``. For profiles that take an argument, such as ``sb``, this is passed in with
    an underscore e.g. ``p['sb_b']`` for b-band surface brightnesses.

    **Storing profiles**: Use the :func:`~pynbody.analysis.profile.Profile.write` function to write the current
    profiles with all the necessary information to a file. Initialize a profile with the ``load_from_file=True``
    keyword to automatically load a previously saved profile. The filename is chosen automatically and corresponds to
    a hash generated from the positions of the particles used in the profile. This is to ensure that you are always
    looking at the same set of particles, centered in the same way. It also means you *must* use the same centering
    method if you want to reuse a saved profile.

    .. versionchanged:: 2.0

      The method ``create_particle_array`` has been removed. Its behaviour was poorly defined in v1, and not believed
      to be widely used.

    """

    _profile_registry = {}

    def _calculate_x(self, sim):
        return ((sim['pos'][:, 0:self.ndim] ** 2).sum(axis=1)) ** (1, 2)

    def __init__(self, sim, load_from_file=False, ndim=2, type='lin', calc_x=None, weight_by='mass', **kwargs):
        """Initialise a profile, determining the binning quantity and bin size.

        The constructor generates the bins without actually calculating any profiles. The profiles are calculated
        lazily when requested.

        Parameters
        ----------

        sim : pynbody.snapshot.SimSnap
            The simulation snapshot to generate a profile for

        ndim : int, optional:
            Specifies whether it's a 2D or 3D profile - in the 2D case, the bins are generated in the xy plane

        type : str, optional:
            Specifies whether bins should be spaced linearly ('lin', default), logarithmically ('log') or contain
            equal numbers of particles ('equaln')

        rmin : float, optional:
            Minimum value to consider (left-hand-edge of lowest bin). Default is the minimum value of the binning
            quantity.

        rmax : float, optional:
            Maximum value to consider (right-hand-edge of highest bin). Default is the maximum value of the binning
            quantity.

        nbins : int, optional:
            Number of bins to use. Default is 100.

        bins : array-like, optional:
            Predefined bin edges in units of the binning quantity. If this keyword is set, the values of the keywords
            type, nbins, rmin and rmax will be ignored.

        calc_x : function, optional:
            Function to use to calculate the value for binning. If None it defaults to the radial distance from
            origin (in either 2 or 3 dimensions), but you can specify this function to return any value you want for
            making profiles along arbitrary axes. Depending on your function, the units of certain profiles (such as
            density) might not make sense.

        weight_by : str, optional:
            Name of the array to use for weighting averages across particles in each bin. Default is 'mass'.

        """

        generate_new = True
        if calc_x is None:
            calc_x = self._calculate_x
        self.sim = sim
        self.type = type
        self.ndim = ndim
        self._weight_by = weight_by
        self._x = calc_x(sim)
        x = self._x

        if load_from_file:

            filename = self._get_unique_filepath_from_particle_list()

            try:
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                self._properties = data['properties']
                self.max = data['max']
                self.min = data['min']
                self.nbins = data['nbins']
                self._profiles = data['profiles']
                self.binind = data['binind']

                logger.info("Loaded profile from %s" % filename)

                generate_new = False

            except FileNotFoundError:
                logger.warning(
                    "Existing profile not found -- generating one from scratch instead")

        if generate_new:
            self._properties = {}
            # The profile object is initialized given some array of values
            # and optional keyword parameters

            if 'max' in kwargs:
                kwargs['rmax'] = kwargs.pop('max')
                warnings.warn("Use of max as a keyword argument is deprecated. Use rmax instead.", DeprecationWarning)
            if 'min' in kwargs:
                kwargs['rmin'] = kwargs.pop('min')
                warnings.warn("Use of min as a keyword argument is deprecated. Use rmin instead.", DeprecationWarning)

            if 'rmax' in kwargs:
                if isinstance(kwargs['rmax'], str):
                    self.max = units.Unit(kwargs['rmax']).ratio(x.units,
                                                               **sim.conversion_context())
                else:
                    self.max = kwargs['rmax']
            else:
                self.max = np.max(x)
            if 'bins' in kwargs:
                self.nbins = len(kwargs['bins']) - 1
            elif 'nbins' in kwargs:
                self.nbins = kwargs['nbins']
            else:
                self.nbins = 100

            if 'rmin' in kwargs:
                if isinstance(kwargs['rmin'], str):
                    self.min = units.Unit(kwargs['rmin']).ratio(x.units,
                                                               **sim.conversion_context())
                else:
                    self.min = kwargs['rmin']
            else:
                self.min = np.min(x[x > 0])

            if 'bins' in kwargs:
                self._properties['bin_edges'] = kwargs['bins']
                self.min = kwargs['bins'].min()
                self.max = kwargs['bins'].max()
            elif type == 'log':
                self._properties['bin_edges'] = np.logspace(
                    np.log10(self.min), np.log10(self.max), num=self.nbins + 1)
            elif type == 'lin':
                self._properties['bin_edges'] = np.linspace(
                    self.min, self.max, num=self.nbins + 1)
            elif type == 'equaln':
                self._properties['bin_edges'] = util.equipartition(
                    x, self.nbins, self.min, self.max)
            else:
                raise RuntimeError("Bin type must be one of: lin, log, equaln")

            self['bin_edges'] = array.SimArray(self['bin_edges'], x.units)
            self['bin_edges'].sim = self.sim

            n, bins = np.histogram(self._x, self['bin_edges'])
            self._setup_bins()

            # set up the empty list of profiles
            self._profiles = {'n': n}

    def _setup_bins(self):
        # middle of the bins for convenience

        self._properties['rbins'] = 0.5 * (self['bin_edges'][:-1] +
                                           self['bin_edges'][1:])

        # no idea why this next line was there... for conversion_context
        # self['rbins'].sim = self.sim

        # Width of the bins
        self._properties['dr'] = np.gradient(
            self['rbins']).view(array.SimArray)
        # be extra cautious carrying over stuff because sometimes fails
        self._properties['dr'].units = self['rbins'].units
        self._properties['dr'].sim = self.sim

        self.binind = []
        if len(self._x) > 0:
            self.partbin = np.digitize(self._x, self['bin_edges'])
        else:
            self.partbin = np.array([])

        assert self.ndim in [2, 3]
        if self.ndim == 2:
            self._binsize = np.pi * (self['bin_edges'][1:] ** 2 -
                                     self['bin_edges'][:-1] ** 2)
        else:
            self._binsize = 4. / 3. * np.pi * (self['bin_edges'][1:] ** 3 -
                                               self['bin_edges'][:-1] ** 3)

        # sort the partbin array
        from bisect import bisect
        sortind = self.partbin.argsort()
        sort_pind = self.partbin[sortind]

        # create the bin index arrays
        prev_index = bisect(sort_pind, 0)
        for i in range(self.nbins):
            new_index = bisect(sort_pind, i + 1)
            self.binind.append(np.sort(sortind[prev_index:new_index]))
            prev_index = new_index

    def __len__(self):
        """Returns the number of bins used in this profile object"""
        return self.nbins

    def _get_profile(self, name):
        """Return the profile of a given kind"""
        x = name.split(",")
        if name in self._profiles:
            return self._profiles[name]

        elif x[0] in Profile._profile_registry:
            args = x[1:]
            self._profiles[name] = Profile._profile_registry[x[0]](self, *args)

            try:
                self._profiles[name].sim = self.sim
            except AttributeError:
                pass
            return self._profiles[name]

        elif name in list(self.sim.keys()) or name in self.sim.all_keys():
            self._profiles[name] = self._auto_profile(name)
            self._profiles[name].sim = self.sim
            return self._profiles[name]

        elif name[-5:] == "_disp" and (name[:-5] in list(self.sim.keys()) or name[:-5] in self.sim.all_keys()):
            logger.info("Auto-deriving %s" % name)
            self._profiles[name] = self._auto_profile(
                name[:-5], dispersion=True)
            self._profiles[name].sim = self.sim
            return self._profiles[name]

        elif name[-4:] == "_rms" and (name[:-4] in list(self.sim.keys()) or name[:-4] in self.sim.all_keys()):
            logger.info("Auto-deriving %s" % name)
            self._profiles[name] = self._auto_profile(name[:-4], rms=True)
            self._profiles[name].sim = self.sim
            return self._profiles[name]

        elif name[-4:] == "_med" and (name[:-4] in list(self.sim.keys()) or name[:-4] in self.sim.all_keys()):
            logger.info("Auto-deriving %s" % name)
            self._profiles[name] = self._auto_profile(name[:-4], median=True)
            self._profiles[name].sim = self.sim
            return self._profiles[name]

        elif name[0:2] == "d_" and (name[2:] in list(self.keys()) or name[2:] in self.derivable_keys() or name[2:] in self.sim.all_keys()):
            #            if np.diff(self['dr']).all() < 1e-13 :
            logger.info("Auto-deriving %s/dR" % name)
            self._profiles[name] = np.gradient(self[name[2:]], self['dr'][0])
            self._profiles[name] = self._profiles[name] / self['dr'].units
            return self._profiles[name]
            # else :
            #    raise RuntimeError, "Derivatives only possible for profiles of fixed bin width."

        else:
            raise KeyError(name + " is not a valid profile")

    def _auto_profile(self, name, dispersion=False, rms=False, median=False):
        result = np.zeros(self.nbins)

        # force derivation of array if necessary:
        self.sim[name]

        for i in range(self.nbins):
            subs = self.sim[self.binind[i]]
            name_array = subs[name].view(np.ndarray)
            mass_array = subs[self._weight_by].view(np.ndarray)

            if dispersion:
                sq_mean = (name_array ** 2 * mass_array).sum() / \
                    self['weight_fn'][i]
                mean_sq = (
                    (name_array * mass_array).sum() / self['weight_fn'][i]) ** 2
                try:
                    result[i] = math.sqrt(sq_mean - mean_sq)
                except ValueError:
                    # sq_mean<mean_sq occasionally from numerical roundoff
                    result[i] = 0

            elif rms:
                result[i] = np.sqrt(
                    (name_array ** 2 * mass_array).sum() / self['weight_fn'][i])
            elif median:
                if len(subs) == 0:
                    result[i] = np.nan
                else:
                    sorted_name = sorted(name_array)
                    result[i] = sorted_name[int(np.floor(0.5 * len(subs)))]
            else:
                result[i] = (name_array * mass_array).sum() / self['weight_fn'][i]

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
        else:
            raise KeyError(name + " is not a valid profile or property")

    def __delitem__(self, name):
        del self._profiles[name]

    def __repr__(self):
        return ("<Profile: " +
                str(self.families()) + " ; " +
                str(self.ndim) + "D ; " +
                self.type) + " ; " + str(list(self.keys())) + ">"

    def keys(self):
        """Returns a listing of available profile types"""
        return list(self._profiles.keys())

    def derivable_keys(self):
        """Returns a list of possible profiles"""
        return list(self._profile_registry.keys())

    def families(self):
        """Returns the family of particles used"""
        return self.sim.families()

    def _generate_hash_filename_from_particles(self):
        """Create a filename for the saved profile from a hash using the binning data"""

        import hashlib

        # Changing to the new() method, which will not fail if usedforsecurity is unsupported
        # in a given system configuration (issue 581)
        h = hashlib.new('md5')
        # Reproduce old behaviour, given byte-like data to create the hash.
        h.update(self._x)
        return h.hexdigest()

    def _get_unique_filepath_from_particle_list(self):

        try:
            folder_path = self.sim.base.filename
        except AttributeError:
            folder_path = self.sim.filename

        unique_hash = self._generate_hash_filename_from_particles()
        filename = folder_path + '.profile.' + unique_hash
        return filename

    def write(self):
        """
        Writes all the vital information of the profile to a file.

        To recover the profile, initialize a profile with the ``load_from_file=True`` keyword to automatically
        load a previously saved profile. The filename is chosen automatically and corresponds to a hash generated
        from the positions of the particles used in the profile. This is to ensure that you are always looking at
        the same set of particles, centered in the same way. It also means you *must* use the same centering
        method if you want to reuse a saved profile.

        """

        # record all the important data except for the snapshot itself
        # use the hash generated from the particle list for the file name
        # suffix

        filename = self._get_unique_filepath_from_particle_list()

        logger.info("Writing profile to %s", filename)

        with open(filename, "wb") as f:
            pickle.dump(
                {
                    'properties': self._properties,
                    'max': self.max,
                    'min': self.min,
                    'nbins': self.nbins,
                    'profiles': self._profiles,
                    'binind': self.binind
                },
                f,
            )

    @staticmethod
    def profile_property(fn):
        """Function decorator to define a new profile property.

        For example,

        .. code-block:: python

         @Profile.profile_property
         def x_squared(pro):
             return pro['x']**2

        would define a new profile property 'x_squared' which is the square of the 'x' profile.
        This can then be accessed as ``pro['x_squared']`` for any profile object ``pro``.
        """
        Profile._profile_registry[fn.__name__] = fn
        return fn


@Profile.profile_property
def weight_fn(pro, weight_by=None):
    """
    Calculate mass in each bin
    """
    if weight_by is None:
        weight_by = pro._weight_by
    mass = array.SimArray(np.zeros(pro.nbins), pro.sim[weight_by].units)

    with pro.sim.immediate_mode:
        pmass = pro.sim[weight_by].view(np.ndarray)

    for i in range(pro.nbins):
        mass[i] = (pmass[pro.binind[i]]).sum()

    mass.sim = pro.sim
    mass.units = pro.sim[weight_by].units

    return mass

@Profile.profile_property
def mass(pro):
    return weight_fn(pro, 'mass')



@Profile.profile_property
def density(pro):
    """
    Generate a radial density profile for the current type of profile
    """
    return pro['mass'] / pro._binsize


@Profile.profile_property
def fourier(pro, delta_t ="0.1 Myr", phi_bins=100):
    """
    Generate a profile of fourier coefficients, amplitudes and phases
    """

    delta_t = pynbody.units.Unit(delta_t)


    f = {'c': np.zeros((7, pro.nbins), dtype=complex),
         'c_delta': np.zeros((7, pro.nbins), dtype=complex),
         'amp': np.zeros((7, pro.nbins)),
         'phi': np.zeros((7, pro.nbins)),
         'dphi_dt': np.zeros((7, pro.nbins))}

    for i in range(pro.nbins):
        if pro._profiles['n'][i] > 100:
            phi = np.arctan2(
                pro.sim['y'][pro.binind[i]], pro.sim['x'][pro.binind[i]])
            mass = pro.sim['mass'][pro.binind[i]]

            x1 = pro.sim['x'][pro.binind[i]] + pro.sim['vx'][pro.binind[i]] * delta_t
            y1 = pro.sim['y'][pro.binind[i]] + pro.sim['vy'][pro.binind[i]] * delta_t
            phi1 = np.arctan2(y1,x1)

            hist_range = (-np.pi, np.pi)
            hist, binphi = np.histogram(phi, weights=mass, bins=phi_bins, range=(hist_range))
            hist1, _ = np.histogram(phi1, weights=mass, bins=phi_bins, range=(hist_range))
            binphi = binphi[:-1] + .5*np.diff(binphi)
            for m in range(7):
                f['c'][m, i] = np.sum(hist * np.exp(-1j * m * binphi))
                f['c_delta'][m, i] = np.sum(hist1 * np.exp(-1j * m * binphi))

    f['c'][:, pro['mass'] > 0] /= pro['mass'][pro['mass'] > 0]
    f['amp'] = np.sqrt(np.imag(f['c']) ** 2 + np.real(f['c']) ** 2)
    f['phi'] = np.arctan2(np.imag(f['c']), np.real(f['c']))

    dphi = np.arctan2(np.imag(f['c_delta']), np.real(f['c_delta'])) - f['phi']
    dphi[dphi>np.pi] = dphi[dphi>np.pi]-2*np.pi
    dphi[dphi<-np.pi] = dphi[dphi<-np.pi]+2*np.pi
    dphi = array.SimArray(dphi,"1")
    f['dphi_dt'] = (dphi / delta_t).in_units(pro.sim['vx'].units / pro.sim['x'].units)

    return f

@Profile.profile_property
def pattern_frequency(pro):
    """Estimate the pattern speed from the m=2 Fourier mode"""
    return pro['fourier']['dphi_dt'][2,:]/2

@Profile.profile_property
def mass_enc(pro):
    """
    Generate the enclosed mass profile
    """
    return pro['mass'].cumsum()


@Profile.profile_property
def density_enc(pro):
    """
    Generate the mean enclosed density profile
    """
    return pro['mass_enc'] / ((4. * math.pi / 3) * pro['rbins'] ** 3)


@Profile.profile_property
def dyntime(pro):
    """The dynamical time of the bin, sqrt(R^3/2GM)."""
    dyntime = (pro['rbins'] ** 3 / (2 * units.G * pro['mass_enc'])) ** (1, 2)
    return dyntime


@Profile.profile_property
def g_spherical(pro):
    """The naive gravitational acceleration assuming spherical
    symmetry = GM_enc/r^2"""

    return (units.G * pro['mass_enc'] / pro['rbins'] ** 2)


@Profile.profile_property
def rotation_curve_spherical(pro):
    """
    The naive rotation curve assuming spherical symmetry: vc = sqrt(G M_enc/r)
    """

    # .in_units('km s**-1')
    return ((units.G * pro['mass_enc'] / pro['rbins']) ** (1, 2))


@Profile.profile_property
def j_circ(pro):
    """Angular momentum of particles on circular orbits."""
    return pro['v_circ'] * pro['rbins']


@Profile.profile_property
def v_circ(pro, grav_sim=None):
    """Circular velocity, i.e. rotation curve. Calculated by computing the gravity
    in the midplane - can be expensive"""

    from .. import config, gravity

    global config

    grav_sim = grav_sim or pro.sim

    logger.warning(
        "Profile v_circ -- this routine assumes the disk is in the x-y plane")

    # If this is a cosmological run, go up to the halo level
    # if hasattr(grav_sim,'base') and grav_sim.base.properties.has_key("halo_number") :
    #    while hasattr(grav_sim,'base') and grav_sim.base.properties.has_key("halo_number") :
    #        grav_sim = grav_sim.base

    # elif hasattr(grav_sim,'base') :
    #    grav_sim = grav_sim.base

    start = process_time()
    rc = gravity.midplane_rot_curve(
        grav_sim, pro['rbins']).in_units(pro.sim['vel'].units)
    end = process_time()
    logger.info("Rotation curve calculated in %5.3g s" % (end - start))
    return rc


@Profile.profile_property
def E_circ(pro):
    """Calculates the energy of particles on circular orbits in the z=0 plane."""
    return 0.5 * (pro['v_circ'] ** 2) + pro['pot']


@Profile.profile_property
def pot(pro):
    """Calculates the potential in the z=0 plane"""
    from .. import gravity

    logger.warning(
        "Profile pot -- this routine assumes the disk is in the x-y plane")

    grav_sim = pro.sim
    # Go up to the halo level
    while hasattr(grav_sim, 'base') and "halo_number" in grav_sim.base.properties:
        grav_sim = grav_sim.base

    start = process_time()
    pot = gravity.midplane_potential(
        grav_sim, pro['rbins']).in_units(pro.sim['vel'].units ** 2)
    end = process_time()
    logger.info("Potential calculated in %5.3g s" % (end - start))
    return pot


@Profile.profile_property
def omega(pro):
    """Circular frequency Omega = v_circ/radius (see Binney & Tremaine Sect. 3.2) in the z=0 plane"""
    prof = pro['v_circ'] / pro['rbins']
    prof.set_units_like('km s**-1 kpc**-1')
    return prof


@Profile.profile_property
def kappa(pro):
    """Radial frequency kappa = sqrt(R dOmega^2/dR + 4 Omega^2) (see Binney & Tremaine Sect. 3.2) in the z=0 plane"""
    dOmega2dR = np.gradient(pro['omega'] ** 2) / np.gradient(pro['rbins'])
    return np.sqrt(pro['rbins'] * dOmega2dR + 4 * pro['omega'] ** 2)


@Profile.profile_property
def beta(pro):
    """3D Anisotropy parameter as defined in Binney and Tremaine"""
    assert pro.ndim == 3
    return 1.5 - (pro['vx_disp'] ** 2 + pro['vy_disp'] ** 2 + pro['vz_disp'] ** 2) / pro['vr_disp'] ** 2 / 2.


@Profile.profile_property
def magnitudes(pro, band='v'):
    """Calculate magnitudes in each bin

    When calling this from a profile object, the band can be specified after an underscore, e.g.
    ``p['magnitudes_b']`` for b-band magnitudes.

    For important information about the calculation of magnitudes and surface brightnesses, see
    the module documentation for :mod:`pynbody.analysis.luminosity`.
    """
    from . import luminosity

    magnitudes = np.zeros(pro.nbins)
    for i in range(pro.nbins):
        magnitudes[i] = luminosity.halo_mag(
            pro.sim[pro.binind[i]], band=band)
    magnitudes = array.SimArray(magnitudes, units.Unit('1'))
    magnitudes.sim = pro.sim
    return magnitudes


@Profile.profile_property
def sb(pro, band='v'):
    """Calculate surface brightness in each bin

    When calling this from a profile object, the band can be specified after an underscore, e.g.
    ``p['sb_b']`` for b-band surface brightnesses.

    For important information about the calculation of magnitudes and surface brightnesses, see
    the module documentation for :mod:`pynbody.analysis.luminosity`.
    """
    # At 10 pc (distance for absolute magnitudes), 1 arcsec is 10 AU=1/2.06e4 pc
    # In [5]: (np.tan(np.pi/180/3600)*10.0)**2
    # Out[5]: 2.3504430539466191e-09
    # 1 square arcsecond is thus 2.35e-9 pc^2
    sqarcsec_in_bin = pro._binsize.in_units('pc^2') / 2.3504430539466191e-09
    bin_luminosity = 10.0 ** (-0.4 * pro['magnitudes,' + band])
    #import pdb; pdb.set_trace()
    surfb = -2.5 * np.log10(bin_luminosity / sqarcsec_in_bin)
    surfb = array.SimArray(surfb, units.Unit('1'))
    surfb.sim = pro.sim
    return surfb


@Profile.profile_property
def Q(pro):
    """Toomre Q parameter"""
    return (pro['vr_disp'] * pro['kappa'] / (3.36 * pro['density'] * units.G)).in_units("")


@Profile.profile_property
def X(pro):
    """X parameter defined as kappa^2*R/(2*pi*G*sigma*m), using the rotation curve from the z=0 plane

    See Binney & Tremaine 2008, eq. 6.77"""

    lambda_crit = 4. * np.pi ** 2 * units.G * \
                  pro['density'] / (pro['kappa'] ** 2)
    kcrit = 2. * np.pi / lambda_crit
    return (kcrit * pro['rbins'] / 2).in_units("")


@Profile.profile_property
def jtot(pro):
    """Magnitude of the total angular momentum
    """
    jtot = np.zeros(pro.nbins)

    for i in range(pro.nbins):
        subs = pro.sim[pro.binind[i]]
        jx = (subs['j'][:, 0] * subs['mass']).sum() / pro['mass'][i]
        jy = (subs['j'][:, 1] * subs['mass']).sum() / pro['mass'][i]
        jz = (subs['j'][:, 2] * subs['mass']).sum() / pro['mass'][i]

        jtot[i] = np.sqrt(jx ** 2 + jy ** 2 + jz ** 2)

    return jtot


@Profile.profile_property
def j_theta(pro):
    """Angle that the angular momentum vector of the bin makes with respect to the xy-plane."""

    return np.arccos(pro['jz'] / pro['jtot'])


@Profile.profile_property
def j_phi(pro):
    """Angle that the angular momentum vector of the bin makes with the x-axis in the xy plane."""
    j_phi = np.zeros(pro.nbins)

    for i in range(pro.nbins):
        subs = pro.sim[pro.binind[i]]
        jx = (subs['j'][:, 0] * subs['mass']).sum() / pro['mass'][i]
        jy = (subs['j'][:, 1] * subs['mass']).sum() / pro['mass'][i]
        j_phi[i] = np.arctan(jy, jx)

    return j_phi


class InclinedProfile(Profile):
    """
    A profile object to be used with a snapshot inclined by some known angle to the xy plane.

    In addition to the SimSnap object, it also requires the angle to
    initialize.

    **Example:**

    >>> s = pynbody.load('sim')
    >>> pynbody.analysis.angmom.faceon(s)
    >>> s.rotate_x(60)
    >>> p = pynbody.profile.InclinedProfile(s, 60)

    """

    def _calculate_x(self, sim):
        # calculate an ellipsoidal radius
        return (sim['x'] ** 2 + (sim['y'] / np.cos(np.radians(self.angle))) ** 2) ** (1, 2)

    def __init__(self, sim, angle, load_from_file=False, ndim=2, type='lin', **kwargs):
        self.angle = angle

        Profile.__init__(
            self, sim, load_from_file=load_from_file, ndim=ndim, type=type, **kwargs)

        # define the minor axis
        self._properties['rbins_min'] = np.cos(
            np.radians(self.angle)) * self['rbins']


class VerticalProfile(Profile):
    """A profile class that uses the absolute value of the z coordinate for binning instead of a radial coordinate.
    """

    def _calculate_x(self, sim):
        return array.SimArray(np.abs(sim['z']), sim['z'].units)

    def __init__(self, sim, rmin, rmax, zmax, load_from_file=False, ndim=3, type='lin', **kwargs):
        """Creates a profile object that uses the absolute value of the z-coordinate for binning.

        Parameters
        ----------

        sim : pynbody.snapshot.simsnap.SimSnap
            The snapshot to make a profile from.

        rmin : str, float or pynbody.units.Unit
            Minimum radius for particle selection.

        rmax : str, float or pynbody.units.Unit
            Maximum radius for particle selection.

        zmax : str, float or pynbody.units.Unit
            Maximum height to consider (the upper edge of the binning range)

        ndim : int, optional
            If ndim=2, an edge-on projected profile is produced, i.e. density is in units of mass/length^2.
            If ndim=3 (default) a volume profile is made, i.e. density is in units of mass/length^3.

        type : str, optional
            The type of binning to use. Can be 'lin' (default), 'log', or 'equaln'.

        """
        if isinstance(rmin, str):
            rmin = units.Unit(rmin)
        if isinstance(rmax, str):
            rmax = units.Unit(rmax)
        if isinstance(zmax, str):
            zmax = units.Unit(zmax)
        self.rmin = rmin
        self.rmax = rmax
        self.zmax = zmax

        # create a snapshot that only includes the section of disk we're
        # interested in
        assert ndim in [2, 3]
        if ndim == 3:
            sub_sim = sim[
                pynbody.filt.Disc(rmax, zmax) & ~pynbody.filt.Disc(rmin, zmax)]
        else:
            sub_sim = sim[(pynbody.filt.BandPass('x', rmin, rmax) |
                           pynbody.filt.BandPass('x', -rmax, -rmin)) &
                          pynbody.filt.BandPass('z', -zmax, zmax)]

        Profile.__init__(
            self, sub_sim, load_from_file=load_from_file, ndim=ndim, type=type, **kwargs)

    def _setup_bins(self):
        Profile._setup_bins(self)

        dr = self.rmax - self.rmin

        if self.ndim == 2:
            self._binsize = (
                self['bin_edges'][1:] - self['bin_edges'][:-1]) * dr
        else:
            area = array.SimArray(
                np.pi * (self.rmax ** 2 - self.rmin ** 2), "kpc^2")
            self._binsize = (
                self['bin_edges'][1:] - self['bin_edges'][:-1]) * area


class QuantileProfile(Profile):
    """A profile object that returns requested quantiles instead of means in each bin.
    """

    def __init__(self, sim, q=(0.16, 0.50, 0.84), weights=None, load_from_file = False, ndim = 3, type = 'lin', **kwargs):
        """Creates a profile object that returns the requested quantiles for a given array in a given bin.

        Parameters
        ----------

        sim : pynbody.snapshot.simsnap.SimSnap
            The snapshot to make a profile from.

        q : list of floats, optional
            The quantiles that will be returned. Default is median with 1-sigma on either side.
            q can be of arbitrary length allowing the user to select any quantiles they desire.

        weights : pynbody.array.SimArray, optional
            What should be used to weight the quantile. A likely possibility is to use particle mass: ``sim['mass']``.
            The default is to weight by particle number, weights=None.

        **kwargs :
            Additional keyword arguments are passed onto the underlying :class:`Profile` constructor.

        """
        # create a snapshot that only includes the section of disk we're
        # interested in
        self.quantiles = q
        self.qweights = weights

        Profile.__init__(
            self, sim, load_from_file=load_from_file, ndim=ndim, type=type, **kwargs)

    def _get_profile(self, name):
        """Return the profile of a given kind"""
        x = name.split(",")
        if name in self._profiles:
            return self._profiles[name]

        elif name in list(self.sim.keys()) or name in self.sim.all_keys():
            self._profiles[name] = self._auto_profile(name)
            self._profiles[name].sim = self.sim
            return self._profiles[name]

        else:
            raise KeyError(name + " is not a valid QuantileProfile")

    def _auto_profile(self, name, dispersion=False, rms=False, median=False):
        result = np.zeros((self.nbins, len(self.quantiles)))
        for i in range(self.nbins):
            subs = self.sim[self.binind[i]]
            with self.sim.immediate_mode:
                name_array = subs[name].view(np.ndarray)
                sorted_array = sorted(name_array)
                topind = len(name_array) - 1
                if self.qweights is not None:
                    sorted_weights = self.qweights[np.argsort(name_array)]

            for iq, q in enumerate(self.quantiles):
                #import pdb; pdb.set_trace()
                if len(name_array) > 0:
                    if self.qweights is None:
                        ilow = int(np.floor(q * topind))
                        inc = q * topind - ilow
                        lowval = sorted_array[ilow]
                        hival = sorted_array[ilow + 1]
                        result[i, iq] = lowval + inc * (hival - lowval)
                    else:
                        cumw = np.cumsum(
                            sorted_weights) / np.sum(sorted_weights)
                        imin = min(
                            np.arange(len(sorted_array)), key=lambda x: abs(cumw[x] - q))
                        inc = q - cumw[imin]
                        lowval = sorted_array[imin]
                        if inc > 0:
                            nextval = sorted_array[imin + 1]
                        else:
                            if imin == 0:
                                nextval = lowval
                            else:
                                nextval = sorted_array[imin - 1]

                        result[i, iq] = lowval + inc * (nextval - lowval)
                        #if (result[i,iq] < 1e-6) : import pdb; pdb.set_trace()
                else:
                    result[i, iq] = np.nan
                    self['rbins'][i] = np.nan

        result = result.view(array.SimArray)
        result.units = self.sim[name].units
        result.sim = self.sim
        return result
