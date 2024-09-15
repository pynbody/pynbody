"""
Implements functional forms of known profiles and code to fit them to simulation data.

At present only the NFW profile is implemented, but the code is designed to be easily extensible to other profiles.

"""

import abc

import numpy as np


class AbstractBaseProfile(abc.ABC):
    """
    Represents an analytic profile of a halo, and provides a method to fit the profile to data.

    The class is organised a dictionary, i.e. profile parameters of a given instance can be accessed through
    ``profile['...']`` and the available parameters are listed in ``profile.keys()``. The parameters are set at
    initialisation and cannot be changed afterwards.

    To define a new profile, create a new class inheriting from this base class and define your own profile_functional()
    method. The static version can be handy to avoid having to create and object every time.
    As a example, the NFW functional is implemented.

    A generic fitting function is provided (:meth:`fit`). Given a binned quantity as a function of radius, it
    uses least-squares to fit the given functional form to the data.

    """
    def __init__(self):
        self._parameters = dict()

    @abc.abstractmethod
    def profile_functional(self, radius):
        """Return the value of the profile at a given radius"""
        pass

    @classmethod
    @abc.abstractmethod
    def profile_functional_static(cls, radius, **kwargs):
        """Return the value of a profile with the specified parameters at a given radius"""
        pass

    @classmethod
    @abc.abstractmethod
    def log_profile_functional_static(cls, radius, **kwargs):
        """Return the logarithm of the value of the profile with the specified parameters at a given radius"""
        pass

    @abc.abstractmethod
    def get_dlogrho_dlogr(self, radius):
        """Return the logarithmic slope of the profile at a given radius"""
        pass

    @classmethod
    @abc.abstractmethod
    def get_dlogrho_dlogr_static(cls, radius, **kwargs):
        """Return the logarithmic slope of the profile with the specified parameters at a given radius"""
        pass

    @classmethod
    @abc.abstractmethod
    def jacobian_profile_functional_static(cls, radius, **kwargs):
        """Return the jacobian of the profile with respect to the parameters"""
        pass

    @abc.abstractmethod
    def get_enclosed_mass(self, radius_of_enclosure):
        """Return the mass enclosed within a given radius"""
        pass

    @classmethod
    def fit(cls, radial_data, profile_data, profile_err=None, use_analytical_jac=True, guess=None):
        """Fit the given profile using a least-squares method.

        Parameters
        ----------

        radial_data : array_like
            The central radius of the bins in which the profile data is measured

        profile_data : array_like
            The profile density values

        profile_err : array_like, optional
            The error on the profile data

        use_analytical_jac : bool
            Whether to use the analytical jacobian of the profile function. If False, finite differencing is used.

        guess : array_like, optional
            An initial guess for the parameters of the profile. If None, the initial guess is taken to be all ones,
            according to the underlying ``scipy.optimize.curve_fit`` function.
        """

        import scipy.optimize as so

        # Check data is not corrupted. Some are likely check in curve-fit already
        if np.isnan(radial_data).any() or np.isnan(profile_data).any():
            raise RuntimeError("Provided data contains NaN values")

        if np.count_nonzero(radial_data) != radial_data.size or np.count_nonzero(profile_data) != profile_data.size:
            raise RuntimeError("Provided data contains zeroes. This is likely to make the fit fail.")

        if radial_data.size != profile_data.size != profile_err.size:
            raise RuntimeError("Provided data arrays do not match in shape")

        if use_analytical_jac:
            use_analytical_jac = cls.jacobian_profile_functional_static

        profile_lower_bound = np.amin(profile_data)
        profile_upper_bound = np.amax(profile_data)

        radial_lower_bound = np.amin(radial_data)
        radial_upper_bound = np.amax(radial_data)

        try:
            parameters, cov = so.curve_fit(cls.profile_functional_static,
                                           radial_data,
                                           profile_data,
                                           sigma=profile_err,
                                           p0=guess,
                                           bounds=([profile_lower_bound, radial_lower_bound],
                                                   [profile_upper_bound, radial_upper_bound]),
                                           check_finite=True,
                                           jac=use_analytical_jac,
                                           method='trf',
                                           ftol=1e-10,
                                           xtol=1e-10,
                                           gtol=1e-10,
                                           x_scale=1.0,
                                           loss='linear',
                                           f_scale=1.0,
                                           max_nfev=None,
                                           diff_step=None,
                                           tr_solver=None,
                                           verbose=2)
        except so.OptimizeWarning as w:
            raise RuntimeError(str(w))

        if (guess is None and any(parameters == np.ones(parameters.shape))) or any(parameters == guess):
            raise RuntimeError("Fitted parameters are equal to their initial guess. This is likely a failed fit.")

        return parameters, cov

    def __getitem__(self, item):
        return self._parameters.__getitem__(item)

    def __setitem__(self, key, value):
        raise KeyError('Cannot change a parameter from the profile once set')

    def __delitem__(self, key):
        raise KeyError('Cannot delete a parameter from the profile once set')

    def __repr__(self):
        return "<" + self.__class__.__name__ + str(list(self.keys())) + ">"

    def keys(self):
        """Return the keys of the profile parameters"""
        return list(self._parameters.keys())


class NFWprofile(AbstractBaseProfile):
    """Represents a Navarro-Frenk-White (NFW) profile."""

    def __init__(self, halo_radius, scale_radius=None, density_scale_radius=None, concentration=None,
                 halo_mass=None):
        """Represents a Navarro-Frenk-White (NFW) profile.

        The profile can then be initialised either through *scale_radius* and central_density,
        or *halo_mass* and *concentration*.

        From one mode of initialisation, the derived parameters of the others are calculated, e.g. if you initialise
        with halo_mass + concentration, the scale_radius and central density will be derived.

        Parameters
        ----------

        halo_radius : float
            The outer boundary of the halo (r200m, r200c, rvir ... depending on definitions)

        scale_radius : float, optional
            The radius at which the slope is equal to -2

        density_scale_radius : float, optional
            1/4 of density at r=rs (normalisation).

        halo_mass : float, optional
            The mass enclosed inside the outer halo radius

        concentration : float, optional
            The outer_radius / scale_radius

        """

        super().__init__()

        self._halo_radius = halo_radius

        if scale_radius is None or density_scale_radius is None:
            if concentration is None or halo_mass is None or halo_radius is None:
                raise ValueError("You must provide concentration, virial mass"
                                 " if not providing the central density and scale_radius")
            else:
                self._parameters['concentration'] = concentration
                self._halo_mass = halo_mass

                self._parameters['scale_radius'] = self._derive_scale_radius()
                self._parameters['density_scale_radius'] = self._derive_central_overdensity()

        else:
            if concentration is not None or halo_mass is not None:
                raise ValueError("You can't provide both scale_radius+central_overdensity and concentration")

            self._parameters['scale_radius'] = scale_radius
            self._parameters['density_scale_radius'] = density_scale_radius

            self._parameters['concentration'] = self._derive_concentration()
            self._halo_mass = self.get_enclosed_mass(halo_radius)


    @classmethod
    def profile_functional_static(cls, radius, density_scale_radius, scale_radius):
        # Variable number of argument abstract methods only works because python is lazy with checking.
        # Is this a problem ?
        return density_scale_radius / ((radius / scale_radius) * (1.0 + (radius / scale_radius)) ** 2)

    @classmethod
    def jacobian_profile_functional_static(cls, radius, density_scale_radius, scale_radius):
        d_scale_radius = density_scale_radius * (3 * radius / scale_radius + 1) / (radius * (1 + radius / scale_radius) ** 3)
        d_central_density = 1 / ((radius / scale_radius) * (1 + radius / scale_radius) ** 2)
        return np.transpose([d_central_density, d_scale_radius])

    @classmethod
    def log_profile_functional_static(cls, radius, density_scale_radius, scale_radius):
        return np.log10(NFWprofile.profile_functional_static(radius, density_scale_radius, scale_radius))

    @classmethod
    def get_dlogrho_dlogr_static(cls, radius, scale_radius):
        return - (1.0 + 3.0 * radius / scale_radius) / (1.0 + radius / scale_radius)

    def profile_functional(self, radius):
        return NFWprofile.profile_functional_static(radius, self._parameters['density_scale_radius'],
                                                    self._parameters['scale_radius'])

    def get_enclosed_mass(self, radius_of_enclosure):
        # Eq 7.139 in M vdB W
        return self._parameters['density_scale_radius'] * self._parameters['scale_radius'] ** 3 \
               * NFWprofile._helper_function(self._parameters['concentration'] *
                                             radius_of_enclosure / self._halo_radius)

    def _derive_concentration(self):
        return self._halo_radius / self._parameters['scale_radius']

    def _derive_scale_radius(self):
        return self._halo_radius / self._parameters['concentration']

    def _derive_central_overdensity(self):
        return self._halo_mass / (NFWprofile._helper_function(self._parameters['concentration'])
                                  * self._parameters['scale_radius'] ** 3)

    def get_dlogrho_dlogr(self, radius):
        return NFWprofile.get_dlogrho_dlogr_static(radius, self._parameters['scale_radius'])

    @staticmethod
    def _helper_function(x):
        return 4 * np.pi * (np.log(1.0 + x) - x / (1.0 + x))
