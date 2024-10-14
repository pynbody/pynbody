"""
Implements functional forms of known density profiles and code to fit them to simulation data.

At present only the NFW profile is implemented, but the code is designed to be easily extensible to other profiles.

"""

import abc

import numpy as np

from .. import util


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

    @classmethod
    @abc.abstractmethod
    def parameter_bounds(self, r_values, rho_values):
        """Return bounds on the parameter values for the profile fit

        Parameters
        ----------

        r_values : array_like
            The radii at which the profile is measured

        rho_values : array_like
            The density values of the profile


        Returns
        -------

        bounds : tuple[array_like]
            A 2-tuple containing lower and upper bounds respectively for the parameters of the profile

        """
        pass

    @abc.abstractmethod
    def logarithmic_slope(self, radius):
        """Return the logarithmic slope of the profile, d ln rho / d ln r, at a given radius"""
        pass

    @abc.abstractmethod
    def enclosed_mass(self, radius):
        """Return the mass, M(r), enclosed within a given radius"""
        pass

    @classmethod
    def fit(cls, radial_data, profile_data, profile_err=None, use_analytical_jac=True, guess=None, verbose=0,
            return_profile = True):
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

        verbose : int
            The verbosity level to pass to the underlying ``scipy.optimize.curve_fit`` function.

        return_profile : bool
            Whether to return the profile object or just the parameters

        Returns
        -------

        fitted_profile : array_like | AbstractBaseProfile
            If return_profile is True, the fitted profile object. Otherwise, the fitted parameters.

        cov : array_like
            The covariance matrix of the fit. The diagonal elements are the variance of the parameters.

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
            def jacobian_wrapper(radius, *args):
                return cls(*args).jacobian(radius)
            jac = jacobian_wrapper
        else:
            jac = '3-point'

        lower_bounds, upper_bounds = cls.parameter_bounds(radial_data, profile_data)

        def profile_wrapper(radius, *args):
            return cls(*args)(radius)

        if guess is None:
            guess = (np.asarray(upper_bounds) + np.asarray(lower_bounds)) / 2.0


        try:
            parameters, cov = so.curve_fit(profile_wrapper,
                                           radial_data,
                                           profile_data,
                                           sigma=profile_err,
                                           p0=guess,
                                           bounds=(lower_bounds, upper_bounds),
                                           check_finite=True,
                                           jac=jac,
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
                                           verbose=verbose)
        except so.OptimizeWarning as w:
            raise RuntimeError(str(w))

        if (guess is None and any(parameters == np.ones(parameters.shape))) or any(parameters == guess):
            raise RuntimeError("Fitted parameters are equal to their initial guess. This is likely a failed fit.")

        if return_profile:
            return cls(*parameters), cov
        else:
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

    def __contains__(self, item):
        return item in self._parameters


class NFWProfile(AbstractBaseProfile):
    """Represents a Navarro-Frenk-White (NFW) profile."""

    def __init__(self, density_scale_radius=None, scale_radius=None, halo_radius=None, concentration=None,
                 halo_mass=None):
        """Represents a Navarro-Frenk-White (NFW) profile.

        The profile can then be initialised through one of the following combination of parameters:

        * *scale_radius*, *density_scale_radius* and optionally *halo_radius*;
        * *halo_radius*, *concentration* and *density_scale_radius*;
        * *halo_radius*, *concentration* and *halo_mass*.

        From one mode of initialisation, the derived parameters of the others are calculated, e.g. if you initialise
        with halo_mass + concentration, the scale_radius and central density will be derived. The exception is if
        you initialise with *scale_radius* + *density_scale_radius* without *halo_radius*.

        Units may be passed into the parameters by using scalar arrays.

        Parameters
        ----------

        scale_radius : float | array-like, optional
            The radius at which the slope is equal to -2

        density_scale_radius : float | array-like, optional
            1/4 of density at r=rs (normalisation).

        halo_mass : float | array-like, optional
            The mass enclosed inside the outer halo radius

        halo_radius : float | array-like
            The outer boundary of the halo (r200m, r200c, rvir ... depending on definitions)

        concentration : float | array-like, optional
            The outer_radius / scale_radius

        """

        super().__init__()

        if scale_radius is not None and density_scale_radius is not None and concentration is None and halo_mass is None:
            self._parameters['scale_radius'] = scale_radius
            self._parameters['density_scale_radius'] = density_scale_radius
            if halo_radius is not None:
                self._parameters['halo_radius'] = halo_radius
                self._parameters['concentration'] = halo_radius / scale_radius
                self._parameters['halo_mass'] = self.enclosed_mass(halo_radius)
        elif (halo_radius is not None and concentration is not None and density_scale_radius is not None
              and halo_mass is None):
            self._parameters['halo_radius'] = halo_radius
            self._parameters['concentration'] = concentration
            self._parameters['density_scale_radius'] = density_scale_radius
            self._parameters['scale_radius'] = halo_radius / concentration
            self._parameters['halo_mass'] = self.enclosed_mass(halo_radius)
        elif (halo_radius is not None and concentration is not None and halo_mass is not None
              and density_scale_radius is None):
            self._parameters['halo_radius'] = halo_radius
            self._parameters['concentration'] = concentration
            self._parameters['scale_radius'] = halo_radius / concentration
            self._parameters['halo_mass'] = halo_mass
            self._parameters['density_scale_radius'] = self._derive_central_overdensity()
        else:
            raise ValueError("Invalid combination of parameters for initializing NFWProfile.")

    @classmethod
    def parameter_bounds(cls, r_values, rho_values):
        profile_lower_bound = np.amin(rho_values)
        profile_upper_bound = np.amax(rho_values)

        radial_lower_bound = np.amin(r_values)
        radial_upper_bound = np.amax(r_values)

        return ([profile_lower_bound, radial_lower_bound], [profile_upper_bound, radial_upper_bound])

    def jacobian(self, radius):
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']

        d_scale_radius = density_scale_radius * (3 * radius / scale_radius + 1) / (
                    radius * (1 + radius / scale_radius) ** 3)
        d_central_density = 1 / ((radius / scale_radius) * (1 + radius / scale_radius) ** 2)
        return np.transpose([d_central_density, d_scale_radius])

    def __call__(self, radius):
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']
        return density_scale_radius / ((radius / scale_radius) * (1.0 + (radius / scale_radius)) ** 2)

    def enclosed_mass(self, radius):
        # Eq 7.139 in M vdB W
        return self._parameters['density_scale_radius'] * self._parameters['scale_radius'] ** 3 \
               * self._integral(radius / self._parameters['scale_radius'])

    def _derive_concentration(self):
        return self._parameters['halo_radius'] / self._parameters['scale_radius']

    def _derive_scale_radius(self):
        return self._parameters['halo_radius'] / self._parameters['concentration']

    def _derive_central_overdensity(self):
        return self._parameters['halo_mass'] / (self._parameters['scale_radius']**3 *
                                                self._integral(self._parameters['concentration']))

    def logarithmic_slope(self, radius):
        scale_radius = self._parameters['scale_radius']
        return - (1.0 + 3.0 * radius / scale_radius) / (1.0 + radius / scale_radius)

    @staticmethod
    def _integral(x):
        return 4 * np.pi * (np.log(1.0 + x) - x / (1.0 + x))


@util.deprecated("Deprecated alias for NFWProfile. Use NFWProfile instead.")
def NFWprofile(*args, **kwargs):
    return NFWProfile(*args, **kwargs)
