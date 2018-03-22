"""

theoretical_profiles
====

Functional forms of common profiles (NFW as an example)

"""

from . import cosmology
import numpy as np
import abc


class AbstractBaseProfile:

    def __init__(self):
        self._parameters = dict()

    @abc.abstractclassmethod
    def profile_functional(self, radius):
        pass

    @staticmethod
    @abc.abstractclassmethod
    def profile_functional_static(radius, **kwargs):
        pass

    @classmethod
    def fit(cls, radial_data, profile_data, profile_err=None):
        import scipy.optimize as so
        # Check data is not corrupted. This is likely check in curve-fit already
        if np.isnan(radial_data).any() or np.isnan(profile_data).any():
            raise RuntimeError("Provided data contains NaN values")

        if np.count_nonzero(radial_data) != radial_data.size or np.count_nonzero(profile_data) != profile_data.size:
            raise RuntimeError("Provided data contains zeroes. This is likely to make the fit fail.")

        if radial_data.size != profile_data.size != profile_err.size:
            raise RuntimeError("Provided data arrays do not match in shape")

        radial_guess = np.mean(radial_data)
        profile_guess = np.mean(profile_data)

        profile_lower_bound = np.amin(profile_data)
        profile_upper_bound = np.amax(profile_data)

        radial_lower_bound = np.amin(radial_data)
        radial_upper_bound = np.amax(radial_data)

        try:
            parameters, cov = so.curve_fit(cls.profile_functional_static,
                                           radial_data,
                                           profile_data,
                                           sigma=profile_err,
                                           p0=[profile_guess, radial_guess],
                                           bounds=([profile_lower_bound, radial_lower_bound],
                                                   [profile_upper_bound, radial_upper_bound]),
                                           check_finite=True,
                                           method='trf')
        except so.OptimizeWarning as w:
            raise RuntimeError(str(w))

        if parameters[0] == profile_guess or parameters[1] == radial_guess:
            raise RuntimeError("Fitted parameters are equal to their guess. This is likely a failed fit.")

        return parameters, cov

    def __getitem__(self, item):
        return self._parameters.__getitem__(item)

    def __setitem__(self, key, value):
        raise KeyError('Cannot change a parameter from the profile once set')

    def __delitem__(self, key):
        raise KeyError('Cannot delete a parameter from the profile once set')

    def __repr__(self):
        return "<" + self.__class__.__name__ + str(self.keys()) + ">"

    def keys(self):
        return self._parameters.keys()


class NFWprofile(AbstractBaseProfile):

    def __init__(self, halo_radius, scale_radius=None, central_density=None, concentration=None,
                 halo_mass=None):

        super().__init__()

        self._halo_radius = halo_radius

        if scale_radius is None or central_density is None:
            if concentration is None or halo_mass is None or halo_radius is None:
                raise ValueError("You must provide concentration, virial mass"
                                 " if not providing the central density and scale_radius")
            else:
                self._parameters['concentration'] = concentration
                self._halo_mass = halo_mass

                self._parameters['scale_radius'] = self._derive_scale_radius()
                self._parameters['central_density'] = self._derive_central_overdensity()

        else:
            if concentration is not None or halo_mass is not None:
                raise ValueError("You can't provide both scale_radius+central_overdensity and concentration")

            self._parameters['scale_radius'] = scale_radius
            self._parameters['central_density'] = central_density

            self._parameters['concentration'] = self._derive_concentration()
            self._halo_mass = self.get_enclosed_mass(halo_radius)

    ''' Define static versions for use without initialising the class'''
    @staticmethod
    def profile_functional_static(radius, central_density, scale_radius):
        # Variable number of argument abstract methods only works because python is lazy with checking.
        # Is this a problem ?
        return central_density / ((radius / scale_radius) * (1.0 + (radius / scale_radius)) ** 2)

    @staticmethod
    def log_profile_functional_static(radius, central_density, scale_radius):
        return np.log(NFWprofile.profile_functional_static(radius, central_density, scale_radius))

    @staticmethod
    def get_dlogrho_dlogr_static(radius, scale_radius):
        return - (1.0 + 3.0 * radius / scale_radius) / (1.0 + radius / scale_radius)

    ''' Class methods'''
    def profile_functional(self, radius):
        return NFWprofile.profile_functional_static(radius, self._parameters['central_density'], self._parameters['scale_radius'])

    def get_enclosed_mass(self, radius_of_enclosure):
        # Eq 7.139 in M vdB W
        return self._parameters['central_density'] * self._parameters['scale_radius'] ** 3 \
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
