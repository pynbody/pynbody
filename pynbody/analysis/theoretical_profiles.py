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

    @abc.abstractclassmethod
    def fit(self, data, **kwargs):
        pass

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
            self._halo_mass = self.get_enclosed_mass(self._parameters['scale_radius'])

    ''' Define static versions for use without initialising the class'''
    @staticmethod
    def profile_functional_static(radius, central_density, scale_radius):
        # Variable number of argument abstract methods only works because python is lazy with checking.
        # Is this a problem ?
        return central_density / ((radius / scale_radius) * (1 + (radius / scale_radius)) ** 2)

    @staticmethod
    def log_profile_functional_static(radius, central_density, scale_radius):
        return np.log(NFWprofile.profile_functional_static(radius, central_density, scale_radius))

    @staticmethod
    def get_dlogrho_dlogr_static(radius, scale_radius):
        return - (1 + 3 * radius / scale_radius) / (1 + radius / scale_radius)

    ''' Class methods'''
    def profile_functional(self, radius):
        return NFWprofile.profile_functional_static(radius, self._parameters['central_density'], self._parameters['scale_radius'])

    def fit(self, data, **kwargs):
        pass

    def get_enclosed_mass(self, radius_of_enclosure):
        # Eq 7.139 in M vdB W
        return 4 * np.pi * self._parameters['scale_radius'] ** 3 \
               * NFWprofile._helper_function(self._parameters['concentration'] *
                                             radius_of_enclosure / self._parameters['scale_radius'])

    def _derive_concentration(self):
        return self._halo_radius / self._parameters['scale_radius']

    def _derive_scale_radius(self):
        return self._halo_radius / self._parameters['concentration']

    def _derive_central_overdensity(self):
        return self._halo_mass / (4 * np.pi * self._parameters['scale_radius'] ** 3 *
                                  NFWprofile._helper_function(self._parameters['concentration']))

    def get_dlogrho_dlogr(self, radius):
        return NFWprofile.get_dlogrho_dlogr_static(radius, self._parameters['scale_radius'])

    @staticmethod
    def _helper_function(x):
        return np.log10(1 + x) - x / (1 + x)
