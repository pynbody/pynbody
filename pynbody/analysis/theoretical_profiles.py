"""

theoretical_profiles
====

Functional forms of common profiles (NFW, Einasto, deVaucouleurs etc) and relationships between their parameters

"""

from . import cosmology
import numpy as np
import abc


class AbstractBaseProfile:

    def __init__(self):
        self._minimal_radius = 0
        self._maximal_radius = np.inf

        "Profile definition can depend on cosmology"
        self.cosmological_context = None

    @abc.abstractclassmethod
    def profile_functional(self, radius):
        raise NotImplementedError("Base class does not implement a given functional form")

    @staticmethod
    def profile_functional_static(radius):
        raise NotImplementedError("Base class does not implement a given functional form")

    @abc.abstractclassmethod
    def get_enclosed_value(self, radius_of_enclosure):
        raise NotImplementedError("Base class does can not derive")

    def __getattr__(self, item):
        pass

    def __getitem__(self, item):
        pass

    def __delattr__(self, item):
        pass

    def __delitem__(self, key):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass


class NFWprofile(AbstractBaseProfile):

    def __init__(self, context, scale_radius=None, central_density=None, concentration=None,
                 virial_mass=None):

        super(AbstractBaseProfile, self).__init__()

        if scale_radius is None or central_density is None:
            if concentration is None or virial_mass is None:
                raise ValueError("You must provide concentration, virial mass"
                                 "if not providing the central density and scale_radius")
            else:
                self.concentration = concentration
                self._virial_mass = virial_mass

                self.cosmological_context = context

                self.scale_radius = self._derive_scale_radius()
                self.central_density = self._derive_central_overdensity()

        else:
            if concentration is not None or virial_mass is not None:
                raise ValueError("You can't provide both scale_radius+central_overdensity and concentration")

            self.scale_radius = scale_radius
            self.central_density = central_density

            self.cosmological_context = context
            self.concentration = self._derive_concentration()
            self._virial_mass = self.get_enclosed_value(self.scale_radius)

        self._redshift = self.cosmological_context.properties['z']

        # For now, this assumes virial mass against rho_crit
        self._virial_radius = (self._virial_mass / cosmology.rho_crit(self.cosmological_context)) ** (1/3)



    @staticmethod
    def profile_functional_static(radius, central_density, scale_radius):
        return central_density / ((radius / scale_radius) * (1 + (radius / scale_radius)) ** 2)

    def profile_functional(self, radius):
        return NFWprofile.profile_functional_static(radius, self.central_density, self.scale_radius)

    def _derive_concentration(self):
        return self._virial_radius / self.scale_radius

    def _derive_scale_radius(self):
        return self._virial_radius / self.concentration

    def _derive_central_overdensity(self):
        return self._virial_mass / (4 * np.pi * self.scale_radius ** 3 *
                                    NFWprofile._helper_function(self.concentration))

    def get_enclosed_value(self, radius_of_enclosure):
        # Eq 7.139
        return 4 * np.pi * self.scale_radius ** 3 * NFWprofile._helper_function(self.concentration
                                                                                * radius_of_enclosure / self.scale_radius)

    def get_dlogrho_dlogr(self):
        # Theoretical slope
        pass

    @staticmethod
    def _helper_function(x):
        return np.log10(1 + x) - x / (1 + x)
