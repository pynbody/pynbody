import numpy as np

from . import HaloCatalogue
from .details.number_mapping import create_halo_number_mapper
from .details.particle_indices import HaloParticleIndices


class HaloNumberCatalogue(HaloCatalogue):
    """A generic catalogue using an array of halo numbers, one per particle.

    This is the output format used by SKID, for example.
    """
    def __init__(self, sim, array='grp', ignore=None, **kwargs):
        """Construct a GrpCatalogue, extracting halos based on a simulation-wide integer array with their numbers

        *sim* - the SimSnap for which the halos will be constructed
        *array* - the name of the array which should be present, loadable or derivable across the simulation
        *ignore* - a special value indicating "no halo", or None if no such special value is defined
        """
        sim[array] # noqa - trigger lazy-loading and/or kick up a fuss if unavailable
        self._array = array
        self._ignore = ignore
        self._halo_numbers = np.unique(sim[array])
        number_mapper = create_halo_number_mapper(self._trim_array_for_ignore(self._halo_numbers))
        HaloCatalogue.__init__(self, sim, number_mapper=number_mapper)

    def _trim_array_for_ignore(self, array):
        assert len(array) == len(self._halo_numbers)
        if self._ignore is None:
            return array
        if self._ignore == self._halo_numbers[0]:
            return array[1:]
        elif self._ignore == self._halo_numbers[-1]:
            return array[:-1]
        else:
            raise ValueError(
                "ignore must be either the smallest or largest value in the array")


    def _get_all_particle_indices(self):

        halo_number_per_particle = self.base[self._array]

        particle_index_list = np.argsort(halo_number_per_particle, kind='mergesort')
        start = np.searchsorted(halo_number_per_particle[particle_index_list], self._halo_numbers)
        stop = np.concatenate((start[1:], [len(particle_index_list)]))

        particle_index_list_boundaries = self._trim_array_for_ignore(
            np.hstack((start[:, np.newaxis], stop[:, np.newaxis]))
        )

        return HaloParticleIndices(particle_ids = particle_index_list, boundaries = particle_index_list_boundaries)

    def _get_particle_indices_one_halo(self, halo_number):
        if halo_number == self._ignore:
            self._no_such_halo(halo_number)
        array = np.where(self.base[self._array] == halo_number)[0]
        if len(array) == 0:
            self._no_such_halo(halo_number)
        return array

    def _no_such_halo(self, i):
        raise KeyError(f"No such halo {i}")

    def get_group_array(self, family=None):
        if family is not None:
            return self.base[family][self._array]
        else:
            return self.base[self._array]

    @classmethod
    def _can_load(cls, sim, arr_name='grp'):
        if (arr_name in sim.loadable_keys()) or (arr_name in list(sim.keys())) :
            return True
        else:
            return False

class AmigaGrpCatalogue(HaloNumberCatalogue):
    """A catalogue of halos using Alyson Brooks' post-processed AHF output (turned into a SKID-like array)"""
    def __init__(self, sim):
        super().__init__(sim, array='amiga.grp')

    @classmethod
    def _can_load(cls, sim, arr_name='amiga.grp'):
        return super()._can_load(sim, arr_name)
