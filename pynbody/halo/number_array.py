"""Support for generic halo catalogues based on an array of halo numbers for each particle."""

import numpy as np

from . import HaloCatalogue
from .details.number_mapping import create_halo_number_mapper
from .details.particle_indices import HaloParticleIndices


class HaloNumberCatalogue(HaloCatalogue):
    """A generic catalogue using an array of halo numbers, one per particle.

    This is the output format used by SKID, for example.
    """

    # the array of halo numbers covers only the particles which have been loaded, so a halo which extends
    # outside a partially-loaded snapshot is indistinguishable from one which does not
    _can_determine_completeness = False

    def __init__(self, sim, array='grp', ignore=None, **kwargs):
        """Construct a GrpCatalogue, extracting halos based on a simulation-wide integer array with their numbers

        *sim* - the SimSnap for which the halos will be constructed
        *array* - the name of the array which should be present, loadable or derivable across the simulation
        *ignore* - a special value indicating "no halo", or None if no such special value is defined
        """
        sim[array] # noqa - trigger lazy-loading and/or kick up a fuss if unavailable
        self._array = array
        self._ignore = ignore
        # the counts come free from the same pass as the unique numbers, and save re-deriving the halo
        # boundaries from a sorted copy of the whole per-particle array later on
        self._halo_numbers, self._num_particles_per_halo_number = np.unique(sim[array], return_counts=True)
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

        # NB numpy offers no way to argsort into a pre-existing buffer, so when shared arrays are in use
        # _adopt_array has to copy the result once; there is no copy otherwise
        particle_index_list = self._adopt_array(np.argsort(halo_number_per_particle, kind='mergesort'))

        boundaries = self._array_factory((len(self._halo_numbers), 2), np.intp)
        np.cumsum(self._num_particles_per_halo_number, out=boundaries[:, 1])
        boundaries[1:, 0] = boundaries[:-1, 1]
        boundaries[:1, 0] = 0

        # NB a view, so that the shared memory allocated above is the memory actually in use
        particle_index_list_boundaries = self._trim_array_for_ignore(boundaries)

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

    def get_group_array(self, family=None, use_index=False, fill_value=-1):
        if family is not None:
            result = self.base[family][self._array]
        else:
            result = self.base[self._array]

        if use_index:
            result = np.copy(result)
            ignored = result == self._ignore
            not_ignored = ~ignored
            result[ignored] = fill_value
            result[not_ignored] = self.number_mapper.number_to_index(result[not_ignored])

        return result


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
