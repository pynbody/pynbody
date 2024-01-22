from typing import Any

import numpy as np
from numpy import typing as npt

from pynbody.halo import MonotonicHaloNumberMapper


class HaloParticleIndices:
    def __init__(self, /, halo_number_per_particle: npt.NDArray[int] = None, ignore_halo_number: int = None,
                 particle_ids: npt.NDArray[int] = None,
                 halo_number_mapper: MonotonicHaloNumberMapper = None,
                 boundaries: np.ndarray[(Any, 2), int] = None):
        """An IndexList represents abstract information about halo membership

        Either an halo_number_per_particle can be specified, which is an array of halo IDs for each particle
        (length should be the number of particles in the simulation). If this is the case, ignore_halo_number specifies
        a special value that indicates "no halo" (e.g. -1), or None if no such special value is defined. Note that
        ignore_halo_number must be either the smallest or largest value in halo_number_per_particle.

        Alternatively, an array of particle_ids can be specified, which is then sub-divided into the
        particle ids for each halo according to the boundaries array passed. The halos are then labelled
        by the specified halo_number_mapper, which maps halo numbers (arbitrary unique integers) to zero-based
        indices within the halo catalogue.

        To summarise, the parameters are:

        * halo_number_per_particle: array of halo numbers for each particle, length Npart
        * ignore_halo_number: a special value that indicates "no halo" (e.g. -1), or None if no such special value is defined
        OR
        * particle_ids: array of particle IDs (mutually exclusive with halo_number_per_particle), length Npart_in_halos
        * halo_number_mapper: object to map from halo numbers to zero-based indices within the halo catalogue
        * boundaries: array of indices in particle_ids where each halo starts, length Nhalo; must be monotonically
                      increasing

        """
        if halo_number_per_particle is not None:
            assert particle_ids is None
            assert halo_number_mapper is None
            assert boundaries is None
            self._setup_internals_from_halo_number_array(halo_number_per_particle, ignore_halo_number)

        else:
            assert particle_ids is not None
            assert halo_number_mapper is not None
            assert boundaries is not None
            self.particle_index_list = particle_ids
            self.halo_number_mapper = halo_number_mapper
            self.particle_index_list_boundaries = boundaries

            # should have a start and stop for each halo in the index list:
            assert self.particle_index_list_boundaries.shape == (len(self.halo_number_mapper), 2)


    def _setup_internals_from_halo_number_array(self, halo_number_per_particle: npt.NDArray[int], ignore_halo_number: int):
        self.particle_index_list = np.argsort(halo_number_per_particle, kind='mergesort')  # mergesort for stability
        halo_numbers = np.unique(halo_number_per_particle)

        start = np.searchsorted(halo_number_per_particle[self.particle_index_list], halo_numbers)
        stop = np.concatenate((start[1:], [len(self.particle_index_list)]))

        self.particle_index_list_boundaries = np.hstack((start[:,np.newaxis],stop[:,np.newaxis]))
        if ignore_halo_number is not None and ignore_halo_number in halo_numbers:
            if ignore_halo_number == halo_numbers[0]:
                self.particle_index_list_boundaries = self.particle_index_list_boundaries[1:]
                halo_numbers = halo_numbers[1:]
            elif ignore_halo_number == halo_numbers[-1]:
                self.particle_index_list_boundaries = self.particle_index_list_boundaries[:-1]
                halo_numbers = halo_numbers[:-1]
            else:
                raise ValueError("ignore_halo_number must be either the smallest or largest value in halo_number_per_particle")

        self.halo_number_mapper = MonotonicHaloNumberMapper(halo_numbers)

    def get_index_list_for_halo_number(self, halo_number):
        """Get the index list for the specified halo/object ID"""
        halo_offset = self.halo_number_mapper.number_to_index(halo_number)
        return self.particle_index_list[self._get_index_slice_for_halo_number(halo_offset)]

    def _get_index_slice_for_halo_number(self, obj_offset):
        """Get the slice for the index array corresponding to the object *offset* (not ID),
        i.e. the one whose index list starts at self.boundaries[obj_offset]"""
        ptcl_start, ptcl_end = self.particle_index_list_boundaries[obj_offset]
        return slice(ptcl_start, ptcl_end)

    def get_halo_number_per_particle(self, sim_length, fill_value=-1, dtype=int):
        """Return an array of object IDs, one per particle.

        Where a particle belongs to more than one object, the smallest object is favoured on the assumption that
        will identify the sub-halos etc in any reasonable case."""
        lengths = np.diff(self.particle_index_list_boundaries, axis=1).ravel()
        ordering = np.argsort(-lengths, kind='stable')

        id_array = np.empty(sim_length, dtype=dtype)
        id_array.fill(fill_value)

        for halo_index in ordering:
            object_id = self.halo_number_mapper.index_to_number(halo_index)
            indexing_slice = self._get_index_slice_for_halo_number(halo_index)
            id_array[self.particle_index_list[indexing_slice]] = object_id

        return id_array

    def __iter__(self):
        yield from self.halo_number_mapper

    def __getitem__(self, obj_number):
        return self.get_index_list_for_halo_number(obj_number)

    def __len__(self):
        return len(self.halo_number_mapper)
