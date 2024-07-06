from typing import Any

import numpy as np
from numpy import typing as npt


class HaloParticleIndices:
    def __init__(self, particle_ids: npt.NDArray[int] = None, boundaries: np.ndarray[(Any, 2), int] = None):
        """An IndexList represents abstract information about halo membership

        * particle_ids: array of particle IDs (mutually exclusive with halo_number_per_particle), length Npart_in_halos
        * boundaries: a Nhalo x 2 array of start and stop indices for each halo in the particle_ids array

        NB throughout this class halo indices (zero-based, continuous integer numbers) are used, NOT halo numbers.
        For accessing halos by halo number, one must additionally use the HaloNumberMapper class to get the index
        before passing it in here.
        """

        self.particle_index_list = particle_ids
        self.particle_index_list_boundaries = boundaries


    def get_particle_index_list_for_halo(self, halo_index):
        """Get the index list for the specified halo index"""
        return self.particle_index_list[self._get_index_slice_for_halo(halo_index)]

    def _get_index_slice_for_halo(self, obj_offset):
        """Get the slice for the index array corresponding to the object *offset* (not ID),
        i.e. the one whose index list starts at self.boundaries[obj_offset]"""
        ptcl_start, ptcl_end = self.particle_index_list_boundaries[obj_offset]
        return slice(ptcl_start, ptcl_end)

    def get_halo_number_per_particle(self, sim_length, number_mapper, fill_value=-1, dtype=int):
        """Return an array of halo numbers, one per particle.

        Requires a HaloNumberMapper to map halo indices to halo numbers. If None is passed for the number_mapper,
        the halo indices are returned instead."""
        lengths = np.diff(self.particle_index_list_boundaries, axis=1).ravel()
        ordering = np.argsort(-lengths, kind='stable')

        id_array = np.empty(sim_length, dtype=dtype)
        id_array.fill(fill_value)

        if number_mapper is not None:
            halo_numbers = number_mapper.index_to_number(ordering)
        else:
            halo_numbers = ordering

        for halo_number, halo_index in zip(halo_numbers, ordering):
            indexing_slice = self._get_index_slice_for_halo(halo_index)
            id_array[self.particle_index_list[indexing_slice]] = halo_number

        return id_array

    def __len__(self):
        return len(self.particle_index_list_boundaries)
