from typing import Any

import numpy as np
from numpy import typing as npt


class IncompleteHaloError(RuntimeError):
    """Raised when a halo cannot be constructed because some of its particles are not in the snapshot.

    This normally means the snapshot has been partially loaded (e.g. only a region, or a subset of families,
    was read from disk) while the halo catalogue still refers to the particles that were left out."""

    def __init__(self, halo_number=None, num_missing_particles=None):
        self.halo_number = halo_number
        self.num_missing_particles = num_missing_particles

        halo_description = "This halo" if halo_number is None else f"Halo {halo_number}"
        if num_missing_particles is None:
            num_description = "some of its particles are"
        elif num_missing_particles == 1:
            num_description = "one of its particles is"
        else:
            num_description = f"{num_missing_particles} of its particles are"

        super().__init__(f"{halo_description} cannot be loaded because {num_description} not present in the "
                         f"snapshot. This normally means the snapshot has been partially loaded. Use the "
                         f"catalogue's complete_keys() method to get the halo numbers which can be loaded.")


class HaloParticleIndices:
    def __init__(self, particle_ids: npt.NDArray[int] = None, boundaries: np.ndarray[(Any, 2), int] = None,
                 num_missing_particles: npt.NDArray[int] = None):
        """An IndexList represents abstract information about halo membership

        * particle_ids: array of particle IDs (mutually exclusive with halo_number_per_particle), length Npart_in_halos
        * boundaries: a Nhalo x 2 array of start and stop indices for each halo in the particle_ids array
        * num_missing_particles: an optional array of length Nhalo, giving the number of particles that the halo
          finder assigns to each halo but which are not present in the snapshot (normally because it has been
          partially loaded). Such particles are simply absent from particle_ids; halos with a non-zero count are
          regarded as incomplete, and attempting to access them raises an IncompleteHaloError.

        NB throughout this class halo indices (zero-based, continuous integer numbers) are used, NOT halo numbers.
        For accessing halos by halo number, one must additionally use the HaloNumberMapper class to get the index
        before passing it in here.
        """

        self.particle_index_list = particle_ids
        self.particle_index_list_boundaries = boundaries
        self.num_missing_particles = num_missing_particles

    def get_num_missing_particles_for_halo(self, halo_index) -> int:
        """Get the number of particles that are missing from the snapshot for the specified halo index"""
        if self.num_missing_particles is None:
            return 0
        return int(self.num_missing_particles[halo_index])

    def is_halo_incomplete(self, halo_index) -> bool:
        """Return True if particles are missing from the snapshot for the specified halo index"""
        return self.get_num_missing_particles_for_halo(halo_index) > 0

    def get_complete_mask(self) -> npt.NDArray[bool]:
        """Return a boolean mask, in halo index order, of the halos with no missing particles"""
        if self.num_missing_particles is None:
            return np.ones(len(self), dtype=bool)
        return np.asarray(self.num_missing_particles) == 0

    def get_particle_index_list_for_halo(self, halo_index, halo_number=None, allow_incomplete=False):
        """Get the index list for the specified halo index

        Raises an IncompleteHaloError if particles are missing from the snapshot for this halo, unless
        allow_incomplete is True, in which case only the particles that are present are returned.

        The halo_number, if provided, is used only to generate a more helpful error message."""
        if not allow_incomplete and self.is_halo_incomplete(halo_index):
            raise IncompleteHaloError(halo_number, self.get_num_missing_particles_for_halo(halo_index))
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
