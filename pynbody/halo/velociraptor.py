from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from . import Halo, HaloCatalogue


class VelociraptorCatalogue(HaloCatalogue):
    """
    Velociraptor catalogue -- tested only with swift at present
    """

    _zero_offset = 1

    @classmethod
    def _catalogue_path(cls, sim) -> Optional[Path]:


        simpath = Path(sim.filename)

        filename = simpath.name

        # from filename snap_nnnn[.m].hdf5 get integer n
        try:
            snapshot_num = int(filename.split('_')[1].split('.')[0])
        except ValueError:
            return None

        halos_filename = f'catalogue_{snapshot_num:04d}/halos_{snapshot_num:04d}'

        possible_paths = [
            simpath / halos_filename,
            simpath / 'VR' / halos_filename,
            simpath.parent / halos_filename,
            simpath.parent / 'VR' / halos_filename,
        ]

        for p in possible_paths:
            if p.with_suffix('.properties.0').is_file():
                return p

        return None
    @classmethod
    def _can_load(cls, sim, **kwargs):
        path = cls._catalogue_path(sim)
        if path is None:
            return False

        for suffix in ['.catalog_groups.0', '.catalog_particles.0', '.properties.0']:
            if not (path.with_suffix(suffix).is_file()):
                return False

        return True

    def __init__(self, sim, include_unbound=False):
        super().__init__(sim)
        self._include_unbound = include_unbound
        self._path = self._catalogue_path(sim)
        self._grps = h5py.File(str(self._path.with_suffix('.catalog_groups.0')), 'r')
        self._part_ids = h5py.File(str(self._path.with_suffix('.catalog_particles.0')), 'r')
        if include_unbound:
            self._part_ids_unbound = h5py.File(str(self._path.with_suffix('.catalog_particles.unbound.0')), 'r')
        self._props = h5py.File(str(self._path.with_suffix('.properties.0')), 'r')

        assert self._grps['Num_of_files'][0] == 1, "Multi-file catalogues not supported at present"

        self._num_halos = self._grps['Num_of_groups'][0]
        self._init_iord_to_fpos()
        self._calculate_children()

    def __len__(self):
        return self._num_halos

    def _calculate_children(self):
        self._parents = self._grps['Parent_halo_ID'][:] - 1 # -1 to convert to zero-based indexing
        _all_children = np.arange(self._num_halos, dtype=np.int32)[self._parents != -2]
        self._all_children_ordered_by_parent = _all_children[np.argsort(self._parents[_all_children])]
        self._children_start_index = np.searchsorted(self._parents[self._all_children_ordered_by_parent],
                                                     np.arange(self._num_halos),
                                                     side='left')
        self._children_stop_index = np.concatenate((self._children_start_index[1:],
                                                    np.array([self._num_halos],
                                                             dtype=self._children_start_index.dtype)))

    def _get_halo_properties(self, i):
        i -= self._zero_offset
        parent = self._parents[i]
        children = self._all_children_ordered_by_parent[self._children_start_index[i]:self._children_stop_index[i]]
        return {'parent': parent, 'children': children}
    def _get_halo(self, i):
        assert i<self._num_halos

        ptcl_fpos = self._get_particle_offsets_for_halo(i)
        if self._include_unbound:
            ptcl_fpos_unbound = self._get_particle_offsets_for_halo(i, unbound=True)
            ptcl_fpos = np.concatenate((ptcl_fpos, ptcl_fpos_unbound))

        h = Halo(i, self, self.base, index_array=ptcl_fpos, allow_family_sort=True)
        h.properties.update(self._get_halo_properties(i))

        return h

    def _get_particle_offsets_for_halo(self, i, unbound=False):
        if unbound:
            grps_hdf_array = self._grps['Offset_unbound']
            particle_ids_hdf_array = self._part_ids_unbound['Particle_IDs']
        else:
            grps_hdf_array = self._grps['Offset']
            particle_ids_hdf_array = self._part_ids['Particle_IDs']

        ptcl_start = grps_hdf_array[i]

        if i == self._num_halos - 1:
            ptcl_end = particle_ids_hdf_array.shape[0]
        else:
            ptcl_end = grps_hdf_array[i + 1]

        ptcl_ids_this_halo = particle_ids_hdf_array[ptcl_start:ptcl_end]
        ptcl_fpos = self._iord_to_fpos[ptcl_ids_this_halo]
        return ptcl_fpos
