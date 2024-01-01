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

        possible_paths = [
            simpath.parent,
            simpath.parent / 'VR',
            simpath.parent.parent,
            simpath.parent.parent / 'VR'
        ]

        for basepath in possible_paths:
            if basepath.is_dir():
                for p in basepath.iterdir():
                    if p.is_dir() and f'{snapshot_num:04d}' in p.name:
                        possible_paths.append(p)
                    if 'catalog_groups.0' in p.name and p.is_file():
                        return basepath / str(p.name)[:-(len('.catalog_groups.0'))]
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

    def __init__(self, sim, vr_basename=None, include_unbound=False):
        super().__init__(sim)
        self._include_unbound = include_unbound
        if vr_basename is None:
            self._path = self._catalogue_path(sim)
        else:
            self._path = Path(vr_basename)

        if self._path is None:
            raise IOError("Could not find velociraptor catalogue. Try specifying vr_basename='path/to/output', where the velociraptor outputs are output.properties.0 etc")
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
        self._parents = self._grps['Parent_halo_ID'][:]
        _all_children_zero_based = np.arange(self._num_halos, dtype=np.int32)[self._parents != -1]
        self._all_children_ordered_by_parent = (
                _all_children_zero_based[np.argsort(self._parents[_all_children_zero_based])]
                + self._zero_offset
        )

        self._children_start_index = np.searchsorted(self._parents[self._all_children_ordered_by_parent-self._zero_offset],
                                                     np.arange(self._num_halos+self._zero_offset),
                                                     side='left')
        self._children_stop_index = np.concatenate((self._children_start_index[1:],
                                                    np.array([self._num_halos],
                                                             dtype=self._children_start_index.dtype)))

    def _get_halo_properties(self, i):
        i_zerobased = i - self._zero_offset
        parent = self._parents[i_zerobased]
        children = self._all_children_ordered_by_parent[self._children_start_index[i]:self._children_stop_index[i]]
        return {'parent': parent, 'children': children}

    def _get_halo(self, i):
        if i >= self._num_halos+self._zero_offset or i < self._zero_offset:
            raise IndexError(f"Halo index out of range (must be between {self._zero_offset}"
                             f" and {self._num_halos+self._zero_offset-1})")


        ptcl_fpos = self._get_particle_offsets_for_halo(i)
        if self._include_unbound:
            ptcl_fpos_unbound = self._get_particle_offsets_for_halo(i, unbound=True)
            ptcl_fpos = np.concatenate((ptcl_fpos, ptcl_fpos_unbound))

        h = Halo(i, self, self.base, index_array=ptcl_fpos, allow_family_sort=True)
        h.properties.update(self._get_halo_properties(i))

        return h

    def _get_particle_offsets_for_halo(self, i, unbound=False):
        i_zerobased = i - self._zero_offset
        if unbound:
            grps_hdf_array = self._grps['Offset_unbound']
            particle_ids_hdf_array = self._part_ids_unbound['Particle_IDs']
        else:
            grps_hdf_array = self._grps['Offset']
            particle_ids_hdf_array = self._part_ids['Particle_IDs']

        ptcl_start = grps_hdf_array[i_zerobased]

        if i_zerobased == self._num_halos - 1:
            ptcl_end = particle_ids_hdf_array.shape[0]
        else:
            ptcl_end = grps_hdf_array[i_zerobased + 1]

        ptcl_ids_this_halo = particle_ids_hdf_array[ptcl_start:ptcl_end]
        ptcl_fpos = self._iord_to_fpos[ptcl_ids_this_halo]
        return ptcl_fpos
