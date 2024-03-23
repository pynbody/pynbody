from __future__ import annotations

import functools
import warnings
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from numpy.typing import NDArray

from .. import array, units, util
from . import HaloCatalogue, HaloParticleIndices
from .details import number_mapping


class VelociraptorCatalogue(HaloCatalogue):
    """
    Velociraptor catalogue -- tested only with swift at present
    """

    @classmethod
    def _catalogue_path(cls, sim) -> Path | None:


        simpath = Path(sim.filename)

        filename = simpath.name

        # from filename snap_nnnn[.m].hdf5 get integer n
        try:
            snapshot_num = int(filename.split('_')[1].split('.')[0])
        except (ValueError, IndexError):
            snapshot_num = None

        possible_paths = [
            simpath.parent,
            simpath.parent / 'VR',
            simpath.parent.parent,
            simpath.parent.parent / 'VR'
        ]

        for basepath in possible_paths:
            if basepath.is_dir():
                for p in basepath.iterdir():
                    if snapshot_num and p.is_dir() and f'{snapshot_num:04d}' in p.name:
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

        self._include_unbound = include_unbound
        if vr_basename is None:
            self._path = self._catalogue_path(sim)
        else:
            self._path = Path(vr_basename)

        if self._path is None:
            raise OSError("Could not find velociraptor catalogue. Try specifying vr_basename='path/to/output', where the velociraptor outputs are output.properties.0 etc")
        self._grps = h5py.File(str(self._path.with_suffix('.catalog_groups.0')), 'r')
        self._part_ids = h5py.File(str(self._path.with_suffix('.catalog_particles.0')), 'r')
        self._properties_hdf_file = h5py.File(str(self._path.with_suffix('.properties.0')), 'r')

        if include_unbound:
            self._part_ids_unbound = h5py.File(str(self._path.with_suffix('.catalog_particles.unbound.0')), 'r')
        self._props = h5py.File(str(self._path.with_suffix('.properties.0')), 'r')

        assert self._grps['Num_of_files'][0] == 1, "Multi-file catalogues not supported at present"

        self._num_halos = self._grps['Num_of_groups'][0]

        super().__init__(sim, number_mapping.create_halo_number_mapper(self._properties_hdf_file['ID']))

        self._setup_property_keys()
        self._setup_property_units()
        self._init_iord_to_fpos()
        self._calculate_children()

    def _setup_property_keys(self):
        self._property_keys = []
        for k in self._properties_hdf_file.keys():
            if len(self._properties_hdf_file[k]) == self._num_halos:
                self._property_keys.append(k)

    def _setup_property_units(self):
        unitinfo = self._props['UnitInfo'].attrs
        comoving = int(unitinfo['Comoving_or_Physical'])!=0
        length = units.Unit("kpc")*float(unitinfo['Length_unit_to_kpc'])
        if comoving:
            # https://github.com/pelahi/VELOCIraptor-STF/blob/6f4b760ef5043b959a922a8e7ae453fd0a9f988f/src/io.cxx#L1591
            length *= units.Unit("a h^-1")
        mass = units.Unit("Msol")*float(unitinfo['Mass_unit_to_solarmass'])
        vel = units.Unit("km s^-1")*float(unitinfo['Velocity_unit_to_kms'])
        time = length / vel
        # the above is actually a guess - but don't think the Dimension_Time is anyway ever used?
        # if any time dimension is found, a warning will be issued

        self._property_units = []
        dims_attr_names = ("Dimension_Length", "Dimension_Mass", "Dimension_Time", "Dimension_Velocity")
        dims_units = (length, mass, time, vel)
        for k in self._property_keys:
            powers = [util.fractions.Fraction.from_float(float(self._props[k].attrs[d])).limit_denominator()
                      for d in dims_attr_names]
            if powers[2] != 0.0:
                warnings.warn("Time dimension found in property %s, but no time conversion factor is stored in the velociraptor output. Guessing an appropriate conversion." % k)
            final_unit = functools.reduce(lambda x, y: x * y,
                                     [u**p for u, p in zip(dims_units, powers)])
            self._property_units.append(final_unit)



    def _calculate_children(self):
        self._parents = self._grps['Parent_halo_ID'][:]
        _all_children_zero_based = np.arange(self._num_halos, dtype=np.int32)[self._parents != -1]
        self._all_children_ordered_by_parent = self.number_mapper.index_to_number(
                _all_children_zero_based[np.argsort(self._parents[_all_children_zero_based])]
        )

        self._children_start_index = np.searchsorted(self._parents[
                                                         self.number_mapper.number_to_index(self._all_children_ordered_by_parent)
                                                     ],
                                                     self.number_mapper.all_numbers,
                                                     side='left')
        self._children_stop_index = np.concatenate((self._children_start_index[1:],
                                                    np.array([self._num_halos],
                                                             dtype=self._children_start_index.dtype)))

    def get_properties_one_halo(self, halo_number) -> dict:
        i_zerobased = self.number_mapper.number_to_index(halo_number)
        properties = {k: self._props[k][i_zerobased] * u
                      for k, u in zip(self._property_keys, self._property_units)}
        parent = self._parents[i_zerobased]
        children = self._all_children_ordered_by_parent[self._children_start_index[i_zerobased]:self._children_stop_index[i_zerobased]]
        properties.update({'parent': parent, 'children': children})
        return properties

    def get_properties_all_halos(self, with_units=True) -> dict:
        if with_units:
            all_properties_hdf_file = {k: array.SimArray(self._properties_hdf_file[k][:], u)
                              for k, u in zip(self._property_keys, self._property_units)}
        else:
            all_properties_hdf_file = {k: self._properties_hdf_file[k] for k in self._property_keys}

        all_properties_hdf_file.update(
               {'parent': self._parents,
                'children': [self._all_children_ordered_by_parent[start:end]
                             for start, end in zip(self._children_start_index, self._children_stop_index)]}
        )
        return all_properties_hdf_file


    def _get_particle_indices_one_halo(self, halo_number) -> NDArray[int]:
        i_zerobased = self.number_mapper.number_to_index(halo_number)
        ptcl_fpos = self.__get_particle_indices_from_halo_index(i_zerobased, False)

        if self._include_unbound:
            ptcl_fpos_unbound =  self.__get_particle_indices_from_halo_index(i_zerobased, True)
            ptcl_fpos = np.concatenate((ptcl_fpos, ptcl_fpos_unbound))

        return np.sort(ptcl_fpos)

    def _get_all_particle_indices(self) -> HaloParticleIndices | tuple[np.ndarray, np.ndarray]:
        particle_ids_hdf_array = self._part_ids['Particle_IDs']
        offsets = np.concatenate((self._grps['Offset'][:], [particle_ids_hdf_array.shape[0]]),
                                 dtype=np.intp)
        boundaries = np.vstack((offsets[:-1], offsets[1:])).T

        if self._include_unbound:
            num_ids = particle_ids_hdf_array.shape[0] + self._part_ids_unbound['Particle_IDs'].shape[0]
            offsets_unbound = np.concatenate((self._grps['Offset_unbound'][:],
                                              [self._part_ids_unbound['Particle_IDs'].shape[0]]),
                                             dtype=np.intp)
            boundaries_unbound = np.vstack((offsets_unbound[:-1], offsets_unbound[1:])).T
            output_boundaries = boundaries + boundaries_unbound
            particle_ids = np.empty(num_ids, dtype=np.intp)

            for (a,b), (a_unbound, b_unbound), (a_out, b_out) in zip(boundaries, boundaries_unbound, output_boundaries):
                particle_ids[a_out:b_out] = np.sort(self._iord_to_fpos.map_ignoring_order(
                    np.concatenate((self._part_ids['Particle_IDs'][a:b],
                                    self._part_ids_unbound['Particle_IDs'][a_unbound:b_unbound])))
                )
        else:
            output_boundaries = boundaries

            particle_ids = np.empty(particle_ids_hdf_array.shape[0], dtype=np.intp)

            for a,b in boundaries:
                particle_ids[a:b] = np.sort(self._iord_to_fpos.map_ignoring_order(self._part_ids['Particle_IDs'][a:b]))

        return particle_ids, output_boundaries


    def __get_particle_indices_from_halo_index(self, i_zerobased, unbound):
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
        ptcl_fpos = self._iord_to_fpos.map_ignoring_order(ptcl_ids_this_halo)
        return ptcl_fpos
