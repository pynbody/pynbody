"""HBT+ halo catalogue support.

.. _hbt_plus_parent_groups:

HBT+ and parent groups
----------------------

HBT+ identifies halos, not the parent groups (using the SubFind terminology). As a result, it may be used alongside a
parent group catalogue. The *pynbody* reader can present a unified interface to the combined hierarchy, presenting
groups from the parent catalogue as the top-level objects, with the HBT+ halos as children of these groups, and
HBT+ subhalos as children of the HBT+ halos.

For example:

.. ipython::

  In [1]: import pynbody

  In [2]: s = pynbody.load('testdata/gadget4_subfind_HBT/snapshot_034.hdf5')

  In [3]: hbtplus_halos = s.halos(priority=["HBTPlusCatalogue"])

  In [4]: subfind_groups = s.halos(priority=["Gadget4SubfindHDFCatalogue"])

  In [5]: combined_catalogue = hbtplus_halos.with_groups_from(subfind_groups)

  In [6]: combined_catalogue[0]
  Out[6]: <SimSnap "testdata/gadget4_subfind_HBT/snapshot_034.hdf5:halo_0" len=307386>

We can tell that this is actually a SubFind group by inspecting its properties:

.. ipython::

  In [7]: combined_catalogue[0].properties['GroupMass'] # <-- a SubFind property
  Out[7]: Unit("7.80e+44 g h**-1")

  In [8]: subfind_groups[0].properties['GroupMass'] # <-- the same property accessed directly from SubFind
  Out[8]: Unit("7.80e+44 g h**-1")

The ``subhalos`` attribute of each halo in the ``combined_catalogue`` will return the HBT+ subhalos that are children of
the corresponding SubFind group:

.. ipython::

  In [14]: combined_catalogue[0].subhalos[0].properties['TrackId'] # <-- an HBT+ property
  Out[14]: 54

Naturally, not only the properties but the particle information is available in the combined catalogue.


"""
from __future__ import annotations

import pathlib
import re

import h5py
import numpy as np
from numpy.typing import NDArray

from . import HaloCatalogue, HaloParticleIndices
from .details import number_mapping


class HBTPlusCatalogue(HaloCatalogue):
    """A class to represent a HBT+ halo catalogue."""
    def __init__(self, sim, halo_numbers=None, filename=None):
        """Initialize a HBTPlusCatalogue object.

        Parameters
        ----------
        sim : SimSnap
            The simulation snapshot to which this catalogue applies.

        halo_numbers : str, optional
            How to number the halos. If None (default), use a zero-based indexing.

            * If ``track``, use the TrackId from the catalogue.
            * If ``length-order``, order by Nbound (descending), similar to the AHF option of the same name.

        filename : str, optional
            The filename of the HBTPlus catalogue. If the file is spanned across multiple outputs (e.g.
            ``path/to/SubSnap_034.0.hdf5``, ``SubSnap_034.1.hdf5``, etc.), pass the filename of the first file or
            the common prefix (``path/to/SubSnap_034``). If None (default), attempt to find the file automatically.
        """
        if filename is None:
            filename = self._infer_hbt_filename(sim)
        else:
            filename = self._map_user_filename_to_file_0(filename)

        self._file = h5py.File(filename, 'r')

        num_halos = int(self._file["NumberOfSubhalosInAllFiles"][0])
        if int(self._file["NumberOfFiles"][0])>1:
            from ..util import hdf_vds
            filename = str(filename)[:-7]
            all_files = [f"{filename}.{num}.hdf5" for num in range(int(self._file["NumberOfFiles"][0]))]
            maker = hdf_vds.HdfVdsMaker(all_files)
            self._file = maker.get_temporary_hdf_vfile()

        self._trackid_number_mapper = number_mapping.create_halo_number_mapper(self._file["Subhalos"]["TrackId"])

        if halo_numbers is None:
            number_mapper = number_mapping.SimpleHaloNumberMapper(0, num_halos)
        elif halo_numbers == 'track':
            number_mapper = self._trackid_number_mapper
        elif halo_numbers == 'length-order':
            osort = np.argsort(-self._file["Subhalos"]["Nbound"][:], kind='stable')
            number_mapper = number_mapping.NonMonotonicHaloNumberMapper(osort, ordering=True, start_index=0)
        else:
            raise ValueError(f"Invalid value for halo_numbers: {halo_numbers}")
        super().__init__(sim, number_mapper)

        self._setup_parents()

    def __del__(self):
        if hasattr(self, "_file"):
            self._file.close()

    @classmethod
    def _infer_hbt_filename(cls, sim):
        sim_filename: pathlib.Path  = sim.filename
        try:
            snap_num = int(re.search(r'_(\d+)', sim_filename.name).group(1))
        except AttributeError:
            raise FileNotFoundError(f'Could not infer HBT filename from {sim_filename}. '
                                    f'Try passing hbt_filename explicitly.') from None

        candidate_paths = [sim_filename.with_name(f'SubSnap_{snap_num:03d}.0.hdf5'),
                            sim_filename.parent / f'{snap_num:03d}' / f'SubSnap_{snap_num:03d}.0.hdf5']

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path

        raise FileNotFoundError(f'Could not find HBTPlus catalogue for {sim_filename}. Try passing hbt_filename explicitly.')

    @classmethod
    def _map_user_filename_to_file_0(cls, filename):
        filename = pathlib.Path(filename)
        if not filename.exists():
            dot_hdf5 = filename.parent / (filename.name + '.hdf5')
            dot_0_hdf5 = filename.parent / (filename.name + '.0.hdf5')
            if dot_hdf5.exists():
                filename = dot_hdf5
            elif dot_0_hdf5.exists():
                filename = dot_0_hdf5
        return filename

    def _setup_parents(self):
        parents = np.empty(len(self), dtype=np.intp)
        parents.fill(-1)
        for i in range(len(self)):
            for child in self._trackid_number_mapper.number_to_index(self._file["NestedSubhalos"][i]):
                parents[child] = self.number_mapper.index_to_number(i)

        self._parents = parents

    def _map_trackid_to_pynbody_number(self, trackid: NDArray[int]) -> NDArray[int]:
        if self.number_mapper is self._trackid_number_mapper:
            return trackid
        else:
            return self.number_mapper.index_to_number(
                self._trackid_number_mapper.number_to_index(
                  trackid
                )
            )

    def _get_particle_indices_one_halo(self, halo_number) -> NDArray[int]:
        self._init_iord_to_fpos()

        iords = self._file["SubhaloParticles"][self.number_mapper.number_to_index(halo_number)]
        return np.sort(self._iord_to_fpos.map_ignoring_order(iords))

    def _get_all_particle_indices(self) -> HaloParticleIndices | tuple[np.ndarray, np.ndarray]:
        self._init_iord_to_fpos()
        indices = np.empty(self._file["Subhalos"]["Nbound"].sum(),
                           dtype=self._file["SubhaloParticles"][0].dtype)
        boundaries = np.empty((len(self._file["SubhaloParticles"]), 2),
                              dtype=np.intp)
        start = 0
        for i, halo_parts in enumerate(self._file['SubhaloParticles']):
            end = start + len(halo_parts)
            indices[start:end] = np.sort(self._iord_to_fpos.map_ignoring_order(halo_parts))
            boundaries[i] = (start,end)
            start = end
        return indices, boundaries

    def with_groups_from(self, other: HaloCatalogue) -> HaloCatalogue:
        """Return a new catalogue that combines an HBT+ halo catalogue with a parent group catalogue.

        For example:

        >>> grps = hbt_cat.with_groups_from(subfind_cat)
        >>> grps[0] # -> returns subfind_cat[0]
        >>> grps[0].properties['children'] # -> returns relevant halo numbers in hbt_cat
        >>> grps[0].subhalos # -> returns HBT+ subhalos that are children of subfind_cat[0]

        See also :ref:`hbt_plus_parent_groups`.
        """
        return HBTPlusCatalogueWithGroups(self, other)

    def get_properties_one_halo(self, halo_number) -> dict:
        index = self.number_mapper.number_to_index(halo_number)
        result = {}
        for k in self._file["Subhalos"].dtype.names:
            result[k] = self._file["Subhalos"][k][index]
        result['children'] = self._get_children_one_halo(index)
        result['parent'] = self._parents[index]
        return result

    def get_properties_all_halos(self, with_units=True) -> dict:
        result = {}
        for k in self._file["Subhalos"].dtype.names:
            result[k] = np.asarray(self._file["Subhalos"][k])
        result['children'] = [self._get_children_one_halo(i) for i in range(len(self))]
        result['parent'] = self._parents
        return result

    def _get_children_one_halo(self, index) -> NDArray[int]:
        return self._map_trackid_to_pynbody_number(self._file["NestedSubhalos"][index])



    @classmethod
    def _can_load(cls, sim, halo_numbers=None, filename=None):
        if filename is not None:
            filename = cls._map_user_filename_to_file_0(filename)
        try:
            hbt_filename = filename or cls._infer_hbt_filename(sim)
            if h5py.is_hdf5(hbt_filename):
                with h5py.File(hbt_filename, 'r') as f:
                    if "NumberOfFiles" in f:
                        return True
        except OSError:
            pass
        return False




class HBTPlusCatalogueWithGroups(HaloCatalogue):
    """A class to represent a HBT+ halo catalogue with parent groups from another finder.

    For more information about parent groups, see :ref:`hbt_plus_parent_groups`.
    """

    def __init__(self, hbt_cat: HBTPlusCatalogue, group_cat: HaloCatalogue):
        if hbt_cat.base is not group_cat.base:
            raise ValueError("The two catalogues must have the same base simulation snapshot")
        if not isinstance(hbt_cat, HBTPlusCatalogue):
            raise ValueError("The first catalogue must be a HBTPlusCatalogue")

        self._hbt_cat = hbt_cat
        self._group_cat = group_cat
        self._hbt_host_groups = np.asarray(hbt_cat._file["Subhalos"]["HostHaloId"])
        if self._hbt_host_groups.max() >= len(group_cat):
            raise ValueError("The HBT+ catalogue contains host groups that are not in the group catalogue")
        self._children = [[] for _ in range(len(group_cat))]
        for hbt_index, hbt_number in enumerate(hbt_cat.number_mapper):
            self._children[self._hbt_host_groups[hbt_index]].append(hbt_number)
        self._children = [np.array(c) for c in self._children]

        super().__init__(group_cat.base, group_cat.number_mapper)

    def load_all(self):
        self._hbt_cat.load_all()
        self._group_cat.load_all()

    def _get_particle_indices_one_halo(self, halo_number) -> NDArray[int]:
        return self._group_cat._get_particle_indices_one_halo(halo_number)

    def get_properties_one_halo(self, halo_number) -> dict:
        group_properties = self._group_cat.get_properties_one_halo(halo_number)
        group_properties['children'] = self._children[self.number_mapper.number_to_index(halo_number)]
        return group_properties

    def get_properties_all_halos(self, with_units=True) -> dict:
        properties_all = self._group_cat.get_properties_all_halos(with_units)
        properties_all['children'] = self._children
        return properties_all

    def _get_subhalo_catalogue(self, halo_number):
        index = self._group_cat.number_mapper.number_to_index(halo_number)
        return self._hbt_cat[self._children[index]]
