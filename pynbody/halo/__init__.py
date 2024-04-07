"""

halo
====

Implements halo catalogue functions. If you have a supported halo
catalogue on disk or a halo finder installed and correctly configured,
you can access a halo catalogue through f.halos() where f is a
SimSnap.

See the `halo tutorial
<http://pynbody.github.io/pynbody/tutorials/halos.html>`_ for some
examples.

Halo catalogues act like a dictionary, mapping from a _halo number_ to a
Halo object. The halo number is typically determined by the halo finder, and
is often (but not always) the same as the _halo index_ which is the zero-based
offset within the catalogue.
"""
from __future__ import annotations

import copy
import logging
import warnings
import weakref
from typing import TYPE_CHECKING, Iterable

import numpy as np
from numpy.typing import NDArray

from .. import iter_subclasses, snapshot, units, util
from .details.iord_mapping import make_iord_to_offset_mapper
from .details.number_mapping import MonotonicHaloNumberMapper, create_halo_number_mapper
from .details.particle_indices import HaloParticleIndices

if TYPE_CHECKING:
    from .subhalo_catalogue import SubhaloCatalogue

logger = logging.getLogger("pynbody.halo")

class DummyHalo(snapshot.util.ContainerWithPhysicalUnitsOption):

    def __init__(self):
        self.properties = {}

    def physical_units(self, *args, **kwargs):
        pass


class Halo(snapshot.subsnap.IndexedSubSnap):

    """
    Generic class representing a halo.
    """

    def __init__(self, halo_number, properties, halo_catalogue, *args, **kwa):
        super().__init__(*args, **kwa)
        self._halo_catalogue = halo_catalogue
        self._halo_number = halo_number
        self._descriptor = "halo_" + str(halo_number)
        self.properties = copy.copy(self.properties)
        self.properties['halo_number'] = halo_number
        self.properties.update(properties)

        # Inherit autoconversion from parent
        self._autoconvert_properties()

    @property
    @util.deprecated("The sub property has been renamed to subhalos")
    def sub(self):
        return self.subhalos

    @property
    def subhalos(self) -> SubhaloCatalogue:
        return self._halo_catalogue._get_subhalo_catalogue(self._halo_number)

    def physical_units(self, distance='kpc', velocity='km s^-1', mass='Msol', persistent=True, convert_parent=True):
        """
        Converts all array's units to be consistent with the
        distance, velocity, mass basis units specified.

        Base units can be specified using keywords.

        **Optional Keywords**:

           *distance*: string (default = 'kpc')

           *velocity*: string (default = 'km s^-1')

           *mass*: string (default = 'Msol')

           *persistent*: boolean (default = True); apply units change to future lazy-loaded arrays if True

           *convert_parent*: boolean (default = None); if True, propagate units change to parent snapshot. See note below.

        **Note**:

            When convert_parent is True, the unit conversion is propagated to
            the parent halo catalogue and the halo properties *are not
            converted*. The halo catalogue is in charge of calling
            physical_units with convert_parent=False for all halo objects
            (including this one).

            When convert_parent is False, the properties are converted
            immediately.

        """
        if convert_parent:
            self._halo_catalogue.physical_units(
                distance=distance,
                velocity=velocity,
                mass=mass,
                persistent=persistent
            )
        else:
            # Convert own properties
            self._autoconvert_properties()


class HaloCatalogue(snapshot.util.ContainerWithPhysicalUnitsOption,
                    iter_subclasses.IterableSubclasses):

    """Generic halo catalogue object.

    To the user, this presents a simple interface where calling h[i] returns halo i.

    By convention, i should use the halo finder's own indexing scheme, e.g. if the halo-finder is one-based then
    h[1] should return the first halo.

    To support a new format, subclass this and implement the following methods:
      __init__, which must pass a HaloNumberMapper into the base constructor, to specify what halos are available
      _get_all_particle_indices [essential]
      _get_particle_indices_one_halo [optional, if it's possible to do this more efficiently than _get_all_particle_indices]
      get_properties_one_halo [only if you have halo finder-provided properties to expose]
      get_properties_all_halos [only if you have halo finder-provided properties to expose]
      _get_halo [only if you want to add further customization to halos]
      _get_num_halos [optional, if it's possible to do this more efficiently than calling _get_index_list_all_halos]
      get_group_array [only if it's possible to do this more efficiently than the default implementation]
      _get_subhalo_catalogue [optional, if this halo catalogue supports subhalos]; user-accessed via .subhalos on a Halo

    Note that particle indices are zero-relative offsets within pynbody's representation of the snapshot. They are not
    the same as particle IDs or 'iord's which are the particle IDs stored in the simulation file. To aid converting
    between iords/IDs (which are used by many halo finder outputs) and pynbody's particle indices, call
    _init_iord_to_fpos, which creates a mapper as _iord_to_fpos. See details/iord_mapping.py for more information.
    """

    def __init__(self, sim, number_mapper):
        self._base: weakref[snapshot.SimSnap] = weakref.ref(sim)
        self.number_mapper: MonotonicHaloNumberMapper = number_mapper
        self._index_lists: HaloParticleIndices | None = None
        self._properties: dict | None = None
        self._cached_halos: dict[int, Halo] = {}

    def load_all(self):
        """Loads all halos, which is normally more efficient if a large fraction of them will be accessed."""
        if not self._index_lists:
            index_lists = self._get_all_particle_indices()
            properties = self.get_properties_all_halos(with_units=True)
            if isinstance(index_lists, tuple):
                index_lists = HaloParticleIndices(*index_lists)
            self._index_lists = index_lists
            if len(properties)>0:
                self._properties = properties

    @util.deprecated("precalculate has been renamed to load_all")
    def precalculate(self):
        self.load_all()

    def _get_num_halos(self):
        return len(self.number_mapper)

    def _get_all_particle_indices_cached(self):
        """Get the index information for all halos, using a cached version if available"""
        self.load_all()
        return self._index_lists

    def _get_all_particle_indices(self) -> HaloParticleIndices | tuple[np.ndarray, np.ndarray]:
        """Returns information about the index list for all halos.

        Returns an HaloParticleIndices object, which is a container for the following information:
        - particle_ids: particle IDs contained in halos, sorted by halo ID
        - boundaries: the indices in particle_ids where each halo starts and ends
        """
        raise NotImplementedError("This halo catalogue does not support loading all halos at once")

    def get_properties_one_halo(self, halo_number) -> dict:
        """Returns a dictionary of properties for a single halo, given a halo_number """
        return {}

    def get_properties_all_halos(self, with_units=True) -> dict:
        """Returns a dictionary of properties for all halos.

        If with_units is True, the properties are returned as SimArrays with units if possible. Otherwise, numpy arrays
        are returned.

        Note that the returned properties are in contiguous arrays, and as a result may be in a different order to the
        halo numbers which are used to access individual halos. To map between halo numbers and properties, use the
        .number_mapper object; or access individual property dictionaries by halo number using get_properties_one_halo."""
        return {}

    def _get_particle_indices_one_halo(self, halo_number) -> NDArray[int]:
        """Get the index list for a single halo, given a halo_number.

        A generic implementation is provided that fetches index lists for all halos and then extracts the one"""
        self.load_all()
        return self._index_lists.get_particle_index_list_for_halo(
            self.number_mapper.number_to_index(halo_number)
        )

    def _get_particle_indices_one_halo_using_list_if_available(self, halo_number, halo_index) -> NDArray[int]:
        if self._index_lists:
            return self._index_lists.get_particle_index_list_for_halo(halo_index)
        else:
            if len(self._cached_halos) == 5:
                warnings.warn("Accessing multiple halos may be more efficient if you call load_all() on the "
                              "halo catalogue", RuntimeWarning)
            return self._get_particle_indices_one_halo(halo_number)
            # NB subclasses may implement loading one halo direct from disk in the above
            # if not, the default implementation will populate _cached_index_lists

    def _get_halo_cached(self, halo_number) -> Halo:
        if halo_number not in self._cached_halos:
            self._cached_halos[halo_number] = self._get_halo(halo_number)
        return self._cached_halos[halo_number]

    def _get_properties_one_halo_using_cache_if_available(self, halo_number, halo_index):
        if self._properties is None:
            return self.get_properties_one_halo(halo_number)
        else:
            return {k: units.get_item_with_unit(self._properties[k],halo_index)
                    for k in self._properties}

    def _get_halo(self, halo_number) -> Halo:
        halo_index = self.number_mapper.number_to_index(halo_number)
        return Halo(halo_number,
                    self._get_properties_one_halo_using_cache_if_available(halo_number, halo_index),
                    self, self.base,
                    self._get_particle_indices_one_halo_using_list_if_available(halo_number, halo_index))

    def get_dummy_halo(self, halo_number) -> DummyHalo:
        """Return a DummyHalo object containing only the halo properties, no particle information"""
        h = DummyHalo()
        h.properties.update(self.get_properties_one_halo(halo_number))
        return h

    def __len__(self) -> int:
        return self._get_num_halos()

    def __iter__(self) -> Iterable[Halo]:
        self.load_all()
        for i in self.number_mapper:
            yield self[i]

    def __getitem__(self, item) -> Halo | SubhaloCatalogue:
        from .subhalo_catalogue import SubhaloCatalogue
        if isinstance(item, slice):
            return SubhaloCatalogue(self, np.arange(*item.indices(len(self))))
        elif hasattr(item, "__len__"):
            return SubhaloCatalogue(self, item)
        else:
            return self._get_halo_cached(item)

    @property
    def base(self) -> snapshot.SimSnap:
        return self._base()

    def _init_iord_to_fpos(self):
        """Create a member array, _iord_to_fpos, that maps particle IDs to file positions.

        This is a convenience function for subclasses to use."""
        if not hasattr(self, "_iord_to_fpos"):
            if 'iord' in self.base.loadable_keys() or 'iord' in self.base.keys():
                self._iord_to_fpos = make_iord_to_offset_mapper(self.base['iord'])

            else:
                warnings.warn("No iord array available; assuming halo catalogue is using sequential particle IDs",
                              RuntimeWarning)

                class OneToOneIndex:
                    def __getitem__(self, i):
                        return i

                self._iord_to_fpos = OneToOneIndex()

    def _get_subhalo_catalogue(self, parent_halo_number: int) -> SubhaloCatalogue:
        from .subhalo_catalogue import SubhaloCatalogue
        props = self.get_properties_one_halo(parent_halo_number)
        if 'children' in props:
            return SubhaloCatalogue(self, props['children'])
        else:
            raise ValueError(f"This halo catalogue does not support subhalos")

    def contains(self, halo_number: int) -> bool:
        return halo_number in self.number_mapper

    def __contains__(self, haloid):
        return self.contains(haloid)

    def get_group_array(self, family=None):
        """Return an array with an integer for each particle in the simulation
        indicating which halo that particle is associated with. If there are multiple
        levels (i.e. subhalos), the number returned corresponds to the lowest level, i.e.
        the smallest subhalo."""
        self.load_all()
        number_per_particle = self._index_lists.get_halo_number_per_particle(len(self.base), self.number_mapper)
        if family is not None:
            return number_per_particle[self.base._get_family_slice(family)]
        else:
            return number_per_particle

    def load_copy(self, halo_number):
        """Load a fresh SimSnap with only the particles in specified halo

        This relies on the underlying SimSnap being capable of partial loading."""
        from .. import load
        halo_index = self.number_mapper.number_to_index(halo_number)
        return load(self.base.filename,
                    take=self._get_particle_indices_one_halo_using_list_if_available(halo_number, halo_index))

    def physical_units(self, distance='kpc', velocity='km s^-1', mass='Msol', persistent=True, convert_parent=False):
        """
        Converts all array's units to be consistent with the
        distance, velocity, mass basis units specified.

        Base units can be specified using keywords.

        **Optional Keywords**:

           *distance*: string (default = 'kpc')

           *velocity*: string (default = 'km s^-1')

           *mass*: string (default = 'Msol')

           *persistent*: boolean (default = True); apply units change to future lazy-loaded arrays if True

           *convert_parent*: boolean (default = None); ignored for HaloCatalogue objects

        """
        self.base.physical_units(distance=distance, velocity=velocity, mass=mass, persistent=persistent)

        # Convert all instantiated subhalos
        for halo in self._cached_halos.values():
            halo.physical_units(
                distance,
                velocity,
                mass,
                persistent=persistent,
                convert_parent=False
            )

    @classmethod
    def _can_load(cls, sim):
        return False


from . import (
    adaptahop,
    ahf,
    hbtplus,
    hop,
    number_array,
    rockstar,
    subfind,
    subfindhdf,
    velociraptor,
)
