"""
Support for halo and group catalogues.

Halo catalogues act like a dictionary, mapping from halo numbers to a Halo objects. The halo *number* is typically
determined by the halo finder, and is often (but not always) the same as the halo *index* which is the zero-based
offset within the catalogue.

If you have a supported halo catalogue on disk or a halo finder installed and correctly configured, you can access a
halo catalogue through ``f.halos()`` where ``f`` is a SimSnap.

See the :ref:`halo catalogue tutorial <halo_tutorial>` for introductory information and guidance.


.. _v2_0_halo_changes:

.. versionchanged:: 2.0

  Backwards-incompatible changes to the halo catalogue system

  For version 2.0, the halo catalogue loading system was substantially rewritten. The new system is more robust and
  more consistent across different halo finders. However, this means that some defaults have changed, most significantly
  in the AHF halo numbering. Backward-compatibility can be achieved by passing ``halo_numbers='v1'`` to the
  :class:`~pynbody.halo.ahf.AHFCatalogue` constructor. For more information, read the documentation for that class.

  Furthermore, older versions of pynbody (i.e. v1.x) could be configured to create a halo catalogue if one was not
  found, using AHF. This is no longer the case, as creating a halo catalogue requires choosing a halo finder and its
  parameters carefully for the task in hand and it was not possible to provide a one-size-fits-all solution.

  Finally, options to write ``.stat`` files and ``.grp`` files have been removed. However it is still possible to
  generate a ``.grp`` file by  calling :meth:`~HaloCatalogue.get_group_array` and writing out the resulting
  array of integers using a tool like ``numpy.savetxt``.

  By paring back the less-used functionality of the halo catalogue system, the remaining functionality is more
  consistent, robust, and extensible to new halo finders.


.. _supported_halo_finders:

Supported halo-finder formats
-----------------------------

The currently-supported formats are:

- Adaptahop (:class:`~pynbody.halo.adaptahop.AdaptaHOPCatalogue`);
- AHF (:class:`~pynbody.halo.ahf.AHFCatalogue`);
- HBT+ (:class:`~pynbody.halo.hbtplus.HBTPlusCatalogue`);
- HOP (:class:`~pynbody.halo.hop.HOPCatalogue`);
- Rockstar (:class:`~pynbody.halo.rockstar.RockstarCatalogue`);
- Subfind (old format :class:`~pynbody.halo.subfind.SubfindCatalogue`, or various HDF5 variants
  as :class:`~pynbody.halo.subfindhdf.SubfindHDFCatalogue`);
- VELOCIraptor (:class:`~pynbody.halo.velociraptor.VelociraptorCatalogue`).

In addition, generic halo finders which output a list of halo numbers for each particle are supported via
:class:`~pynbody.halo.number_array.HaloNumberCatalogue`.



.. note::

    The principal development of ``pynbody`` took place in the UK, and the spelling of "catalogue" is British English.
    However, since much code is written in American English, v2.0.0 introduced aliases such that all
    classes can be accessed with the American spelling ``HaloCatalog``, ``AdaptaHOPCatalog`` etc.


"""
from __future__ import annotations

import copy
import logging
import warnings
import weakref
from typing import TYPE_CHECKING, Iterable

import numpy as np
from numpy.typing import NDArray

from .. import array, snapshot, units, util
from ..util import iter_subclasses
from .details.iord_mapping import make_iord_to_offset_mapper
from .details.number_mapping import (
    HaloNumberMapper,
    MonotonicHaloNumberMapper,
    create_halo_number_mapper,
)
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
    Represents a single halo from a halo catalogue.

    Note that pynbody refers to groups, halos and subhalos interchangably, with the term "halo" being used to cover
    all of these.
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
        """Deprecated alias for :property:`subhalos`."""
        return self.subhalos

    @property
    def subhalos(self) -> SubhaloCatalogue:
        """A HaloCatalogue object containing only the subhalos of this halo."""
        return self._halo_catalogue._get_subhalo_catalogue(self._halo_number)


    def physical_units(self, distance='kpc', velocity='km s^-1', mass='Msol', persistent=True, convert_parent=True):
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

    To the user, this presents a simple interface where calling ``h[i]`` returns halo ``i``. Properties of halos
    can be retrieved without loading the halo via :meth:`get_properties_one_halo` or :meth:`get_properties_all_halos`.

    More information for users can be found in the :ref:`halo catalogue tutorial <halo_tutorial>`; see also the
    :ref:`supported halo finders <supported_halo_finders>`.

    Implementing a new format
    ^^^^^^^^^^^^^^^^^^^^^^^^^

    To support a new format, subclass :class:`HaloCatalogue` and implement the following methods:

    * :meth:`__init__`
    * :meth:`_can_load`
    * :meth:`_get_all_particle_indices`
    * :meth:`_get_particle_indices_one_halo` [only if it's possible to do this more efficiently than
      :meth:`_get_all_particle_indices` for users accessing only a few halos]
    * :meth:`get_properties_all_halos` [only if you have halo finder-provided properties to expose]
    * :meth:`get_properties_one_halo` [only if you have halo finder-provided properties to expose, and it's efficient
      to expose them one halo at a time; the default implementation will call get_properties_all_halos and extract]
    * :meth:`get_group_array` [only if it's possible to do this more efficiently than the default implementation]

    Nomenclature/conventions are worth being aware of if you are implementing a new format:

    * The halo number is the user-exposed identifier for a halo. It is typically assigned by the halo finder, although
      subclasses are free to assign their own (e.g. some have a `halo_number` option that can be passed to the
      constructor to override the halo finder's numbering). The halo numbers are used to access individual halos via
      the [] operator.
    * The halo *index* is the zero-based offset within the catalogue, which may be different from the halo number.
      Internally, *pynbody* converts between these using a :class:`details.number_mapping.HaloNumberMapper` object,
      which is set up in the :meth:`__init__` method.
    * Particle indices should be returned from methods like :meth:`_get_particle_indices_one_halo` as zero-relative
      offsets within the snapshot, not particle IDs or 'iord's. Many halo finders output particle IDs which must
      therefore be mapped. To aid this, call :meth:`_init_iord_to_fpos` in your :meth:`__init__` method, which creates
      a mapper as :attr:`_iord_to_fpos`. See :mod:`details.iord_mapping` for more information.

    """

    def __init__(self, sim, number_mapper):
        self._base: weakref[snapshot.SimSnap] = weakref.ref(sim)
        self.number_mapper: HaloNumberMapper = number_mapper
        self._index_lists: HaloParticleIndices | None = None
        self._properties: dict | None = None
        self._cached_halos: dict[int, Halo] = {}
        self._persistent_units = None

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

            if self._persistent_units is not None:
                self._cached_properties_to_physical_units(self._persistent_units)

    @util.deprecated("precalculate has been renamed to load_all")
    def precalculate(self):
        """Deprecated alias for :meth:`load_all`"""
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

        # Default implementation: extract from all halos. Subclasses may override this if they can load properties
        # for a single halo more efficiently.
        self._properties = self.get_properties_all_halos(with_units=True)
        return self._get_properties_one_halo_using_cache_if_available(halo_number,
                                                                      self.number_mapper.number_to_index(halo_number))

    def get_properties_all_halos(self, with_units=True) -> dict:
        """Returns a dictionary of properties for all halos.

        If with_units is True, the properties are returned as SimArrays with units if possible. Otherwise, numpy arrays
        are returned.

        Note that the returned properties are in contiguous arrays, and as a result may be in a different order to the
        halo numbers which are used to access individual halos. To map between halo numbers and properties, use the
        .number_mapper object; or access individual property dictionaries by halo number using get_properties_one_halo."""
        return {}

    def _get_properties_one_halo_using_cache_if_available(self, halo_number, halo_index):
        if self._properties is None:
            return self.get_properties_one_halo(halo_number)
        else:
            return {k: units.get_item_with_unit(self._properties[k],halo_index)
                    for k in self._properties}

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

    def __repr__(self):
        return f"<{type(self).__name__}, length {len(self)}>"

    def keys(self):
        """Return an iterable of all halo numbers in the catalogue."""
        return self.number_mapper.all_numbers

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
        """The snapshot object that this halo catalogue is based on."""
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

    @util.deprecated("This method is deprecated and will be removed in a future release. Use python `in` syntax instead.")
    def contains(self, halo_number: int) -> bool:
        """Deprecated alias; instead of ``h.contains(number)`` use ``number in h``."""
        return halo_number in self

    def __contains__(self, halo_number) -> bool:
        """Returns True if the halo catalogue contains the specified halo number."""
        return halo_number in self.number_mapper

    def get_group_array(self, family=None, use_index=False, fill_value=-1):
        """Return an array with an integer for each particle in the simulation, indicating the halo of that particle.

        If there are multiple levels (i.e. subhalos), the number returned corresponds to the lowest level, i.e.
        the smallest subhalo.

        Parameters
        ----------

        family : str, optional
            If specified, return only the group array for the specified family.

        use_index: bool, optional
            If True, return the halo index rather than the halo number. (See the class documentation for the
            distinction between halo numbers and indices.)

        fill_value : int, optional
            The value to fill for particles not in any halo.

        """
        self.load_all()
        number_per_particle = self._index_lists.get_halo_number_per_particle(len(self.base),
                                                                             None if use_index else self.number_mapper,
                                                                             fill_value = fill_value)
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

        all_units = [units.Unit(x) for x in (distance, velocity, mass, 'a', 'h', 'K')]

        if persistent:
            self._persistent_units = all_units

        self._cached_properties_to_physical_units(all_units)

    def _cached_properties_to_physical_units(self, all_units):
        if self._properties is not None:
            for k in self._properties:
                if isinstance(self._properties[k], array.SimArray) and units.has_unit(self._properties[k]):
                    self.base._autoconvert_array_unit(self._properties[k], all_units)


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


def _fix_american_spelling(p):
    """Map American to British spelling (used by SimSnap.halos to allow flexible spelling)"""
    if isinstance(p, str) and p.endswith('Catalog'):
        return p.replace('Catalog', 'Catalogue')
    else:
        return p
def _alias_american_spelling():
    """Create American spelling aliases for all HaloCatalogue subclasses."""
    for c in HaloCatalogue.iter_subclasses():
        american_name = c.__name__.replace("Catalogue", "Catalog")
        # put american_name into the same module as c (not this module)

        if c.__module__.startswith('pynbody.halo.'):
            module = eval(c.__module__.replace('pynbody.halo.', ''))

        setattr(module, american_name, c)

    globals()['HaloCatalog'] = HaloCatalogue

_alias_american_spelling()
