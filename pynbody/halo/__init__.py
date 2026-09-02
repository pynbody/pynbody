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


Transferring a catalogue between processes
------------------------------------------

Any halo catalogue can be expressed as a dictionary of numpy arrays and python primitives, using
:meth:`HaloCatalogue.get_portable_state`, and turned back into a live catalogue on a specified simulation using
:meth:`HaloCatalogue.from_portable_state`. This makes it possible to load a catalogue once and hand it to other
processes -- for example through shared memory -- without those processes needing access to the halo finder's
files, or any knowledge of the role of the individual arrays. See :mod:`pynbody.halo.portable` for more
information.


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
from .details.iord_mapping import (
    NO_OFFSET,
    IllegalIordError,
    make_iord_to_offset_mapper,
)
from .details.number_mapping import (
    HaloNumberMapper,
    MonotonicHaloNumberMapper,
    create_halo_number_mapper,
)
from .details.particle_indices import (
    HaloParticleIndices,
    IncompleteHaloError,
    PartialLoadingNotSupportedError,
)

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
      offsets within the in-memory SimSnap representation of the snapshot, not particle IDs, 'iord's, or the
      position on disk. Many halo finders output particle IDs which must therefore be mapped. To aid this, 
      call :meth:`_init_iord_to_fpos` in your :meth:`__init__` method, which creates a mapper as 
      :attr:`_iord_to_fpos`. See :mod:`details.iord_mapping` for more information.
    * If the snapshot is partially loaded, a halo may refer to particles that are not present. Rather than mapping
      particle IDs with :attr:`_iord_to_fpos` directly, use :meth:`_map_iords_to_fpos_one_halo` (in
      :meth:`_get_particle_indices_one_halo`), which raises an :class:`IncompleteHaloError`; and
      :meth:`_map_iords_to_fpos` (in :meth:`_get_all_particle_indices`), which discards and counts the missing
      particles. The counts are passed to :class:`details.particle_indices.HaloParticleIndices` as
      ``num_missing_particles``, most easily by assembling the index list with
      :meth:`_assemble_particle_indices`, so that loading all halos does not fail, while an
      :class:`IncompleteHaloError` is still raised if an affected halo is accessed. Those counts are also what
      :meth:`complete_keys` reports to the user, so no further work is needed to support it.
    * If, on the other hand, the format identifies its particles by position within the snapshot rather than
      by ID, those positions refer to particles which are not all present, and cannot in general be mapped
      onto those which are. Such subclasses should set the class attribute
      :attr:`_uses_file_position_addressing` to True, so that construction is refused with a
      :class:`details.particle_indices.PartialLoadingNotSupportedError` rather than the wrong particles being
      returned.
    * A third category expresses halo membership through an array covering only the particles which were
      loaded, such as a halo number per particle. Such a catalogue is usable, but has no way of knowing that
      the halo finder assigned further particles which are absent. These subclasses should set
      :attr:`_can_determine_completeness` to False, so that users are warned instead of being told that every
      halo is complete.
    * Catalogues which are views onto another catalogue, and therefore delegate :meth:`load_all` rather than
      populating their own particle index lists, must provide their own :meth:`_get_complete_mask` (mapping the
      underlying catalogue's answer into their own halo indexing) and :meth:`_is_loaded`.

    """

    _can_determine_completeness = True
    """Whether this catalogue is able to tell that particles are missing from the snapshot.

    Catalogues which identify their particles by ID can tell; those which rely on an array defined only for
    the particles which have been loaded cannot. Subclasses in the latter category should set this to False,
    so that users are warned rather than being given a completeness answer which is really an assumption.
    See :meth:`complete_keys`."""

    _uses_file_position_addressing = False
    """Whether this catalogue identifies its particles by position within the file rather than by ID.

    Such a catalogue cannot be used with a partially loaded snapshot, since its positions refer to particles
    which are not all present, and there is in general no way to map them onto those which are. Subclasses in
    this category should set this to True (before calling ``super().__init__``, if it depends on how the
    catalogue is configured), and construction is then refused for such snapshots."""

    def _refuse_if_snapshot_partially_loaded(self, sim):
        """Raise if this catalogue addresses particles by file position but the snapshot is partially loaded.

        This is called by :meth:`__init__`. Subclasses which do significant work (such as locating and opening
        their files) before calling ``super().__init__`` may call it earlier, so that the user gets this
        explanation rather than a confusing failure of that earlier work."""
        if self._uses_file_position_addressing and sim.is_partially_loaded():
            raise PartialLoadingNotSupportedError(type(self).__name__)

    def __init__(self, sim, number_mapper):
        self._refuse_if_snapshot_partially_loaded(sim)

        self._base: weakref[snapshot.SimSnap] = weakref.ref(sim)
        self.number_mapper: HaloNumberMapper = number_mapper
        self._index_lists: HaloParticleIndices | None = None
        self._complete_mask: NDArray[bool] | None = None
        self._properties: dict | None = None
        self._cached_halos: dict[int, Halo] = {}
        self._persistent_units = None

    def load_all(self):
        """Loads all halos, which is normally more efficient if a large fraction of them will be accessed."""
        if self._index_lists is None:
            index_lists = self._get_all_particle_indices()
            properties = self.get_properties_all_halos(with_units=True)
            if isinstance(index_lists, tuple):
                index_lists = HaloParticleIndices(*index_lists)
            self._index_lists = index_lists
            if len(properties)>0:
                self._properties = properties

            if self._persistent_units is not None:
                self._cached_properties_to_physical_units(self._persistent_units)

    def get_portable_state(self) -> dict:
        """Express the entire catalogue as numpy arrays and python primitives.

        The result is a dictionary whose values are numpy arrays, python primitives (such as strings and
        integers), or nested dictionaries of the same. Nothing in it is tied to the present process, or to the
        halo finder's files, so it may be transferred elsewhere -- for example to another process, by putting
        the arrays into shared memory. The recipient does not need to know the role of any individual array;
        it needs only to reproduce the structure at the other end, where
        :meth:`from_portable_state` turns it back into a live halo catalogue attached to a specified
        simulation.

        The state includes the halo numbering, the particles belonging to each halo, any information about
        particles that are missing from the snapshot (see
        :class:`~pynbody.halo.details.particle_indices.IncompleteHaloError`), whether this catalogue is able
        to detect such particles at all (:attr:`_can_determine_completeness`, so that a catalogue which
        cannot does not appear to have become able to once transferred), and the halo finder properties
        that are available from :meth:`get_properties_all_halos`. Properties which are only available one halo
        at a time, i.e. those provided by catalogues which implement :meth:`get_properties_one_halo` without
        :meth:`get_properties_all_halos`, are not included.

        Note that halo membership is expressed as offsets into the snapshot, so the state is only meaningful
        alongside a snapshot with the same particle ordering as the present one.

        .. versionadded:: 2.7.0

        """
        from .portable import PORTABLE_STATE_VERSION, properties_to_portable_state

        self.load_all()

        if self._index_lists is None:
            raise TypeError(f"A {type(self).__name__} cannot be turned into a portable state, because it does "
                            f"not expose the particles belonging to all its halos")

        state = {'version': PORTABLE_STATE_VERSION,
                 'catalogue_class': type(self).__name__,
                 'can_determine_completeness': self._can_determine_completeness,
                 'number_mapper': self.number_mapper.get_portable_state(),
                 'particle_indices': self._index_lists.get_portable_state()}

        if self._properties is not None:
            state['properties'] = properties_to_portable_state(self._properties)

        return state

    @classmethod
    def from_portable_state(cls, state: dict, sim: snapshot.SimSnap) -> HaloCatalogue:
        """Recreate a halo catalogue from a state dictionary, attaching it to the specified simulation.

        Parameters
        ----------

        state : dict
            A dictionary previously returned by :meth:`get_portable_state`, possibly having been transferred
            from another process.

        sim : :class:`~pynbody.snapshot.simsnap.SimSnap`
            The simulation to attach the catalogue to. It must have the same particle ordering as the
            simulation the state was generated from, since halo membership is expressed as offsets into it.

        Returns
        -------

        :class:`~pynbody.halo.portable.PortableHaloCatalogue`
            A halo catalogue which behaves like the original, but does not refer to the halo finder's files.

        .. versionadded:: 2.7.0

        """
        from .portable import PortableHaloCatalogue
        return PortableHaloCatalogue(sim, state)

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
            self.number_mapper.number_to_index(halo_number), halo_number
        )

    def _get_particle_indices_one_halo_using_list_if_available(self, halo_number, halo_index) -> NDArray[int]:
        if self._index_lists is not None:
            return self._index_lists.get_particle_index_list_for_halo(halo_index, halo_number)
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

    def keys(self) -> NDArray[int]:
        """Return an array of all halo numbers in the catalogue, whether or not they can be loaded.

        If the snapshot has been partially loaded, some of these halos may refer to particles that are not
        present, and cannot be retrieved; see :meth:`complete_keys`.

        The returned array is read-only, since it may be a view of the catalogue's internal numbering.

        .. versionchanged:: 2.7.0

          A read-only numpy array is returned. Previously this was whatever the catalogue's number mapper
          held, and was writable, so code which modified it in place must now take a copy first.

        """
        all_numbers = np.asarray(self.number_mapper.all_numbers).view()
        all_numbers.flags.writeable = False
        return all_numbers

    def complete_keys(self, load_all_if_required=True) -> NDArray[int]:
        """Return an array of the halo numbers which can actually be retrieved from the snapshot.

        This is a subset of :meth:`keys`, in the same order, from which halos referring to particles that are
        not present in the snapshot have been excluded. Accessing such halos raises an
        :class:`~pynbody.halo.details.particle_indices.IncompleteHaloError`; this normally happens only if the
        snapshot has been partially loaded.

        Note that halo *properties* remain available for all halos, including incomplete ones; it is only
        access to the particles which fails.

        Some catalogue formats express halo membership through an array covering only the particles which
        were loaded, such as a halo number per particle, and so are unable to tell whether the halo finder
        assigned further particles which are absent. These report all their halos as complete, and issue a
        RuntimeWarning to that effect. (Formats which instead identify their particles by position within the
        snapshot cannot be used with a partially loaded snapshot at all; they refuse to load.)

        Parameters
        ----------

        load_all_if_required : bool
            Establishing which halos are complete requires the particle lists for all halos, so
            :meth:`load_all` is called if it has not been already. If this is undesirable (e.g. because the
            catalogue is very large), pass False to raise a RuntimeError instead.

        .. versionadded:: 2.7.0

        """
        all_numbers = self.keys()
        complete_mask = self.get_complete_mask(load_all_if_required)

        # NB the halo numbers are not necessarily in halo index order (see NonMonotonicHaloNumberMapper),
        # so the mask must be explicitly mapped rather than applied directly
        all_indices = self.number_mapper.number_to_index(all_numbers)

        return all_numbers[complete_mask[all_indices]]

    def is_complete(self, halo_number, load_all_if_required=True) -> bool:
        """Return True if the specified halo can actually be retrieved from the snapshot.

        See :meth:`complete_keys` for more information, including the meaning of the
        ``load_all_if_required`` argument.

        .. versionadded:: 2.7.0

        """
        halo_index = self.number_mapper.number_to_index(halo_number)
        return bool(self.get_complete_mask(load_all_if_required)[halo_index])

    def get_complete_mask(self, load_all_if_required=True) -> NDArray[bool]:
        """Return a boolean mask, in halo index order, of the halos which can be retrieved from the snapshot.

        This is the underlying information from which :meth:`complete_keys` and :meth:`is_complete` are
        derived. Since it is in index order, it can be used to filter the arrays returned by
        :meth:`get_properties_all_halos`. (See the class documentation for the distinction between halo
        numbers and indices.)

        See :meth:`complete_keys` for the meaning of the ``load_all_if_required`` argument. The returned array
        is cached and read-only.

        .. versionadded:: 2.7.0

        """
        if not self._can_determine_completeness:
            warnings.warn(f"{type(self).__name__} is unable to tell whether particles are missing from the "
                          f"snapshot, so all halos are being reported as complete. If the snapshot has been "
                          f"partially loaded, halos may silently contain fewer particles than the halo finder "
                          f"assigned to them.", RuntimeWarning)
            if self._complete_mask is None:
                complete_mask = np.ones(len(self), dtype=bool)
                complete_mask.flags.writeable = False
                self._complete_mask = complete_mask
            return self._complete_mask

        if self._complete_mask is None:
            if not self._is_loaded():
                if not load_all_if_required:
                    raise RuntimeError("Establishing which halos are complete requires the particle lists for "
                                       "all halos. Either call load_all() first, or pass "
                                       "load_all_if_required=True.")
                self.load_all()

            complete_mask = np.asarray(self._get_complete_mask())

            if complete_mask.dtype != bool or complete_mask.shape != (len(self),):
                raise ValueError(f"{type(self).__name__}._get_complete_mask returned a {complete_mask.dtype} "
                                 f"array of shape {complete_mask.shape}; expected a boolean array of shape "
                                 f"({len(self)},)")

            complete_mask.flags.writeable = False # it is cached, and handed out repeatedly
            self._complete_mask = complete_mask

        return self._complete_mask

    def _is_loaded(self) -> bool:
        """Return True if the particle lists underlying this catalogue have been loaded.

        Catalogues which are views onto another catalogue should override this alongside
        :meth:`_get_complete_mask`."""
        return self._index_lists is not None

    def _get_complete_mask(self) -> NDArray[bool]:
        """Return a boolean array, in halo index order, flagging the halos whose particles are all present.

        This is the hook underlying :meth:`get_complete_mask`, and is called only once the catalogue has been
        loaded; it should not itself trigger loading. The default implementation uses the particle index
        lists, so catalogues which are views onto another catalogue must override it (see also
        :meth:`_is_loaded`)."""
        if self._index_lists is None:
            raise NotImplementedError(f"{type(self).__name__} is unable to establish which of its halos are "
                                      f"complete; it should provide its own _get_complete_mask implementation")

        return self._index_lists.get_complete_mask()

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
                if self.base.is_partially_loaded():
                    # without particle IDs we would have to take the catalogue's values as file positions,
                    # which are meaningless if not all the particles are present
                    raise PartialLoadingNotSupportedError(
                        f"{type(self).__name__}, with no iord array available,")

                warnings.warn("No iord array available; assuming halo catalogue is using sequential particle IDs",
                              RuntimeWarning)

                class OneToOneIndex:
                    def __getitem__(self, i):
                        return i

                    def map_ignoring_order(self, i, allow_missing=False):
                        # without iords there is no way to tell whether a particle is missing, so all
                        # iords are taken at face value
                        return i

                self._iord_to_fpos = OneToOneIndex()

    def _map_iords_to_fpos(self, iords) -> tuple[NDArray[int], int]:
        """Map particle IDs to file positions, discarding any particles that are not in the snapshot.

        This is a convenience function for subclasses to use when implementing
        :meth:`_get_all_particle_indices`; see also :meth:`_map_iords_to_fpos_one_halo`. It requires
        :meth:`_init_iord_to_fpos` to have been called first.

        Returns the file positions of the particles that are present, together with the number of particles
        that were discarded. A non-zero count should be recorded in the ``num_missing_particles`` argument of
        :class:`~pynbody.halo.details.particle_indices.HaloParticleIndices`, so that the affected halos are
        flagged as incomplete rather than silently returning too few particles.

        Particles may only go missing because the snapshot is partially loaded. If it is not, everything the
        catalogue names must be present, and an unmatched ID means the IDs are not being interpreted as the
        snapshot's iords; an :class:`~pynbody.halo.details.iord_mapping.IllegalIordError` is then raised
        rather than misreporting the halo as incomplete.
        """
        fpos = self._iord_to_fpos.map_ignoring_order(iords, allow_missing=True)

        # all genuine file positions are non-negative, so a single reduction tells us whether anything is
        # missing, without allocating a mask in the usual case that nothing is
        if len(fpos) > 0 and fpos.min() < 0:
            present = fpos != NO_OFFSET
            num_missing = len(fpos) - int(present.sum())
            if not self.base.is_partially_loaded():
                raise IllegalIordError(
                    f"{num_missing} of the {len(fpos)} particle ID(s) in a halo of this catalogue do not "
                    f"correspond to any particle in the snapshot. Most likely {type(self).__name__} is not "
                    f"interpreting the catalogue's particle IDs as iords of this snapshot correctly."
                )
            return fpos[present], num_missing
        else:
            return fpos, 0

    def _map_iords_to_fpos_one_halo(self, iords, halo_number=None) -> NDArray[int]:
        """Map the particle IDs of a single halo to file positions.

        This is a convenience function for subclasses to use when implementing
        :meth:`_get_particle_indices_one_halo`; see also :meth:`_map_iords_to_fpos`. It requires
        :meth:`_init_iord_to_fpos` to have been called first.

        Raises an :class:`IncompleteHaloError` if any of the particles are not in the snapshot.
        """
        fpos, num_missing = self._map_iords_to_fpos(iords)
        if num_missing > 0:
            raise IncompleteHaloError(halo_number, num_missing)
        return fpos

    def _assemble_particle_indices(self, mapped_iords_per_halo, num_halos, num_particles,
                                   sort=False) -> HaloParticleIndices:
        """Assemble the index list for all halos, given their particles' file positions.

        This is a convenience function for subclasses to use when implementing
        :meth:`_get_all_particle_indices`. Particles which are not present in the snapshot have already been
        discarded by :meth:`_map_iords_to_fpos`, so the index list is compacted, and the halos which have lost
        particles are flagged as incomplete.

        Parameters
        ----------

        mapped_iords_per_halo : iterable
            An iterable of ``(file_positions, num_missing)`` pairs, one per halo in halo index order, as
            returned by :meth:`_map_iords_to_fpos`.

        num_halos : int
            The number of halos, i.e. the number of items in ``mapped_iords_per_halo``.

        num_particles : int
            The total number of particles assigned to halos by the halo finder. This is used to allocate the
            index list, and may be an overestimate if particles are missing from the snapshot.

        sort : bool
            If True, sort each halo's file positions into ascending order.

        """
        particle_ids = np.empty(num_particles, dtype=np.intp)
        boundaries = np.empty((num_halos, 2), dtype=np.intp)
        num_missing_particles = np.zeros(num_halos, dtype=np.intp)

        start = 0
        for halo_index, (fpos, num_missing) in enumerate(mapped_iords_per_halo):
            stop = start + len(fpos)
            particle_ids[start:stop] = np.sort(fpos) if sort else fpos
            boundaries[halo_index] = (start, stop)
            num_missing_particles[halo_index] = num_missing
            start = stop

        # NB this is a view rather than a copy, so that nothing is duplicated in the usual case that all the
        # particles are present and the whole array is in use anyway
        return HaloParticleIndices(particle_ids[:start], boundaries,
                                   num_missing_particles=num_missing_particles)

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
    portable,
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
