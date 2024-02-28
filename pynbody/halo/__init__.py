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

"""
from __future__ import annotations

import copy
import logging
import warnings
import weakref

import numpy as np
from numpy.typing import NDArray

from .. import iter_subclasses, snapshot, util
from .details.iord_mapping import make_iord_to_offset_mapper
from .details.number_mapper import MonotonicHaloNumberMapper, create_halo_number_mapper
from .details.particle_indices import HaloParticleIndices

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

    def is_subhalo(self, otherhalo):
        """
        Convenience function that calls the corresponding function in
        a halo catalogue.
        """

        return self._halo_catalogue.is_subhalo(self._halo_number, otherhalo._halo_number)

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
      _get_index_list_one_halo [optional, if it's possible to do this more efficiently than _get_index_list_all_halos]
      _get_properties_one_halo [only if you have halo finder-provided properties to expose]
      _get_halo [only if you want to add further customization to halos]
      _get_num_halos [optional, if it's possible to do this more efficiently than calling _get_index_list_all_halos]
      get_group_array [only if it's possible to do this more efficiently than the default implementation]

    """

    def __init__(self, sim, number_mapper):
        self._base: weakref[snapshot.SimSnap] = weakref.ref(sim)
        self._number_mapper: MonotonicHaloNumberMapper = number_mapper
        self._index_lists: HaloParticleIndices | None = None
        self._cached_halos: dict[Halo] = {}

    def load_all(self):
        """Loads all halos, which is normally more efficient if a large fraction of them will be accessed."""
        if not self._index_lists:
            index_lists = self._get_all_particle_indices()
            if isinstance(index_lists, tuple):
                index_lists = HaloParticleIndices(*index_lists)
            self._index_lists = index_lists

    @util.deprecated("precalculate has been renamed to load_all")
    def precalculate(self):
        self.load_all()

    def _get_num_halos(self):
        return len(self._number_mapper)

    def _get_index_list_all_halos_cached(self):
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

    def _get_properties_one_halo(self, halo_number) -> dict:
        """Returns a dictionary of properties for a single halo, given a halo_number """
        return {}

    def _get_index_list_one_halo(self, halo_number) -> NDArray[int]:
        """Get the index list for a single halo, given a halo_number.

        A generic implementation is provided that fetches index lists for all halos and then extracts the one"""
        self.load_all()
        return self._index_lists.get_particle_index_list_for_halo(
            self._number_mapper.number_to_index(halo_number)
        )

    def _get_index_list_via_most_efficient_route(self, halo_number) -> NDArray[int]:
        if self._index_lists:
            return self._index_lists.get_particle_index_list_for_halo(
                self._number_mapper.number_to_index(halo_number)
            )
        else:
            if len(self._cached_halos) == 5:
                warnings.warn("Accessing multiple halos may be more efficient if you call load_all() on the "
                              "halo catalogue", RuntimeWarning)
            return self._get_index_list_one_halo(halo_number)
            # NB subclasses may implement loading one halo direct from disk in the above
            # if not, the default implementation will populate _cached_index_lists

    def _get_halo_cached(self, halo_number) -> Halo:
        if halo_number not in self._cached_halos:
            self._cached_halos[halo_number] = self._get_halo(halo_number)
        return self._cached_halos[halo_number]

    def _get_halo(self, halo_number) -> Halo:
        return Halo(halo_number, self._get_properties_one_halo(halo_number), self, self.base,
                    self._get_index_list_via_most_efficient_route(halo_number))

    def get_dummy_halo(self, halo_number) -> DummyHalo:
        """Return a DummyHalo object containing only the halo properties, no particle information"""
        h = DummyHalo()
        h.properties.update(self._get_properties_one_halo(halo_number))
        return h

    def __len__(self) -> int:
        return self._get_num_halos()

    def __iter__(self):
        self.load_all()
        for i in self._number_mapper:
            yield self[i]

    def __getitem__(self, item) -> Halo:
        if isinstance(item, slice):
            return (self._get_halo_cached(i) for i in range(*item.indices(len(self))))
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

    def is_subhalo(self, childid, parentid):
        """Checks whether the specified 'childid' halo is a subhalo
        of 'parentid' halo.
        """
        if (childid in self._halos[parentid].properties['children']):
            return True
        else:
            return False

    def contains(self, halo_number):
        if halo_number in self._halos:
            return True
        else:
            return False

    def __contains__(self, haloid):
        return self.contains(haloid)

    def get_group_array(self, family=None):
        """Return an array with an integer for each particle in the simulation
        indicating which halo that particle is associated with. If there are multiple
        levels (i.e. subhalos), the number returned corresponds to the lowest level, i.e.
        the smallest subhalo."""
        self.load_all()
        number_per_particle = self._index_lists.get_halo_number_per_particle(len(self.base), self._number_mapper)
        if family is not None:
            return number_per_particle[self.base._get_family_slice(family)]
        else:
            return number_per_particle

    def load_copy(self, halo_number):
        """Load a fresh SimSnap with only the particles in specified halo

        This relies on the underlying SimSnap being capable of partial loading."""
        from .. import load
        return load(self.base.filename, take=self._get_index_list_via_most_efficient_route(halo_number))

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

    @staticmethod
    def _can_load(self):
        return False


class GrpCatalogue(HaloCatalogue):
    """
    A generic catalogue using a .grp file to specify which particles
    belong to which group.
    """
    def __init__(self, sim, array='grp', ignore=None, **kwargs):
        """Construct a GrpCatalogue, extracting halos based on a simulation-wide integer array with their IDs

        *sim* - the SimSnap for which the halos will be constructed
        *array* - the name of the array which should be present, loadable or derivable across the simulation
        *ignore* - a special value indicating "no halo", or None if no such special value is defined
        """
        sim[array] # noqa - trigger lazy-loading and/or kick up a fuss if unavailable
        self._array = array
        self._ignore = ignore
        self._halo_numbers = np.unique(sim[array])
        number_mapper = create_halo_number_mapper(self._trim_array_for_ignore(self._halo_numbers))
        HaloCatalogue.__init__(self, sim, number_mapper=number_mapper)

    def _trim_array_for_ignore(self, array):
        assert len(array) == len(self._halo_numbers)
        if self._ignore is None:
            return array
        if self._ignore == self._halo_numbers[0]:
            return array[1:]
        elif self._ignore == self._halo_numbers[-1]:
            return array[:-1]
        else:
            raise ValueError(
                "ignore must be either the smallest or largest value in the array")


    def _get_all_particle_indices(self):

        halo_number_per_particle = self.base[self._array]

        particle_index_list = np.argsort(halo_number_per_particle, kind='mergesort')
        start = np.searchsorted(halo_number_per_particle[particle_index_list], self._halo_numbers)
        stop = np.concatenate((start[1:], [len(particle_index_list)]))

        particle_index_list_boundaries = self._trim_array_for_ignore(
            np.hstack((start[:, np.newaxis], stop[:, np.newaxis]))
        )

        return HaloParticleIndices(particle_ids = particle_index_list, boundaries = particle_index_list_boundaries)

    def _get_index_list_one_halo(self, halo_number):
        if halo_number == self._ignore:
            self._no_such_halo(halo_number)
        array = np.where(self.base[self._array] == halo_number)[0]
        if len(array) == 0:
            self._no_such_halo(halo_number)
        return array

    def _no_such_halo(self, i):
        raise KeyError(f"No such halo {i}")

    def get_group_array(self, family=None):
        if family is not None:
            return self.base[family][self._array]
        else:
            return self.base[self._array]

    @staticmethod
    def _can_load(sim, arr_name='grp'):
        if (arr_name in sim.loadable_keys()) or (arr_name in list(sim.keys())) :
            return True
        else:
            return False


class AmigaGrpCatalogue(GrpCatalogue):
    def __init__(self, sim):
        GrpCatalogue.__init__(self, sim, array='amiga.grp')

    @staticmethod
    def _can_load(sim,arr_name='amiga.grp'):
        return GrpCatalogue._can_load(sim, arr_name)


from . import adaptahop, ahf, hop, rockstar, subfind, subfindhdf
