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

import copy
import logging
import warnings
import weakref

import numpy as np
import numpy.typing as npt

from typing import Any

import pynbody.snapshot.subsnap

from .. import snapshot, util

logger = logging.getLogger("pynbody.halo")

class DummyHalo(snapshot.util.ContainerWithPhysicalUnitsOption):

    def __init__(self):
        self.properties = {}

    def physical_units(self, *args, **kwargs):
        pass


class Halo(pynbody.snapshot.subsnap.IndexedSubSnap):

    """
    Generic class representing a halo.
    """

    def __init__(self, halo_id, properties, halo_catalogue, *args, **kwa):
        super().__init__(*args, **kwa)
        self._halo_catalogue = halo_catalogue
        self._halo_id = halo_id
        self._descriptor = "halo_" + str(halo_id)
        self.properties = copy.copy(self.properties)
        self.properties['halo_id'] = halo_id
        self.properties.update(properties)

        # Inherit autoconversion from parent
        self._autoconvert_properties()

    def is_subhalo(self, otherhalo):
        """
        Convenience function that calls the corresponding function in
        a halo catalogue.
        """

        return self._halo_catalogue.is_subhalo(self._halo_id, otherhalo._halo_id)

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


class IndexList:
    def __init__(self, /, halo_id_per_particle: npt.NDArray[int] = None, ignore_halo_id: int = None,
                 particle_ids: npt.NDArray[int] = None, halo_numbers: npt.NDArray[int]= None,
                 boundaries: np.ndarray[(Any, 2), int] = None):
        """An IndexList represents abstract information about halo membership

        Either an halo_id_per_particle can be specified, which is an array of halo IDs for each particle
        (length should be the number of particles in the simulation). If this is the case, ignore_halo_id specifies
        a special value that indicates "no halo" (e.g. -1), or None if no such special value is defined. Note that
        ignore_halo_id must be either the smallest or largest value in halo_id_per_particle.

        Alternatively, an array of particle_ids can be specified, which is then sub-divided into the
        particle ids for each halo according to the boundaries array passed. The halos are then labelled
        by the specified halo_numbers (which must be unique).

        To summarise, the parameters are:

        * halo_id_per_particle: array of halo IDs for each particle, length Npart
        * ignore_halo_id: a special value that indicates "no halo" (e.g. -1), or None if no such special value is defined
        * particle_ids: array of particle IDs (mutually exclusive with halo_id_per_particle), length Npart_in_halos
        * halo_numbers: array of halo IDs (mutually exclusive with halo_id_per_particle), length Nhalo, must be unique
        * boundaries: array of indices in particle_ids where each halo starts (mutually exclusive with
                      halo_id_per_particle), length Nhalo; must be monotonically increasing

        """
        if halo_id_per_particle is not None:
            assert particle_ids is None
            assert halo_numbers is None
            assert boundaries is None
            self._setup_internals_from_halo_id_array(halo_id_per_particle, ignore_halo_id)

        else:
            assert particle_ids is not None
            assert halo_numbers is not None
            assert boundaries is not None
            self.particle_index_list = particle_ids
            self.halo_numbers = halo_numbers

            # check the halo numbers really are unique:
            assert len(np.unique(self.halo_numbers)) == len(self.halo_numbers)

            self.particle_index_list_boundaries = boundaries

            # should have a start and stop for each halo in the index list:
            assert self.particle_index_list_boundaries.shape == (len(self.halo_numbers), 2)


    def _setup_internals_from_halo_id_array(self, halo_id_per_particle, ignore_halo_id):
        self.particle_index_list = np.argsort(halo_id_per_particle, kind='mergesort')  # mergesort for stability
        self.halo_numbers = np.unique(halo_id_per_particle)
        start = np.searchsorted(halo_id_per_particle[self.particle_index_list], self.halo_numbers)
        stop = np.concatenate((start[1:], [len(self.particle_index_list)]))
        self.particle_index_list_boundaries = np.hstack((start[:,np.newaxis],stop[:,np.newaxis]))
        if ignore_halo_id is not None and ignore_halo_id in self.halo_numbers:
            if ignore_halo_id == self.halo_numbers[0]:
                self.particle_index_list_boundaries = self.particle_index_list_boundaries[1:]
                self.halo_numbers = self.halo_numbers[1:]
            elif ignore_halo_id == self.halo_numbers[-1]:
                self.particle_index_list_boundaries = self.particle_index_list_boundaries[:-1]
                self.halo_numbers = self.halo_numbers[:-1]
            else:
                raise ValueError("ignore_halo_id must be either the smallest or largest value in halo_id_per_particle")

    def get_index_list(self, halo_number):
        """Get the index list for the specified halo/object ID"""
        halo_offset = self._get_halo_offset_from_id(halo_number)
        return self.particle_index_list[self._get_index_slice_from_halo_offset(halo_offset)]

    def _get_halo_offset_from_id(self, obj_number):
        """Get the offset in the index boundary array for the specified object number"""
        obj_offset = np.searchsorted(self.halo_numbers, obj_number)
        if obj_offset >= len(self.halo_numbers) or self.halo_numbers[obj_offset] != obj_number:
            raise KeyError(f"No such halo {obj_number}")
        return obj_offset

    def _get_index_slice_from_halo_offset(self, obj_offset):
        """Get the slice for the index array corresponding to the object *offset* (not ID),
        i.e. the one whose index list starts at self.boundaries[obj_offset]"""
        ptcl_start, ptcl_end = self.particle_index_list_boundaries[obj_offset]
        return slice(ptcl_start, ptcl_end)

    def get_object_id_per_particle(self, sim_length, fill_value=-1, dtype=int):
        """Return an array of object IDs, one per particle.

        Where a particle belongs to more than one object, the smallest object is favoured on the assumption that
        will identify the sub-halos etc in any reasonable case."""
        lengths = np.diff(self.particle_index_list_boundaries, axis=1).ravel()
        ordering = np.argsort(-lengths, kind='stable')

        id_array = np.empty(sim_length, dtype=dtype)
        id_array.fill(fill_value)

        for obj_offset in ordering:
            object_id = self.halo_numbers[obj_offset]
            indexing_slice = self._get_index_slice_from_halo_offset(obj_offset)
            id_array[self.particle_index_list[indexing_slice]] = object_id

        return id_array

    def __iter__(self):
        yield from self.halo_numbers

    def __getitem__(self, obj_number):
        return self.get_index_list(obj_number)

    def __len__(self):
        return len(self.halo_numbers)



class HaloCatalogue(snapshot.util.ContainerWithPhysicalUnitsOption):

    """Generic halo catalogue object.

    To the user, this presents a simple interface where calling h[i] returns halo i.

    By convention, i should use the halo finder's own indexing scheme, e.g. if the halo-finder is one-based then
    h[1] should return the first halo.

    To support a new format, subclass this and implement the following methods:
      _get_index_list_all_halos [essential]
      _get_halo_ids [optional, if it's possible to do this more efficiently than calling _get_index_list_all_halos]
      _get_index_list_one_halo [optional, if it's possible to do this more efficiently than _get_index_list_all_halos]
      _get_properties_one_halo [only if you have halo finder-provided properties to expose]
      _get_halo [only if you want to add further customization to halos]
      _get_num_halos [optional, if it's possible to do this more efficiently than calling _get_index_list_all_halos]
      get_group_array [only if it's possible to do this more efficiently than the default implementation]

    """

    def __init__(self, sim):
        self._base = weakref.ref(sim)
        self._cached_index_lists = None
        self._cached_halos = {}

    def load_all(self):
        """Loads all halos, which is normally more efficient if a large fraction of them will be accessed."""
        if not self._cached_index_lists:
            self._cached_index_lists = self._get_index_list_all_halos()

    @util.deprecated("precalculate has been renamed to load_all")
    def precalculate(self):
        self.load_all()

    def _get_num_halos(self):
        if self._cached_index_lists is not None:
            return len(self._cached_index_lists)
        else:
            return len(self._get_halo_ids())

    def _get_index_list_all_halos_cached(self):
        """Get the index information for all halos, using a cached version if available"""
        self.load_all()
        return self._cached_index_lists

    def _get_halo_ids(self):
        """Get the IDs of all halos.

        A default implementation is provided but subclasses may override this if they can do it more efficiently."""
        self.load_all()
        return self._cached_index_lists.halo_numbers

    def _get_index_list_all_halos(self):
        """Returns information about the index list for all halos.

        Returns an IndexList object, which is a container for the following information:
        - particle_ids: particle IDs contained in halos, sorted by halo ID
        - unique_obj_numbers: the halo IDs, in ascending order
        - boundaries: the indices in particle_ids where each halo starts and ends
        """
        raise NotImplementedError("This halo catalogue does not support loading all halos at once")

    def _get_properties_one_halo(self, i):
        """Returns a dictionary of properties for a single halo"""
        return {}

    def _get_index_list_one_halo(self, i):
        """Get the index list for a single halo.

        A generic implementation is provided that fetches index lists for all halos and then extracts the one"""
        self.load_all()
        return self._cached_index_lists[i]

    def _get_index_list_via_most_efficient_route(self, i):
        if self._cached_index_lists:
            return self._cached_index_lists[i]
        else:
            return self._get_index_list_one_halo(i)
            # NB subclasses may implement loading one halo direct from disk in the above
            # if not, the default implementation will populate _cached_index_lists

    def _get_halo_cached(self, i):
        if i not in self._cached_halos:
            self._cached_halos[i] = self._get_halo(i)
        return self._cached_halos[i]

    def _get_halo(self, i):
        return Halo(i, self._get_properties_one_halo(i), self, self.base,
                 self._get_index_list_via_most_efficient_route(i))



    def get_dummy_halo(self, i):
        """Return a DummyHalo object containing only the halo properties, no particle information"""
        h = DummyHalo()
        h.properties.update(self._get_properties_one_halo(i))
        return h

    def __len__(self):
        return self._get_num_halos()

    def __iter__(self):
        self.load_all()
        for i in self._cached_index_lists:
            yield self[i]


    def __getitem__(self, item):
        if isinstance(item, slice):
            return (self._get_halo_cached(i) for i in range(*item.indices(len(self))))
        else:
            return self._get_halo_cached(item)

    @property
    def base(self) -> pynbody.snapshot.SimSnap:
        return self._base()

    def _init_iord_to_fpos(self):
        """Create a member array, _iord_to_fpos, that maps particle IDs to file positions.

        This is a convenience function for subclasses to use."""
        if not hasattr(self, "_iord_to_fpos"):
            if 'iord' in self.base.loadable_keys():
                self._iord_to_fpos = np.empty(self.base['iord'].max()+1,dtype=np.int64)
                self._iord_to_fpos[self.base['iord']] = np.arange(len(self.base))
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

    def contains(self, haloid):
        if (haloid in self._halos):
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
        id_per_particle = self._cached_index_lists.get_object_id_per_particle(len(self.base))
        if family is not None:
            return id_per_particle[self.base._get_family_slice(family)]
        else:
            return id_per_particle

    def load_copy(self, i):
        """Load a fresh SimSnap with only the particles in halo i

        This relies on the underlying SimSnap being capable of partial loading."""
        from .. import load
        return load(self.base.filename, take=self._get_index_list_via_most_efficient_route(i))

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
        HaloCatalogue.__init__(self,sim)

    def _get_index_list_all_halos(self):
        return IndexList(halo_id_per_particle=self.base[self._array], ignore_halo_id=self._ignore)

    def _get_index_list_one_halo(self, i):
        if i == self._ignore:
            self._no_such_halo(i)
        array = np.where(self.base[self._array] == i)[0]
        if len(array) == 0:
            self._no_such_halo(i)
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


from pynbody.halo.adaptahop import AdaptaHOPCatalogue, NewAdaptaHOPCatalogue
from pynbody.halo.ahf import AHFCatalogue
from pynbody.halo.hop import HOPCatalogue
from pynbody.halo.legacy import RockstarIntermediateCatalogue
from pynbody.halo.rockstar import RockstarCatalogue
from pynbody.halo.subfind import SubfindCatalogue
from pynbody.halo.subfindhdf import (
    ArepoSubfindHDFCatalogue,
    Gadget4SubfindHDFCatalogue,
    SubFindHDFHaloCatalogue,
    TNGSubfindHDFCatalogue,
)


def _get_halo_classes():
    # AmigaGrpCatalogue MUST be scanned first, because if it exists we probably
    # want to use it, but an AHFCatalogue will probably be on-disk too.
    _halo_classes = [
        GrpCatalogue, AmigaGrpCatalogue, AHFCatalogue,
        RockstarCatalogue, SubfindCatalogue, SubFindHDFHaloCatalogue,
        NewAdaptaHOPCatalogue, AdaptaHOPCatalogue,
        RockstarIntermediateCatalogue, HOPCatalogue, Gadget4SubfindHDFCatalogue,
        ArepoSubfindHDFCatalogue, TNGSubfindHDFCatalogue
    ]

    return _halo_classes
