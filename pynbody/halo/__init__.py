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
    def __init__(self, /, object_id_per_particle=None, particle_ids=None, unique_obj_numbers=None, boundaries=None):

        if object_id_per_particle is not None:
            assert particle_ids is None
            assert unique_obj_numbers is None
            assert boundaries is None
            self.sort_key = np.argsort(object_id_per_particle, kind='mergesort')  # mergesort for stability
            self.unique_obj_numbers = np.unique(object_id_per_particle)
            self.boundaries = np.searchsorted(object_id_per_particle[self.sort_key], self.unique_obj_numbers)
        else:
            assert particle_ids is not None
            assert unique_obj_numbers is not None
            assert boundaries is not None
            self.sort_key = particle_ids
            self.unique_obj_numbers = unique_obj_numbers
            self.boundaries = boundaries

    def get_index_list(self, obj_number):
        """Get the index list for the specified halo/object ID"""
        obj_offset = np.searchsorted(self.unique_obj_numbers, obj_number)
        if obj_offset >= len(self.unique_obj_numbers) or self.unique_obj_numbers[obj_offset] != obj_number:
            raise KeyError(f"No such halo {obj_number}")

        ptcl_start = self.boundaries[obj_offset]

        if obj_offset == len(self.unique_obj_numbers) - 1:
            ptcl_end = len(self.sort_key)
        else:
            ptcl_end = self.boundaries[obj_offset + 1]

        return self.sort_key[ptcl_start:ptcl_end]

    def __getitem__(self, obj_number):
        return self.get_index_list(obj_number)

    def __len__(self):
        return len(self.unique_obj_numbers)




# ----------------------------#
# General HaloCatalogue class #
#-----------------------------#

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
        return self._cached_index_lists.unique_obj_numbers

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
                 self._get_index_list_one_halo(i))



    def get_dummy_halo(self, i):
        """Return a DummyHalo object containing only the halo properties, no particle information"""
        h = DummyHalo()
        h.properties.update(self._get_properties_one_halo(i))
        return h

    def __len__(self):
        return self._get_num_halos()

    def __iter__(self):
        return self._halo_generator()

    def __getitem__(self, item):
        if isinstance(item, slice):
            return (self._get_halo_cached(i) for i in range(*item.indices(len(self))))
        else:
            return self._get_halo_cached(item)

    @property
    def base(self):
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

    def get_group_array(self):
        """Return an array with an integer for each particle in the simulation
        indicating which halo that particle is associated with. If there are multiple
        levels (i.e. subhalos), the number returned corresponds to the lowest level, i.e.
        the smallest subhalo."""
        raise NotImplementedError

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
        sim[array] # trigger lazy-loading and/or kick up a fuss if unavailable
        self._halos = {}
        self._array = array
        self._sorted = None
        self._ignore = ignore
        HaloCatalogue.__init__(self,sim)

    def __len__(self):
        if self._ignore is None:
            N = self.base[self._array].max()
        else:
            N = self.base[self._array]
            N = N[N!=self._ignore]
            N = N.max()
        if N<0:
            N=0
        return N

    def precalculate(self):
        """Speed up future operations by precalculating the indices
        for all halos in one operation. This is slow compared to
        getting a single halo, however."""
        self._sorted = np.argsort(
            self.base[self._array], kind='mergesort')  # mergesort for stability
        self._unique_i = np.unique(self.base[self._array])
        self._boundaries = np.searchsorted(self.base[self._array][self._sorted],self._unique_i)

    def get_group_array(self, family=None):
        if family is not None:
            return self.base[family][self._array]
        else:
            return self.base[self._array]

    def _get_halo_indices(self, i):
        if self.base is None:
            raise RuntimeError("Parent SimSnap has been deleted")

        no_exist = ValueError("Halo %s does not exist" % (str(i)))

        if self._sorted is None:
            # one-off selection
            index = np.where(self.base[self._array] == i)
            return index
        else:
            # pre-calculated
            if not np.isin(i,self._unique_i):
                raise no_exist

            match = np.where(self._unique_i==i)[0]

            start = self._boundaries[match][0]

            if start == self._boundaries[-1]:
                # This is the final halo
                end = None
            else:
                end = self._boundaries[match+1][0]

            return self._sorted[start:end]


    def _get_halo(self, i):
        x = Halo(i, self, self.base, self._get_halo_indices(i))
        if len(x) == 0:
            raise ValueError("Halo %s does not exist" % (str(i)))
        x._descriptor = "halo_" + str(i)
        return x


    def load_copy(self, i):
        """Load the a fresh SimSnap with only the particle in halo i"""
        from .. import load
        return load(self.base.filename, take=self._get_halo_indices(i))

    @property
    def base(self):
        return self._base()

    @staticmethod
    def _can_load(sim, arr_name='grp'):
        if (arr_name in sim.loadable_keys()) or (arr_name in list(sim.keys())) :
            return True
        else:
            return False


class AmigaGrpCatalogue(GrpCatalogue):
    def __init__(self, sim, arr_name='amiga.grp',**kwargs):
        GrpCatalogue.__init__(self, sim, arr_name)

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
