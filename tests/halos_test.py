"""Test the generic halo catalogue mechanisms, using a very simple reference implementation"""

import gc
import pickle
import warnings

import numpy as np
import pytest

import pynbody
import pynbody.array.shared
import pynbody.halo.details.iord_mapping
import pynbody.halo.details.number_mapping
import pynbody.halo.details.particle_indices
import pynbody.test_utils
from pynbody import halo


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("ramses", "gasoline_ahf")


class SimpleHaloCatalogue(halo.HaloCatalogue):

    def __init__(self, sim):
        number_mapper = pynbody.halo.details.number_mapping.SimpleHaloNumberMapper(1, 9)
        super().__init__(sim, number_mapper)

    def _get_all_particle_indices(self):
        np.random.seed(123)

        indexes = np.random.permutation(np.arange(len(self.base)))[:(len(self.base)*80)//100]
        # above means ~80% of particles should be in a halo

        start = np.sort(np.random.randint(0, len(indexes), len(self.number_mapper)))
        start[0] = 0
        # above makes up some boundaries in the index list to divide up into halos
        # nb the last boundary is always implicitly len(indexes)

        stop = np.concatenate((start[1:], [len(indexes)]))

        boundaries = np.vstack((start,stop)).T

        # NB _adopt_array puts these into shared memory if the snapshot is using it, which is what lets the
        # catalogue be handed to another process without copying; a real implementation should do the same
        return pynbody.halo.details.particle_indices.HaloParticleIndices(
            particle_ids=self._adopt_array(indexes), boundaries=self._adopt_array(boundaries))

    def get_properties_one_halo(self, i):
        return {'testproperty': 1.5*i, 'testproperty_with_units': 2.0*i*pynbody.units.Mpc}

class SimpleHaloCatalogueWithMultiMembership(halo.HaloCatalogue):

    def __init__(self, sim):
        number_mapper = pynbody.halo.details.number_mapping.SimpleHaloNumberMapper(1, 10)
        super().__init__(sim, number_mapper)

    def _get_all_particle_indices(self):
        np.random.seed(123)
        num_objs = 10
        lengths = np.random.randint(0,len(self.base)//5,num_objs)
        ptrs = np.concatenate(([0], np.cumsum(lengths)))
        boundaries = np.vstack((ptrs[:-1], ptrs[1:])).T
        members = np.concatenate([np.sort(np.random.choice(len(self.base), length)) for length in lengths])

        return pynbody.halo.details.particle_indices.HaloParticleIndices(particle_ids = members, boundaries = boundaries)

class SimpleHaloCatalogueWithIncompleteHalos(SimpleHaloCatalogue):
    """A catalogue where halos 2 and 5 refer to particles that are not in the snapshot.

    This mimics what a real catalogue does when the snapshot has been partially loaded: the missing
    particles are simply absent from the index list, but the halos are flagged so that accessing them
    raises rather than silently returning too few particles."""

    incomplete_halo_numbers = (2, 5)

    def _get_all_particle_indices(self):
        indices = super()._get_all_particle_indices()
        num_missing = self._array_factory(len(self.number_mapper), np.intp, zeros=True)
        for halo_number in self.incomplete_halo_numbers:
            num_missing[self.number_mapper.number_to_index(halo_number)] = halo_number
        indices.num_missing_particles = num_missing
        return indices


class SimpleHaloCatalogueWithAllIncompleteHalos(SimpleHaloCatalogueWithIncompleteHalos):
    """A catalogue where no halo at all can be constructed"""

    incomplete_halo_numbers = tuple(range(1, 10))


class SimpleHaloCatalogueWithReorderedNumbers(SimpleHaloCatalogueWithIncompleteHalos):
    """A catalogue where the halo numbers are not in the same order as the halo indices.

    This mimics catalogues loaded with halo_numbers='length-order', where the halo numbers are assigned
    according to a reordering of the underlying halos."""

    ordering = np.array([3, 1, 8, 0, 6, 2, 7, 4, 5])
    incomplete_halo_numbers = (2, 5)

    def __init__(self, sim):
        number_mapper = pynbody.halo.details.number_mapping.NonMonotonicHaloNumberMapper(
            self.ordering, ordering=True, start_index=1)
        halo.HaloCatalogue.__init__(self, sim, number_mapper)


@pytest.fixture
def snap_for_incomplete_halos():
    return pynbody.new(dm=100)


@pytest.fixture
def incomplete_halos(snap_for_incomplete_halos):
    return SimpleHaloCatalogueWithIncompleteHalos(snap_for_incomplete_halos)


@pytest.mark.parametrize("load_all", [True, False])
def test_incomplete_halos_raise(incomplete_halos, load_all):
    if load_all:
        incomplete_halos.load_all() # must not itself raise

    for halo_number in incomplete_halos.keys():
        if halo_number in SimpleHaloCatalogueWithIncompleteHalos.incomplete_halo_numbers:
            with pytest.raises(halo.IncompleteHaloError):
                _ = incomplete_halos[halo_number]
        else:
            assert len(incomplete_halos[halo_number]) > 0


def test_incomplete_halo_error_carries_details(incomplete_halos):
    with pytest.raises(halo.IncompleteHaloError) as excinfo:
        _ = incomplete_halos[2]

    assert excinfo.value.halo_number == 2
    assert excinfo.value.num_missing_particles == 2
    assert "Halo 2" in str(excinfo.value)


def test_incomplete_halos_do_not_affect_other_functionality(incomplete_halos, snap_for_incomplete_halos):
    """Properties, group arrays and complete halos remain available even when some halos are incomplete"""
    assert incomplete_halos.get_properties_one_halo(2)['testproperty'] == 3.0

    complete_halos = SimpleHaloCatalogue(snap_for_incomplete_halos)
    assert (incomplete_halos.get_group_array() == complete_halos.get_group_array()).all()
    assert (incomplete_halos[1].get_index_list(snap_for_incomplete_halos)
            == complete_halos[1].get_index_list(snap_for_incomplete_halos)).all()


def test_incomplete_halo_index_list_accessible_if_requested():
    """The particles that *are* present can still be retrieved, for callers that ask explicitly"""
    from pynbody.halo.details.particle_indices import (
        HaloParticleIndices,
        IncompleteHaloError,
    )

    index_list = HaloParticleIndices(particle_ids=np.array([0, 1, 2, 3, 4]),
                                     boundaries=np.array([[0, 2], [2, 5]]),
                                     num_missing_particles=np.array([0, 3]))

    assert not index_list.is_halo_incomplete(0)
    assert index_list.is_halo_incomplete(1)
    assert index_list.get_num_missing_particles_for_halo(1) == 3

    assert (index_list.get_particle_index_list_for_halo(0) == [0, 1]).all()

    with pytest.raises(IncompleteHaloError):
        index_list.get_particle_index_list_for_halo(1)

    assert (index_list.get_particle_index_list_for_halo(1, allow_incomplete=True) == [2, 3, 4]).all()


def test_index_list_without_missing_particle_information():
    """Catalogues that don't provide missing particle counts behave exactly as before"""
    from pynbody.halo.details.particle_indices import HaloParticleIndices

    index_list = HaloParticleIndices(particle_ids=np.array([0, 1, 2, 3, 4]),
                                     boundaries=np.array([[0, 2], [2, 5]]))

    assert not index_list.is_halo_incomplete(1)
    assert (index_list.get_particle_index_list_for_halo(1) == [2, 3, 4]).all()


def test_keys_unaffected_by_incompleteness(incomplete_halos):
    """keys() always describes the whole catalogue, whether or not the halos can be constructed"""
    assert (incomplete_halos.keys() == np.arange(1, 10)).all()
    incomplete_halos.load_all()
    assert (incomplete_halos.keys() == np.arange(1, 10)).all()


def test_keys_is_read_only(incomplete_halos):
    """keys() may be a view of the catalogue's own numbering, so it must not be modifiable"""
    keys = incomplete_halos.keys()
    with pytest.raises(ValueError):
        keys[0] = 99

    assert incomplete_halos.keys()[0] == 1


def test_complete_keys_excludes_incomplete_halos(incomplete_halos):
    complete_keys = incomplete_halos.complete_keys()
    assert (complete_keys == [1, 3, 4, 6, 7, 8, 9]).all()
    assert np.issubdtype(complete_keys.dtype, np.integer)


def test_complete_keys_matches_accessible_halos(incomplete_halos):
    """The halos reported as complete are exactly those which can actually be retrieved"""
    complete_keys = incomplete_halos.complete_keys()

    with warnings.catch_warnings():
        # we are deliberately accessing halos one at a time, which may prompt an efficiency warning
        warnings.filterwarnings("ignore", message="Accessing multiple halos")
        for halo_number in incomplete_halos.keys():
            try:
                incomplete_halos[halo_number]
                accessible = True
            except halo.IncompleteHaloError:
                accessible = False

            assert accessible == (halo_number in complete_keys)


def test_complete_keys_triggers_load_all(incomplete_halos):
    assert incomplete_halos._index_lists is None
    assert (incomplete_halos.complete_keys() == [1, 3, 4, 6, 7, 8, 9]).all()
    assert incomplete_halos._index_lists is not None


def test_complete_keys_can_decline_to_load_all(incomplete_halos):
    with pytest.raises(RuntimeError):
        incomplete_halos.complete_keys(load_all_if_required=False)

    with pytest.raises(RuntimeError):
        incomplete_halos.is_complete(1, load_all_if_required=False)

    incomplete_halos.load_all()

    assert (incomplete_halos.complete_keys(load_all_if_required=False) == [1, 3, 4, 6, 7, 8, 9]).all()
    assert incomplete_halos.is_complete(1, load_all_if_required=False)


def test_complete_keys_when_completeness_is_unknown(snap_for_incomplete_halos):
    """Catalogues that don't report missing particles describe all their halos as complete"""
    halos = SimpleHaloCatalogue(snap_for_incomplete_halos)
    assert (halos.complete_keys() == halos.keys()).all()
    assert halos.get_complete_mask().all()


def test_complete_keys_when_nothing_is_complete(snap_for_incomplete_halos):
    halos = SimpleHaloCatalogueWithAllIncompleteHalos(snap_for_incomplete_halos)
    complete_keys = halos.complete_keys()
    assert len(complete_keys) == 0
    assert np.issubdtype(complete_keys.dtype, np.integer)
    assert not halos.get_complete_mask().any()


def test_is_complete(incomplete_halos):
    assert incomplete_halos.is_complete(1) is True
    assert incomplete_halos.is_complete(2) is False

    with pytest.raises(KeyError):
        incomplete_halos.is_complete(100)


def test_complete_mask_is_in_index_order(incomplete_halos):
    mask = incomplete_halos.get_complete_mask()

    assert mask.dtype == bool
    assert len(mask) == len(incomplete_halos)
    assert (incomplete_halos.number_mapper.index_to_number(np.flatnonzero(mask))
            == incomplete_halos.complete_keys()).all()


def test_iterating_over_incomplete_catalogue_still_raises(incomplete_halos):
    """Incomplete halos must not be silently omitted from iteration; the error points at the alternative"""
    with pytest.raises(halo.IncompleteHaloError) as excinfo:
        list(incomplete_halos)

    assert "complete_keys" in str(excinfo.value)


def test_complete_keys_with_reordered_halo_numbers(snap_for_incomplete_halos):
    """Halo numbers are not necessarily in halo index order, and completeness must follow the numbers"""
    halos = SimpleHaloCatalogueWithReorderedNumbers(snap_for_incomplete_halos)

    complete_keys = halos.complete_keys()

    # the answer must be expressible either via the halo numbers or via the underlying indices
    complete_indices = np.flatnonzero(halos.get_complete_mask())
    assert sorted(complete_keys) == sorted(halos.number_mapper.index_to_number(complete_indices))

    # ... and it must be a subsequence of keys(), i.e. simply those halos which can be constructed
    assert list(complete_keys) == [n for n in halos.keys() if n not in halos.incomplete_halo_numbers]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Accessing multiple halos")
        for halo_number in halos.keys():
            try:
                halos[halo_number]
                accessible = True
            except halo.IncompleteHaloError:
                accessible = False

            assert accessible == (halo_number in complete_keys)


def test_complete_mask_is_cached_and_read_only(incomplete_halos):
    """The mask is handed out repeatedly, so it must not be recalculated or modifiable"""
    mask = incomplete_halos.get_complete_mask()
    assert incomplete_halos.get_complete_mask() is mask

    with pytest.raises(ValueError):
        mask[0] = False


def test_complete_mask_of_wrong_shape_is_rejected(snap_for_incomplete_halos):
    class BrokenCatalogue(SimpleHaloCatalogue):
        def _get_complete_mask(self):
            return np.ones(len(self) + 1, dtype=bool)

    with pytest.raises(ValueError, match="_get_complete_mask"):
        BrokenCatalogue(snap_for_incomplete_halos).get_complete_mask()


def test_delegating_catalogue_without_hook_is_reported(snap_for_incomplete_halos):
    """A catalogue that never populates its own index lists must say so, rather than failing obscurely"""
    class DelegatingCatalogue(SimpleHaloCatalogue):
        def load_all(self):
            pass

    with pytest.raises(NotImplementedError, match="_get_complete_mask"):
        DelegatingCatalogue(snap_for_incomplete_halos).complete_keys()


def test_complete_keys_avoids_efficiency_warning(incomplete_halos):
    """Since complete_keys loads all halos, the recommended idiom does not trip the efficiency warning"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for halo_number in incomplete_halos.complete_keys():
            incomplete_halos[halo_number]


class SimpleHaloCatalogueWithoutCompletenessInformation(SimpleHaloCatalogue):
    """A catalogue of the kind whose membership is an array covering only the particles that were loaded.

    That is the halo-number-per-particle case, e.g. a .grp file: usable, but with no way to tell that the
    finder assigned further particles which are absent. Catalogues addressing particles by *file position*
    are a different category, and set _uses_file_position_addressing instead."""

    _can_determine_completeness = False


def test_warning_when_completeness_cannot_be_determined(snap_for_incomplete_halos):
    """Catalogues which cannot detect missing particles must say so rather than claiming completeness"""
    halos = SimpleHaloCatalogueWithoutCompletenessInformation(snap_for_incomplete_halos)

    with pytest.warns(RuntimeWarning, match="unable to tell whether particles are missing"):
        assert (halos.complete_keys() == halos.keys()).all()

    with pytest.warns(RuntimeWarning, match="unable to tell whether particles are missing"):
        assert halos.is_complete(1)

    with pytest.warns(RuntimeWarning, match="unable to tell whether particles are missing"):
        assert halos.get_complete_mask().all()


def test_completeness_warning_does_not_require_load_all(snap_for_incomplete_halos):
    """There is nothing to be gained by loading the halos, so the question is answered without doing so"""
    halos = SimpleHaloCatalogueWithoutCompletenessInformation(snap_for_incomplete_halos)

    with pytest.warns(RuntimeWarning, match="unable to tell whether particles are missing"):
        halos.complete_keys(load_all_if_required=False)

    assert halos._index_lists is None


def test_subhalo_catalogue_completeness(incomplete_halos):
    """A subhalo catalogue is a view onto its parent, and must report completeness in its own numbering"""
    from pynbody.halo.subhalo_catalogue import SubhaloCatalogue

    # parent halos 1, 2, 3 become subhalo catalogue entries 0, 1, 2; of these, parent halo 2 is incomplete
    subhalos = SubhaloCatalogue(incomplete_halos, np.array([1, 2, 3]))

    assert (subhalos.complete_keys() == [0, 2]).all()
    assert subhalos.is_complete(0) is True
    assert subhalos.is_complete(1) is False

    assert len(subhalos[0]) > 0
    with pytest.raises(halo.IncompleteHaloError):
        subhalos[1]


def test_subhalo_catalogue_completeness_with_no_subhalos(incomplete_halos):
    from pynbody.halo.subhalo_catalogue import SubhaloCatalogue

    subhalos = SubhaloCatalogue(incomplete_halos, np.array([]))
    assert len(subhalos.complete_keys()) == 0


class SimpleIordBasedHaloCatalogue(halo.HaloCatalogue):
    """A catalogue which, like the real ones, names its particles by iord rather than by position"""

    def __init__(self, sim, iords_per_halo):
        self._iords_per_halo = [np.asarray(iords) for iords in iords_per_halo]
        number_mapper = pynbody.halo.details.number_mapping.SimpleHaloNumberMapper(1, len(self._iords_per_halo))
        super().__init__(sim, number_mapper)
        self._init_iord_to_fpos()

    def _get_all_particle_indices(self):
        return self._assemble_particle_indices(
            (self._map_iords_to_fpos(iords) for iords in self._iords_per_halo),
            num_halos=len(self._iords_per_halo),
            num_particles=sum(len(iords) for iords in self._iords_per_halo)
        )

    def _get_particle_indices_one_halo(self, halo_number):
        index = self.number_mapper.number_to_index(halo_number)
        return self._map_iords_to_fpos_one_halo(self._iords_per_halo[index], halo_number)

    def get_properties_one_halo(self, i):
        return {}


@pytest.fixture
def snap_with_iords():
    f = pynbody.new(dm=100)
    f['iord'] = np.arange(100)
    return f


@pytest.mark.parametrize("load_all", [True, False])
def test_unmatched_iords_in_complete_snapshot_are_an_error(snap_with_iords, load_all):
    """A fully loaded snapshot has nothing missing, so an unmatched ID means the IDs aren't iords at all.

    Reporting the halo as incomplete would send the user looking for a partial load that isn't there, so
    the mismatch is raised instead."""
    halos = SimpleIordBasedHaloCatalogue(snap_with_iords, [[0, 1, 2], [3, 4, 1000]])

    with pytest.raises(pynbody.halo.details.iord_mapping.IllegalIordError,
                       match="particle IDs as iords"):
        if load_all:
            halos.load_all()
        else:
            halos[2]


def test_unmatched_iords_in_partial_snapshot_are_incompleteness(snap_with_iords):
    """The same catalogue against a partially loaded snapshot reports incompleteness rather than raising"""
    partial = snap_with_iords[:50]
    halos = SimpleIordBasedHaloCatalogue(partial, [[0, 1, 2], [3, 4, 60]])

    assert (halos.complete_keys() == [1]).all()
    assert len(halos[1]) == 3

    with pytest.raises(halo.IncompleteHaloError):
        halos[2]


class SimpleHaloCatalogueWithAllProperties(SimpleHaloCatalogue):
    """A catalogue exposing properties for all halos at once, of the various kinds a halo finder may provide"""

    def get_properties_all_halos(self, with_units=True) -> dict:
        num_halos = len(self.number_mapper)
        mass = np.arange(num_halos, dtype=float) * 1e10
        if with_units:
            mass = pynbody.array.SimArray(mass, "Msol")
        return {'mass': mass,
                'pos': np.arange(3 * num_halos, dtype=float).reshape((num_halos, 3)),
                'children': [np.arange(i) for i in range(num_halos)]}


def _assert_portable(state):
    """Check that a portable state contains only things that can be transferred without further interpretation"""
    assert isinstance(state, dict)
    for key, value in state.items():
        assert isinstance(key, str)
        if isinstance(value, dict):
            _assert_portable(value)
        elif isinstance(value, list):
            _assert_portable({str(i): v for i, v in enumerate(value)})
        else:
            assert isinstance(value, (np.ndarray, np.number, np.bool_, str, bytes, bool, int, float, complex,
                                      type(None)))


def _transfer(state):
    """Mimic sending a portable state to another process, as a parallel task manager would"""
    return pickle.loads(pickle.dumps(state))


def test_portable_state_round_trip():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogueWithAllProperties(f)

    state = h.get_portable_state()
    _assert_portable(state)

    h_recreated = pynbody.halo.HaloCatalogue.from_portable_state(_transfer(state), f)

    assert isinstance(h_recreated, pynbody.halo.portable.PortableHaloCatalogue)
    assert len(h_recreated) == len(h)
    assert (h_recreated.keys() == h.keys()).all()

    for halo_number in h.keys():
        assert (h_recreated[halo_number].get_index_list(f) == h[halo_number].get_index_list(f)).all()

    assert (h_recreated.get_group_array() == h.get_group_array()).all()


def test_portable_state_round_trip_preserves_properties():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogueWithAllProperties(f)

    h_recreated = pynbody.halo.HaloCatalogue.from_portable_state(_transfer(h.get_portable_state()), f)

    properties = h_recreated.get_properties_all_halos()
    original_properties = h.get_properties_all_halos()

    assert (properties['mass'] == original_properties['mass']).all()
    assert properties['mass'].units == "Msol"
    assert (properties['pos'] == original_properties['pos']).all()

    for children, original_children in zip(properties['children'], original_properties['children']):
        assert (children == original_children).all()

    # properties should also be accessible one halo at a time, with units
    assert h_recreated[3].properties['mass'] == h[3].properties['mass']
    assert (h_recreated[3].properties['children'] == h[3].properties['children']).all()

    # ... and without units, if requested
    assert not pynbody.units.has_unit(h_recreated.get_properties_all_halos(with_units=False)['mass'])


def test_portable_properties_encoding():
    """Halo properties of the various shapes a finder may provide survive a round trip"""
    from pynbody.halo.portable import (
        properties_from_portable_state,
        properties_to_portable_state,
    )

    properties = {'mass': pynbody.array.SimArray(np.arange(3, dtype=float), "Msol"),
                  'plain': np.arange(3),
                  'children': [np.array([1, 2]), np.array([], dtype=np.int64), np.array([0])],
                  'names': ['a', 'bb', 'ccc'],
                  'finder': 'simple'}

    state = properties_to_portable_state(properties)
    _assert_portable(state)

    recreated = properties_from_portable_state(_transfer(state))

    assert (recreated['mass'] == properties['mass']).all()
    assert recreated['mass'].units == "Msol"
    assert (recreated['plain'] == properties['plain']).all()
    assert not pynbody.units.has_unit(recreated['plain'])
    for children, original_children in zip(recreated['children'], properties['children']):
        assert (children == original_children).all()
    assert list(recreated['names']) == properties['names']
    assert recreated['finder'] == 'simple'


def test_portable_state_round_trip_without_properties():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f) # only offers properties one halo at a time, so they can't be transferred

    h_recreated = pynbody.halo.HaloCatalogue.from_portable_state(_transfer(h.get_portable_state()), f)

    assert h_recreated.get_properties_all_halos() == {}
    assert len(h_recreated[1]) == len(h[1])


def test_portable_state_round_trip_preserves_incompleteness():
    """Halos which cannot be loaded must still be flagged after a transfer, not silently truncated"""
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogueWithIncompleteHalos(f)

    h_recreated = pynbody.halo.HaloCatalogue.from_portable_state(_transfer(h.get_portable_state()), f)

    assert (h_recreated.complete_keys() == h.complete_keys()).all()
    assert (h_recreated.get_complete_mask() == h.get_complete_mask()).all()

    for halo_number in SimpleHaloCatalogueWithIncompleteHalos.incomplete_halo_numbers:
        with pytest.raises(halo.IncompleteHaloError):
            _ = h_recreated[halo_number]


def test_portable_state_round_trip_preserves_inability_to_determine_completeness():
    """A catalogue which cannot detect missing particles must not appear able to once transferred"""
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogueWithoutCompletenessInformation(f)

    state = h.get_portable_state()
    assert state['can_determine_completeness'] is False

    h_recreated = pynbody.halo.HaloCatalogue.from_portable_state(_transfer(state), f)

    assert h_recreated._can_determine_completeness is False
    with pytest.warns(RuntimeWarning, match="unable to tell whether particles are missing"):
        assert (h_recreated.complete_keys() == h_recreated.keys()).all()


def test_portable_catalogue_does_not_address_particles_by_file_position():
    """The offsets in a state are resolved against the snapshot, so a partially loaded one is no obstacle.

    What matters is that the simulation presents the same particles in the same order as the one the state
    came from, which is the caller's responsibility; there is nothing here that refers to a file, so the
    refusal that applies to file-position catalogues must not fire."""
    f = pynbody.new(dm=100)
    state = SimpleHaloCatalogueWithAllProperties(f).get_portable_state()

    subsnap = f[:]
    assert subsnap.is_partially_loaded() # a view, so it counts as partial; see SimSnap.is_partially_loaded

    h_recreated = pynbody.halo.HaloCatalogue.from_portable_state(_transfer(state), subsnap)

    assert h_recreated._uses_file_position_addressing is False
    assert len(h_recreated[1]) > 0


def test_portable_state_arrays_can_be_mapped():
    """A consumer can replace all the arrays in a state (e.g. with shared memory) without knowing their roles"""
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogueWithAllProperties(f)

    mapped_arrays = []

    def mapper(ar):
        mapped_arrays.append(ar)
        return ar.copy()

    state = pynbody.halo.portable.map_arrays(h.get_portable_state(), mapper)

    assert len(mapped_arrays) > 0
    assert all(isinstance(ar, np.ndarray) for ar in mapped_arrays)

    h_recreated = pynbody.halo.HaloCatalogue.from_portable_state(state, f)
    assert (h_recreated[1].get_index_list(f) == h[1].get_index_list(f)).all()


def _rebuild_catalogue_in_subprocess(packed_state, result_queue):
    """Rebuild a catalogue that has been transferred through shared memory, and report on what it contains"""
    import pynbody
    from pynbody.array.shared import SharedArrayReference, from_shared_reference

    state = pynbody.halo.portable.map_arrays(packed_state, from_shared_reference,
                                            types=SharedArrayReference)

    f = pynbody.new(dm=100)
    h = pynbody.halo.HaloCatalogue.from_portable_state(state, f)

    incomplete = []
    index_lists = {}
    for halo_number in h.keys():
        try:
            index_lists[int(halo_number)] = h[halo_number].get_index_list(f)
        except pynbody.halo.IncompleteHaloError:
            incomplete.append(int(halo_number))

    result_queue.put((len(h), incomplete, index_lists))


def test_portable_state_transferred_through_shared_memory():
    """The intended use case: another process rebuilds the catalogue from arrays in shared memory"""
    import multiprocessing as mp

    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogueWithIncompleteHalos(f)

    def to_shared_memory(ar):
        shared_ar = pynbody.array.shared.make_shared_array(ar.shape, ar.dtype)
        shared_ar[:] = ar
        return shared_ar

    state = pynbody.halo.portable.map_arrays(h.get_portable_state(), to_shared_memory)
    packed_state = pynbody.halo.portable.map_arrays(state, pynbody.array.shared.to_shared_reference)

    context = mp.get_context('spawn')
    result_queue = context.Queue()
    process = context.Process(target=_rebuild_catalogue_in_subprocess, args=(packed_state, result_queue))
    process.start()
    try:
        num_halos, incomplete, index_lists = result_queue.get(timeout=120)
    finally:
        process.join(120)

    assert process.exitcode == 0
    assert num_halos == len(h)
    assert incomplete == list(SimpleHaloCatalogueWithIncompleteHalos.incomplete_halo_numbers)

    for halo_number, index_list in index_lists.items():
        assert (index_list == h[halo_number].get_index_list(f)).all()

    del state, packed_state
    gc.collect()


def test_portable_state_preserves_incompleteness(snap_for_incomplete_halos, incomplete_halos):
    """Halos which are incomplete because of partial loading remain flagged as such after a round trip"""
    state = incomplete_halos.get_portable_state()
    assert 'num_missing_particles' in state['particle_indices']

    h_recreated = pynbody.halo.HaloCatalogue.from_portable_state(_transfer(state), snap_for_incomplete_halos)

    for halo_number in h_recreated.keys():
        if halo_number in SimpleHaloCatalogueWithIncompleteHalos.incomplete_halo_numbers:
            with pytest.raises(halo.IncompleteHaloError) as excinfo:
                _ = h_recreated[halo_number]
            assert excinfo.value.halo_number == halo_number
            assert excinfo.value.num_missing_particles == halo_number
        else:
            assert len(h_recreated[halo_number]) > 0


def test_portable_state_without_incompleteness_information():
    """Catalogues with no missing particles don't carry any completeness information around"""
    f = pynbody.new(dm=100)
    state = SimpleHaloCatalogue(f).get_portable_state()

    assert 'num_missing_particles' not in state['particle_indices']

    h_recreated = pynbody.halo.HaloCatalogue.from_portable_state(_transfer(state), f)
    assert not h_recreated._index_lists.is_halo_incomplete(0)


@pytest.mark.parametrize("halo_numbers", [np.arange(1, 10), np.array([-5, -3, 0, 10]), np.array([5, 2, 9, 1])])
def test_portable_number_mapper_round_trip(halo_numbers):
    mapper = pynbody.halo.details.number_mapping.create_halo_number_mapper(halo_numbers)

    recreated = pynbody.halo.details.number_mapping.HaloNumberMapper.from_portable_state(
        _transfer(mapper.get_portable_state()))

    assert type(recreated) is type(mapper)
    assert len(recreated) == len(mapper)
    assert (recreated.all_numbers == mapper.all_numbers).all()
    assert (recreated.number_to_index(halo_numbers) == mapper.number_to_index(halo_numbers)).all()
    assert (recreated.index_to_number(np.arange(len(mapper)))
            == mapper.index_to_number(np.arange(len(mapper)))).all()


def test_portable_number_mapper_default_implementation():
    """A mapper which doesn't describe itself can still be transferred, via its halo numbers"""

    class ReversedHaloNumberMapper(pynbody.halo.details.number_mapping.HaloNumberMapper):
        def __init__(self, num_halos):
            self.num_halos = num_halos

        def number_to_index(self, halo_number):
            return self.num_halos - 1 - halo_number

        def index_to_number(self, halo_index):
            return self.num_halos - 1 - halo_index

        def __len__(self):
            return self.num_halos

        def __iter__(self):
            yield from self.all_numbers

        @property
        def all_numbers(self):
            return np.arange(self.num_halos)[::-1]

    mapper = ReversedHaloNumberMapper(5)
    recreated = pynbody.halo.details.number_mapping.HaloNumberMapper.from_portable_state(
        _transfer(mapper.get_portable_state()))

    assert (recreated.all_numbers == mapper.all_numbers).all()
    for halo_index in range(len(mapper)):
        assert recreated.index_to_number(halo_index) == mapper.index_to_number(halo_index)
        assert recreated.number_to_index(halo_index) == mapper.number_to_index(halo_index)


def test_portable_number_mapper_with_no_halos():
    mapper = pynbody.halo.details.number_mapping.HaloNumberMapper.from_portable_state(
        {'type': 'halo_numbers', 'halo_numbers': np.array([], dtype=np.intp)})
    assert len(mapper) == 0


def test_portable_number_mapper_unknown_type():
    with pytest.raises(ValueError, match="Unknown halo number mapper type"):
        pynbody.halo.details.number_mapping.HaloNumberMapper.from_portable_state({'type': 'NotAMapper'})


def test_portable_state_warns_about_untransferable_property():
    class CatalogueWithUnusualProperty(SimpleHaloCatalogue):
        def get_properties_all_halos(self, with_units=True) -> dict:
            return {'mass': np.arange(len(self.number_mapper), dtype=float),
                    'unusual': [{'not': 'transferable'}]}

    f = pynbody.new(dm=100)
    h = CatalogueWithUnusualProperty(f)

    with pytest.warns(RuntimeWarning, match="cannot be transferred"):
        state = h.get_portable_state()

    assert 'mass' in state['properties']
    assert 'unusual' not in state['properties']


def test_portable_state_unavailable_for_view_catalogues():
    """Catalogues that don't expose the particles of all their halos can't be made portable"""
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)

    with pytest.raises(TypeError, match="cannot be turned into a portable state"):
        h[1:3].get_portable_state()


def test_portable_state_rejects_future_version():
    f = pynbody.new(dm=100)
    state = SimpleHaloCatalogue(f).get_portable_state()
    state['version'] = pynbody.halo.portable.PORTABLE_STATE_VERSION + 1

    with pytest.raises(ValueError, match="format version"):
        pynbody.halo.HaloCatalogue.from_portable_state(state, f)


def test_portable_catalogue_repr():
    f = pynbody.new(dm=100)
    h_recreated = pynbody.halo.HaloCatalogue.from_portable_state(SimpleHaloCatalogue(f).get_portable_state(), f)
    assert repr(h_recreated) == "<PortableHaloCatalogue from SimpleHaloCatalogue, length 9>"

def _snapshot_with_grp_array(shared):
    f = pynbody.new(dm=100)
    f['grp'] = np.arange(100) % 10
    if shared:
        # NB must come before the catalogue is loaded, since it only affects subsequent allocations
        f.enable_shared_arrays()
    return f


def _all_arrays_in_state(state, path=''):
    """Yield (path, value) for every numpy array or shared array reference in a portable state"""
    if isinstance(state, dict):
        for key, value in state.items():
            yield from _all_arrays_in_state(value, f"{path}/{key}")
    elif isinstance(state, list):
        for i, value in enumerate(state):
            yield from _all_arrays_in_state(value, f"{path}/{i}")
    elif isinstance(state, (np.ndarray, pynbody.array.shared.SharedArrayReference)):
        yield path, state


def test_shared_arrays_flag_is_visible_from_subsnaps():
    """The flag lives on the underlying snapshot, so a view must give the same answer as its ancestor"""
    f = pynbody.new(dm=100)
    assert not f.uses_shared_arrays()
    assert not f[:50].uses_shared_arrays()

    f.enable_shared_arrays()
    assert f.uses_shared_arrays()
    assert f[:50].uses_shared_arrays()
    assert f.dm.uses_shared_arrays()


@pytest.mark.parametrize("shared", [True, False])
def test_index_lists_allocated_in_shared_memory(shared):
    """A catalogue on a snapshot using shared arrays puts its index lists in shared memory directly"""
    f = _snapshot_with_grp_array(shared)
    h = f.halos()
    h.load_all()

    index_lists = h._index_lists
    for array in (index_lists.particle_index_list, index_lists.particle_index_list_boundaries):
        assert pynbody.array.shared.is_shared_array(array) == shared

    del h, f
    gc.collect()


@pytest.mark.parametrize("shared", [True, False])
def test_assembled_index_lists_allocated_in_shared_memory(shared):
    """The same for a catalogue which assembles its index lists from per-halo particle IDs"""
    f = pynbody.new(dm=100)
    f['iord'] = np.arange(100)
    if shared:
        f.enable_shared_arrays()

    h = SimpleHaloCatalogueWithIncompleteHalos(f)
    h.load_all()

    index_lists = h._index_lists
    assert index_lists.num_missing_particles is not None
    for array in (index_lists.particle_index_list, index_lists.particle_index_list_boundaries,
                  index_lists.num_missing_particles):
        assert pynbody.array.shared.is_shared_array(array) == shared

    del h, f
    gc.collect()


def test_index_lists_in_shared_memory_from_file_based_finder():
    """The same, for a catalogue read from a halo finder's own files"""
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    f.enable_shared_arrays()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = f.halos()
        h.load_all()

    assert isinstance(h, pynbody.halo.ahf.AHFCatalogue)
    index_lists = h._index_lists
    assert pynbody.array.shared.is_shared_array(index_lists.particle_index_list)
    assert pynbody.array.shared.is_shared_array(index_lists.particle_index_list_boundaries)

    # ... including the finder's own properties, which are read from file and so have to be copied in
    state = h.get_portable_state()
    assert len(state['properties']) > 0
    for path, array in _all_arrays_in_state(state):
        assert pynbody.array.shared.is_shared_array(array), f"{path} is not in shared memory"

    del h, f, state, index_lists
    gc.collect()


def test_portable_state_of_shared_catalogue_needs_no_copy():
    """Every array of the state can be referenced directly, without an intervening copy"""
    f = _snapshot_with_grp_array(shared=True)
    h = SimpleHaloCatalogueWithAllProperties(f)

    state = h.get_portable_state()

    # the state must hand out the arrays the catalogue is really using, not copies of them
    assert state['particle_indices']['particle_index_list'] is h._index_lists.particle_index_list

    arrays = dict(_all_arrays_in_state(state))
    assert len(arrays) > 0
    for path, array in arrays.items():
        assert pynbody.array.shared.is_shared_array(array), f"{path} is not in shared memory"

    references = pynbody.halo.portable.map_arrays(state, pynbody.array.shared.to_shared_reference)
    referenced = dict(_all_arrays_in_state(references))

    assert set(referenced.keys()) == set(arrays.keys())
    for path, reference in referenced.items():
        assert isinstance(reference, pynbody.array.shared.SharedArrayReference), f"{path} is not a reference"

    del h, f, state, arrays, references, referenced
    gc.collect()


def test_shared_portable_state_round_trips_in_process():
    """Rebuilding from a state whose arrays are in shared memory gives back an equivalent catalogue"""
    f = _snapshot_with_grp_array(shared=True)
    h = SimpleHaloCatalogueWithAllProperties(f)

    # NB the state must be kept alive: it owns the shared memory that the references merely describe
    state = h.get_portable_state()
    references = pynbody.halo.portable.map_arrays(state, pynbody.array.shared.to_shared_reference)
    recreated_state = pynbody.halo.portable.map_arrays(references,
                                                       pynbody.array.shared.from_shared_reference,
                                                       types=pynbody.array.shared.SharedArrayReference)

    h_recreated = pynbody.halo.HaloCatalogue.from_portable_state(recreated_state, f)

    assert (h_recreated.keys() == h.keys()).all()
    for halo_number in h.keys():
        assert (h_recreated[halo_number].get_index_list(f) == h[halo_number].get_index_list(f)).all()

    properties = h_recreated.get_properties_all_halos()
    original_properties = h.get_properties_all_halos()
    assert (properties['mass'] == original_properties['mass']).all()
    assert properties['mass'].units == original_properties['mass'].units
    assert (properties['pos'] == original_properties['pos']).all()
    for children, original_children in zip(properties['children'], original_properties['children']):
        assert (children == original_children).all()

    del h, h_recreated, f, state, recreated_state, references, properties, original_properties
    gc.collect()


def test_shared_catalogue_transferred_to_another_process_without_copying():
    """End-to-end: the catalogue allocates in shared memory, and the state is referenced rather than copied"""
    import multiprocessing as mp

    f = pynbody.new(dm=100)
    f.enable_shared_arrays()
    h = SimpleHaloCatalogueWithIncompleteHalos(f)

    # NB no copying pass here, unlike test_portable_state_transferred_through_shared_memory
    state = h.get_portable_state()
    packed_state = pynbody.halo.portable.map_arrays(state, pynbody.array.shared.to_shared_reference)

    context = mp.get_context('spawn')
    result_queue = context.Queue()
    process = context.Process(target=_rebuild_catalogue_in_subprocess, args=(packed_state, result_queue))
    process.start()
    try:
        num_halos, incomplete, index_lists = result_queue.get(timeout=120)
    finally:
        process.join(120)

    assert process.exitcode == 0
    assert num_halos == len(h)
    assert incomplete == list(SimpleHaloCatalogueWithIncompleteHalos.incomplete_halo_numbers)

    for halo_number, index_list in index_lists.items():
        assert (index_list == h[halo_number].get_index_list(f)).all()

    del h, f, state, packed_state
    gc.collect()


def test_portable_state_unshared_by_default():
    """Without shared arrays enabled, nothing goes into shared memory"""
    f = _snapshot_with_grp_array(shared=False)
    h = SimpleHaloCatalogueWithAllProperties(f)

    state = h.get_portable_state()

    arrays = dict(_all_arrays_in_state(state))
    assert len(arrays) > 0
    for path, array in arrays.items():
        assert not pynbody.array.shared.is_shared_array(array), f"{path} unexpectedly in shared memory"

    with pytest.raises(TypeError):
        pynbody.halo.portable.map_arrays(state, pynbody.array.shared.to_shared_reference)


def test_shared_array_identity_survives_portable_encoding():
    """np.asarray would strip the shared-memory identity that the whole scheme depends on"""
    from pynbody.halo.details.portable_arrays import as_portable_array

    shared_array = pynbody.array.shared.make_shared_array((6,), np.int64, zeros=True)

    assert as_portable_array(shared_array) is shared_array
    assert not pynbody.array.shared.is_shared_array(np.asarray(shared_array))

    # ... but a subclass pynbody doesn't know about is still reduced to a plain ndarray
    assert type(as_portable_array(np.ma.masked_array([1, 2, 3]))) is np.ndarray
    assert type(as_portable_array([1, 2, 3])) is np.ndarray

    del shared_array
    gc.collect()


def test_halo_number_mapper():
    halo_numbers = np.array([-5, -3, 0, 10])
    mapper = pynbody.halo.details.number_mapping.MonotonicHaloNumberMapper(halo_numbers)
    assert mapper.number_to_index(-5) == 0
    assert mapper.number_to_index(-3) == 1
    assert mapper.number_to_index(0) == 2
    assert mapper.number_to_index(10) == 3
    with pytest.raises(KeyError):
        _ = mapper.number_to_index(5)

    assert (mapper.number_to_index([-5,0]) == [0,2]).all()

    with pytest.raises(KeyError):
        _ = mapper.number_to_index([-5,5])

    assert mapper.index_to_number(0) == -5
    assert mapper.index_to_number(1) == -3
    assert (mapper.index_to_number([0,1]) == [-5,-3]).all()

    with pytest.raises(IndexError):
        mapper.index_to_number(5)

    with pytest.raises(IndexError):
        mapper.index_to_number([1,5])

    assert len(mapper) == 4

    assert (list(mapper) == halo_numbers).all()

    assert (mapper.all_numbers == halo_numbers).all()

def test_non_monotonic_halo_number_mapper():
    halo_numbers = np.array([10, -5, 0, -3])
    mapper = pynbody.halo.details.number_mapping.NonMonotonicHaloNumberMapper(halo_numbers)
    assert mapper.number_to_index(-5) == 1
    assert mapper.number_to_index(-3) == 3
    assert mapper.number_to_index(0) == 2
    assert mapper.number_to_index(10) == 0
    with pytest.raises(KeyError):
        _ = mapper.number_to_index(5)

    assert (mapper.number_to_index([10,0]) == [0,2]).all()

    with pytest.raises(KeyError):
        _ = mapper.number_to_index([10,5])

    assert mapper.index_to_number(0) == 10
    assert mapper.index_to_number(1) == -5
    assert mapper.index_to_number(3) == -3
    assert (mapper.index_to_number([0,1,3]) == [10,-5,-3]).all()

    with pytest.raises(IndexError):
        mapper.index_to_number(4)

    with pytest.raises(IndexError):
        mapper.index_to_number([1,4])

    assert len(mapper) == 4

    assert (list(mapper) == halo_numbers).all()

    assert (mapper.all_numbers == [10, -5, 0, -3]).all()

def test_simple_halo_number_mapper():
    mapper = pynbody.halo.details.number_mapping.SimpleHaloNumberMapper(1, 10)
    assert mapper.number_to_index(1) == 0
    assert mapper.number_to_index(10) == 9
    with pytest.raises(KeyError):
        _ = mapper.number_to_index(11)

    assert (mapper.number_to_index([1,10]) == [0,9]).all()

    with pytest.raises(KeyError):
        _ = mapper.number_to_index([1, 11])

    assert mapper.index_to_number(0) == 1
    assert mapper.index_to_number(9) == 10
    with pytest.raises(IndexError):
        _ = mapper.index_to_number(10)

    assert (mapper.index_to_number([0,9]) == [1,10]).all()

    with pytest.raises(IndexError):
        _ = mapper.index_to_number([0,10])

    assert len(mapper) == 10

    assert (list(mapper) == np.arange(1,11)).all()

    assert (mapper.all_numbers == np.arange(1,11)).all()

def test_create_halo_number_mapper():
    from pynbody.halo.details.number_mapping import (
        MonotonicHaloNumberMapper,
        NonMonotonicHaloNumberMapper,
        SimpleHaloNumberMapper,
        create_halo_number_mapper,
    )

    # Test SimpleHaloNumberMapper
    halo_numbers = np.array([1, 2, 3, 4, 5])
    mapper = create_halo_number_mapper(halo_numbers)
    assert isinstance(mapper, SimpleHaloNumberMapper)
    assert mapper.zero_offset == 1
    assert len(mapper) == 5

    # Test SimpleHaloNumberMapper with non-zero offset
    halo_numbers = np.array([2, 3, 4, 5, 6])
    mapper = create_halo_number_mapper(halo_numbers)
    assert isinstance(mapper, SimpleHaloNumberMapper)
    assert mapper.zero_offset == 2
    assert len(mapper) == 5

    # Test MonotonicHaloNumberMapper
    halo_numbers = np.array([1, 3, 5, 7, 9])
    mapper = create_halo_number_mapper(halo_numbers)
    assert isinstance(mapper, MonotonicHaloNumberMapper)
    assert len(mapper) == 5

    # Test NonMonotonicHaloNumberMapper
    halo_numbers = np.array([1, 9, 5, 7, 3])
    mapper = create_halo_number_mapper(halo_numbers)
    assert isinstance(mapper, NonMonotonicHaloNumberMapper)
    assert len(mapper) == 5

def test_get_halo():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)
    assert len(h) == 9
    assert len(h[1])==15
    assert (h[1].get_index_list(f) == [8, 70, 82, 28, 63, 0, 5, 50, 81, 4, 23, 65, 76, 60, 24]).all()

def test_nonexistent_halo():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)
    with pytest.raises(KeyError):
        _ = h[0]

def test_get_halocat_slice():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)

    h_range = h[1:6:2]

    expected_halos = [h[1], h[3], h[5]]

    for halo, expected_halo in zip(h_range, expected_halos):
        assert halo == expected_halo

def test_get_halocat_indexed():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)

    h_range = h[[1,2,4,7]]
    expected_halos = [h[1], h[2], h[4], h[7]]

    for halo, expected_halo in zip(h_range, expected_halos):
        assert halo == expected_halo

def test_property_access():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)
    assert h[1].properties['testproperty'] == 1.5
    assert h[2].properties['testproperty'] == 3.0

def test_property_from_dummy():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)
    assert h.get_dummy_halo(1).properties['testproperty'] == 1.5

def test_halocat_keys():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)
    assert (h.keys() == np.arange(1,10)).all()

def test_property_units():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)
    assert "kpc" not in str(h[1].properties['testproperty_with_units'])

    h = SimpleHaloCatalogue(f)
    f.physical_units()
    assert np.allclose(h[1].properties['testproperty_with_units'].in_units('Mpc'), 2.0)
    assert "kpc" in str(h[1].properties['testproperty_with_units'])

def test_dummyhalo_property():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)
    assert h.get_dummy_halo(1).properties['testproperty'] == 1.5
    assert h._index_lists is None

def test_halo_iterator():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)
    for i, this_h in enumerate(h, 1):
        assert this_h == h[i]

def test_last_halo():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)
    assert len(h[9]) == 21

def test_get_group_array():
    f = pynbody.new(dm=100,gas=100)
    h = SimpleHaloCatalogueWithMultiMembership(f)
    grp = h.get_group_array()
    dm_grp = h.get_group_array(pynbody.family.dm)
    gas_grp = h.get_group_array(pynbody.family.gas)
    # grp should contain the halo id of the smallest halo to which each particle belongs

    # let's construct an independent test of tht
    f['comparison_grp'] = np.empty(len(f), dtype=int)
    f['comparison_grp'].fill(-1)
    all_halos_ordered = sorted(h, key=lambda x: len(x), reverse=True)
    for halo in all_halos_ordered:
        halo['comparison_grp'] = halo.properties['halo_number']

    assert (f['comparison_grp'] == grp).all()
    assert (f.dm['comparison_grp'] == dm_grp).all()
    assert (f.gas['comparison_grp'] == gas_grp).all()



@pytest.fixture
def snap_with_grp():
    f = pynbody.new(dm=100, gas=100)
    f['grp'] = np.random.randint(0,10,200)
    f['id'] = np.arange(200)
    yield f


@pytest.mark.parametrize("do_load_all", [True, False])
def test_grp_catalogue_single_halo(snap_with_grp, do_load_all):
    f = snap_with_grp
    h = pynbody.halo.number_array.HaloNumberCatalogue(f)
    if do_load_all:
        h.load_all()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        for halo_number in range(1,10):
            assert (h[halo_number]['id'] == f[f['grp'] == halo_number]['id']).all()

        with pytest.raises(KeyError):
            _ = h[10]

    if do_load_all:
        assert len(record) == 0



@pytest.mark.parametrize("do_load_all", [True, False])
@pytest.mark.parametrize("ignore_value", [0, 9])
def test_grp_catalogue_with_ignore_value(snap_with_grp, do_load_all, ignore_value):
    f = snap_with_grp
    h = pynbody.halo.number_array.HaloNumberCatalogue(snap_with_grp, ignore=ignore_value)
    if do_load_all:
        h.load_all()

    assert len(h) == 9 # NOT 10!

    with pytest.raises(KeyError):
        _ = h[ignore_value]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for halo_number in range(1,10):
            if halo_number != ignore_value:
                assert (h[halo_number]['id'] == f[f['grp'] == halo_number]['id']).all()

def test_grp_catalogue_cannot_determine_completeness(snap_with_grp):
    """The halo number array covers only the loaded particles, so completeness cannot be established"""
    h = pynbody.halo.number_array.HaloNumberCatalogue(snap_with_grp)

    with pytest.warns(RuntimeWarning, match="unable to tell whether particles are missing"):
        assert (h.complete_keys() == h.keys()).all()


def test_grp_catalogue_generated(snap_with_grp):
    h = snap_with_grp.halos()
    assert isinstance(h, pynbody.halo.number_array.HaloNumberCatalogue)

def test_amiga_grp_catalogue_generated(snap_with_grp):
    snap_with_grp['amiga.grp'] = snap_with_grp['grp']
    del snap_with_grp['grp']
    h = snap_with_grp.halos()

    assert isinstance(h, pynbody.halo.number_array.AmigaGrpCatalogue)

@pytest.mark.parametrize("load_all", [True, False])
def test_warning_when_inefficient(snap_with_grp, load_all):
    h = snap_with_grp.halos()
    def load_lots():
        for i in range(1,8):
            _ = h[i]

    if load_all:
        h.load_all()

        with warnings.catch_warnings(record=True) as record:
            load_lots()
        assert len(record) == 0, "No warnings should be raised when load_all called"
    else:
        with pytest.warns(RuntimeWarning, match="may be more efficient"):
            load_lots()

def test_short_iord_to_pos_map():
    iord = np.array([0, 5, 4, 2])
    iord_to_fpos = halo.details.iord_mapping.make_iord_to_offset_mapper(iord)
    assert isinstance(iord_to_fpos, halo.details.iord_mapping.IordToOffsetDense)
    assert (iord_to_fpos.map_ignoring_order([5, 2, 0, 4]) == [1, 3, 0, 2]).all()

def test_long_iord_to_pos_map():
    iord = np.array([0, 20, 10, 300])
    iord_to_fpos = halo.details.iord_mapping.make_iord_to_offset_mapper(iord)
    assert isinstance(iord_to_fpos, halo.details.iord_mapping.IordToOffsetSparse)
    assert (iord_to_fpos.map_ignoring_order([0, 10, 20, 300]) == np.array([0, 2, 1, 3])).all()
    assert iord_to_fpos.map_ignoring_order(300) == 3


def test_empty_iord_to_pos_map():
    """A snapshot which loaded nothing can find nothing; the reductions used to choose a mapper have no
    identity on an empty array, so the case is handled explicitly."""
    from pynbody.halo.details import iord_mapping

    mapper = iord_mapping.make_iord_to_offset_mapper(np.array([], dtype=np.int64))

    assert (mapper.map_ignoring_order([0, 5, 100], allow_missing=True) == iord_mapping.NO_OFFSET).all()
    assert mapper.map_ignoring_order(3, allow_missing=True) == iord_mapping.NO_OFFSET
    assert len(mapper.map_ignoring_order(np.array([], dtype=np.int64), allow_missing=True)) == 0

    with pytest.raises(iord_mapping.IllegalIordError):
        mapper.map_ignoring_order([0, 5, 100])


def test_catalogue_against_empty_selection(snap_with_iords):
    """Selecting no particles at all leaves every halo incomplete, rather than failing to build a mapper"""
    empty = snap_with_iords[:0]
    halos = SimpleIordBasedHaloCatalogue(empty, [[0, 1, 2], [3, 4, 5]])

    assert len(halos.complete_keys()) == 0
    assert not halos.get_complete_mask().any()

    with pytest.raises(halo.IncompleteHaloError):
        halos[1]


@pytest.fixture(params=['dense', 'sparse'])
def iord_mapper_and_missing(request):
    """An iord mapper alongside iord values which are, and are not, present in the underlying array"""
    from pynbody.halo.details import iord_mapping
    if request.param == 'dense':
        iord = np.array([0, 5, 4, 2])
        mapper = iord_mapping.make_iord_to_offset_mapper(iord)
        assert isinstance(mapper, iord_mapping.IordToOffsetDense)
        # 1 and 3 are inside the range covered by the lookup table but absent; 100 is beyond its end
        return mapper, [5, 2, 0, 4], [1, 3, 2], [100, 4], [-1, 0]
    else:
        iord = np.array([0, 20, 10, 300])
        mapper = iord_mapping.make_iord_to_offset_mapper(iord)
        assert isinstance(mapper, iord_mapping.IordToOffsetSparse)
        return mapper, [0, 10, 20, 300], [5, 11, 10], [1000, 300], [-1, 0]


def test_iord_to_pos_map_rejects_missing(iord_mapper_and_missing):
    from pynbody.halo.details import iord_mapping
    mapper, present, with_interior_missing, with_exterior_missing, with_negative = iord_mapper_and_missing

    # no exception when everything is present:
    mapper.map_ignoring_order(present)

    for values in (with_interior_missing, with_exterior_missing, with_negative):
        with pytest.raises(iord_mapping.IllegalIordError):
            mapper.map_ignoring_order(values)

    # ... and the same for scalar queries
    for value in (with_interior_missing[0], with_exterior_missing[0], with_negative[0]):
        with pytest.raises(iord_mapping.IllegalIordError):
            mapper.map_ignoring_order(value)


def test_iord_to_pos_map_allowing_missing(iord_mapper_and_missing):
    from pynbody.halo.details import iord_mapping
    mapper, present, with_interior_missing, with_exterior_missing, with_negative = iord_mapper_and_missing

    # results are only defined up to ordering, so compare sorted results with sorted expectations
    for values in (with_interior_missing, with_exterior_missing, with_negative):
        result = mapper.map_ignoring_order(values, allow_missing=True)
        expected = [mapper.map_ignoring_order(v, allow_missing=True) for v in values]
        assert (np.sort(result) == np.sort(expected)).all()
        assert (result == iord_mapping.NO_OFFSET).sum() == sum(e == iord_mapping.NO_OFFSET for e in expected)

    # scalars map to scalars, and missing ones map to the sentinel
    assert mapper.map_ignoring_order(present[0], allow_missing=True) >= 0
    assert np.ndim(mapper.map_ignoring_order(present[0], allow_missing=True)) == 0
    assert mapper.map_ignoring_order(with_interior_missing[0], allow_missing=True) == iord_mapping.NO_OFFSET
    assert mapper.map_ignoring_order(with_exterior_missing[0], allow_missing=True) == iord_mapping.NO_OFFSET
    assert mapper.map_ignoring_order(with_negative[0], allow_missing=True) == iord_mapping.NO_OFFSET


def test_iord_to_pos_map_missing_message(iord_mapper_and_missing):
    from pynbody.halo.details import iord_mapping
    mapper, _, with_interior_missing, _, _ = iord_mapper_and_missing

    # the exception should name the offending iords, not the ones that were found
    with pytest.raises(iord_mapping.IllegalIordError) as excinfo:
        mapper.map_ignoring_order(with_interior_missing)

    message = str(excinfo.value)
    present_values = {v for v in with_interior_missing
                      if mapper.map_ignoring_order(v, allow_missing=True) != iord_mapping.NO_OFFSET}
    missing_values = set(with_interior_missing) - present_values

    for value in missing_values:
        assert str(value) in message
    assert message.startswith(f"{len(missing_values)} particle ID")


def test_iord_to_pos_map_all_present_unaffected(iord_mapper_and_missing):
    """Check that allow_missing doesn't change the result when nothing is missing"""
    mapper, present, _, _, _ = iord_mapper_and_missing
    assert (mapper.map_ignoring_order(present)
            == mapper.map_ignoring_order(present, allow_missing=True)).all()


def test_iord_offset_modifier_with_missing():
    from pynbody.halo.details import iord_mapping
    iord = np.array([0, 5, 4, 2])
    mapper = iord_mapping.IordOffsetModifier(iord_mapping.make_iord_to_offset_mapper(iord), 100)

    assert (mapper.map_ignoring_order([5, 2, 0, 4]) == [101, 103, 100, 102]).all()

    with pytest.raises(iord_mapping.IllegalIordError):
        mapper.map_ignoring_order([5, 3])

    # the sentinel must not be shifted by the offset
    result = mapper.map_ignoring_order([5, 3, 100], allow_missing=True)
    assert (result == [101, iord_mapping.NO_OFFSET, iord_mapping.NO_OFFSET]).all()

    assert mapper.map_ignoring_order(5, allow_missing=True) == 101
    assert mapper.map_ignoring_order(3, allow_missing=True) == iord_mapping.NO_OFFSET
    with pytest.raises(iord_mapping.IllegalIordError):
        mapper.map_ignoring_order(3)

def test_load_halo_priority():
    from pynbody.halo.adaptahop import AdaptaHOPCatalogue
    from pynbody.halo.hop import HOPCatalogue
    f = pynbody.load("testdata/ramses/output_00080")

    # check that the priority ordering is respected
    halos = f.halos(priority=['HOPCatalogue'])
    assert isinstance(halos, HOPCatalogue)

    halos = f.halos(priority=["AdaptaHOPCatalogue", "HOPCatalogue"])
    assert isinstance(halos, AdaptaHOPCatalogue)

    # check we can pass a class instead of its name
    halos = f.halos(priority=[AdaptaHOPCatalogue])
    assert isinstance(halos, AdaptaHOPCatalogue)

    # check that classes not in the priority order are still scanned
    halos = f.halos(priority=["AHFCatalogue"])
    assert isinstance(halos, HOPCatalogue) or isinstance(halos, AdaptaHOPCatalogue)

def test_load_halo_priority_americanised():
    from pynbody.halo.adaptahop import AdaptaHOPCatalogue
    from pynbody.halo.hop import HOPCatalogue
    f = pynbody.load("testdata/ramses/output_00080")

    # check that the priority ordering is respected
    halos = f.halos(priority=['HOPCatalog'])
    assert isinstance(halos, HOPCatalogue)

    halos = f.halos(priority=["AdaptaHOPCatalog"])
    assert isinstance(halos, AdaptaHOPCatalogue)

def test_repr():
    f = pynbody.load("testdata/ramses/output_00080")
    halos = f.halos()
    assert repr(halos) == "<AdaptaHOPCatalogue, length 170>"
