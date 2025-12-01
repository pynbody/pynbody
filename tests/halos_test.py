"""Test the generic halo catalogue mechanisms, using a very simple reference implementation"""

import warnings

import numpy as np
import pytest

import pynbody
import pynbody.halo.details.iord_mapping
import pynbody.halo.details.number_mapping
import pynbody.halo.details.particle_indices
import pynbody.test_utils
from pynbody import halo


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("ramses")


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

        return pynbody.halo.details.particle_indices.HaloParticleIndices(particle_ids=indexes, boundaries=boundaries)

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
