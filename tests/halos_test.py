"""Test the generic halo catalogue mechanisms, using a very simple reference implementation"""

import numpy as np
import pynbody
import pytest

from pynbody import halo


class SimpleHaloCatalogue(halo.HaloCatalogue):
    def _get_index_list_all_halos(self):
        np.random.seed(123)
        halo_ids = np.arange(1,10)
        indexes = np.random.permutation(np.arange(len(self.base)))[:(len(self.base)*80)//100]
        # above means ~80% of particles should be in a halo

        boundaries = np.sort(np.random.randint(0, len(indexes), len(halo_ids)))
        boundaries[0] = 0
        # above makes up some boundaries in the index list to divide up into halos
        # nb the last boundary is always implicitly len(indexes)

        return halo.IndexList(particle_ids=indexes, boundaries=boundaries, unique_obj_numbers=halo_ids)

    def _get_properties_one_halo(self, i):
        return {'testproperty': 1.5*i, 'testproperty_with_units': 2.0*i*pynbody.units.Mpc}

class SimpleHaloCatalogueWithMultiMembership(halo.HaloCatalogue):
    def _get_index_list_all_halos(self):
        np.random.seed(123)
        num_objs = 10
        halo_ids = np.arange(1,num_objs+1)
        lengths = np.random.randint(0,len(self.base)//5,num_objs)
        boundaries = np.concatenate(([0], np.cumsum(lengths)))[:-1]

        members = np.concatenate([np.random.choice(len(self.base), length) for length in lengths])

        return halo.IndexList(particle_ids = members, boundaries = boundaries, unique_obj_numbers=halo_ids)

def test_get_halo():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)
    assert len(h) == 9
    assert len(h[1])==15
    assert (h[1].get_index_list(f) == [8, 70, 82, 28, 63,  0,  5, 50, 81,  4, 23, 65, 76, 60, 24]).all()

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

    assert next(h_range, None) is None

def test_property_access():
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogue(f)
    assert h[1].properties['testproperty'] == 1.5
    assert h[2].properties['testproperty'] == 3.0

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
    assert h._cached_index_lists is None

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
    f = pynbody.new(dm=100)
    h = SimpleHaloCatalogueWithMultiMembership(f)
    grp = h.get_group_array()
    # grp should contain the halo id of the smallest halo to which each particle belongs

    # let's construct an independent test of tht
    f['comparison_grp'] = np.empty(len(f), dtype=int)
    f['comparison_grp'].fill(-1)
    all_halos_ordered = sorted(h, key=lambda x: len(x), reverse=True)
    for halo in all_halos_ordered:
        halo['comparison_grp'] = halo.properties['halo_id']

    assert (f['comparison_grp'] == grp).all()



@pytest.fixture
def snap_with_grp():
    f = pynbody.new(dm=100, gas=100)
    f['grp'] = np.random.randint(0,10,200)
    f['id'] = np.arange(200)
    yield f


def test_grp_catalogue_single_halo(snap_with_grp):
    f = snap_with_grp
    h = pynbody.halo.GrpCatalogue(f)
    assert (f[f['grp'] == 1]['id'] == h[1]['id']).all()

def test_grp_catalogue_multihalo(snap_with_grp):
    f = snap_with_grp
    h = pynbody.halo.GrpCatalogue(f)
    h.load_all()

    assert (f[f['grp'] == 1]['id'] == h[1]['id']).all()
    assert (f[f['grp'] == 5]['id'] == h[5]['id']).all()