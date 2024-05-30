import numpy as np
import pytest

import pynbody


def test_order_bridge():
    f1 = pynbody.new(dm=1000)
    f2 = pynbody.new(dm=3000)
    f1['iord'] = np.arange(5, 2005, 2, dtype=int)
    f2['iord'] = np.arange(3000, dtype=int)

    b = pynbody.bridge.OrderBridge(f1, f2)

    h1 = f1[:50:2]
    assert b(h1).ancestor is f2
    assert (b(h1)['iord'] == h1['iord']).all


def test_bridge_factory():
    f1 = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    f2 = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    b = pynbody.bridge.bridge_factory(f1, f2)

    assert isinstance(b, pynbody.bridge.OrderBridge)


def test_nonmonotonic_transfer_matrix():
    # test the case where a non-monotonic bridge between two simulations is queries for particle transfer
    f1 = pynbody.new(dm=10)
    f2 = pynbody.new(dm=10)
    f1['iord'] = np.arange(0,10)
    f2['iord'] = np.array([0,2,4,6,8,1,3,5,7,9],dtype=np.int32)


    # 100% of mass in group 1 transfers to group 1
    # 80% of mass in group 0 transfers to group 0
    # 20% of mass in group 0 transfers to group 1

    f1['grp'] = np.array([0,0,0,0,0,1,1,1,1,1],dtype=np.int32)
    f2['grp'] = np.array([0,0,0,0,1,1,1,1,1,1],dtype=np.int32)[f2['iord']]

    b = pynbody.bridge.OrderBridge(f1,f2,monotonic=False)

    h1 = pynbody.halo.number_array.HaloNumberCatalogue(f1)
    h2 = pynbody.halo.number_array.HaloNumberCatalogue(f2)

    with pytest.warns(DeprecationWarning):
        xfer = b.catalog_transfer_matrix(0,1,h1,h2)

    assert (xfer==[[4,1],[0,5]]).all()

def test_missing_values_transfer_matrix():
    """Test making a transfer matrix where some particles are in no halo at all"""
    f1 = pynbody.new(dm=10)
    f2 = pynbody.new(dm=10)

    f1['grp'] = np.array([-1,0,-1,0,0,0,1,1,1,1],dtype=np.int32)
    f2['grp'] = np.array([0,-1,-1,0,0,1,1,1,1,1],dtype=np.int32)
    b = pynbody.bridge.OneToOneBridge(f1,f2)

    h1 = pynbody.halo.number_array.HaloNumberCatalogue(f1, ignore=-1)
    h2 = pynbody.halo.number_array.HaloNumberCatalogue(f2, ignore=-1)

    b.count_particles_in_common(h1, h2)

def test_nonmonotonic_incomplete_bridge():
    # Test the non-monotonic bridge where not all the particles are shared
    f1 = pynbody.new(dm=10)
    f2 = pynbody.new(dm=9)
    f1['iord'] = np.arange(0,10)
    f2['iord'] = np.array([0,2,4,6,8,1,5,7,9])
    b = pynbody.bridge.OrderBridge(f1,f2,monotonic=False)

    assert (b(f1[:5])['iord']==np.array([0,1,2,4])).all()


def test_monotonic_incomplete_bridge():
    # Test the monotonic bridge where not all the particles are shared
    f1 = pynbody.new(dm=10)
    f2 = pynbody.new(dm=9)
    f1['iord'] = np.arange(0, 10)
    f2['iord'] = np.array([0, 1, 2, 4, 5 ,6, 7, 8, 9])
    b = pynbody.bridge.OrderBridge(f1, f2, monotonic=False)

    assert (b(f1[:5])['iord'] == np.array([0, 1, 2, 4])).all()


def test_family_morphing():
    f1 = pynbody.new(dm=5, gas=5)
    f2 = pynbody.new(dm=10)

    # set dm and gas iords separately as it's possible the new command initialises them out of order
    f1.dm['iord'] = np.arange(0,5)
    f1.gas['iord'] = np.arange(5,10)
    f2['iord'] = np.array([0,2,4,6,8,1,3,5,7,9])

    b = pynbody.bridge.OrderBridge(f1,f2,monotonic=False,allow_family_change=True)

    assert (b(f2).dm['iord']==np.array([0,2,4,1,3])).all()
    assert (b(f2).gas['iord'] == np.array([6, 8, 5, 7, 9])).all()


def test_bridging_with_more_families():
    # Test that we can create a group array for snapshots that have a complex family structure,
    # and bridge only with one family (DM). This is necessary for e.g. Tangos linking
    f1 = pynbody.load("testdata/ramses_new_format_cosmo_with_ahf_output_00110")
    g1 = pynbody.halo.ahf.AHFCatalogue(f1, halo_numbers='v1')
    g1.load_all()
    # Work only on one family and create a useless bridge which is enough to break the code
    f1 = f1.dm
    b = pynbody.bridge.OrderBridge(f1, f1)

    with pytest.warns(DeprecationWarning):
        # The line below would fail with IndexError: index is out of bound when called for a single family
        # The family slicing is done wrto to the base ancestor, but the IDs were not offset
        # by family.start and would run over the array. This is now fixed
        mat = b.catalog_transfer_matrix(max_index=5, groups_1=g1, groups_2=g1,
                                        use_family=pynbody.family.dm, only_family=pynbody.family.dm)

    # Now it works and produces a diagonla matrix  since we are mapping the same snapshot
    assert(np.count_nonzero(mat - np.diag(np.diagonal(mat))) == 0)
    assert(mat[0, 0] == 12)

def test_fuzzy_match_only_one_family():
    f = pynbody.new(dm=10, gas=10)
    f['iord'] = np.arange(0,20)
    f['grp'] = np.array([0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1],dtype=np.int32)

    f2 = pynbody.new(dm=10, gas=10)
    f2['iord'] = np.arange(0, 20)
    f2['grp'] = f['grp']
    f2.dm['grp'] = np.array([0,0,0,0,0,1,1,1,2,2],dtype=np.int32)

    h = pynbody.halo.number_array.HaloNumberCatalogue(f)
    h2 = pynbody.halo.number_array.HaloNumberCatalogue(f2)
    b = pynbody.bridge.OrderBridge(f,f2)

    with pytest.warns(DeprecationWarning):
        assert b.fuzzy_match_catalog(use_family=pynbody.family.gas,groups_1=h, groups_2=h2)[1]==[(1, 1.0)]
        assert b.fuzzy_match_catalog(use_family=pynbody.family.dm, groups_1=h, groups_2=h2)[1] == [(1, 0.6), (2, 0.4)]
        assert b.fuzzy_match_catalog(groups_1=h, groups_2=h2)[1] == [(1, 0.8), (2, 0.2)]

        # Test that it also works with only_family:
        assert b.fuzzy_match_catalog(only_family=pynbody.family.gas, groups_1=h, groups_2=h2)[1] == [(1, 1.0)]
        assert b.fuzzy_match_catalog(only_family=pynbody.family.dm, groups_1=h, groups_2=h2)[1] == [(1, 0.6), (2, 0.4)]

@pytest.fixture
def snapshot_pair():
    f1 = pynbody.new(dm=100)
    f1['iord'] = np.arange(0, 100)
    f1['grp'] = np.array([0] * 20 + [1] * 10 + [2] * 30 + [4] * 40, dtype=np.int32)

    f2 = pynbody.new(dm=100)
    order = np.random.permutation(100)
    f2['iord'] = order
    f2['grp'] = f1['grp'][order]
    f2['grp'][f2['grp'] == 0] = 1  # imagine that halo 0 merged with halo 1

    # imagine that halo 4 split into lots of little things so we won't get a match
    f2['grp'][f2['grp'] == 4] = np.arange(4, 44)

    return f1, f2

def test_match_halos(snapshot_pair):

    f1, f2 = snapshot_pair

    b = pynbody.bridge.OrderBridge(f1, f2, monotonic=False)
    h1 = f1.halos()
    h2 = f2.halos()

    matched = b.match_halos(h1,h2)

    assert matched.keys() == {0,1,2,4}
    expected_matches = {0:1,1:1,2:2,4:-1}

    for k,v in expected_matches.items():
        assert matched[k] == v


    matched_on_index = b.match_halos(h1,h2,use_halo_indexes=True)

    for k,v in expected_matches.items():
        k_id = h1.number_mapper.number_to_index(k)
        if v != -1:
            v_id = h2.number_mapper.number_to_index(v)
        else:
            v_id = -1
        assert matched_on_index[k_id] == v_id

    assert matched_on_index[0] == 0


    b = pynbody.bridge.OrderBridge(f2, f1, monotonic=False)
    matched_reverse = b.match_halos(h2,h1)
    assert matched_reverse.keys() == set(f2['grp'])
    assert matched_reverse[1] == 0
    assert matched_reverse[2] == 2
    for i in range(4,44):
        assert matched_reverse[i] == 4

def test_fuzzy_match_halos(snapshot_pair):
    f1, f2 = snapshot_pair

    b = pynbody.bridge.OrderBridge(f1, f2, monotonic=False)
    h1 = f1.halos()
    h2 = f2.halos()

    fuzzy_match = b.fuzzy_match_halos(h1, h2)

    assert fuzzy_match.keys() == {0, 1, 2, 4}

    assert fuzzy_match[0] == [(1, 1.0)]
    assert fuzzy_match[1] == [(1, 1.0)]
    assert fuzzy_match[2] == [(2, 1.0)]

    for i in range(4, 44):
        assert (i, 1. / 40) in fuzzy_match[4]

    b = pynbody.bridge.OrderBridge(f2, f1, monotonic=False)
    fuzzy_match_rev = b.fuzzy_match_halos(h2, h1)

    assert fuzzy_match_rev.keys() == set(f2['grp'])

    assert fuzzy_match_rev[1] == [(0, 2./3), (1, 1./3)]
