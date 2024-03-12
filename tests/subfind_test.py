import numpy as np
import pytest
from numpy.testing import assert_allclose

import pynbody


@pytest.fixture(params=[True, False])
def snap(request):
    f = pynbody.load("testdata/subfind/snapshot_019")
    if request.param:
        f['iord'] = np.random.permutation(f['iord'])
    return f

@pytest.fixture
def halos(snap):
    return pynbody.halo.subfind.SubfindCatalogue(snap, subs=True)

@pytest.fixture
def groups(snap):
    return pynbody.halo.subfind.SubfindCatalogue(snap, subs=False)


def test_lengths(halos, groups):
    assert len(groups)==2853
    assert len(halos)==3290

def test_autodetect_subfind(snap):
    assert isinstance(snap.halos(), pynbody.halo.subfind.SubfindCatalogue)

def test_group_properties(groups):
    assert_allclose(float(groups[3].properties['mmean_200']), 1.22e14, rtol=1.e-2)
    assert_allclose(float(groups[3].properties['rtop_200']), 1.02, rtol=1.e-2)

@pytest.mark.parametrize('with_units', [True, False])
def test_all_group_properties(groups, with_units):
    properties = groups.get_properties_all_halos(with_units=with_units)
    assert_allclose(properties['mmean_200'][3], 12181.62, rtol=1.e-4)
    assert_allclose(properties['rtop_200'][3], 1.02, rtol=1.e-2)
    assert np.all(properties['children'][3] == np.arange(89,109))
    if with_units:
        assert properties['mmean_200'].units == '1e10 Msol h^-1'
        assert properties['rtop_200'].units == 'Mpc a h^-1'
    else:
        assert not hasattr(properties['mmean_200'], 'units')
        assert not hasattr(properties['rtop_200'], 'units')

def test_group_children(groups):
    assert np.all(groups[0].properties['children'] == np.arange(0, 39))
    assert np.all(groups[3].properties['children'] == np.arange(89,109))
    halo_nums_in_subs = [sh.properties['halo_number'] for sh in groups[3].subhalos]
    assert (halo_nums_in_subs == np.arange(89,109)).all()

    halos = groups.base.halos(subs=True)
    for sh_from_group, sh in zip(groups[3].subhalos, halos[89:109]):
        assert sh.properties['halo_number'] == sh_from_group.properties['halo_number']
        assert np.all(sh['iord'] == sh_from_group['iord'])

    # subs of subs also tested below in test_halo_children
    assert groups[1].subhalos[0].subhalos[0].properties['halo_number'] == 39

def test_halo_properties(halos):
    assert_allclose(halos[3].properties['sub_pos'], [ 0.4759067,  1.862322 , 33.249245 ])
    assert_allclose(float(halos[0].properties['sub_VMax']), 685., rtol=1.e-2)

@pytest.mark.parametrize('with_units', [True, False])
def test_all_halo_properties(halos, with_units):
    properties = halos.get_properties_all_halos(with_units=with_units)
    assert_allclose(properties['sub_pos'][3], [ 0.4759067,  1.862322 , 33.249245 ])
    assert_allclose(float(properties['sub_VMax'][0]), 685., rtol=1.e-2)
    for i in range(20):
        # NB this assumes the halo number = the halo offset, but that's ok for this file
        assert (properties['children'][i] == halos[i].properties['children']).all()
    if with_units:
        assert properties['sub_pos'].units == 'Mpc a h^-1'
        assert properties['sub_VMax'].units == 'km a**1/2 s^-1'

def test_halo_properties_physical(halos):
    halos.physical_units()
    assert_allclose(halos[3].properties['sub_pos'], [  558.2842,  2184.6821, 39004.55  ])

def test_halo_children(halos):
    assert np.all(halos[0].properties['children'] == [ 0,  1,  2,  3,  4,  6,  7, 12, 13, 14, 15, 16, 17, 20, 21, 22, 24,
        26, 32, 33, 35, 36, 37])
    assert np.all(halos[1].properties['children'] == [ 5,  8,  9, 11, 18, 19, 23, 25, 29, 30, 31, 34])
    assert halos[39].properties['sub_groupNr'] == 1
    assert np.all(halos[39].properties['children'] == [39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        58, 59, 61, 62, 63, 64, 65, 66, 69])

    for sh_a, expected_halo_num in zip(halos[39].subhalos,
                          [39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                           58, 59, 61, 62, 63, 64, 65, 66, 69]
                          ):
        sh_b = halos[expected_halo_num]
        assert sh_a.properties['halo_number'] == sh_b.properties['halo_number']
        assert np.all(sh_a['iord'] == sh_b['iord'])

def test_group_particles(groups):
    assert np.all(groups[0]['iord'][::1000] == [1902806, 2048342,   66771, 2083281, 2066768, 2000586,   35672,
          1966932, 1901524,   83913, 2035150, 1835347,   49759, 1836627,
            84691, 1900752,  132302, 2050529,   67936,  132548, 1950435,
          1936453, 2047847, 1851964,  214227, 1981505,  145348, 1883081,
          2095318, 1965514, 2079065,   46789, 2062554, 1914447,  228794,
          1932736, 2095170,  211516,   46020, 1914966, 1901016, 1933909,
            50884,  115538, 2028737,   81869,   79680, 1802047,  178240,
           146263])
    assert np.all(groups[10]['iord'][::1000] == [ 848941,  897584,  996144,  882349,  848173,  847673,  946356,
           733368,  997292, 1126063])


def test_halo_particles(halos):
    assert np.all(halos[0]['iord'][::1000] == [1902806, 2048342,   66771, 2083281, 2066768, 2000586,   35672,
          1966932, 1901524,   83913, 2035150, 1835347,   49759, 1836627,
            84691, 1900752,  132302, 2050529,   67936,  132548, 1950435,
          1936453, 2047847, 1851964,  214227])
    assert np.all(halos[39]['iord'][::1000] == [493394, 198751, 396116, 462038, 394458, 328930, 623070, 525789,
          280024, 623448, 245976, 608093, 608598, 494178, 491350, 507866,
          264776, 558562, 362319, 591560, 706259, 348265, 249821, 231233,
          474710, 231023, 380767, 413674, 589153, 459711, 576598, 671059,
          523615])
