import warnings

import numpy.testing as npt
import pytest

import pynbody

pytestmark = pytest.mark.filterwarnings("ignore:Unable to infer units from HDF attributes")
@pytest.fixture
def snap():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pynbody.load("testdata/gadget4_subfind_HBT/snapshot_034.hdf5")

@pytest.fixture
def halos(snap):
    return pynbody.halo.hbtplus.HBTPlusCatalogue(snap)

@pytest.fixture
def halos_length_ordered(snap):
    return pynbody.halo.hbtplus.HBTPlusCatalogue(snap, halo_numbers='length-order')

@pytest.fixture
def subfind_groups(snap):
    return pynbody.halo.subfindhdf.Gadget4SubfindHDFCatalogue(snap)

@pytest.mark.parametrize('load_all', [True, False])
def test_membership(halos, load_all):
    if load_all:
        halos.load_all()
    assert (halos[0]['iord'] == [3107002, 3148974, 3148457, 3148464, 3106492, 3147952, 3106995,
                                 3106485, 3148466, 3147949, 3148462, 3148975, 3106482, 3148463,
                                 3106475, 3147956, 3148472, 3188900, 3147944, 3148468, 3147429,
                                 3106484, 3230888, 3147935, 3148469, 3147946, 3148467, 3230879,
                                 3147945, 3148465, 3189411, 3189407, 3189418, 3147957, 3106480,
                                 3148982, 3106991, 3230878, 3106997, 3105964, 3106994]).all()
    assert ( halos[100]['iord'] == [4510177, 4467806, 4467811, 4467165, 4510176, 4467802, 4510807,
                                    4467809, 4468446, 4467161, 4468439, 4467807, 4467805, 4510814,
                                    4467808, 4467812, 4467813, 4467163, 4467803, 4468450, 4424805,
                                    4467172, 4467799, 4467810, 4468443, 4510809, 4424804, 4424802,
                                    4510816, 4510167, 4467164, 4425435, 4510174, 4510169, 4467804,
                                    4510818, 4424803]).all()

def test_load(snap):
    h = snap.halos(priority=["HBTPlusCatalogue", "Gadget4SubfindHDFCatalogue"])
    assert isinstance(h, pynbody.halo.hbtplus.HBTPlusCatalogue)

def test_properties_one_halo(halos):
    assert halos[0].properties['Nbound'] == 41
    assert halos[0].properties['TrackId'] == 0
    assert halos[79].properties['TrackId'] == 92
    npt.assert_allclose(halos[0].properties['Mbound'], 0.0052275728)
    npt.assert_allclose(halos[79].properties['Mbound'], 0.004462562)
    npt.assert_allclose(halos[5].properties['ComovingAveragePosition'],
                        [24.523391, 28.355526, 25.766893])

def test_properties_all_halos(halos):
    properties = halos.get_properties_all_halos()
    assert properties['Nbound'][0] == 41
    assert properties['TrackId'][0] == 0
    assert properties['TrackId'][79] == 92
    npt.assert_allclose(properties['Mbound'][0], 0.0052275728)
    npt.assert_allclose(properties['Mbound'][79], 0.004462562)
    npt.assert_allclose(properties['ComovingAveragePosition'][5],
                        [24.523391, 28.355526, 25.766893])

    assert properties['ComovingAveragePosition'].shape == (len(halos), 3)

def test_number_by_trackid(snap):
    halos_by_trackid = pynbody.halo.hbtplus.HBTPlusCatalogue(snap, halo_numbers='track')
    assert halos_by_trackid[3].properties['TrackId'] == 3

def test_number_by_length(halos_length_ordered):
    h = halos_length_ordered

    assert h[0].properties['Nbound'] == 240058
    assert len(h[0]) == 240058

def test_subsub_halos(halos, halos_length_ordered):
    h = halos_length_ordered
    assert (h[0].properties['children'] == [  75,   79,  106,  111,  147,  148,  174,  240,
                                              460,  216,  179,  496,  481, 2301,
                                              783,  516,  552,  865,  233,  341,  291,  980,
                                              949,  856,  889,  420, 1121, 1001, 1801, 1513, 1195,
                                              39, 2084, 862]).all()
    assert h[0].properties['parent'] == -1
    assert h[75].properties['parent'] == 0

    assert len(h[0].subhalos) == len(h[0].properties['children'])



    assert (halos[237].properties['children'] == [1600]).all()
    assert len(halos[2].properties['children'])==0
    assert halos[1600].properties['parent'] == 237
    assert halos[0].properties['parent'] == -1

    all_properties = halos.get_properties_all_halos()
    assert all_properties['parent'][1600] == 237
    assert (all_properties['children'][237] == [1600]).all()

def test_with_group_cat(halos, subfind_groups):
    combined_catalogue = halos.with_groups_from(subfind_groups)

    assert len(combined_catalogue) == len(subfind_groups)

    assert combined_catalogue[0] == subfind_groups[0]

    children_of_0 = [  47,  209,  328,  418,  419,  479,  558,  633,  634,  675,
                   697, 732,  800,  801,  925,  951, 1233, 1341, 1358, 1739,
                   1764, 1767, 1775, 1804, 1818, 1821, 1841, 1866, 1875, 1882,
                   1923, 1951, 1953, 1996, 2029]
    assert (combined_catalogue[0].properties['children'] ==
                children_of_0).all()
    assert combined_catalogue[0].subhalos[0] == halos[47]
    assert combined_catalogue[0].subhalos[1] == halos[209]
    assert len(combined_catalogue[0].subhalos) == 35

    # check subfind halos are undisturbed
    assert subfind_groups[0].properties['children'][0] == 0

    properties = combined_catalogue.get_properties_all_halos()
    assert (properties['children'][0] == children_of_0).all()
