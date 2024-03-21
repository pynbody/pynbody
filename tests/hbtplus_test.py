import warnings

import numpy as np
import numpy.testing as npt
import pytest

import pynbody

pytestmark = pytest.mark.filterwarnings("ignore:Unable to infer units from HDF attributes")

@pytest.fixture(params=[True, False], ids=['multifile', 'single-file'])
def snap(request):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if request.param:
            # multifile
            return pynbody.load("testdata/gadget4_subfind_HBT_multifile/snapshot_034.hdf5")
        else:
            return pynbody.load("testdata/gadget4_subfind_HBT/snapshot_034.hdf5")

@pytest.fixture
def snap_type(request):
    if 'multifile' in request.node.callspec.id:
        mode = 'multifile'
    else:
        mode = 'single-file'
    return mode

@pytest.fixture
def halos(snap):
    return pynbody.halo.hbtplus.HBTPlusCatalogue(snap)

@pytest.fixture
def halos_length_ordered(snap):
    return pynbody.halo.hbtplus.HBTPlusCatalogue(snap, halo_numbers='length-order')

@pytest.fixture
def subfind_groups(snap):
    return pynbody.halo.subfindhdf.Gadget4SubfindHDFCatalogue(snap)

@pytest.mark.parametrize('load_all', [True, False],
                         ids=['load_all', 'no_load_all'])
def test_membership(halos, load_all, snap_type):
    if load_all:
        halos.load_all()

    h0_iords = {
        'single-file': [3107002, 3148974, 3148457, 3148464, 3106492, 3147952, 3106995,
                                 3106485, 3148466, 3147949, 3148462, 3148975, 3106482, 3148463,
                                 3106475, 3147956, 3148472, 3188900, 3147944, 3148468, 3147429,
                                 3106484, 3230888, 3147935, 3148469, 3147946, 3148467, 3230879,
                                 3147945, 3148465, 3189411, 3189407, 3189418, 3147957, 3106480,
                                 3148982, 3106991, 3230878, 3106997, 3105964, 3106994],
        'multifile': [2421127, 2390931, 2390926, 2390928, 2390929, 2390930, 2390537,
          2421135, 2421519, 2421131, 2391310, 2390932, 2390923, 2390544,
          2390538, 2420741, 2390927, 2391311, 2391315, 2421134, 2391316,
          2421137, 2390919, 2390933, 2421518, 2390539, 2390531, 2390541,
          2391317]
    }[snap_type]

    assert (halos[0]['iord'] == h0_iords).all()

    h100_iords = {
        'single-file': [4510177, 4467806, 4467811, 4467165, 4510176, 4467802, 4510807,
                                    4467809, 4468446, 4467161, 4468439, 4467807, 4467805, 4510814,
                                    4467808, 4467812, 4467813, 4467163, 4467803, 4468450, 4424805,
                                    4467172, 4467799, 4467810, 4468443, 4510809, 4424804, 4424802,
                                    4510816, 4510167, 4467164, 4425435, 4510174, 4510169, 4467804,
                                    4510818, 4424803],
        'multifile': [3954820, 3954182, 3865157, 3954827, 3865156, 3909185, 3954189,
          3954176, 3909826, 3909827, 3865785, 3954828, 3865787, 3909184,
          3909831, 3910463, 3998209, 3909182, 3998849, 3865794, 3954179,
          3954180, 3865143, 4042248, 3955466, 3865152, 3865802, 4042881,
          3998214, 3908549, 3865145, 3998852, 3909197, 3997576, 3954818,
          3954814, 3998847, 3908556, 3777730, 3909189, 3953548, 3865796,
          3909822, 3998208, 3909196, 4042890, 3954184, 3955457, 3998861,
          3864512, 3954824, 3954178, 3955459, 3955463, 4086921, 3821115,
          3954174, 3909183, 3909192, 3909195, 3955467, 3998207, 3955458,
          3998851, 3999491, 3865799, 3910470, 4042885, 3999500, 3998853,
          4086918, 3864516, 3865154, 3908545, 3954185, 3909187, 3821766,
          3864517, 3864521, 4043523, 3954829, 3821126, 3909186, 3954181,
          3953550, 4086917, 3821128, 3999495, 3821131, 3954815, 4042252,
          4043530, 3865165, 3954188, 4086284, 3955461, 3954832, 3954822,
          3821763, 3909823, 3866435, 3821764, 3820485, 3999498, 3998850,
          3998857, 3909190, 3910466, 3953543, 3955455, 3821118, 3998854,
          3909828, 3864520, 4042246, 4087551, 3998217, 3999499, 4043528,
          3953537, 3910462, 3954817, 3998856, 3909194, 3998220, 3954819,
          3909188, 4042888, 3909191, 4086911, 3954834, 3865803, 3998213,
          3910467, 3998212, 3953535, 3998222, 3954836, 3821117, 4086920,
          3866431, 3953547, 3821119, 3909204, 4042891, 3909830, 3954825,
          3955454, 3865158, 4087558, 4042878, 3910471, 3910458, 3954177,
          3865792, 4042241, 3865150, 3955464, 3865805, 3908555, 4086285,
          4086912, 3955460, 3909835, 4087554, 4086281, 3909824, 3864524,
          3908542, 3953545, 3954821, 3999487, 3864522, 3821125, 3998860,
          3865804, 3865153, 3821122, 4086276, 3821130, 4042886, 3909836,
          3998221, 3866442, 3956098, 4041601, 3999493, 4087550, 4086916,
          3910465, 3998215, 3997580, 3865163, 3954816, 3908552, 4042239,
          3821759, 3999492, 3998218, 4086924, 3909825, 4086278, 4000135,
          3821123, 3998864, 4086914, 3998216, 4086277, 4042249, 3955474,
          3865144, 3821133, 3821757, 3866423, 4042883, 4042256, 3997577,
          3821770, 3910475, 4042244, 3953549, 3955465, 3821109, 3777092,
          3865155, 3866430, 3999494, 4042879, 3821758, 3865164, 3864511,
          3908537, 3908564, 3909832, 3999489, 4042880, 3998855, 3997581,
          4043526, 4086913, 3865782, 3954175, 4042247, 3908553, 3821121,
          3909174, 4086910, 3866437, 4043527, 3998858, 3999486, 4042896,
          3909829, 3998846, 3954186, 3999496, 3865162, 3865798, 3821120,
          3821756, 3953536, 3865789, 3998859, 4043522, 3864514, 3910476,
          3821114, 3909842, 3999488, 3863880, 3955468, 3866436, 3864513,
          3999490, 3998862, 3864525, 4086919, 3998211, 3821755, 3954194,
          3865793, 3909837, 3997573, 3910469, 4086273, 3909179, 4000126,
          3821750, 3865783, 3908557, 3956100, 3865151, 3998848, 3956094,
          4042893, 3998226, 3907917, 3910464, 3997572, 3953544, 4086279,
          3999506, 3997584, 4000132, 4042253, 3954187, 3998228, 3866434,
          3909202, 3911108, 3865160, 3777090, 4043519, 3956103, 3777085,
          3821132, 3867067, 3865148, 3953546, 3909818, 4086282, 3998219,
          3908543, 4042254, 3909181, 3953552, 3999501, 3821113, 4042892,
          3909198, 3909834, 4043518, 3997567, 3866427, 4087559, 3777722,
          3954823, 4086292, 3907913, 3956107, 3777084, 3777724, 3955456,
          4043521, 3864509, 3908550, 3911102, 4086915, 3777093, 4000127,
          4088190, 4041612, 3956102, 3866425, 3956097, 4042882, 4086928,
          3821762, 3820479, 4131586, 4086275, 3910477, 4044162, 3821111,
          3955470, 3821112, 3820484, 4131598, 4044167, 4131585, 3866429,
          3909821, 3821752, 3821116, 3955462, 3864523, 3820478, 4041616,
          3865142, 3821751, 3819845, 3952905, 4087557, 3910474, 4042245,
          4042240, 4086923, 3865791, 3956099, 3998866, 4174985, 4043533,
          3865797, 3820481, 4041609, 3956101, 3954826, 3820488, 4131584,
          3909177, 4175617, 4131589, 3910459, 4087552, 3910468, 3910454,
          3910472, 3998868, 3910456, 4042894, 4131602, 4042887, 3865147,
          4131591, 3996937, 4042900, 3865788, 4042260, 3863884, 4131583,
          4130951, 3864519, 3999502, 4000128, 3999497, 4175623, 4130949,
          3866426, 3820483, 4086922, 3820473, 3911106, 3956104, 3821747,
          4000129, 4086926, 3820482, 4086288, 3865795, 3777717, 4000774,
          3822383, 3820490, 4085641, 3998206, 3998210, 4043534, 4000767,
          4041605, 3909844, 4044166, 3821742, 3911107, 4088191, 3956106,
          3822390, 4130957, 4044163, 4000772, 3777720, 4042898, 3778362,
          4043524, 4088194, 3954190, 4043520, 3821107, 3777713, 4044168,
          3955469, 4041606, 4000131, 3778355, 3867062, 3911101, 4000773,
          3953556, 3733683, 4000766, 4041614, 4041608, 3777081, 3777725,
          3909178, 3953554, 3822396, 4000769, 3954192, 3952916, 3996941,
          3997575, 3777088, 3822387, 4042884, 3910461, 3908551, 4000136,
          4130953, 3907911, 3997574, 3997586, 3908554, 3822391, 3956740,
          3911095, 3821744, 3956734, 3777077, 3865149, 4042258]
    }[snap_type]
    assert ( halos[100]['iord'] == h100_iords ).all()

def test_load(snap):
    h = snap.halos(priority=["HBTPlusCatalogue", "Gadget4SubfindHDFCatalogue"])
    assert isinstance(h, pynbody.halo.hbtplus.HBTPlusCatalogue)

@pytest.fixture
def expected_properties(snap_type):
    return {
        'single-file': {0: {'Nbound': 41, 'TrackId': 0, 'Mbound': 0.0052275728,},
                        79: {'TrackId': 92, 'Mbound': 0.004462562},
                        5: {'ComovingAveragePosition': [24.523391, 28.355526, 25.766893]}},
        'multifile': {0: {'Nbound': 29,  'TrackId': 333},
                      2000: {'Nbound': 31, 'TrackId': 1998}}
    }[snap_type]

def test_properties_one_halo(halos, expected_properties):
    for hnum, properties in expected_properties.items():
        for k, v in properties.items():
            if isinstance(v, int):
                assert halos[hnum].properties[k] == v
            else:
                npt.assert_allclose(halos[hnum].properties[k], v)

def test_properties_all_halos(halos, expected_properties):
    all_props = halos.get_properties_all_halos()
    for hnum, properties in expected_properties.items():
        for k, v in properties.items():
            if isinstance(v, int):
                assert all_props[k][hnum] == v
            else:
                npt.assert_allclose(all_props[k][hnum], v)

def test_number_by_trackid(snap):
    halos_by_trackid = pynbody.halo.hbtplus.HBTPlusCatalogue(snap, halo_numbers='track')
    assert halos_by_trackid[3].properties['TrackId'] == 3

def test_number_by_length(halos_length_ordered, snap_type):
    h = halos_length_ordered

    if snap_type == 'single-file':
        assert h[0].properties['Nbound'] == 240058
        assert len(h[0]) == 240058
    else:
        assert h[0].properties['Nbound'] == 239880
        assert len(h[0]) == 239880



@pytest.fixture
def subsub_halos_expectations(snap_type):
    expectations = {
        'multifile': {
            'children-of-0': [3, 11, 39, 50, 80, 81, 121, 192, 214, 238, 258,
                      545, 579, 612, 614, 655, 668, 533, 279, 747, 702, 918,
                      986, 894, 1279, 691, 1376, 789, 1675, 1670, 1556, 2256, 2229,
                      1302],
            'children-of-200': [333],
            'children-of-unsorted-477': [ 480,  486,  564,  594,  647,  608,  671,  732,  730,  708,  667,
                                       1378, 1292,  894, 1490, 1414,  599, 1297,  675,  991,  806, 1674,
                                       1680, 1426, 1949, 1148, 1904, 1091, 1717, 1373, 1682, 1456, 1678,
                                       1428],
            'children-of-unsorted-1449': [1707]


        },
        'single-file': {
            'children-of-0': [3, 11, 39, 50, 80, 81, 121, 192, 214, 238, 258,
                        544, 579, 614, 613, 656, 667, 533, 279, 753, 707, 916,
                        983, 893, 1265, 695, 1361, 788, 1665, 1662, 1547, 2251, 2223,
                        1283],
            'children-of-200': [332],
            'children-of-unsorted-1764': [1767, 1775, 1804, 1818, 1841, 1821, 1875, 1953, 1951, 1923, 1866,
                                          675, 633, 2029, 732, 697, 209, 634, 1882, 479, 1996, 800,
                                          801, 418, 951, 558, 925, 328, 1358, 1233, 1341, 1739, 47,
                                          419],
            'children-of-unsorted-1805': [1273]
        }
    }[snap_type]

    return expectations

@pytest.mark.parametrize('load_all', [True, False])
@pytest.mark.filterwarnings("ignore:Accessing multiple halos")
def test_subsub_halos(halos, halos_length_ordered, subsub_halos_expectations,
                      load_all):
    if load_all:
        halos.load_all()
        halos_length_ordered.load_all()

    for k, children in subsub_halos_expectations.items():
        if 'unsorted' in k:
            use_halos = halos
        else:
            use_halos = halos_length_ordered

        parent_num = int(k.split('-')[-1])

        assert (use_halos[parent_num].properties['children'] == children).all()

        assert use_halos[parent_num].properties['parent'] == -1
        for subsub in children:
            assert use_halos[subsub].properties['parent'] == parent_num

@pytest.fixture
def children_of_group_0(snap_type):
    children_of_0 = {
        'multifile': [1419, 1308, 1302, 1485,  917,  523, 1683,  859,  239, 1527,  959,
        232,  918, 1063, 1151, 1426, 1922, 1977,   91,  801, 1305, 1654,
       2212, 1713, 2347,  146,  397,  931, 1644, 1840, 1901, 2085, 2052,
       1062, 1254],
        'single-file': [1813, 1980,  662,  607,  627,  643,  217,  211,  215,  309,  280,
        347,  457,  462, 2178, 1037,  963, 1184, 1485, 1614,  949, 1378,
       1534, 1275, 1656, 1693,   31, 1642, 1751, 2318, 1757,  156,  181,
        132, 2236],
    }
    return children_of_0[snap_type]

def test_with_group_cat(halos_length_ordered, subfind_groups, children_of_group_0):
    combined_catalogue = halos_length_ordered.with_groups_from(subfind_groups)

    assert len(combined_catalogue) == len(subfind_groups)

    assert combined_catalogue[0] == subfind_groups[0]

    assert (combined_catalogue[0].properties['children'] ==
                children_of_group_0).all()


    assert combined_catalogue[0].subhalos[0] == halos_length_ordered[children_of_group_0[0]]
    assert combined_catalogue[0].subhalos[1] == halos_length_ordered[children_of_group_0[1]]
    assert len(combined_catalogue[0].subhalos) == len(children_of_group_0)

    # check subfind halos are undisturbed
    assert subfind_groups[0].properties['children'][0] == 0

    properties = combined_catalogue.get_properties_all_halos()
    assert (properties['children'][0] == children_of_group_0).all()
