import numpy as np
import pytest

import pynbody
from pynbody.test_utils.gadget4_subfind_reader import Halos

# tell pytest not to raise warnings across this module
pytestmark = pytest.mark.filterwarnings("ignore:Masses are either stored")


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gadget", "arepo", "hbt", "tng_subfind")


@pytest.fixture
def snap():
    with pytest.warns(UserWarning, match="Masses are either stored in the header or have another dataset .*"):
        return pynbody.load('testdata/gadget4_subfind/snapshot_000.hdf5')

@pytest.fixture
def halos(snap):
    return pynbody.halo.subfindhdf.Gadget4SubfindHDFCatalogue(snap)

@pytest.fixture
def subhalos(snap):
    return pynbody.halo.subfindhdf.Gadget4SubfindHDFCatalogue(snap, subhalos=True)

@pytest.fixture
def htest():
    return Halos('testdata/gadget4_subfind/', 0)

@pytest.fixture
def snap_arepo():
    with pytest.warns(UserWarning, match="Masses are either stored in the header or have another dataset .*"):
        return pynbody.load('testdata/arepo/cosmobox_015.hdf5')

@pytest.fixture
def halos_arepo(snap_arepo):
    return pynbody.halo.subfindhdf.ArepoSubfindHDFCatalogue(snap_arepo)

@pytest.fixture
def subhalos_arepo(snap_arepo):
    return pynbody.halo.subfindhdf.ArepoSubfindHDFCatalogue(snap_arepo, subhalos=True)

@pytest.fixture
def htest_arepo():
    return Halos('testdata/arepo/', 15)


def test_catalogue(snap, snap_arepo, halos, subhalos, halos_arepo, subhalos_arepo):
    _h_nogrp = snap.halos()
    _subh_nogrp = snap.halos(subhalos=True)
    _harepo_nogrp = snap_arepo.halos()
    _subharepo_nogrp = snap_arepo.halos(subhalos=True)
    for h in [halos, subhalos, _h_nogrp, _subh_nogrp, halos_arepo, subhalos_arepo, _harepo_nogrp, _subharepo_nogrp]:
        assert(isinstance(h, pynbody.halo.subfindhdf.Gadget4SubfindHDFCatalogue)), \
            "Should be a Gadget4SubfindHDFCatalogue catalogue but instead it is a " + str(type(h))

def test_lengths(halos, subhalos, halos_arepo, subhalos_arepo):
    assert len(halos)==299
    assert len(subhalos)==343
    assert len(halos_arepo)==447
    assert len(subhalos_arepo)==475

def test_catalogue_from_filename_gadget4():
    snap = pynbody.load('testdata/gadget4_subfind/snapshot_000.hdf5')
    snap._filename = ""

    halos = snap.halos(filename='testdata/gadget4_subfind/fof_subhalo_tab_000.hdf5')
    assert isinstance(halos, pynbody.halo.subfindhdf.Gadget4SubfindHDFCatalogue)

def test_catalogue_from_filename_arepo():
    snap = pynbody.load('testdata/arepo/cosmobox_015.hdf5')
    snap._filename = ""

    halos = snap.halos(filename='testdata/arepo/fof_subhalo_tab_015.hdf5')
    assert isinstance(halos, pynbody.halo.subfindhdf.ArepoSubfindHDFCatalogue)

@pytest.mark.parametrize('mode', ('gadget4', 'arepo'))
@pytest.mark.parametrize('subhalo_mode', (True, False))
def test_halo_or_subhalo_properties(mode, subhalo_mode, halos, snap, htest, halos_arepo, snap_arepo, htest_arepo):

    halos_str = 'subhalos' if subhalo_mode else 'halos'
    if mode == 'gadget4':
        comparison_catalogue, pynbody_catalogue = htest.load()[halos_str], snap.halos(subhalos=subhalos)
    elif mode=='arepo':
        comparison_catalogue, pynbody_catalogue = htest_arepo.load()[halos_str], snap_arepo.halos(subhalos=subhalos)
    else:
        raise ValueError("Invalid mode")

    np.random.seed(1)
    hids = np.random.choice(range(len(pynbody_catalogue)), 20)

    for hid in hids:
        for key in list(comparison_catalogue.keys()):
            props = pynbody_catalogue.get_dummy_halo(hid).properties
            if key in list(props.keys()):
                value = props[key]
                if pynbody.units.is_unit(value):
                    orig_units = pynbody_catalogue.base.infer_original_units(value)
                    value = value.in_units(orig_units)
                np.testing.assert_allclose(value, comparison_catalogue[key][hid])

    pynbody_all = pynbody_catalogue.get_properties_all_halos()
    for key in list(comparison_catalogue.keys()):
        if key in pynbody_all.keys():
            np.testing.assert_allclose(pynbody_all[key], comparison_catalogue[key])

@pytest.mark.filterwarnings("ignore:Unable to infer units from HDF attributes")
def test_halo_loading(halos, htest, halos_arepo, htest_arepo) :
    """ Check that halo loading works """
    # check that data loading for individual fof groups works
    _ = halos[0]['pos']
    _ = halos[1]['pos']
    _ = halos[0]['mass'].sum()
    _ = halos[1]['mass'].sum()
    _ = halos_arepo[0]['pos']
    _ = halos_arepo[1]['pos']
    _ = halos_arepo[0]['mass'].sum()
    _ = halos_arepo[1]['mass'].sum()
    assert(len(halos[0]['iord']) == len(halos[0]) == htest.load()['halos']['GroupLenType'][0, 1])
    arepo_halos = htest_arepo.load()['halos']
    assert(len(halos_arepo[0]['iord']) == len(halos_arepo[0]) == np.sum(arepo_halos['GroupLenType'][0, :], axis=-1))

def test_subhalos(halos):
    assert len(halos[1].subhalos) == 8
    assert len(halos[1].subhalos[2]) == 91
    assert halos[1].subhalos[2].properties['halo_number'] == 22

@pytest.mark.filterwarnings("ignore:Unable to infer units from HDF attributes", "ignore:Accessing multiple halos")
def test_particle_data(halos, htest):
    hids = np.random.choice(range(len(halos)), 5)
    for hid in hids:
        assert(np.allclose(halos[hid].dm['iord'], htest[hid]['iord']))

@pytest.mark.filterwarnings("ignore:Masses are either stored")
def test_progenitors_and_descendants():
    # although this uses the HBT snapshot, we actually test for the subfind properties...
    f = pynbody.load("testdata/gadget4_subfind_HBT/snapshot_034.hdf5")
    h = f.halos()
    assert isinstance(h, pynbody.halo.subfindhdf.Gadget4SubfindHDFCatalogue)
    p = h[0].subhalos[0].properties
    match = {'FirstProgSubhaloNr': 0, 'NextDescSubhaloNr': 127, 'ProgSubhaloNr': 0,
             'SubhaloNr': 0, 'DescSubhaloNr': 0, 'FirstDescSubhaloNr': 0, 'NextProgSubhaloNr': 74}
    for k, v in match.items():
        assert p[k] == v

    p = h[3].subhalos[1].properties

    match = {'FirstProgSubhaloNr': 167, 'NextDescSubhaloNr': -1, 'ProgSubhaloNr': 167, 'SubhaloNr': 205,
             'DescSubhaloNr': 221, 'FirstDescSubhaloNr': 221, 'NextProgSubhaloNr': -1}
    for k, v in match.items():
        assert p[k] == v


@pytest.mark.filterwarnings("ignore:Masses are either stored")
@pytest.mark.filterwarnings("ignore:Incorrect number of ") # test data has only some of the files
def test_multifile_multipart_tng_halos():
    # This addresses a number of linked bugs in reading subfind data where there are multiple files, and
    # more than one particle type mapping into the same family. See #839
    f = pynbody.load("testdata/arepo/tng/snapdir_261/snap_261")
    h = f.halos()
    assert (h[0]['iord'][::100000] == [1079368176, 1073346509, 1079169082, 1076024094, 1078414281,
          1079318645, 1060773038, 1079036573, 1076276269, 1045575677,
          1056566404, 1063954410, 1078469798,   23152184,   20085564,
            24029526,   23927910,   20018798,   24197828,   21084457,
            23217921,   22226505,   18947605,   25089761,   21155098,
            18961540,   24820839,   21071365,   25913167,   18242487,
            22815450,   20701085,   22021963,   18445394,   22331098,
            18828642,   25965289,   19799793,   24274707,   20909980,
            23210270,   23462569,   19879522,   20108215,   24434760,
            17475958,   20648454,   18793266,   22666648,   20739052,
            23397149,   22953590,   17407270,   16811069,   20600621,
            23023164,   17306789,   21346551,   23646591,   20535924,
          1038202167, 1038385095, 1041854620, 1041212679, 1033619448,
          1038549232, 1033659336, 1043699534, 1031042903, 1029945620,
          1051418078, 1055911654, 1055831064, 1065659516, 1049027467,
          1061288281, 1056286673, 1067075166, 1028317439, 1077239322]).all()

    # here are two low-res particles that are actually PartType2 but lumped into the DM family by pynbody
    # these were previously assigned to wrong particles (PartType2 mapped into PartType1 particles by mistake)
    assert (h[0].dm['iord'][-2:] == [5801691, 5801450]).all()

    assert (h[0].subhalos[2]['iord'][::1000] == [1076590348, 1079252330, 1079343916, 1079237207, 1075658142,
          1075226456, 1074535510,   21365624,   21358512,   21406282,
            22432720,   21411320,   20340518,   20288656,   22321526,
            22377688,   21399799,   21349549,   22323202,   24413954,
            22256512,   21218148,   21360613,   21400353,   22373060,
            20343482,   22322281,   23345354,   22322474,   23341431,
            19413719,   22380145,   20346069,   22259777,   20346110,
          1061149399, 1076605146, 1048386188, 1019409154, 1019367044,
          1034899938]).all()

    assert (h[0].dm['iord'][-2:] == [5801691, 5801450]).all()


    assert ((h[1]['iord'][::100000]) == [1078937127, 1076029549, 1078029745,   19522035,   18616445,
            21423889,   22453094]).all()

    assert ((h[1].subhalos[3]['iord'][::100] == [26360186, 26359643, 26360908, 26356085, 25978477, 26396901,
          26356628, 26022586, 26360378, 25977706, 26360175, 26355601,
          26356851, 26356882, 26396677, 25978240]).all())

    assert (h[3].s['iord'][:5] == [1062632156, 1057152317, 1049782182, 1038218046, 1039576336]).all()
    # another test that PartType2 lumped into DM particles get picked up OK

    assert (h[1].subhalos[0].dm['iord'][-5:] == [5477082,  5477081,  5476835,  5346238,  5346239]).all()

def test_inconsistent_dtype_loading(snap_arepo):
    # this looks like an artificial example but can arise e.g. if rho is written float32

    snap_arepo.dm['rho'] = np.zeros(len(snap_arepo.dm), dtype='f4')

    assert snap_arepo.dm['rho'].dtype == np.dtype('float32')

    assert snap_arepo.gas['rho'].dtype == np.dtype('float64')

    with pytest.warns(RuntimeWarning, match="Data types of family arrays do not match"):
        assert snap_arepo.star['rho'].dtype == np.dtype('float32')

    assert snap_arepo['rho'].dtype == np.dtype('float32')

def test_eps_array(snap_arepo, snap):
    assert np.allclose(snap['eps'], 0.002)
    assert np.allclose(snap_arepo['eps'][[0,-1000,-1]],[0.0025, 0.0005, 0.04])
    assert snap['eps'].units == '3.085678e+24 cm a h^-1'
    assert snap_arepo['eps'].units == '3.085678e+24 cm a h^-1'
