import numpy as np
import pytest

import pynbody
from pynbody.test_utils.gadget4_subfind_reader import Halos


def setup_module():
    global snap, halos, subhalos, htest, snap_arepo, halos_arepo, subhalos_arepo, htest_arepo
    with pytest.warns(UserWarning, match="Masses are either stored in the header or have another dataset .*"):
        snap = pynbody.load('testdata/gadget4_subfind/snapshot_000.hdf5')
        snap_arepo = pynbody.load('testdata/arepo/cosmobox_015.hdf5')

    halos = pynbody.halo.subfindhdf.Gadget4SubfindHDFCatalogue(snap)
    subhalos = pynbody.halo.subfindhdf.Gadget4SubfindHDFCatalogue(snap, subs=True)
    htest = Halos('testdata/gadget4_subfind/', 0)

    halos_arepo = pynbody.halo.subfindhdf.ArepoSubfindHDFCatalogue(snap_arepo)
    subhalos_arepo = pynbody.halo.subfindhdf.ArepoSubfindHDFCatalogue(snap_arepo, subs=True)
    htest_arepo = Halos('testdata/arepo/', 15)


def teardown_module():
    global snap, halos, subhalos, htest, snap_arepo, halos_arepo, subhalos_arepo, htest_arepo
    del snap, halos, subhalos, htest, snap_arepo, halos_arepo, subhalos_arepo, htest_arepo


def test_catalogue():
    _h_nogrp = snap.halos()
    _subh_nogrp = snap.halos(subs=True)
    _harepo_nogrp = snap_arepo.halos()
    _subharepo_nogrp = snap_arepo.halos(subs=True)
    for h in [halos, subhalos, _h_nogrp, _subh_nogrp, halos_arepo, subhalos_arepo, _harepo_nogrp, _subharepo_nogrp]:
        assert(isinstance(h, pynbody.halo.subfindhdf.Gadget4SubfindHDFCatalogue)), \
            "Should be a Gadget4SubfindHDFCatalogue catalogue but instead it is a " + str(type(h))

def test_lengths():
    assert len(halos)==299
    assert len(subhalos)==343
    assert len(halos_arepo)==447
    assert len(subhalos_arepo)==475


@pytest.mark.parametrize('mode', ('gadget4', 'arepo'))
@pytest.mark.parametrize('subhalos', (True, False))
def test_halo_or_subhalo_properties(mode, subhalos):

    halos_str = 'subhalos' if subhalos else 'halos'
    if mode == 'gadget4':
        comparison_catalogue, pynbody_catalogue = htest.load()[halos_str], snap.halos(subs=subhalos)
    elif mode=='arepo':
        comparison_catalogue, pynbody_catalogue = htest_arepo.load()[halos_str], snap_arepo.halos(subs=subhalos)
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
def test_halo_loading() :
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

def test_subhalos():
    assert len(halos[1].subhalos) == 8
    assert len(halos[1].subhalos[2]) == 91
    assert halos[1].subhalos[2].properties['halo_number'] == 22

@pytest.mark.filterwarnings("ignore:Unable to infer units from HDF attributes", "ignore:Accessing multiple halos")
def test_particle_data():
    hids = np.random.choice(range(len(halos)), 5)
    for hid in hids:
        assert(np.allclose(halos[hid].dm['iord'], htest[hid]['iord']))

@pytest.mark.filterwarnings("ignore:Masses are either stored")
def test_progenitors_and_descendants():
    f = pynbody.load("testdata/gadget4_subfind_HBT/snapshot_034.hdf5")
    h = f.halos()
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
