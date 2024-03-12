import pytest
import warnings
import numpy.testing as npt

import pynbody

pytestmark = pytest.mark.filterwarnings("ignore:Unable to infer units from HDF attributes")
@pytest.fixture
def snap():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pynbody.load("/Users/app/Science/pynbody/HBT/tutorial_gadget4/snapshot_034.hdf5")

@pytest.fixture
def halos(snap):
    return pynbody.halo.hbtplus.HBTPlusCatalogue(snap)


def test_membership(halos):
    halos[0]['iord']

def test_properties_one_halo(halos):
    assert halos[0].properties['Nbound'] == 41
    assert halos[0].properties['TrackId'] == 0
    assert halos[79].properties['TrackId'] == 92
    npt.assert_allclose(halos[0].properties['Mbound'], 0.0052275728)
    npt.assert_allclose(halos[79].properties['Mbound'], 0.004462562)

def test_number_by_trackid(snap):
    halos_by_trackid = pynbody.halo.hbtplus.HBTPlusCatalogue(snap, halo_numbers='track')
    assert halos_by_trackid[3].properties['TrackId'] == 3

def test_number_by_length(snap):
    h = pynbody.halo.hbtplus.HBTPlusCatalogue(snap, halo_numbers='length-order')
    assert h[0].properties['Nbound'] == 240058
    assert len(h[0]) == 240058