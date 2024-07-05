import pytest

import pynbody
import pynbody.test_utils
from pynbody.halo.hop import HOPCatalogue


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("ramses")


@pytest.fixture
def f():
    yield pynbody.load("testdata/ramses/output_00080")


@pytest.fixture
def halos(f):
    yield HOPCatalogue(f)

def test_get_halo(halos):
    assert len(halos[0]) == 37927
    assert len(halos[1]) == 20870
    assert (halos[1].dm['iord'][::1000] == [950469, 178234, 570544, 602770, 983503, 185854, 189546, 183353,
                                           971918, 189890, 981389, 978381, 184919, 201405, 227622, 180288,
                                           194977, 195167, 187596, 192792, 590108]).all()
    assert len(halos) == 369

def test_autoload_halos(f):
    halos = f.halos(priority=["HOPCatalogue"])
    assert isinstance(halos, HOPCatalogue)
    assert len(halos[0]) == 37927

def test_autoload_halos_by_filename(f):
    halos = f.halos(filename="testdata/ramses/output_00080/grp00080.tag")

    assert isinstance(halos, HOPCatalogue)
    assert len(halos[0]) == 37927
