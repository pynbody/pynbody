import numpy as np
import pytest

import pynbody
import pynbody.analysis.morphology as morph
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gasoline_ahf", "arepo")


@pytest.fixture
def gasoline_h0():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    h = f.halos()
    return h[0]

@pytest.fixture
def agora():
    f = pynbody.load("testdata/arepo/agora_100.hdf5")
    return f

@pytest.mark.filterwarnings("ignore:.*:UserWarning")
def test_jcirc(gasoline_h0, agora):
    for f in (gasoline_h0, agora):
        pynbody.analysis.faceon(f)
        morph.estimate_jcirc_from_energy(f)
        star_ratio = f.s['jz'] / f.s['j_circ']
        assert (star_ratio>-1.1).all()
        assert (star_ratio>1.1).sum() < 0.01 * len(star_ratio)
        assert (star_ratio>0.8).sum() > 0.2 * len(star_ratio)

        #p.hist(star_ratio, bins=100, histtype='step', density=True, range=(-1.5,1.5))
    #p.savefig('test.png')

@pytest.mark.filterwarnings("ignore:.*:UserWarning")
def test_jcirc_from_r(agora):

    # this really only makes sense in the disk plane, so we only check the isolated agora run
    f = agora

    pynbody.analysis.faceon(f)
    morph.estimate_jcirc_from_rotation_curve(f)

    star_ratio = f.s['jz'] / f.s['j_circ']

    assert (star_ratio > -1).sum() > 0.99 * len(star_ratio)
    assert (star_ratio > 1.1).sum() < 0.2 * len(star_ratio)
    assert (star_ratio > 0.5).sum() > 0.7 * len(star_ratio)

    #p.hist(star_ratio, bins=100, histtype='step', density=True, range=(-1.5, 4.5))
    #p.savefig('test.png')

def test_decomp(gasoline_h0):
    """This checks that decomp at least runs and produces some output. Whether the output is sensible or not
    is a physics question that is not addressed here and could do with looking at again in the future."""

    morph.decomp(gasoline_h0)
    assert (gasoline_h0.s['decomp'][::1000] == [3, 2, 2, 2, 2, 2, 3, 3, 2, 3, 3, 2, 3, 4, 3, 3, 3, 2, 5, 3, 2,
          2, 3, 3, 3, 2, 2, 5, 2, 1, 2, 1, 2, 4, 3, 3, 3, 3, 2, 4, 2, 3,
          2, 4, 3, 3, 1, 3, 2, 3, 5, 3, 2, 4, 2, 3, 2, 4, 5, 3, 5, 3, 2,
          5, 4, 3, 2, 2, 2, 2, 3, 2, 4, 3, 1, 3, 2, 5, 5, 3, 5, 4, 3, 3,
          5, 5, 3, 5, 3, 5, 4, 2, 4, 3, 3, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3,
          3, 3, 2, 4, 2, 3, 5, 3, 3, 3, 3, 3, 2, 5, 2, 2, 2, 4, 5, 3, 2,
          2, 4, 5, 3, 2, 2, 4, 5, 3, 3, 2, 1, 1, 4, 2, 2, 3, 4, 3, 5, 2,
          4, 5, 1, 4, 1, 2, 2, 2, 3, 2, 5, 3, 5, 4, 2, 5, 5, 3, 3, 5, 2,
          1, 4, 4, 4, 4, 3, 5, 1, 5, 1, 1, 3, 4, 3, 2, 1, 2, 2, 4, 2, 1,
          4, 3, 4, 1, 2, 1, 1, 2, 5, 5, 4, 2, 2, 1, 5, 3, 4, 4, 4, 1, 3,
          4, 4, 5, 3, 4, 1, 3, 1, 1, 4, 3, 1, 1, 4, 3, 3, 3, 4, 4, 1, 3,
          5, 3, 1, 1, 1, 1, 5, 5, 3, 2, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1]).all()
