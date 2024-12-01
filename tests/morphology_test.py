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
