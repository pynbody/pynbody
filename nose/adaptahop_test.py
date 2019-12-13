import pynbody
from pynbody.halo.adaptahop import AdaptaHOPCatalogue

import numpy as np


def test_load_adaptahop_catalogue():
    f = pynbody.load('testdata/output_00080')
    h = f.halos()
    assert len(h) == h._headers['nhalos'] + h._headers['nsubs']

def test_load_one_halo():
    f = pynbody.load('testdata/output_00080')
    h = f.halos()
    np.testing.assert_allclose(h[1].properties['members'], h[1]['iord'])