import pynbody
import numpy as np
import glob
import os


def setup():
    global f, h
    f = pynbody.load("testdata/g15784.lr.01024")
    h = f.halos()


def teardown():
    global f, h
    del f, h


def test_center():
    global f, h
    with pynbody.analysis.halo.center(h[1]):
        np.testing.assert_allclose(
            f['pos'][0], [-0.0137471,  -0.00208458, -0.04392379], atol=1e-4)


def test_align():
    global f, h
    with pynbody.analysis.angmom.faceon(h[1]):
        np.testing.assert_allclose(f['pos'][:2], [[-0.01069893, -0.00150329, -0.04478709],
                                                  [-0.01000654,  0.00244104, -0.04465359]], atol=1e-5)

        np.testing.assert_allclose(f['vel'][:2], [[0.02047303,  0.01907281, -0.01987804],
                                                  [0.05459918,  0.02794922, -0.01030767]], atol=1e-5)


def test_virialradius():
    global f, h
    with pynbody.analysis.halo.center(h[1]):
        vrad = pynbody.analysis.halo.virial_radius(f)
        np.testing.assert_allclose(vrad, 0.005946911872, atol=1.e-5)
