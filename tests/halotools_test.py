import glob
import os
import time

import numpy as np
import pytest

import pynbody


def setup_module():
    global f, h
    f = pynbody.load("testdata/g15784.lr.01024")
    h = f.halos()


def teardown_module():
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
        np.testing.assert_allclose(f['pos'][:2], [[-0.010718, -0.001504, -0.044783],
                                                  [-0.010026,  0.002441, -0.04465 ]], atol=1e-5)

        np.testing.assert_allclose(f['vel'][:2], [[ 0.017203,  0.01848 , -0.019859],
                                                  [ 0.051333,  0.027357, -0.010303]], atol=1e-5)


def test_virialradius():
    global f, h
    with pynbody.analysis.halo.center(h[1]):
        start = time.time()
        vrad = pynbody.analysis.halo.virial_radius(f)
        print ("time=",time.time()-start)
        np.testing.assert_allclose(vrad, 0.005946911872, atol=1.e-5)


def test_ssc_bighalo():
    s = pynbody.load('testdata/Test_NOSN_NOZCOOL_L010N0128/data/subhalos_103/subhalo_103')
    s.physical_units()
    h = s.halos()
    pynbody.analysis.halo.center(h[1])
    assert h[1]['r'].min()<0.02


def test_binning_hmf():
    subfind = pynbody.load('testdata/Test_NOSN_NOZCOOL_L010N0128/data/subhalos_103/subhalo_103')

    h = subfind.halos()
    assert len(h) == 4226

    center, means, err = pynbody.analysis.hmf.simulation_halo_mass_function(subfind,
                                                                            log_M_min=8,
                                                                            log_M_max=14,
                                                                            delta_log_M=0.5,
                                                                            subsample_catalogue=100)

    assert(len(means) == len(center) == len(err) == 12)

    np.testing.assert_allclose(center, [2.08113883e+08,   6.58113883e+08,   2.08113883e+09,   6.58113883e+09,
                                        2.08113883e+10,   6.58113883e+10,   2.08113883e+11,   6.58113883e+11,
                                        2.08113883e+12,   6.58113883e+12,   2.08113883e+13,   6.58113883e+13], atol=1e-5)

    np.testing.assert_allclose(means, [0., 0.05600012,  0.02200005,  0.00600001,  0, 0., 0.002,
                                        0., 0.,          0.,          0.,          0.], atol=1e-5)

    np.testing.assert_allclose(err, [0., 0.01058303,  0.00663326, 0.00346411,  0.,  0., 0.002,
                                     0.,          0.,         0.,          0.,          0.], atol=1e-5)
