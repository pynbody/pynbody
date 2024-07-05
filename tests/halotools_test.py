import glob
import os
import time

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gasoline_ahf", "gadget")

@pytest.fixture
def snap():
    return pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")

@pytest.fixture
def halos(snap):
    return snap.halos()

def test_center(snap, halos):
    with pynbody.analysis.halo.center(halos[0]):
        np.testing.assert_allclose(
            snap['pos'][0], [-0.0137471, -0.00208458, -0.04392379], rtol=1e-4)

def test_center_wrapped_halo():
    npart = 10000
    f = pynbody.new(dm=npart)
    np.random.seed(1337)
    f['pos'] = np.random.normal(scale=0.1, size=f['pos'].shape)
    f['vel'] = np.random.normal(scale=0.1, size=f['vel'].shape)
    f['pos'][0] = [0.0, 0.0, 0.0] # this is the known centre!
    f['vel'][0] = [0.0, 0.0, 0.0] # again, a known centre
    f['vel'] += 5.0

    f['mass'] = np.ones(npart)
    f['x']+=0.95
    f['y']+=0.5
    f['z']-=0.94
    f.properties['boxsize'] = 2.0
    f.wrap()

    with pynbody.analysis.halo.center(f) as t:
        npt.assert_almost_equal(f['pos'][0], [0.0, 0.0, 0.0], decimal=1)
        npt.assert_almost_equal(f['vel'][0], [0.0, 0.0, 0.0], decimal=1)



def test_align(snap, halos):
    with pynbody.analysis.angmom.faceon(halos[0]) as t:
        print(repr(t))
        np.testing.assert_allclose(snap['pos'][:2], [[-0.010711, -0.001491, -0.044785],
                                                     [-0.010019,  0.002454, -0.04465 ]],
                                   atol=1e-5)

        np.testing.assert_allclose(snap['vel'][:2], [[0.019214, 0.024604, -0.020356],
                                                     [ 0.053343,  0.033478, -0.010793]], atol=1e-5)


def test_virialradius(snap, halos):
    with pynbody.analysis.halo.center(halos[0]):
        start = time.time()
        vrad = pynbody.analysis.halo.virial_radius(snap)
        print ("time=",time.time()-start)
        np.testing.assert_allclose(vrad, 0.005946911872, atol=1.e-5)


def test_ssc_bighalo():
    s = pynbody.load('testdata/gadget3/data/subhalos_103/subhalo_103')
    s.physical_units()
    h = s.halos()
    pynbody.analysis.halo.center(h[1])
    assert h[1]['r'].min()<0.02


def test_binning_hmf():
    subfind = pynbody.load('testdata/gadget3/data/subhalos_103/subhalo_103')

    h = subfind.halos()
    assert len(h) == 4226

    with pytest.warns(UserWarning, match=r"Halo finder masses not provided\. Calculating them.*"):
        center, means, err = pynbody.analysis.hmf.simulation_halo_mass_function(
            subfind,
            log_M_min=8,
            log_M_max=14,
            delta_log_M=0.5,
            subsample_catalogue=100
        )

    assert len(means) == len(center) == len(err) == 12

    np.testing.assert_allclose(center, [2.08113883e+08,   6.58113883e+08,   2.08113883e+09,   6.58113883e+09,
                                        2.08113883e+10,   6.58113883e+10,   2.08113883e+11,   6.58113883e+11,
                                        2.08113883e+12,   6.58113883e+12,   2.08113883e+13,   6.58113883e+13], atol=1e-5)

    np.testing.assert_allclose(means, [0., 0.05600012,  0.02200005,  0.00600001,  0, 0., 0.002,
                                        0., 0.,          0.,          0.,          0.], atol=1e-5)

    np.testing.assert_allclose(err, [0., 0.01058303,  0.00663326, 0.00346411,  0.,  0., 0.002,
                                     0.,          0.,         0.,          0.,          0.], atol=1e-5)
