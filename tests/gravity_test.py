import numpy as np
import numpy.testing as npt
import pytest

import pynbody


def test_gravity():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    h = f.halos()
    pynbody.analysis.angmom.faceon(h[0])
    pro = pynbody.analysis.profile.Profile(
        h[0], type='equaln', nbins=50, rmin='100 pc', rmax='50 kpc')

    v_circ_correct = np.array([ 57.19445406, 103.07628964, 132.11223368, 155.58932808,
          175.48202525, 193.19929044, 209.50654738, 224.36527055,
          237.93659247, 250.68636441, 262.72949256, 273.84865161,
          283.90032809, 293.10420183, 301.32998659, 308.540641  ,
          314.80948167, 320.0281338 , 324.2028914 , 327.40124838,
          329.64625769, 330.92733193, 331.25336332, 330.57151552,
          328.97485235, 326.54886075, 323.60357994, 321.03602331,
          318.69193219, 316.16834582, 312.98912943, 308.66044926,
          303.84896218, 299.75583821, 296.77125812, 290.91251525,
          283.84275867, 277.87033036, 271.31481248, 267.90817792,
          263.89271744, 259.83899922, 256.07340296, 251.42403094,
          247.00347206, 244.89404984, 240.3383157 , 238.17681295,
          233.90977942, 229.66882597])

    v_circ = pro['v_circ'].in_units('km s^-1')

    npt.assert_allclose(v_circ, v_circ_correct, rtol=1e-2, atol=0)

def test_gravity_float():
    f = pynbody.new(100)
    np.random.seed(0)
    coords = np.random.normal(size=(100,3))
    del f['pos']
    del f['mass']
    f['pos'] = np.array(coords,dtype=np.float32)
    f['eps'] = np.ones(100,dtype=np.float32)
    f['mass'] = np.ones(100,dtype=np.float32)
    pynbody.gravity.calc.all_direct(f)

def test_eps_retrieval_str():
    f = pynbody.load("testdata/gadget2/test_g2_snap.0")
    f.properties['eps'] = "0.3 kpc"
    pynbody.gravity.calc.all_direct(f)
    true_phi_10 = np.array([-0.06696571, -0.07087147, -0.07049192,
                            -0.06739005, -0.06748439, -0.0695245,
                            -0.06803885, -0.0679833,  -0.07277965, -0.07189107])
    npt.assert_allclose(f['phi'][:10], true_phi_10)


def test_eps_retrieval_unit():
    f = pynbody.load("testdata/gadget2/test_g2_snap.0")
    f.properties['eps'] = 0.3 * pynbody.units.kpc
    pynbody.gravity.calc.all_direct(f)
    true_phi_10 = np.array([-0.06696571, -0.07087147, -0.07049192,
                            -0.06739005, -0.06748439, -0.0695245,
                            -0.06803885, -0.0679833,  -0.07277965, -0.07189107])
    npt.assert_allclose(f['phi'][:10], true_phi_10)


def test_eps_retrieval_number():
    f = pynbody.load("testdata/gadget2/test_g2_snap.0")
    f.properties['eps'] = 0.3
    pynbody.gravity.calc.all_direct(f)
    true_phi_10 = np.array([-0.06696571, -0.07087147, -0.07049192,
                            -0.06739005, -0.06748439, -0.0695245,
                            -0.06803885, -0.0679833,  -0.07277965, -0.07189107])
    npt.assert_allclose(f['phi'][:10], true_phi_10)
