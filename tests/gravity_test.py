import numpy as np
import numpy.testing as npt

import pynbody


def test_gravity():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    h = f.halos()
    pynbody.analysis.angmom.faceon(h[0])
    pro = pynbody.analysis.profile.Profile(
        h[0], type='equaln', nbins=50, rmin='100 pc', rmax='50 kpc')

    v_circ_correct = np.array([ 57.19474878, 103.04044155, 132.12043214, 155.64841818,
          175.52360981, 193.25563697, 209.51834466, 224.37374187,
          237.98002897, 250.72845462, 262.76179453, 273.83990563,
          283.89607552, 293.11008797, 301.34304997, 308.53973799,
          314.79814513, 320.02418914, 324.20335678, 327.401287  ,
          329.64696837, 330.92989093, 331.25359281, 330.57145353,
          328.97259722, 326.53482184, 323.58771045, 321.04396878,
          318.7005073 , 316.17077031, 312.98903071, 308.64512521,
          303.84960576, 299.74476138, 296.7460243 , 290.8896671 ,
          283.84522191, 277.86495435, 271.30411848, 267.89978861,
          263.90881642, 259.82491209, 256.08762092, 251.42180937,
          247.00142313, 244.88890993, 240.34945886, 238.18884383,
          233.90900022, 229.68158801])

    v_circ = pro['v_circ'].in_units('km s^-1')

    print(repr(v_circ))

    npt.assert_allclose(v_circ, v_circ_correct,atol=1e-5)

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
