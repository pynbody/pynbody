import numpy as np
import numpy.testing as npt

import pynbody


def test_gravity():
    f = pynbody.load("testdata/g15784.lr.01024")
    h = f.halos()
    pynbody.analysis.angmom.faceon(h[1])
    pro = pynbody.analysis.profile.Profile(
        h[1], type='equaln', nbins=50, rmin='100 pc', rmax='50 kpc')

    v_circ_correct = np.array([  57.19634349,  103.04331454,  132.12411594,  155.65275799,
        175.52850379,  193.26102536,  209.52418648,  224.3799979 ,
        237.98666437,  250.73544548,  262.7691209 ,  273.84754088,
        283.90399116,  293.11826051,  301.35145207,  308.54834075,
        314.80692238,  320.03311211,  324.21239628,  327.41041566,
        329.65615964,  330.93911797,  331.26282888,  330.58067058,
        328.98176969,  326.54392634,  323.59673278,  321.05292019,
        318.70939336,  316.17958584,  312.99775752,  308.6537309 ,
        303.85807775,  299.75311891,  296.75429822,  290.89777773,
        283.85313613,  277.87270183,  271.31168303,  267.90725824,
        263.91617477,  259.83215657,  256.09476119,  251.42881956,
        247.00831007,  244.89573797,  240.35616032,  238.19548506,
        233.91552211,  229.68799203])

    v_circ = pro['v_circ'].in_units('km s^-1')

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
    f = pynbody.load("testdata/test_g2_snap.0")
    f.properties['eps'] = "0.3 kpc"
    pynbody.gravity.calc.all_direct(f)
    true_phi_10 = np.array([-0.06696571, -0.07087147, -0.07049192,
                            -0.06739005, -0.06748439, -0.0695245,
                            -0.06803885, -0.0679833,  -0.07277965, -0.07189107])
    npt.assert_allclose(f['phi'][:10], true_phi_10)


def test_eps_retrieval_unit():
    f = pynbody.load("testdata/test_g2_snap.0")
    f.properties['eps'] = 0.3 * pynbody.units.kpc
    pynbody.gravity.calc.all_direct(f)
    true_phi_10 = np.array([-0.06696571, -0.07087147, -0.07049192,
                            -0.06739005, -0.06748439, -0.0695245,
                            -0.06803885, -0.0679833,  -0.07277965, -0.07189107])
    npt.assert_allclose(f['phi'][:10], true_phi_10)


def test_eps_retrieval_number():
    f = pynbody.load("testdata/test_g2_snap.0")
    f.properties['eps'] = 0.3
    pynbody.gravity.calc.all_direct(f)
    true_phi_10 = np.array([-0.06696571, -0.07087147, -0.07049192,
                            -0.06739005, -0.06748439, -0.0695245,
                            -0.06803885, -0.0679833,  -0.07277965, -0.07189107])
    npt.assert_allclose(f['phi'][:10], true_phi_10)
