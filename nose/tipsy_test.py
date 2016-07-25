import pynbody
import numpy as np
import glob
import os


def setup():
    X = glob.glob("testdata/test_out.*")
    for z in X:
        os.unlink(z)

    global f, h
    f = pynbody.load("testdata/g15784.lr.01024")
    h = f.halos()


def teardown():
    global f
    del f


def test_get():
    current = f['pos'][0:100:10]
    print(current)

    correct = np.array([[0.01070931, -0.03619793, -0.16635996],
                        [0.01066598, -0.0328698, -0.16544016],
                        [0.0080902, -0.03409814, -0.15953901],
                        [0.01125323, -0.03251356, -0.14957215],
                        [0.01872441, -0.03908035, -0.16008312],
                        [0.01330984, -0.03552091, -0.14454767],
                        [0.01438289, -0.03428916, -0.13781759],
                        [0.01499815, -0.03602122, -0.13986239],
                        [0.0155305, -0.0332876, -0.14802328],
                        [0.01534095, -0.03525123, -0.14457548]])

    print("Position error of ", np.abs(current - correct).sum())
    assert (np.abs(current - correct).sum() < 1.e-7)


def test_standard_arrays():
    # just check all standard arrays are there
    with f.lazy_off:
        f['x']
        f['y']
        f['z']
        f['pos']
        f['vel']
        f['vx']
        f['vy']
        f['vz']
        f['eps']
        f['phi']
        f['mass']
        f.gas['rho']
        f.gas['temp']
        f.gas['metals']
        f.star['tform']
        f.star['metals']


def test_halo():
    print("Length=", len(h[1]))
    assert len(h[1]) == 502300


def test_loadable_array():
    assert 'HI' in f.loadable_keys()
    f['HI']
    assert 'HI' in f.keys()

    assert 'HeI' in f.loadable_keys()
    f['HeI']
    assert 'HeI' in f.keys()

    assert f['HI'].dtype == np.float32
    assert f['HeI'].dtype == np.float32
    assert f['igasorder'].dtype == np.int32

    HI_correct = np.array([5.35406599e-08,   4.97452731e-07,   5.73000014e-01,
                           5.73000014e-01,   5.73000014e-01,   5.73000014e-01,
                           5.73000014e-01,   5.73000014e-01,   5.73000014e-01,
                           5.73000014e-01,   5.73000014e-01,   5.73000014e-01,
                           5.73000014e-01,   5.73000014e-01,   5.73000014e-01,
                           4.18154418e-01,   5.86960971e-01,   3.94545615e-01], dtype=np.float32)
    HeI_correct = np.array([3.51669648e-12,   2.28513852e-09,   3.53999995e-03,
                            3.53999995e-03,   3.53999995e-03,   3.53999995e-03,
                            3.53999995e-03,   3.53999995e-03,   3.53999995e-03,
                            3.53999995e-03,   3.53999995e-03,   3.53999995e-03,
                            3.53999995e-03,   3.53999995e-03,   3.53999995e-03,
                            3.94968614e-02,   5.48484921e-02,   4.77905162e-02], dtype=np.float32)
    igasorder_correct = np.array([0,      0,      0,      0,      0,      0,      0,      0,
                                  0,      0,      0,      0,      0,      0,      0,  67264,
                                  72514, 177485], dtype=np.int32)

    assert (f['igasorder'][::100000] == igasorder_correct).all()
    assert abs(f['HI'][::100000] - HI_correct).sum() < 1.e-10
    assert abs(f['HeI'][::100000] - HeI_correct).sum() < 1.e-10


def _assert_unit(u, targ, eps=0.01):
    assert abs(u.ratio(targ) - 1.0) < eps


def test_units():
    _assert_unit(f['pos'].units, "6.85e+04 kpc a")
    _assert_unit(f['vel'].units, "1.73e+03 km a s**-1")
    _assert_unit(f['phi'].units, "2.98e+06 km**2 a**-1 s**-2")
    _assert_unit(f.gas['rho'].units, "1.48e+02 Msol kpc**-3 a**-3")
    _assert_unit(f.star['tform'].units, "38.76 Gyr")


def test_halo_unit_conversion():
    f.gas['rho'].convert_units('Msol kpc^-3')
    assert str(h[1].gas['rho'].units) == 'Msol kpc**-3'

    h[1].gas['rho'].convert_units('m_p cm^-3')
    assert str(h[1].gas['rho'].units) == 'm_p cm**-3'


def test_write():
    f2 = pynbody.new(gas=20, star=11, dm=9, order='gas,dm,star')
    f2.dm['test_array'] = np.ones(9)
    f2['x'] = np.arange(0, 40)
    f2['vx'] = np.arange(40, 80)
    f2.write(fmt=pynbody.tipsy.TipsySnap, filename="testdata/test_out.tipsy")

    f3 = pynbody.load("testdata/test_out.tipsy")
    assert all(f3['x'] == f2['x'])
    assert all(f3['vx'] == f3['vx'])
    assert all(f3.dm['test_array'] == f2.dm['test_array'])


def test_array_write():

    f['array_write_test'] = np.random.rand(len(f))
    f['array_write_test'].write(overwrite=True)
    f['array_read_test'] = f['array_write_test']
    del f['array_write_test']

    # will re-lazy-load
    assert all(np.abs(f['array_write_test'] - f['array_read_test']) < 1.e-5)


def test_isolated_read():
    s = pynbody.load('testdata/isolated_ics.std')


def test_array_metadata():
    f1 = pynbody.load("testdata/test_out.tipsy")

    f1.gas['zog'] = np.ones(len(f1.gas))
    f1.gas['zog'].units = "Msol kpc^-1"
    f1.gas['zog'].write()

    f1['banana'] = np.ones(len(f1)) * 0.5
    f1['banana'].units = "kpc^3 Myr^-1"
    f1['banana'].write()

    del f1

    f1 = pynbody.load("testdata/test_out.tipsy")
    assert "banana" in f1.loadable_keys()
    assert "zog" not in f1.loadable_keys()
    assert "banana" in f1.gas.loadable_keys()
    assert "zog" in f1.gas.loadable_keys()

    try:
        f1.star["zog"]  # -> KeyError
        assert False  # Shouldn't have been able to load gas-only array zog
    except KeyError:
        pass

    f1.gas['zog']
    assert f1.gas['zog'][0] == 1.0
    assert f1.gas['zog'].units == "Msol kpc^-1"

    f1.star['banana']
    f1.gas['banana']
    f1.dm['banana']
    assert f1['banana'].units == "kpc^3 Myr^-1"


def test_array_update():
    f1 = pynbody.load("testdata/test_out.tipsy")

    f1['bla'] = np.zeros(len(f1))
    f1['bla'].units = 'km'
    f1['bla'].write()

    del(f1['bla'])

    f1['bla']

    f1.g['bla'] = 1
    f1.d['bla'] = 2
    f1.s['bla'] = 3

    # test the case where bla is a snapshot-level array

    try:
        f1.g['bla'].write()
        assert False  # should not be allowed to overwrite here
    except IOError:
        pass

    f1.write_array(
        'bla', [pynbody.family.gas, pynbody.family.dm], overwrite=True)

    del(f1['bla'])

    f1['bla']

    assert all(f1.g['bla'] == 1)
    assert all(f1.d['bla'] == 2)
    assert all(f1.s['bla'] == 0)

    # test the case where bla2 is a family-level array

    f1.g['bla2'] = np.zeros(len(f1.g))
    f1.g['bla2'].units = 'km'
    f1.s['bla2'] = np.ones(len(f1.s))

    f1.write_array('bla2', [pynbody.family.gas, pynbody.family.star])

    del(f1)

    f1 = pynbody.load("testdata/test_out.tipsy")

    assert all(f1.g['bla2'] == 0)
    assert all(f1.s['bla2'] == 1)


def test_snapshot_update():
    f1 = pynbody.load("testdata/test_out.tipsy")
    f1['pos']
    f1['pos'] = np.arange(0, len(f1) * 3).reshape(len(f1), 3)

    # convert units -- the array should get written out in simulation units
    f1.g['pos'].convert_units('Mpc')

    f1['pos'].write(overwrite=True)
    f1.gas['metals'] = np.ones(len(f1.gas)) * 123.
    f1.star['metals'] = np.ones(len(f1.star)) * 345.

    f1.gas['metals'].write(overwrite=True)
    del f1

    f2 = pynbody.load("testdata/test_out.tipsy")
    assert (f2['pos'] == np.arange(0, len(f2) * 3).reshape(len(f2), 3)).all()
    assert (f2.gas['metals'] == 123.).all()  # should have updated gas.metals
    # should not have written out changes to star.metals
    assert not (f2.star['metals'] == 345.).any()

    # this is a completion:
    f2.dm['metals'] = np.ones(len(f2.dm)) * 789.1

    # should now be a simulation-level array... write it out
    f2['metals'].write(overwrite=True)

    del f2['metals']

    f3 = pynbody.load("testdata/test_out.tipsy")
    assert (f3.dm['metals'] == 789.1).all()


def test_unit_persistence():
    f1 = pynbody.load("testdata/g15784.lr.01024")
    f1['pos']
    f1.physical_units()
    assert f1['pos'].units == 'kpc'
    del f1['pos']
    f1['pos']
    assert f1['pos'].units == 'kpc'
    del f1['pos']
    f1[[4, 6, 10]]['pos']  # test for fail when autoconverting on subarrays
    assert f1['pos'].units == 'kpc'
    del f1['pos']
    f1.original_units()
    f1['pos']
    assert f1['pos'].units != 'kpc'


def test_3d_interpolation():
    # this is the result using scipy.interpolate.interp
    ref3d = np.array([0.07527991,  0.06456315,  0.08380653,  0.15143689,  0.15568259,
                      0.09593283,  0.15549068,  0.13407668,  0.15177078,  0.14166734,
                      0.06554151,  0.13197516,  0.08851156,  0.06210035,  0.09224193,
                      0.162001,  0.16017458,  0.06572785,  0.16384523,  0.1621215,
                      0.15005151,  0.15224003,  0.1580059,  0.1606176,  0.16919882,
                      0.15257504,  0.15542396,  0.15950106,  0.15208175,  0.06277352,
                      0.16297019,  0.16536367,  0.15913841,  0.16852261,  0.16592073,
                      0.16994837,  0.13997742,  0.16381589,  0.16592951,  0.16631843,
                      0.12965855,  0.16823705,  0.16560094,  0.16819444,  0.15564206,
                      0.15821792,  0.1596779,  0.15242394,  0.16538892,  0.16265422,
                      0.1698232,  0.16823449,  0.16036913,  0.13258314,  0.16295899,
                      0.16926536,  0.16349818,  0.07734309,  0.16657909,  0.16847497,
                      0.16370438,  0.17016328,  0.16424645,  0.14292659,  0.17001509,
                      0.16708861,  0.17015931,  0.06657929,  0.16794139,  0.16821759,
                      0.16128866,  0.14454934,  0.16137588,  0.16934885,  0.17008926,
                      0.16986069,  0.16963654,  0.14639736,  0.16590415,  0.16879675,
                      0.16349512,  0.16999227,  0.17003994,  0.15351041,  0.16446416,
                      0.14130995,  0.1636267,  0.17015097,  0.16989277,  0.16982946,
                      0.17006758,  0.16906539,  0.16315827,  0.17021533,  0.16991691,
                      0.17006688,  0.17006756,  0.16753875,  0.15553802,  0.15892623])

    arr = pynbody.analysis.ionfrac.calculate(f, 'ovi')
    assert(np.allclose(arr[0:100], ref3d))

    ref2d = np.array([-8.43178697, -8.43437937, -8.43515772, -8.43554701, -8.43801415,
                      -8.44009401, -8.44048422, -
                      8.4410046, -8.44178542, -8.44204576,
                      -8.44217594, -8.44217594, -
                      8.44243633, -8.44269675, -8.4429572,
                      -8.44334794, -8.4434782, -
                      8.44412964, -8.44412964, -8.44439027,
                      -8.4445206, -8.44465094, -
                      8.44504199, -8.44530274, -8.44530274,
                      -8.44530274, -8.44543312, -
                      8.44595474, -8.44608516, -8.44608516,
                      -8.4462156, -8.4462156, -8.4462156, -
                      8.44634604, -8.44634604,
                      -8.44647649, -8.44686789, -
                      8.44686789, -8.44699837, -8.44712886,
                      -8.44725936, -8.44738987, -
                      8.44738987, -8.44752038, -8.44765091,
                      -8.44765091, -8.44791198, -
                      8.44830364, -8.44830364, -8.44843421,
                      -8.44869538, -8.44869538, -
                      8.4490872, -8.4490872, -8.44934845,
                      -8.44947909, -8.44960973, -
                      8.44987105, -8.44987105, -8.45000172,
                      -8.45000172, -8.4501324, -
                      8.45052448, -8.45091663, -8.45104737,
                      -8.45117811, -8.45130886, -
                      8.45130886, -8.45130886, -8.45157039,
                      -8.45183196, -8.45196275, -
                      8.45196275, -8.45209355, -8.45209355,
                      -8.45235518, -8.45235518, -
                      8.452486, -8.452486, -8.452486,
                      -8.45261684, -8.45261684, -
                      8.45287853, -8.45287853, -8.45300939,
                      -8.45300939, -8.45300939, -
                      8.45314025, -8.45340201, -8.4536638,
                      -8.4536638, -8.4536638, -8.45379471, -
                      8.45379471, -8.45379471,
                      -8.45379471, -8.45392562, -8.45392562, -8.45405654, -8.45405654])

    arr = pynbody.analysis.luminosity.calc_mags(f.s)
    assert(np.allclose(arr[0:100], ref2d))


def test_issue_313():
    f = pynbody.load("testdata/g15784.lr.01024")
    f.physical_units()
    f['vtheta']


def test_issue_315():
    assert np.allclose(f.g['cs'][:3], [187.36890472,  210.86151107,  176.04044173])