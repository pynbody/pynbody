import glob
import os
import warnings

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
from pynbody.dependencytracker import DependencyError


def setup_module():
    X = glob.glob("testdata/test_out.*")
    for z in X:
        os.unlink(z)


@pytest.fixture
def snap():
    return pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")


def teardown_module():
    if os.path.exists("testdata/gasoline_ahf/g15784.lr.log"):
        os.remove("testdata/gasoline_ahf/g15784.lr.log")


def test_get(snap):
    current = snap['pos'][0:100:10]

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

    npt.assert_allclose(current, correct, atol=1e-7)

def test_get_physical_units(snap):
    correct = [[   733.51516346,  -2479.31184326, -11394.52608302],
          [  1796.29124006,  -2579.39183919,  -8380.35581037],
          [  1391.62924878,  -2431.20529787,  -7931.36665997],
          [  1996.67781137,  -2490.91383448,  -7701.69724457],
          [   698.98807645,  -2179.99399308, -11311.06012831],
          [  -179.47289453,   -436.97412369,  -8760.92459688],
          [   446.38441151,   -752.9511779 ,  -8676.36146684],
          [  1076.73469837,  -1073.03453836, -10011.13598307],
          [   949.04364737,  -1414.14226592,  -7630.23693955],
          [   367.36147468,    108.40128938,  -9824.32938046],
          [  1867.13993363,  -2161.14171418,  -8546.99530979],
          [  2704.85332785,  -2265.0704991 ,  -5874.00573962],
          [  3779.85346857,  -9568.42624776,  -9349.93730413],
          [  -323.18262561,    125.80149341, -15521.88942478],
          [ -1357.39061682,   7618.33284238,  -5232.83076561],
          [ -1139.66574099,   4813.36532755,  -6998.1766858 ],
          [   776.35915841,  -4303.63910246,  -7973.70691111],
          [  1002.44463783,   8010.83787585,    471.30764218],
          [ -3229.72184828,   4091.20605634,    891.32393242],
          [  3246.1802437 ,  -1081.66573601,   -619.75766505],
          [  1762.73234649,    944.71854395,   3925.78391122],
          [  -329.95407918,     82.34072804, -15432.01188363],
          [  3060.44785228,  -3769.70638287, -11514.41621742],
          [  -271.28527566,    207.93381396, -15573.8118439 ],
          [  3702.94249857,  -5468.07485231, -12456.56934856],
          [ -5355.64908235,  16549.25990864,  -1065.88350678],
          [ -7197.32982792,   2606.86510945,   9063.50922282],
          [ 14803.87543505, -11033.76666951, -12276.19474495],
          [ 11337.12799352,   5105.16905767, -21948.358665  ],
          [ 30674.84828657, -33314.61334649,  27617.40337765],
          [  1674.72408593,  -2336.26161343,  -8385.37934244],
          [  1683.66096172,  -2345.44777767,  -8386.21982977],
          [  1673.26292906,  -2340.94323505,  -8384.98793152],
          [  1679.94026157,  -2342.84287931,  -8386.23207731],
          [  1674.8270418 ,  -2344.21179691,  -8386.0942925 ]]
    current = snap['pos'][::50000]
    snap.physical_units()

    npt.assert_allclose(current, correct, atol=1e-4)


def test_standard_arrays(snap):
    # just check all standard arrays are there
    snap['pos']
    with snap.lazy_off:
        snap['x']
        snap['y']
        snap['z']
        snap['pos']
        snap['vel']
        snap['vx']
        snap['vy']
        snap['vz']
        snap['eps']
        snap['phi']
        snap['mass']
        snap.gas['rho']
        snap.gas['temp']
        snap.gas['metals']
        snap.star['tform']
        snap.star['metals']


def test_loadable_array(snap):
    assert 'HI' in snap.loadable_keys()
    snap['HI']
    assert 'HI' in snap.keys()

    assert 'HeI' in snap.loadable_keys()
    snap['HeI']
    assert 'HeI' in snap.keys()

    assert snap['HI'].dtype == np.float32
    assert snap['HeI'].dtype == np.float32
    assert snap['igasorder'].dtype == np.int32

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

    assert (snap['igasorder'][::100000] == igasorder_correct).all()
    assert abs(snap['HI'][::100000] - HI_correct).sum() < 1.e-10
    assert abs(snap['HeI'][::100000] - HeI_correct).sum() < 1.e-10


def _assert_unit(u, targ, eps=0.01):
    assert abs(u.ratio(targ) - 1.0) < eps


def test_units(snap):
    _assert_unit(snap['pos'].units, "6.85e+04 kpc a")
    _assert_unit(snap['vel'].units, "1.73e+03 km a s**-1")
    _assert_unit(snap['phi'].units, "2.98e+06 km**2 a**-1 s**-2")
    _assert_unit(snap.gas['rho'].units, "1.48e+02 Msol kpc**-3 a**-3")
    _assert_unit(snap.star['tform'].units, "38.76 Gyr")


def test_halo_unit_conversion(snap):
    h = snap.halos()
    snap.gas['rho'].convert_units('Msol kpc^-3')
    assert str(h[1].gas['rho'].units) == 'Msol kpc**-3'

    h[1].gas['rho'].convert_units('m_p cm^-3')
    assert str(h[1].gas['rho'].units) == 'm_p cm**-3'

@pytest.fixture
def test_output():
    f2 = pynbody.new(gas=20, star=11, dm=9, order='gas,dm,star')
    f2.dm['test_array'] = np.ones(9)
    f2['x'] = np.arange(0, 40)
    f2['vx'] = np.arange(40, 80)
    f2.properties['a'] = 0.5
    f2.properties['time'] = 12.0
    f2.write(fmt=pynbody.snapshot.tipsy.TipsySnap, filename="testdata/gasoline_ahf/test_out.tipsy")
    yield "testdata/gasoline_ahf/test_out.tipsy"
    # unlink anything matching testdata/gasoline_ahf/test_out.tipsy*
    for f in glob.glob("testdata/gasoline_ahf/test_out.tipsy*"):
        os.unlink(f)

def gen_output(cosmological):
    f2 = pynbody.new(gas=20, star=11, dm=9, order='gas,dm,star')
    f2.dm['test_array'] = np.ones(9)
    f2['x'] = np.arange(0, 40)
    f2['vx'] = np.arange(40, 80)
    f2.properties['a'] = 0.5
    f2.properties['time'] = 12.0
    f2.write(fmt=pynbody.snapshot.tipsy.TipsySnap, filename="testdata/gasoline_ahf/test_out.tipsy",
             cosmological=cosmological)
    yield "testdata/gasoline_ahf/test_out.tipsy"
    # unlink anything matching testdata/gasoline_ahf/test_out.tipsy*
    for f in glob.glob("testdata/gasoline_ahf/test_out.tipsy*"):
        os.unlink(f)
@pytest.fixture
def test_noncosmo_output():
    yield from gen_output(False)

@pytest.fixture
def test_output():
    yield from gen_output(True)

def test_write(test_output):
    with warnings.catch_warnings():
        f3 = pynbody.load(test_output)
    assert all(f3['x'] == np.arange(0, 40))
    assert all(f3['vx'] == np.arange(40, 80))
    assert all(f3.dm['test_array'] == np.ones(9))
    assert f3.properties['a']==0.5

def test_write_noncosmo(test_noncosmo_output):
    # this looks strange, but it's because the .param file in the testdata folder implies a cosmological tipsy snap
    # whereas we have just written the snapshot asserting it is non-cosmological
    f3 = pynbody.load(test_noncosmo_output)
    assert f3.properties['a']==12


def test_array_write(snap):
    f = snap
    f['array_write_test'] = np.random.rand(len(f))
    f['array_write_test'].write(overwrite=True)
    f['array_read_test'] = f['array_write_test']
    del f['array_write_test']

    # will re-lazy-load
    assert all(np.abs(f['array_write_test'] - f['array_read_test']) < 1.e-5)


@pytest.mark.filterwarnings("ignore:Paramfile suggests time is cosmological.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:No readable param file.*:RuntimeWarning")
def test_isolated_read():
    s = pynbody.load('testdata/isolated_ics.std')


def test_array_metadata(test_output):
    f1 = pynbody.load(test_output)

    f1.gas['zog'] = np.ones(len(f1.gas))
    f1.gas['zog'].units = "Msol kpc^-1"
    f1.gas['zog'].write()

    f1['banana'] = np.ones(len(f1)) * 0.5
    f1['banana'].units = "kpc^3 Myr^-1"
    f1['banana'].write()

    del f1

    f1 = pynbody.load(test_output)
    assert "banana" in f1.loadable_keys()
    assert "zog" not in f1.loadable_keys()
    assert "banana" in f1.gas.loadable_keys()
    assert "zog" in f1.gas.loadable_keys()

    with pytest.raises(KeyError):
        f1.star["zog"]  # -> KeyError

    f1.gas['zog']
    assert f1.gas['zog'][0] == 1.0
    assert f1.gas['zog'].units == "Msol kpc^-1"

    f1.star['banana']
    f1.gas['banana']
    f1.dm['banana']
    assert f1['banana'].units == "kpc^3 Myr^-1"


def test_array_update(test_output):
    f1 = pynbody.load(test_output)

    f1['bla'] = np.zeros(len(f1))
    f1['bla'].units = 'km'
    f1['bla'].write()

    del(f1['bla'])

    f1['bla']

    f1.g['bla'] = 1
    f1.d['bla'] = 2
    f1.s['bla'] = 3

    # test the case where bla is a snapshot-level array

    with pytest.raises(IOError):
        f1.g['bla'].write()

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

    f1 = pynbody.load(test_output)

    assert all(f1.g['bla2'] == 0)
    assert all(f1.s['bla2'] == 1)


def test_snapshot_update(test_output):
    f1 = pynbody.load(test_output)
    f1['pos']
    f1['pos'] = np.arange(0, len(f1) * 3).reshape(len(f1), 3)

    # convert units -- the array should get written out in simulation units
    f1.g['pos'].convert_units('Mpc')

    f1['pos'].write(overwrite=True)
    f1.gas['metals'] = np.ones(len(f1.gas)) * 123.
    f1.star['metals'] = np.ones(len(f1.star)) * 345.

    f1.gas['metals'].write(overwrite=True)
    del f1

    f2 = pynbody.load(test_output)
    assert (f2['pos'] == np.arange(0, len(f2) * 3).reshape(len(f2), 3)).all()
    assert (f2.gas['metals'] == 123.).all()  # should have updated gas.metals
    # should not have written out changes to star.metals
    assert not (f2.star['metals'] == 345.).any()

    # this is a completion:
    f2.dm['metals'] = np.ones(len(f2.dm)) * 789.1

    # should now be a simulation-level array... write it out
    f2['metals'].write(overwrite=True)

    del f2['metals']

    f3 = pynbody.load(test_output)
    assert (f3.dm['metals'] == 789.1).all()

def test_unit_persistence():
    f1 = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
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


@pytest.mark.filterwarnings("ignore:No log file found:UserWarning")
def test_3d_interpolation(snap):
    # this is the result using scipy.interpolate.interp
    ref3d = np.array([0.07527991, 0.06456315, 0.08380653, 0.15143689, 0.15568259,
       0.09593283, 0.15549068, 0.13407668, 0.15177078, 0.14166734,
       0.06554151, 0.13197516, 0.08851156, 0.06210035, 0.09224193,
       0.162001  , 0.16017458, 0.06572785, 0.16384523, 0.1621215 ,
       0.15005151, 0.15224003, 0.1580059 , 0.1606176 , 0.16919882,
       0.15257504, 0.15542396, 0.15950106, 0.15208175, 0.06277352,
       0.16297019, 0.16536367, 0.15913841, 0.16852261, 0.16592073,
       0.16994837, 0.13997742, 0.16381589, 0.16592951, 0.16631843,
       0.12965855, 0.16823705, 0.16560094, 0.16819444, 0.15564206,
       0.15821792, 0.1596779 , 0.15242394, 0.16538892, 0.16265422,
       0.1698232 , 0.16823449, 0.16036913, 0.13258314, 0.16295899,
       0.16926536, 0.16349818, 0.07734905, 0.16657909, 0.16847497,
       0.16370438, 0.17016328, 0.16424645, 0.14292659, 0.17001509,
       0.16708861, 0.17015931, 0.06657929, 0.16794139, 0.16821759,
       0.16128866, 0.14454934, 0.16137588, 0.16934885, 0.17008926,
       0.16986069, 0.16963654, 0.14639736, 0.16590415, 0.16879675,
       0.16349512, 0.16999227, 0.17003994, 0.15351041, 0.16446416,
       0.14130995, 0.1636267 , 0.17015097, 0.16989277, 0.16982946,
       0.17006758, 0.16906539, 0.16315827, 0.17021533, 0.16991691,
       0.17006688, 0.17006756, 0.16753875, 0.15553802, 0.15892623])

    arr = pynbody.analysis.ionfrac.calculate(snap, 'ovi')
    assert(np.allclose(arr[0:100], ref3d))

def test_alternative_cmd(snap):
    """A very basic test that the alternative cmd path is respected"""
    with pytest.raises(IOError):
        pynbody.analysis.luminosity.calc_mags(snap.s, cmd_path="/nonexistent/path")

def test_issue_313(snap):
    snap.physical_units()
    snap['vtheta']


def test_issue_315(snap):
    assert np.allclose(snap.g['cs'][:3], [ 319.46246429,  359.4923197,   300.13751002])

@pytest.mark.filterwarnings("ignore:No log file found:UserWarning")
def test_read_starlog_no_log(snap):
    rhoform = snap.s['rhoform'].in_units('Msol kpc**-3')[:1000:100]
    correct = np.array([ 2472533.42024787,  5336799.91228041, 64992197.77874849,
            16312128.27530325,  2514281.61028265, 14384682.79060703,
             3334026.53876033, 30041215.63578063, 24977741.02041524,
            10758492.68887316])
    assert np.all(np.abs(rhoform - correct) < 1e-7)
    # h2form should not be in the available starlog keys
    with pytest.raises(KeyError):
        h2form = snap.s['h2form']

def test_read_starlog_with_log(snap):
    # the last key is incorrectly labeled in order to ensure
    # that the log file is being read
    with open('testdata/gasoline_ahf/g15784.lr.log', 'w') as logf:
        logf.write('# ilbDumpIteration: 0\n# bDoSimulateLB: 0\n# starlog data:\n'
                   '# iOrdStar i4\n# iOrdGas i4\n# timeForm f8\n# rForm[0] f8\n'
                   '# rForm[1] f8\n# rForm[2] f8\n# vForm[0] f8\n# vForm[1] f8\n#'
                   ' vForm[2] f8\n# massForm f8\n# rhoForm f8\n# H2FracForm f8\n# end'
                   ' starlog data\n# TimeOut: none\n\n')
    rhoform = snap.s['rhoform'].in_units('Msol kpc**-3')[:1000:100]
    correct = np.array([ 2472533.42024787,  5336799.91228041, 64992197.77874849,
            16312128.27530325,  2514281.61028265, 14384682.79060703,
             3334026.53876033, 30041215.63578063, 24977741.02041524,
            10758492.68887316])
    assert np.all(np.abs(rhoform - correct) < 1e-7)

    correct = np.array([11696.64846193, 11010.78848271, 11035.32739625, 10133.30674776,
          10795.18204699, 10549.8055167 , 10365.82267086, 10389.80826619,
          10766.11592458, 10514.57288485])
    h2form = snap.s['h2form'][:1000:100]
    assert np.all(np.abs(h2form - correct) < 1e-7)

@pytest.fixture
def no_paramfile_snap():
    import pathlib
    import shutil

    # create a folder, gasoline_ahf_noparam
    gasoline_ahf = pathlib.Path("testdata/gasoline_ahf")
    gasoline_ahf_noparam = pathlib.Path("testdata/gasoline_ahf_noparam")
    shutil.rmtree(gasoline_ahf_noparam, ignore_errors=True)
    gasoline_ahf_noparam.mkdir()

    # symlink all files from gasoline_ahf

    for file in gasoline_ahf.iterdir():
        if file.is_file() and file.suffix != ".param":
            (gasoline_ahf_noparam / file.name).symlink_to(file.absolute())

    yield gasoline_ahf_noparam / "g15784.lr.01024"

    shutil.rmtree(gasoline_ahf_noparam, ignore_errors=True)

def test_load_without_paramfile(no_paramfile_snap):
    with pytest.warns(RuntimeWarning, match="No readable param"):
        f = pynbody.load(no_paramfile_snap)
        assert isinstance(f, pynbody.snapshot.tipsy.TipsySnap)
        assert len(f) == 1717156
