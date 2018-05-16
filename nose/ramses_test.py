import pynbody
import numpy as np


def setup():
    global f
    f = pynbody.load("testdata/ramses_partial_output_00250")


def test_lengths():
    assert len(f.gas) == 152667
    assert len(f.star) == 2655
    assert len(f.dm) == 51887

def test_properties():
    np.testing.assert_almost_equal(f.properties['a'], 1.0)
    np.testing.assert_almost_equal(f.properties['h'], 0.01)
    np.testing.assert_almost_equal(f.properties['omegaM0'], 1.0)

def test_particle_arrays():
    f['pos']
    f['vel']
    np.testing.assert_allclose(f.star['pos'][50], [ 29.93861623,  29.29166795,  29.77920022])
    np.testing.assert_allclose(f.dm['pos'][50], [ 23.76016295,  21.64945726,   7.70719058])
    np.testing.assert_equal(f.dm['iord'][-50:-40],[126079, 679980, 602104, 352311, 306943, 147989, 121521, 915870,
       522489, 697169])
    np.testing.assert_equal(f.star['iord'][-50:-40],[124122,  65978, 160951,  83281, 120237, 117882, 124849, 111615,
       144166,  26147])
    np.testing.assert_allclose(f.dm['vel'][50], [ 0.32088361, -0.82660566, -0.32874243])

def test_array_unit_sanity():
    """Picks up on problems with converting arrays as they
    get promoted from family to simulation level"""

    f.gas['pos']
    f.star['pos']
    f.dm['pos']
    f.physical_units()

    f2 = pynbody.load("testdata/ramses_partial_output_00250")
    f2.physical_units()
    f2.gas['pos']
    f2.dm['pos']
    f2.star['pos']

    np.testing.assert_allclose(f2['pos'], f['pos'], atol=1e-5)

def test_key_error():
    """Tests that the appropriate KeyError is raised when a
    hydro array is not found. This is a regression test for
    a problem where AttributeError could be raised instead because
    _rt_blocks_3d was missing for non-RT runs"""

    with np.testing.assert_raises(KeyError):
        f.gas['nonexistentarray']

def test_mass_unit_sanity():
    """Picks up on problems with converting array units as
    mass array gets loaded (which is a combination of a derived
    array and a loaded array)"""

    f1 = pynbody.load("testdata/ramses_partial_output_00250")
    f1['mass']
    f1.physical_units()

    f2 = pynbody.load("testdata/ramses_partial_output_00250")
    f2.physical_units()
    f2['mass']

    np.testing.assert_allclose(f1.dm['mass'], f2.dm['mass'], atol=1e-5)

def test_rt_arrays():
    f1 = pynbody.load("testdata/ramses_rt_partial_output_00002",cpus=[1,2,3])

    for group in range(4):
        assert 'rad_%d_rho'%group in f1.gas.loadable_keys()
        assert 'rad_%d_flux'%group in f1.gas.loadable_keys()

    f1.gas['rad_0_flux'] # ensure 3d name triggers loading

    np.testing.assert_allclose(f1.gas['rad_0_rho'][::5000],
      [  8.63987256e-02,   3.73498855e-04,   3.46061505e-04,
         2.13979002e-04,   3.22825503e-04,   3.29494226e-04,
         2.26216739e-04,   3.30639509e-06,   3.61922553e-05,
         8.25142141e-06,   1.25595394e-05,   7.11374568e-07,
         3.89547811e-05,   5.17086871e-06,   6.90921297e-08,
         1.59989054e-09,   7.61815782e-07,   7.09372161e-08,
         7.76265288e-09,   4.32642383e-09])

def test_all_dm():
    f1 = pynbody.load("testdata/ramses_dmo_partial_output_00051")
    assert len(f1.families())==1
    assert len(f1.dm)==274004
    np.testing.assert_allclose(f1.dm['x'][::5000],
                               [0.35074016, 0.13963163 , 0.36627742 , 0.3627204  , 0.27095636 , 0.37752211,
                                0.33660182, 0.32650455 , 0.35676858 , 0.34949844,  0.36312999,  0.36769716,
                                0.40389644, 0.35737981 , 0.36942017 , 0.36452039 , 0.43302334 , 0.46196367,
                                0.41216312, 0.4668878  , 0.325703   , 0.01979551 , 0.3423653  , 0.44717895,
                                0.2903733 , 0.07345862 , 0.47979538 , 0.45532793 , 0.46041618 , 0.42501456,
                                0.04263056, 0.47700105,  0.01077027,  0.48121992,  0.28120965,  0.47383826,
                                0.08669572, 0.47441274 , 0.48264947 , 0.32018882 , 0.47762009 , 0.49716835,
                                0.48241899, 0.47935638 , 0.47553442 , 0.47741151 , 0.47750557 , 0.47583932,
                                0.48161516, 0.08597729 , 0.48198888 , 0.4815569  , 0.48042167 , 0.48096334,
                                0.48146743],
                               rtol=1e-5
                               )

def test_forcegas_dmo():
    f_dmo = pynbody.load("testdata/ramses_dmo_partial_output_00051", cpus=[1], force_gas=True)
    assert len(f_dmo.families())==2
    assert len(f_dmo.dm)==274004
    assert len(f_dmo.g)==907818

    np.testing.assert_allclose(f_dmo.g['mass'][::5000], np.ones((182,), dtype=np.float64), rtol = 1e-5)
    np.testing.assert_allclose(f_dmo.g['rho'][::10000],[  2.09715200e+06,   2.09715200e+06,   2.09715200e+06,
            2.09715200e+06,   2.09715200e+06,   2.09715200e+06,
            2.09715200e+06,   2.09715200e+06,   2.09715200e+06,
            2.09715200e+06,   2.09715200e+06,   2.09715200e+06,
            2.09715200e+06,   2.09715200e+06,   2.09715200e+06,
            2.09715200e+06,   2.09715200e+06,   2.09715200e+06,
            1.34217728e+08,   1.07374182e+09,   8.58993459e+09,
            6.87194767e+10,   6.87194767e+10,   5.49755814e+11,
            5.49755814e+11,   5.49755814e+11,   5.49755814e+11,
            5.49755814e+11,   5.49755814e+11,   5.49755814e+11,
            4.39804651e+12,   4.39804651e+12,   4.39804651e+12,
            4.39804651e+12,   4.39804651e+12,   4.39804651e+12,
            4.39804651e+12,   4.39804651e+12,   4.39804651e+12,
            4.39804651e+12,   4.39804651e+12,   4.39804651e+12,
            4.39804651e+12,   4.39804651e+12,   4.39804651e+12,
            4.39804651e+12,   4.39804651e+12,   3.51843721e+13,
            3.51843721e+13,   3.51843721e+13,   3.51843721e+13,
            3.51843721e+13,   3.51843721e+13,   3.51843721e+13,
            3.51843721e+13,   3.51843721e+13,   3.51843721e+13,
            3.51843721e+13,   3.51843721e+13,   3.51843721e+13,
            3.51843721e+13,   3.51843721e+13,   3.51843721e+13,
            3.51843721e+13,   3.51843721e+13,   3.51843721e+13,
            3.51843721e+13,   2.81474977e+14,   2.81474977e+14,
            2.81474977e+14,   2.81474977e+14,   2.81474977e+14,
            2.81474977e+14,   2.81474977e+14,   2.81474977e+14,
            2.81474977e+14,   2.81474977e+14,   2.81474977e+14,
            2.81474977e+14,   2.81474977e+14,   2.81474977e+14,
            2.81474977e+14,   2.25179981e+15,   2.25179981e+15,
            2.25179981e+15,   2.25179981e+15,   2.25179981e+15,
            2.25179981e+15,   2.25179981e+15,   2.25179981e+15,
            1.80143985e+16], rtol=1e-5)
