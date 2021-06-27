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


def test_rt_unit_warning_for_photon_rho():
    # Issue 542 about photon density unit.
    # Check that warning informing user about the missing "reduced speed of light" factor is generated
    # at load time
    f1 = pynbody.load("testdata/ramses_rt_partial_output_00002", cpus=[1, 2, 3])

    with np.testing.assert_warns(UserWarning):
        f1.gas['rad_0_rho']


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

    assert len(f_dmo.dm["cpu"]) == 274004
    assert len(f_dmo.g["cpu"]) == 907818

    assert np.all(f_dmo.dm["cpu"] == 1)
    assert np.all(f_dmo.g["cpu"] == 1)

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


def test_metals_field_correctly_copied_from_metal():
    np.testing.assert_allclose(f.st['metals'][::5000], f.st['metal'][::5000], rtol=1e-5)


def test_tform_and_tform_raw():
    # Standard test output is a non-cosmological run, for which tform should be read from disk,
    # rather than transformed. Tform raw and transformed are therefore the same
    assert len(f.st['tform']) == len(f.st['tform_raw']) == 2655
    np.testing.assert_allclose(f.st['tform_raw'], f.st['tform'])

    # Now loads a cosmological run, for which tforms have a weird format
    # Birth files are however not generated for this output, hence tform is filled with -1
    # Raw tform still carry the original weird units
    fcosmo = pynbody.load("testdata/output_00080")
    with np.testing.assert_warns(UserWarning):
        np.testing.assert_allclose(fcosmo.st['tform'], - np.ones((31990,), dtype=np.float64), rtol=1e-5)
        np.testing.assert_allclose(fcosmo.st['tform_raw'][:10],
                                   [-2.72826591, -1.8400868,  -2.35988485, -3.81799766, -2.67772371, -3.22276503,
                                    -2.5208477,  -2.67845014, -3.17295132, -2.43044642],
                                   rtol=1e-5)


def test_proper_time_loading():
    f_pt = pynbody.load(
        "testdata/prop_time_output_00030", cpus=range(10, 20))

    f_pt._is_using_proper_time = True

    f_pt._load_particle_block('tform')
    f_pt._convert_tform()
    np.testing.assert_allclose(
        f_pt.s["tform"].in_units("Gyr"),
        [2.52501534, 2.57053015, 2.66348155, 2.99452429, 2.49332345,
        3.62452373, 2.22125997, 2.53889974, 2.30228611, 3.45341852,
        2.48534871, 3.42507129, 2.39147047, 2.74341721, 3.07370808,
        2.69028377, 2.96989821, 3.0768944, 2.48748702, 3.79943883,
        3.94957879, 2.24967707, 4.01734689, 3.65785368, 2.63618622,
        2.69290132, 2.59963679, 4.03835932, 2.77991464, 2.71311552,
        2.38078038, 4.3666123, 2.68693346, 3.37377901, 3.27283305,
        3.03470615, 2.4334257, 2.65158796, 2.90785361, 2.56396249],
        rtol=1e-5)


def test_is_cosmological_without_namelist():
    # Load a cosmo run, but without the namelist.txt file and checks that cosmo detection works with a warning
    f_without_namelist = pynbody.load("testdata/ramses_dmo_partial_output_00051")
    f_without_namelist.physical_units()

    with np.testing.assert_warns(UserWarning):
        assert (not f_without_namelist._not_cosmological())


def test_temperature_derivation():
    f.g['mu']
    f.g['temp']

    assert(f.g['mu'].min() != f.g['mu'].max())   # Check that ionized and neutral mu are now generated
    np.testing.assert_allclose(f.g['mu'][:10], 0.59 * np.ones(10))
    np.testing.assert_allclose(f.g['mu'].max(), 1.3)

    np.testing.assert_allclose(f.g['temp'][:10], [25988.332966, 27995.231272, 27995.19821, 30516.467776,
                                                  26931.794949, 29073.739294, 29177.197614, 31917.91444,
                                                  26931.790284, 29177.242923])
