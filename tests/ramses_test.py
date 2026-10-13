import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("ramses")

@pytest.fixture
def ramses_file():
    yield pynbody.load("testdata/ramses/ramses_partial_output_00250")

def test_lengths(ramses_file):
    assert len(ramses_file.gas) == 152667
    assert len(ramses_file.star) == 2655
    assert len(ramses_file.dm) == 51887

def test_properties(ramses_file):
    np.testing.assert_almost_equal(ramses_file.properties['a'], 1.0)
    np.testing.assert_almost_equal(ramses_file.properties['h'], 0.01)
    np.testing.assert_almost_equal(ramses_file.properties['omegaM0'], 1.0)

def test_particle_arrays(ramses_file):
    ramses_file['pos'] # noqa
    ramses_file['vel'] # noqa
    np.testing.assert_allclose(ramses_file.star['pos'][50], [ 29.93861623,  29.29166795,  29.77920022])
    np.testing.assert_allclose(ramses_file.dm['pos'][50], [ 23.76016295,  21.64945726,   7.70719058])
    np.testing.assert_equal(ramses_file.dm['iord'][-50:-40],[126079, 679980, 602104, 352311, 306943, 147989, 121521, 915870,
       522489, 697169])
    np.testing.assert_equal(ramses_file.star['iord'][-50:-40],[124122,  65978, 160951,  83281, 120237, 117882, 124849, 111615,
       144166,  26147])
    np.testing.assert_allclose(ramses_file.dm['vel'][50], [ 0.32088361, -0.82660566, -0.32874243])

def test_array_unit_sanity(ramses_file):
    """Picks up on problems with converting arrays as they
    get promoted from family to simulation level"""

    ramses_file.gas['pos'] # noqa
    ramses_file.star['pos'] # noqa
    ramses_file.dm['pos'] # noqa
    ramses_file.physical_units()

    f2 = pynbody.load("testdata/ramses/ramses_partial_output_00250")
    f2.physical_units()
    f2.gas['pos'] # noqa
    f2.dm['pos'] # noqa
    f2.star['pos'] # noqa

    np.testing.assert_allclose(f2['pos'], ramses_file['pos'], atol=1e-5)

def test_key_error(ramses_file):
    """Tests that the appropriate KeyError is raised when a
    hydro array is not found. This is a regression test for
    a problem where AttributeError could be raised instead because
    _rt_blocks_3d was missing for non-RT runs"""

    with pytest.raises(KeyError):
        ramses_file.gas['nonexistentarray'] # noqa

def test_mass_unit_sanity():
    """Picks up on problems with converting array units as
    mass array gets loaded (which is a combination of a derived
    array and a loaded array)"""

    f1 = pynbody.load("testdata/ramses/ramses_partial_output_00250")
    f1['mass']
    f1.physical_units()

    f2 = pynbody.load("testdata/ramses/ramses_partial_output_00250")
    f2.physical_units()
    f2['mass']

    np.testing.assert_allclose(f1.dm['mass'], f2.dm['mass'], atol=1e-5)

def test_rt_arrays():
    f1 = pynbody.load("testdata/ramses/ramses_rt_partial_output_00002",cpus=[1,2,3])

    for group in range(4):
        assert 'rad_%d_rho'%group in f1.gas.loadable_keys()
        assert 'rad_%d_flux'%group in f1.gas.loadable_keys()

    with pytest.warns(UserWarning, match="Loading RT data from disk. Photon densities are stored in flux units*"):
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
    f1 = pynbody.load("testdata/ramses/ramses_rt_partial_output_00002", cpus=[1, 2, 3])

    warn_msg = (
        "Loading RT data from disk. Photon densities are stored in flux units "
        "by Ramses and need to be multiplied by the reduced speed of light of "
        "the run to obtain a physical number. This is currently left to the user, "
        "see issue 542 for more discussion."
    )
    with pytest.warns(UserWarning, match=warn_msg):
        f1.gas['rad_0_rho']


@pytest.mark.filterwarnings(r"ignore:Using field at offset \d to distinguish.*:UserWarning")
def test_all_dm():
    f1 = pynbody.load("testdata/ramses/ramses_dmo_partial_output_00051")

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

@pytest.mark.filterwarnings(r"ignore:Using field at offset \d to distinguish.*:UserWarning")
def test_forcegas_dmo():
    f_dmo = pynbody.load("testdata/ramses/ramses_dmo_partial_output_00051", cpus=[1], force_gas=True)
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


def test_metals_field_correctly_copied_from_metal(ramses_file):
    np.testing.assert_allclose(ramses_file.st['metals'][::5000], ramses_file.st['metal'][::5000], rtol=1e-5)


def _test_tform_checker(tform_raw):
    np.testing.assert_allclose(
        tform_raw[:10],
        [
            -2.72826591,
            -1.8400868,
            -2.35988485,
            -3.81799766,
            -2.67772371,
            -3.22276503,
            -2.5208477,
            -2.67845014,
            -3.17295132,
            -2.43044642,
        ],
        rtol=1e-5,
    )

def test_tform_and_tform_raw(ramses_file):
    # Standard test output is a non-cosmological run, for which tform should be read from disk,
    # rather than transformed. Tform raw and transformed are therefore the same
    assert len(ramses_file.st["tform"]) == len(ramses_file.st["tform_raw"]) == 2655
    np.testing.assert_allclose(ramses_file.st["tform_raw"], ramses_file.st["tform"])

    fcosmo = pynbody.load("testdata/ramses/output_00080")

    warn_msgs = (
        (
            "Assumed times to be in conformal units because no RT file "
            "was found. If this is incorrect, pass the "
            "`times_are_proper` keyword argument when loading the dataset, "
            "or set the option `proper_time` in your .pynbodyrc."
        ),
        (
            "Namelist file either not found or unable to read. Guessing "
            "whether run is cosmological from cosmological parameters "
            "assuming flat LCDM."
        ),
    )
    with pytest.warns(UserWarning) as record:
        tform = fcosmo.st["tform"]
        tform_raw = fcosmo.st["tform_raw"]

    for rec in record:
        assert rec.category == UserWarning
        assert rec.message.args[0] in warn_msgs

    # Make sure we use conformal times
    assert fcosmo.times_are_proper is False

    # Reference values have been computed with `part2birth`
    np.testing.assert_allclose(
        tform[:10],
        (
            2.7014733811298863,
            4.067542816866577,
            3.177232587827327,
            1.7550122687291079,
            2.760737434676254,
            2.2032690196559432,
            2.956344973463002,
            2.7598682671833252,
            2.2475191609123915,
            3.077730768582204
        ),
        rtol=1e-2,
    )
    _test_tform_checker(tform_raw)


def test_rt_conformal_time_detection(ramses_file):
    path = Path("testdata/ramses/output_00080")
    f = pynbody.load(str(path), cpus=[1, 2, 3])

    rt_path = path / "rt_00080.out00001"

    try:
        rt_path.touch()

        warn_msgs = (
            (
                "Assumed times to be in proper units because one RT file "
                f"was detected ({str(rt_path)}). If this is incorrect, pass the "
                "`times_are_proper` keyword argument when loading the dataset, "
                "or set the option `proper_time` in your .pynbodyrc."
            ),
            (
                "Namelist file either not found or unable to read. Guessing "
                "whether run is cosmological from cosmological parameters "
                "assuming flat LCDM."
            ),
        )

        with pytest.warns(UserWarning) as record:
            assert ramses_file.times_are_proper

        for rec in record:
            assert rec.category == UserWarning
            assert rec.message.args[0] in warn_msgs

    finally:
        rt_path.unlink(missing_ok=True)




@pytest.fixture
def use_part2birth_by_default():
    use_part2birth_by_default = pynbody.config_parser.get("ramses", "use_part2birth_by_default")
    ramses_utils = pynbody.config_parser.get("ramses", "ramses_utils")

    pynbody.config_parser.set("ramses", "use_part2birth_by_default", "True")
    pynbody.config_parser.set("ramses", "ramses_utils", "/this/is/an/invalid/path")

    yield

    pynbody.config_parser.set("ramses", "use_part2birth_by_default", use_part2birth_by_default)
    pynbody.config_parser.set("ramses", "ramses_utils", ramses_utils)

@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
def test_tform_and_tform_raw_without_sidecar_files(use_part2birth_by_default):
    fcosmo = pynbody.load("testdata/ramses/output_00080")

    warn_msg = (
        "Failed to read 'tform' from birth files at .* "
        "and to generate them with utility at .*"
    )
    with pytest.warns(UserWarning, match=warn_msg):
        tform = fcosmo.st["tform"]
        tform_raw = fcosmo.st["tform_raw"]

    np.testing.assert_allclose(tform, np.full(31990, -1))
    _test_tform_checker(tform_raw)

def test_proper_time_loading():
    expected_times = np.array([2.50416037, 2.5497581 , 2.64287886, 2.97452475, 2.47241073,
          3.60567204, 2.19985156, 2.51807006, 2.28102533, 3.43425507,
          2.46442147, 3.4058562 , 2.37037218, 2.72296016, 3.05385281,
          2.66972991, 2.9498538 , 3.05704494, 2.46656367, 3.78090582,
          3.93131933, 2.22832044, 3.9992109 , 3.63906271, 2.61553379,
          2.67235223, 2.57891777, 4.02026162, 2.75952408, 2.69260326,
          2.35966261, 4.34911266, 2.66637349, 3.35447046, 3.25334058,
          3.01477982, 2.41240385, 2.6309636 , 2.88769616, 2.54317847])
    for (load_opts, force) in (
        ({"times_are_proper": True}, False),
        ({}, True),
    ):
        f_pt = pynbody.load(
            "testdata/ramses/prop_time_output_00030",
            cpus=range(10, 20),
            **load_opts,
        )
        if force:
            f_pt.times_are_proper = True

        f_pt._load_particle_block('tform')
        f_pt._convert_tform()

        np.testing.assert_allclose(
            f_pt.s["tform"].in_units("Gyr"),
            expected_times,
            rtol=1e-5
        )


@pytest.mark.filterwarnings(r"ignore:Using field at offset \d to distinguish.*:UserWarning")
def test_is_cosmological_without_namelist():
    # Load a cosmo run, but without the namelist.txt file and checks that cosmo detection works with a warning
    f_without_namelist = pynbody.load("testdata/ramses/ramses_dmo_partial_output_00051")
    f_without_namelist.physical_units()

    warn_msg = (
        "Namelist file either not found or unable to read. Guessing whether run "
        "is cosmological from cosmological parameters assuming flat LCDM."
    )
    with pytest.warns(UserWarning, match=warn_msg):
        assert f_without_namelist.is_cosmological


@pytest.mark.filterwarnings("ignore:No ionization fractions found, assuming.*:UserWarning")
def test_temperature_derivation(ramses_file):
    ramses_file.g['mu']
    ramses_file.g['temp']

    assert(ramses_file.g['mu'].min() != ramses_file.g['mu'].max())   # Check that ionized and neutral mu are now generated
    np.testing.assert_allclose(ramses_file.g['mu'][:10], 0.590406 * np.ones(10), rtol=1e-5)
    np.testing.assert_allclose(ramses_file.g['mu'].max(), 1.225115, rtol=1e-5)

    np.testing.assert_allclose(ramses_file.g['temp'][:10], [26006.242067, 28014.523369, 28014.490285,
                                                                   30537.49731 , 26950.35421 , 29093.774613,
                                                                   29197.304228, 31939.90974 , 26950.349542,
                                                                   29197.349569])


def test_family_array(ramses_file):
    ramses_file.dm['mass']
    assert "mass" in ramses_file.family_keys()
    assert ramses_file.dm['mass'].sim == ramses_file.dm


def test_file_descriptor_reading():
    f = pynbody.load("testdata/ramses/prop_time_output_00030")

    expected_fields = ["rho", "vel", "metal", *(f"scalar_{i+1:02d}" for i in range(2))]
    loadable_fields = f.g.loadable_keys()

    for field in expected_fields:
        assert field in loadable_fields


@pytest.mark.filterwarnings(r"ignore:Using field at offset \d to distinguish.*:UserWarning")
def test_tform_and_metals_do_not_break_loading_when_not_present_in_particle_blocks():
    # DMO snapshot would not have tform or metals in the header or defined on disc
    f_dmo = pynbody.load("testdata/ramses/ramses_dmo_partial_output_00051", force_gas=True)

    # This should break the loading with a Key Error that the array cannot be found
    # Previous to the fix of #689, it would break with
    # ValueError: 'metal' is not in list
    # Because the field is not present in the particle blocks but was attempted to be accessed
    with pytest.raises(KeyError, match="No array.*"):
        f_dmo.st['metals'] # noqa

    # Now define a custom-derived metals array, which should enable us to access the array at all time
    # Previously to #689, this would still break the loading with the same ValueError
    # Now it loads the derived field without issues
    from pynbody.snapshot.ramses import RamsesSnap
    @RamsesSnap.derived_array
    def metals(snap):
        return np.zeros(len(snap))

    f_dmo.st['metals'] # noqa


def array_by_array_test_tipsy_converter(ramses_snap, tipsy_snap):
    # Setup a function to extensively test whether Tipsy snapshot written to disc with ramses_util
    # match their original Ramses values and have correct units
    ramses_snap.physical_units()
    tipsy_snap.physical_units()

    # Test lengths
    assert (len(ramses_snap.d) == len(tipsy_snap.d))
    assert (len(ramses_snap.st) == len(tipsy_snap.st))
    assert (len(ramses_snap.g) == len(tipsy_snap.g))

    # This level of precision is limited by the many hardcoded SI constants
    # in the tipsy/ramses loaders and converters, which are not self-consitent with one another
    rtol = 5e-4

    # Test header properties
    npt.assert_allclose(ramses_snap.properties['time'].in_units("Gyr"),
                        tipsy_snap.properties['time'].in_units("Gyr"),
                        rtol=rtol)
    npt.assert_allclose(ramses_snap.properties['a'], tipsy_snap.properties['a'])
    npt.assert_allclose(ramses_snap.properties['h'], tipsy_snap.properties['h'], rtol=rtol)
    npt.assert_allclose(ramses_snap.properties['omegaM0'], tipsy_snap.properties['omegaM0'])
    npt.assert_allclose(ramses_snap.properties['omegaL0'], tipsy_snap.properties['omegaL0'])
    npt.assert_allclose(ramses_snap.properties['boxsize'].in_units("Mpc"),
                        tipsy_snap.properties['boxsize'].in_units("Mpc"), rtol=rtol)

    # Dark matter
    npt.assert_allclose(ramses_snap.d['pos'], tipsy_snap.d['pos'], rtol=rtol)
    npt.assert_allclose(ramses_snap.d['vel'], tipsy_snap.d['vel'], rtol=rtol)
    npt.assert_allclose(ramses_snap.d['mass'], tipsy_snap.d['mass'], rtol=rtol)

    # Stars
    if len(tipsy_snap.st) > 0:
        npt.assert_allclose(ramses_snap.st['pos'], tipsy_snap.st['pos'], rtol=rtol)
        npt.assert_allclose(ramses_snap.st['vel'], tipsy_snap.st['vel'], rtol=rtol)
        npt.assert_allclose(ramses_snap.st['mass'], tipsy_snap.st['mass'], rtol=rtol)
        npt.assert_allclose(ramses_snap.st['tform'], tipsy_snap.st['tform'], rtol=rtol)

    # Gas
    if len(tipsy_snap.g) > 0:
        npt.assert_allclose(ramses_snap.g['pos'], tipsy_snap.g['pos'], rtol=rtol)
        npt.assert_allclose(ramses_snap.g['vel'], tipsy_snap.g['vel'], rtol=rtol)
        npt.assert_allclose(ramses_snap.g['mass'], tipsy_snap.g['mass'], rtol=rtol)


@pytest.mark.filterwarnings(r"ignore:Using field at offset \d to distinguish.*:UserWarning")
@pytest.mark.parametrize("pass_as_str", [True, False])
def test_tipsy_conversion_for_dmo(pass_as_str):
    path = "testdata/ramses/ramses_dmo_partial_output_00051"
    if pass_as_str:
        f_dmo = path
    else:
        f_dmo = pynbody.load(path)

    with pytest.warns(UserWarning, match=r"This routine currently makes the assumption that the ramses snapshot is cosmological.*"):
        pynbody.analysis.ramses_util.convert_to_tipsy_fullbox(f_dmo, write_param=True)

    if pass_as_str:
        f_dmo = pynbody.load(path)

    # There are many tipsy parameter files that are automatically detected by the loader
    # in the testdata folder, make sure we point the right one
    tipsy_path = path + "_fullbox.tipsy"
    try:
        tipsy_dmo = pynbody.load(tipsy_path, paramfile=tipsy_path + ".param")

        array_by_array_test_tipsy_converter(f_dmo, tipsy_dmo)
    finally:
        # Clean up our created param file to avoid it being detected and picked up by other tipsy tests
        os.remove(tipsy_path + ".param")


@pytest.mark.filterwarnings(r"ignore:No ionization fractions found.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
def test_tipsy_conversion_for_cosmo_gas():
    path = "testdata/ramses/output_00080"
    try:
        # The namelist file is not included in the test data
        # Write a quick one-liner to ensure that we identify cosmo correctly
        # and get the correct time units
        with open(path + os.sep + "namelist.txt", "w") as namelist:
            namelist.write("cosmo=.true.")

        f = pynbody.load(path)
        with pytest.warns(UserWarning, match=r"This routine currently makes the assumption that "
                                             r"the ramses snapshot is cosmological.*"):
            pynbody.analysis.ramses_util.convert_to_tipsy_fullbox(f, write_param=True)


        tipsy_path = path + "_fullbox.tipsy"

        tipsy = pynbody.load(tipsy_path, paramfile=tipsy_path + ".param")

        array_by_array_test_tipsy_converter(f, tipsy)
    finally:
        # Clean up our namelist to avoid any other issues with other tests
        Path(path + os.sep + "namelist.txt").unlink(missing_ok=True)
        Path(tipsy_path + ".param").unlink(missing_ok=True)

@pytest.mark.filterwarnings(r"ignore:Using field at offset \d to distinguish.*:UserWarning")
def test_negative_iords(ramses_file):
    fcosmo = pynbody.load("testdata/ramses/output_00080")

    # Make one iord negative -> bug flag should be true but no warning generated because iord are still unique
    fcosmo.dm['iord'][0] = -1
    assert fcosmo.has_negative_iords == True
    assert fcosmo.is_cosmological == True
    assert fcosmo.has_potential_negative_iords_bug == True


@pytest.fixture
def ramses_read_threads():
    """Set the threads used to read RAMSES files, restoring the configuration afterwards"""
    from pynbody.snapshot.ramses import _execute as _ramses_execute
    saved = dict(_ramses_execute.config)

    def set_threads(parallel_read, io_threads='auto'):
        _ramses_execute.config['parallel-read'] = parallel_read
        _ramses_execute.config['io-threads'] = io_threads

    yield set_threads
    _ramses_execute.config.update(saved)


def _all_ramses_arrays(f):
    arrays = {}
    for fam in f.families():
        for name in sorted(f[fam].loadable_keys()):
            if fam is pynbody.family.gas and name == 'mass':
                continue
            arrays[(fam.name, name)] = np.asarray(f[fam][name]).copy()
    return arrays


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
@pytest.mark.parametrize("kwargs", [{}, {"cpus": [7, 2, 11]}, {"maxlevel": 9}], ids=["all", "cpus", "maxlevel"])
def test_threaded_read_matches_serial(ramses_read_threads, kwargs):
    ramses_read_threads(1)
    serial = _all_ramses_arrays(pynbody.load("testdata/ramses/output_00080", **kwargs))
    ramses_read_threads(4, 3)
    threaded = _all_ramses_arrays(pynbody.load("testdata/ramses/output_00080", **kwargs))
    assert serial.keys() == threaded.keys()
    for k in serial:
        np.testing.assert_array_equal(serial[k], threaded[k], err_msg=str(k))


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
def test_maxlevel_counts_coarser_cells():
    f = pynbody.load("testdata/ramses/output_00080")
    f_coarse = pynbody.load("testdata/ramses/output_00080", maxlevel=9)
    assert len(f_coarse.gas) < len(f.gas)
    # the cells of a coarsened grid still fill the volume the full grid fills
    np.testing.assert_allclose((f_coarse.gas['smooth'] ** 3).sum(), (f.gas['smooth'] ** 3).sum(), rtol=1e-10)
    assert f_coarse.gas['smooth'].min() >= f.gas['smooth'].min()


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
@pytest.mark.parametrize("damage", ["truncate", "mismatch"])
def test_damaged_hydro_file_is_detected(tmp_path, damage):
    """A hydro file that does not have the layout its AMR file implies must raise an error, not load nonsense"""
    import shutil
    source = Path("testdata/ramses/output_00080")
    target = tmp_path / "output_00080"
    target.mkdir()
    for name in os.listdir(source):
        if os.path.isfile(source / name):
            os.symlink((source / name).absolute(), target / name)
    hydro = target / "hydro_00080.out00003"
    hydro.unlink()
    shutil.copy(source / "hydro_00080.out00003", hydro)
    if damage == "truncate":
        with open(hydro, "r+b") as f:
            f.truncate(os.path.getsize(hydro) // 2)
    else:
        # replace the file with another CPU's, whose octs lie elsewhere
        shutil.copy(source / "hydro_00080.out00004", hydro)

    f = pynbody.load(str(target), cpus=[3])
    with pytest.raises(ValueError):
        f.gas['rho'] # noqa
    assert 'rho' not in f.gas.keys()


def _split_runs(selection, layout):
    """A selection of the same cells as *selection* (every leaf), but with each level's octs in two runs"""
    from pynbody.snapshot.ramses import _selection
    twotondim = layout.twotondim
    level_start = np.concatenate([[0], np.cumsum(layout.ncache)[:-1]])
    runs, ncells = [], 0
    for level in np.flatnonzero(layout.ncache > 0):
        n = int(layout.ncache[level])
        for first, count in ((0, n // 2), (n // 2, n - n // 2)):
            if count == 0:
                continue
            mask = selection.mask[level_start[level] + first:level_start[level] + first + count]
            nleaf = [int((mask >> ind & 1).sum()) for ind in range(twotondim)]
            runs.append((level, first, count, level_start[level] + first,
                         ncells + np.concatenate([[0], np.cumsum(nleaf)[:-1]])))
            ncells += sum(nleaf)
    level, first, count, mask_offset, starts = zip(*runs)
    return _selection.CellSelection(level=np.array(level), first=np.array(first), count=np.array(count),
                                    mask_offset=np.array(mask_offset), starts=np.array(starts), mask=selection.mask,
                                    ncells=ncells)


def _cell_plans(f, selections):
    """Plans of reading cell positions and every hydro variable of a snapshot, for the given selections"""
    from pynbody.snapshot.ramses import _layout, _plan
    first = np.cumsum([0] + [sel.ncells for sel in selections])[:-1]
    paths = [f._hydro_filename(cpu) for cpu in f._cpus]
    header, header_nbytes = _layout.read_header(paths[0], pynbody.snapshot.ramses.ramses_hydro_header)
    kind = _layout.HydroLikeFile(header_nbytes, int(header['nvarh']))
    positions = _plan.cell_positions(f._amr_layouts, selections, first, [f._amr_filename(cpu) for cpu in f._cpus])
    values = _plan.cell_values(f._amr_layouts, selections, first, paths, kind, [(v, v) for v in range(kind.nvar)])
    return positions, values, kind


def _check_plan(plan, job_nbytes):
    """Each piece of a plan reads increasing, separate ranges of its file into consecutive places in its buffer, and
    decodes and checks only what lies within the buffer"""
    assert plan.pieces
    for piece in plan.pieces:
        offset, nbytes, at = piece.reads.T
        assert (nbytes > 0).all() and (offset[1:] >= offset[:-1] + nbytes[:-1]).all()
        assert offset[-1] + nbytes[-1] <= os.path.getsize(piece.path)
        np.testing.assert_array_equal(at, np.cumsum(nbytes) - nbytes)
        assert piece.nbytes == nbytes.sum() >= piece.requested
        assert (piece.checks[:, 0] >= 0).all() and (piece.checks[:, 0] + 4 <= piece.nbytes).all()
        assert (piece.rows[:, 0] >= 0).all() and (piece.rows[:, 0] < piece.nbytes).all()
    # pieces reading a few megabytes, gaps included, unless a file holds less
    if job_nbytes is not None:
        assert max(p.nbytes for p in plan.pieces) < 2 * job_nbytes


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
@pytest.mark.parametrize("kwargs", [{}, {"cpus": [7, 2, 11], "maxlevel": 12}], ids=["all", "cpus"])
def test_read_plans_are_consistent(kwargs):
    from pynbody.snapshot.ramses import _execute
    f = pynbody.load("testdata/ramses/output_00080", **kwargs)
    f.dm['pos'], f.star['vel'] # noqa
    for plan in _cell_plans(f, f._cell_selections)[:2] + tuple(f._read_plans.values()):
        _check_plan(plan, _execute.job_nbytes)


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
@pytest.mark.parametrize("job_nbytes", [3000, 200000, 4 * 1024 * 1024])
def test_split_runs_are_read_whole(monkeypatch, job_nbytes):
    """Runs covering part of a level (whose ranges interleave in the file with those of other runs) give the same
    cells, however the reads are cut into pieces"""
    from pynbody.snapshot.ramses import _execute, _plan
    f = pynbody.load("testdata/ramses/output_00080", cpus=[3, 1])
    pos, rho = np.asarray(f.gas['pos']), np.asarray(f.gas['rho'])
    monkeypatch.setattr(_execute, "job_nbytes", job_nbytes)
    selections = [_split_runs(sel, layout) for sel, layout in zip(f._cell_selections, f._amr_layouts)]
    positions, values, kind = _cell_plans(f, selections)
    for plan in positions, values:
        # (the coordinates of each run must be read together, and those of the two runs of a level interleave, so a
        # level's centres are read in one piece, whatever its size)
        _check_plan(plan, None)
    split_pos, smooth = np.zeros_like(pos), np.zeros(len(pos))
    _plan.load_cell_positions(positions, selections, f._amr_geometry, f._info['boxlen'], split_pos, smooth)
    arrays = [np.zeros(len(pos)) for _ in range(kind.nvar)]
    _plan.load_cell_values(values, selections, arrays)
    # (cells come in another order: within each run rather than each level)
    order, split_order = np.lexsort(pos.T), np.lexsort(split_pos.T)
    np.testing.assert_array_equal(pos[order], split_pos[split_order])
    np.testing.assert_array_equal(rho[order], arrays[0][split_order])


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
def test_particle_read_plans_are_reused(monkeypatch):
    from pynbody.snapshot.ramses import _plan
    builds = []
    particle_fields = _plan.particle_fields
    monkeypatch.setattr(_plan, "particle_fields", lambda *args: builds.append(args) or particle_fields(*args))
    f = pynbody.load("testdata/ramses/output_00080")
    pos = f['pos'].copy()
    assert 'particle fields' in [k[0] for k in f._read_plans] and len(builds) == 1
    del f['pos'], f['vel']
    np.testing.assert_array_equal(f['pos'], pos)
    assert len(builds) == 1


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
def test_plans_check_their_destinations():
    """A plan performed into arrays too small for it raises, rather than writing beyond them"""
    from pynbody.snapshot.ramses import _plan
    f = pynbody.load("testdata/ramses/output_00080", cpus=[2])
    positions, values, kind = _cell_plans(f, f._cell_selections)
    n = len(f.gas) // 2
    with pytest.raises(ValueError, match="destination"):
        _plan.load_cell_values(values, f._cell_selections, [np.zeros(n)])
    with pytest.raises(ValueError, match="destination"):
        _plan.load_cell_positions(positions, f._cell_selections, f._amr_geometry, 1.0, np.zeros((n, 3)), np.zeros(n))
    f.dm['pos'] # noqa
    plan, = f._read_plans.values()
    with pytest.raises(ValueError, match="destination"):
        _plan.load_particle_fields(plan, f._particle_selections,
                                   [(np.float64, [np.zeros(len(f.dm) // 2), np.zeros(len(f.star))])] * 3)
    with pytest.raises(ValueError, match="distinct"):
        _plan.cell_values(f._amr_layouts, f._cell_selections, [0], [f._hydro_filename(2)], kind, [(0, 0), (0, 1)])
    with pytest.raises(ValueError, match="distinct"):
        _plan.cell_values(f._amr_layouts, f._cell_selections, [0], [f._hydro_filename(2)], kind, [(kind.nvar, 0)])


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
def test_read_plans_are_reused():
    f = pynbody.load("testdata/ramses/output_00080")
    rho, pos = f.gas['rho'].copy(), f['pos'].copy()
    plans = dict(f._read_plans)
    del f.gas['rho'], f['pos']
    np.testing.assert_array_equal(f.gas['rho'], rho)
    np.testing.assert_array_equal(f['pos'], pos)
    assert f._read_plans.keys() == plans.keys()
    assert all(f._read_plans[k] is plans[k] for k in plans)


def test_negative_particle_record_length_is_detected(tmp_path):
    """A negative Fortran record length (a damaged file, or gfortran's subrecords over 2 GB) raises ValueError"""
    import shutil
    import struct
    source = Path("testdata/ramses/output_00080")
    target = tmp_path / "output_00080"
    target.mkdir()
    for name in os.listdir(source):
        if os.path.isfile(source / name):
            os.symlink((source / name).absolute(), target / name)
    part = target / "part_00080.out00001"
    part.unlink()
    shutil.copy(source / "part_00080.out00001", part)
    with open(part, "r+b") as f:
        pos = 0
        for _ in range(8):  # skip the header records
            f.seek(pos)
            pos += 8 + struct.unpack("i", f.read(4))[0]
        f.seek(pos)
        f.write(struct.pack("i", -8))
    with pytest.raises(ValueError, match="Negative record length"):
        pynbody.load(str(target), cpus=[1])


def test_converted_destinations_start_zeroed():
    """Temporaries for destinations of another dtype start as zeros, so elements left unwritten (such as the unused
    columns of pos in a 2D output) are not uninitialised memory"""
    from pynbody.snapshot.ramses import _plan
    (out,), copies = _plan._destinations([np.full(5, 7, dtype=np.float32)], np.float64)
    assert out.dtype == np.float64 and (out == 0).all() and len(copies) == 1


def _assemble(ranges, gap):
    """Plan reads of byte ranges (file, offset, number of bytes), each to be decoded alone, from files 'a' and 'b'"""
    from pynbody.snapshot.ramses import _plan
    ranges = _plan._ranges(*np.array(ranges).T)
    return _plan._assemble(['a', 'b'], ranges, np.zeros((len(ranges), 1), dtype=np.int64),
                           np.arange(len(ranges))[:, None], [0], np.zeros((0, 4), dtype=np.int64), gap)


def test_ranges_closer_than_the_gap_are_read_together():
    ranges = [(0, 0, 10), (0, 30, 10), (0, 1000, 10), (1, 0, 10)]
    a, b = _assemble(ranges, 20)
    np.testing.assert_array_equal(a.reads, [[0, 40, 0], [1000, 10, 40]])
    np.testing.assert_array_equal(a.rows[:, 0], [0, 30, 40])  # (where each range lands in the buffer)
    assert (a.nbytes, a.requested) == (50, 30)
    np.testing.assert_array_equal(b.reads, [[0, 10, 0]])  # (another file is never merged in)
    a, b = _assemble(ranges, 2000)
    np.testing.assert_array_equal(a.reads, [[0, 1010, 0]])
    np.testing.assert_array_equal(a.rows[:, 0], [0, 30, 1000])
    assert (a.nbytes, a.requested) == (1010, 30)


def test_pieces_are_sized_by_the_bytes_they_read(monkeypatch):
    """With the gaps between ranges read too, a piece reads no more than job_nbytes, rather than holding ranges of
    that much"""
    from pynbody.snapshot.ramses import _execute
    monkeypatch.setattr(_execute, "job_nbytes", 100)
    ranges = [(0, 60 * k, 10) for k in range(20)]  # (10 bytes, then 50 between)
    pieces = _assemble(ranges, 60)
    # each piece reads 10 + 60 = 70 bytes (two ranges: a third would take it to 130), where counting only the
    # ranges would take ten
    assert [len(p.rows) for p in pieces] == [2] * 10
    assert [p.nbytes for p in pieces] == [70] * 10
    # with a gap too small to merge them, ranges count alone
    assert [len(p.rows) for p in _assemble(ranges, 40)] == [10, 10]
    # a range bigger than a job is a piece of its own
    assert [len(p.rows) for p in _assemble([(0, 0, 10), (0, 20, 500), (0, 530, 10)], 60)] == [1, 1, 1]


def test_reads_cover_every_range():
    """A read reaches the end of every range in it, even one that is empty or sorts after a longer one"""
    piece, = _assemble([(0, 42, 4), (0, 42, 0), (0, 0, 10)], 64)
    np.testing.assert_array_equal(piece.reads, [[0, 46, 0]])


def test_scan_reads_are_bounded(tmp_path):
    """Reads whose size is not known in advance (refinement maps, when a snapshot is opened) merge no more bytes
    between ranges than the ranges hold, whatever the read gap"""
    from pynbody.snapshot.ramses import _layout
    path = tmp_path / "data"
    path.write_bytes(bytes(range(256)) * 4096)  # 1 MB
    offsets, nbytes = [0, 100, 300000, 300100, 900000], [10, 10, 1000, 1000, 10]
    with open(path, "rb") as f:
        buf, at = _layout._read_ranges(f, offsets, nbytes, 10 ** 9)
    data = path.read_bytes()
    for o, n, a in zip(offsets, nbytes, at):
        assert buf[a:a + n].tobytes() == data[o:o + n]
    assert len(buf) <= 2 * sum(nbytes)



@pytest.mark.parametrize("filesystem, gap", [("ext4", 64 * 1024), (None, 64 * 1024), ("tmpfs", 64 * 1024),
                                             ("nfs4", 256 * 1024), ("9p", 256 * 1024), ("lustre", 1024 * 1024),
                                             ("gpfs", 1024 * 1024), ("fuse.s3fs", 1024 * 1024)])
def test_read_gap_depends_on_the_filesystem(monkeypatch, caplog, filesystem, gap):
    from pynbody.snapshot.ramses import _execute
    path = ["/data/output_00001/amr_00001.out00001"]
    monkeypatch.setattr(_execute.strategy, "filesystem_type", lambda path: filesystem)
    monkeypatch.setitem(_execute.config, "read-gap", "auto")
    assert _execute.read_gap(path) == gap
    monkeypatch.setitem(_execute.config, "read-gap", 12345)
    assert _execute.read_gap(path) == 12345
    # a gap is never more than a quarter of a job, so that pieces stay near the size of a job
    monkeypatch.setitem(_execute.config, "read-gap", 10 ** 9)
    assert _execute.read_gap(path) == _execute.job_nbytes // 4
    # a value set during a session that is not a number of bytes counts as 'auto'
    for value in ("1MB", None, -5):
        monkeypatch.setitem(_execute.config, "read-gap", value)
        assert _execute.read_gap(path) == gap
    assert "Ignoring the read gap" in caplog.text


@pytest.mark.parametrize("value, expected", [(None, "auto"), ("auto", "auto"), ("262144", 262144), ("0", 0),
                                             ("-1", "auto"), ("1MB", "auto")])
def test_read_gap_option(monkeypatch, caplog, value, expected):
    from pynbody.snapshot.ramses import _execute

    class Config:
        def get(self, section, name, fallback):
            return value if name == "read-gap" and value is not None else fallback

    monkeypatch.setattr(_execute, "config_parser", Config())
    assert _execute._read_config()["read-gap"] == expected
    assert ("Ignoring" in caplog.text) == (value in ("-1", "1MB"))


def test_thread_options_accept_any_number(monkeypatch):
    """io-threads of 0 or less means one input thread (as it always has), not 'auto'"""
    from pynbody.snapshot.ramses import _execute

    class Config:
        def get(self, section, name, fallback):
            return {"io-threads": "0", "parallel-read": "-1"}.get(name, fallback)

    monkeypatch.setattr(_execute, "config_parser", Config())
    config = _execute._read_config()
    assert (config["io-threads"], config["parallel-read"]) == (0, -1)
    monkeypatch.setattr(_execute, "config", config)
    assert _execute.choose_strategy(["/data/a"], 10, 1000).io_threads == 1


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
def test_large_read_gap_reads_the_same(monkeypatch, caplog):
    """A large read gap merges more of each file into each read, but loads the same values, in pieces that still
    read only a few megabytes"""
    import logging

    from pynbody.snapshot.ramses import _execute
    f = pynbody.load("testdata/ramses/output_00080")
    expected = {k: np.asarray(f[k]).copy() for k in ('pos', 'vel', 'mass')}
    expected['rho'] = np.asarray(f.gas['rho']).copy()
    monkeypatch.setitem(_execute.config, "read-gap", 1024 * 1024)
    with caplog.at_level(logging.DEBUG, logger='pynbody.snapshot.ramses'):
        f = pynbody.load("testdata/ramses/output_00080")
        f.dm['pos'] # noqa
    assert "read gap 1 MB" in caplog.text
    for k in ('pos', 'vel', 'mass'):
        np.testing.assert_array_equal(f[k], expected[k])
    np.testing.assert_array_equal(f.gas['rho'], expected['rho'])
    from pynbody.snapshot.ramses import _plan
    positions, values, kind = _cell_plans(f, f._cell_selections)
    # (a single variable, as phase 2 will read, with far more between its records than in them)
    rho_only = _plan.cell_values(f._amr_layouts, f._cell_selections, f._gas_i0,
                                 [f._hydro_filename(cpu) for cpu in f._cpus], kind, [(0, 0)])
    for plan in (positions, values, rho_only) + tuple(f._read_plans.values()):
        assert plan.gap == 1024 * 1024
        _check_plan(plan, _execute.job_nbytes)
    rho = np.zeros(len(f.gas))
    _plan.load_cell_values(rho_only, f._cell_selections, [rho])
    np.testing.assert_array_equal(rho, expected['rho'])


@pytest.fixture
def hydro_loading():
    """Set the hydro-loading option, restoring the configuration afterwards"""
    from pynbody.snapshot.ramses import _execute
    saved = _execute.config['hydro-loading']

    def set_mode(mode):
        _execute.config['hydro-loading'] = mode

    yield set_mode
    _execute.config['hydro-loading'] = saved


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:.*hydro variables.*:RuntimeWarning")
@pytest.mark.filterwarnings(r"ignore:Loading RT data from disk.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:No header file found.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Using field at offset.*:UserWarning")
@pytest.mark.parametrize("output, kwargs", [
    ("output_00080", {}), ("output_00080", {"cpus": [5, 2, 9], "maxlevel": 12}), ("ramses_partial_output_00250", {}),
    ("ramses_new_format_partial_output_00001", {}), ("ramses_new_format_cosmo_with_ahf_output_00110", {}),
    ("ramses_rt_partial_output_00002", {"cpus": [3, 1, 2]}), ("ramses_dmo_partial_output_00051",
                                                              {"cpus": [1], "force_gas": True})])
def test_requested_hydro_loading_matches_all(hydro_loading, output, kwargs):
    """Loading each hydro, gravity and RT variable on its own gives what loading them all at once does"""
    path = "testdata/ramses/" + output
    arrays = {}
    for mode in ("all", "requested"):
        hydro_loading(mode)
        f = pynbody.load(path, **kwargs)
        names = sorted(f.gas.loadable_keys())
        if os.path.exists(f._grav_filename(f._cpus[0])) and output != "output_00080":  # (whose gravity files are
            names += ['phi', 'accg']                                                     # in another format)
        arrays[mode] = {k: np.asarray(f.gas[k]).copy() for k in names}
        arrays[mode]['units'] = {k: str(f.gas[k].units) for k in names}
    assert arrays['all'].keys() == arrays['requested'].keys()
    for k in arrays['all']:
        if k == 'units':
            assert arrays['all'][k] == arrays['requested'][k]
        else:
            np.testing.assert_array_equal(arrays['all'][k], arrays['requested'][k], err_msg=k)


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Loading RT data from disk.*:UserWarning")
@pytest.mark.parametrize("read_gap", [64 * 1024, 1024 * 1024])
def test_requested_hydro_loading_reads_one_variable(hydro_loading, monkeypatch, read_gap):
    """Only the variables asked for are read, whatever the read gap"""
    from pynbody.snapshot.ramses import _execute
    monkeypatch.setitem(_execute.config, "read-gap", read_gap)
    hydro_loading("requested")
    f = pynbody.load("testdata/ramses/output_00080")
    f.gas['rho'] # noqa
    assert sorted(f.gas.keys()) == ['rho']
    f.gas['vel'] # noqa
    assert sorted(f.gas.keys()) == ['rho', 'vel', 'vx', 'vy', 'vz']  # (a vector and its components)
    # the synchronised load of pos and vel reads no other hydro variable, and gas mass only what it uses
    f = pynbody.load("testdata/ramses/output_00080")
    f['vel'] # noqa
    assert sorted(f.gas.keys()) == ['pos', 'smooth', 'vel', 'vx', 'vy', 'vz', 'x', 'y', 'z']
    f['mass'] # noqa
    assert 'p' not in f.gas.keys() and 'metal' not in f.gas.keys()
    # RT variables, each flux with its components, and with units only on what is loaded
    f = pynbody.load("testdata/ramses/ramses_rt_partial_output_00002", cpus=[3, 1, 2])
    f.gas['rad_0_flux'] # noqa
    assert sorted(f.gas.keys()) == ['rad_0_flux', 'rad_0_flux_x', 'rad_0_flux_y', 'rad_0_flux_z']
    assert f.gas['rad_0_flux'].units == f._rt_unit
    # and the whole file at once, under the 'all' setting
    hydro_loading("all")
    f = pynbody.load("testdata/ramses/output_00080")
    f.gas['rho'] # noqa
    assert sorted(f.gas.keys()) == ['metal', 'p', 'rho', 'vel', 'vx', 'vy', 'vz']


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
def test_hydro_variable_count_is_warned_about_once(hydro_loading):
    hydro_loading("requested")
    f = pynbody.load("testdata/ramses/ramses_partial_output_00250")
    with pytest.warns(RuntimeWarning, match="More hydro variables") as record:
        f.gas['rho'], f.gas['p'], f.gas['vel'] # noqa
    assert len([w for w in record if "hydro variables" in str(w.message)]) == 1


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
def test_cell_read_plans_are_reused(hydro_loading, monkeypatch):
    from pynbody.snapshot.ramses import _plan
    builds = []
    cell_values = _plan.cell_values
    monkeypatch.setattr(_plan, "cell_values", lambda *args: builds.append(args) or cell_values(*args))
    hydro_loading("requested")
    f = pynbody.load("testdata/ramses/output_00080")
    rho = f.gas['rho'].copy()
    del f.gas['rho']
    np.testing.assert_array_equal(f.gas['rho'], rho)
    f.gas['p'] # noqa
    assert len(builds) == 2  # (rho once, then p)
    assert f._read_plans.nbytes <= _plan.plan_cache_nbytes


def test_plan_cache_lets_go_of_least_recently_used():
    from pynbody.snapshot.ramses import _plan

    def plan(n):
        table = np.zeros((n, 1), dtype=np.int8)
        return _plan.ReadPlan('test', [_plan.PlanPiece('a', 0, table, 0, 0, table, table)])

    cache = _plan.PlanCache(max_nbytes=100)
    cache.get_or_build('a', lambda: plan(40))
    cache.get_or_build('b', lambda: plan(40))
    cache.get_or_build('a', lambda: plan(40))  # (a is now the more recently used)
    cache.get_or_build('c', lambda: plan(40))
    assert list(cache.keys()) == ['a', 'c'] and cache.nbytes == 80
    cache.get_or_build('d', lambda: plan(200))  # (too big ever to keep)
    assert list(cache.keys()) == ['a', 'c']


@pytest.mark.parametrize("value, expected", [(None, "requested"), ("all", "all"), ("Requested", "requested"),
                                             ("some", "requested")])
def test_hydro_loading_option(monkeypatch, caplog, value, expected):
    from pynbody.snapshot.ramses import _execute

    class Config:
        def get(self, section, name, fallback):
            return value if name == "hydro-loading" and value is not None else fallback

    monkeypatch.setattr(_execute, "config_parser", Config())
    assert _execute._read_config()["hydro-loading"] == expected
    assert ("Ignoring" in caplog.text) == (value == "some")



@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
@pytest.mark.parametrize("mode", ["requested", "all"])
def test_loading_more_variables_leaves_loaded_ones_alone(hydro_loading, mode):
    """Arrays already loaded, and changed since, are not read again when other variables are loaded (except, under
    'all', those loaded in the same pass)"""
    hydro_loading(mode)
    f = pynbody.load("testdata/ramses/ramses_partial_output_00250")
    with pytest.warns(RuntimeWarning, match="hydro variables"):
        rho = f.gas['rho']
    rho.convert_units("Msol kpc^-3")
    converted = np.asarray(rho).copy()
    f.gas['p'][:] = -1  # (under 'all', loaded with rho)
    f.gas['vel'] # noqa
    assert f.gas['rho'].units == "Msol kpc^-3"
    np.testing.assert_array_equal(f.gas['rho'], converted)
    assert (f.gas['p'] == -1).all()


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
def test_failed_vector_load_can_be_retried(hydro_loading, monkeypatch):
    """A failure while loading a vector leaves neither the vector nor its components behind, so it can be tried
    again"""
    from pynbody.snapshot.ramses import _plan
    hydro_loading("requested")
    f = pynbody.load("testdata/ramses/output_00080")
    load_cell_values = _plan.load_cell_values

    def fail(*args):
        raise ValueError("Simulated failure")

    monkeypatch.setattr(_plan, "load_cell_values", fail)
    with pytest.raises(ValueError, match="Simulated"):
        f.gas['vel'] # noqa
    assert not {'vel', 'vx', 'vy', 'vz'} & set(f.gas.keys())
    monkeypatch.setattr(_plan, "load_cell_values", load_cell_values)
    assert f.gas['vel'].shape == (len(f.gas), 3)


@pytest.mark.filterwarnings(r"ignore:Assumed times to be in conformal units.*:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Namelist file either not found.*:UserWarning")
def test_read_gap_change_rebuilds_plans(monkeypatch):
    """Plans kept for reuse are not used once the read gap is changed"""
    from pynbody.snapshot.ramses import _execute
    f = pynbody.load("testdata/ramses/output_00080")
    rho = f.gas['rho'].copy()
    gap = _execute.read_gap([f._hydro_filename(1)])
    monkeypatch.setitem(_execute.config, "read-gap", 0)
    del f.gas['rho']
    np.testing.assert_array_equal(f.gas['rho'], rho)
    assert {plan.gap for plan in f._read_plans.values()} == {gap, 0}
