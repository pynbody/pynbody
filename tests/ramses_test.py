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
