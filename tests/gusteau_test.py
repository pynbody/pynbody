"""Tests for the gusteau reader.

The two sample files are transcodings of the same CAMELS box, one from a SWIFT/EAGLE run and
one from an IllustrisTNG/AREPO run, cut down to a single particle of each type. They therefore
exercise both of the unit conventions that gusteau has to carry across from its source formats:
the EAGLE file has no Hubble scalings, while the TNG file retains them. The TNG file also
exercises the spec's convention for a code which cannot supply an absolute simulation time.
"""

import shutil

import h5py
import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.snapshot.gadgethdf
import pynbody.snapshot.gusteau
import pynbody.snapshot.swift
import pynbody.test_utils

EAGLE = "testdata/gusteau_tiny/gusteau.hdf5"
TNG = "testdata/gusteau_tiny/gusteauTNG.hdf5"


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gusteau_tiny", "swift", "arepo")


@pytest.mark.parametrize("filename", [EAGLE, TNG])
def test_load_identifies_gusteau(filename):
    f = pynbody.load(filename)
    assert isinstance(f, pynbody.snapshot.gusteau.GusteauSnap)


def test_gusteau_not_confused_with_other_hdf_formats():
    """Gusteau aliases its particle groups onto PartTypeN, so must not be mistaken for GadgetHDF"""
    for filename in (EAGLE, TNG):
        assert not pynbody.snapshot.swift.SwiftSnap._can_load(pynbody.util.pathlib.Path(filename))

    swift = pynbody.load("testdata/SWIFT/snap_0150.hdf5")
    assert not isinstance(swift, pynbody.snapshot.gusteau.GusteauSnap)
    assert isinstance(swift, pynbody.snapshot.swift.SwiftSnap)

    arepo = pynbody.load("testdata/arepo/agora_100.hdf5")
    assert not isinstance(arepo, pynbody.snapshot.gusteau.GusteauSnap)
    assert isinstance(arepo, pynbody.snapshot.gadgethdf.GadgetHDFSnap)


@pytest.mark.parametrize("break_by", ["remove_group", "remove_source_attr"])
def test_gusteau_identification_requires_spec_mandated_fields(tmp_path, break_by):
    """Identification rests on the five required groups plus /Header.Source"""
    broken = tmp_path / "broken.hdf5"
    shutil.copy(EAGLE, broken)
    with h5py.File(broken, "r+") as h:
        if break_by == "remove_group":
            del h["RunInfo"]
        else:
            del h["Header"].attrs["Source"]

    assert not pynbody.snapshot.gusteau.GusteauSnap._can_load(broken)


@pytest.mark.parametrize("filename", [EAGLE, TNG])
def test_gusteau_families(filename):
    f = pynbody.load(filename)
    assert len(f) == 4
    assert len(f.gas) == 1
    assert len(f.dm) == 1
    assert len(f.star) == 1
    assert len(f.bh) == 1


@pytest.mark.parametrize("filename", [EAGLE, TNG])
def test_gusteau_cosmology(filename):
    """Both samples are transcodings of the same z=0 CAMELS box"""
    f = pynbody.load(filename)
    npt.assert_allclose(f.properties['a'], 1.0)
    npt.assert_allclose(f.properties['z'], 0.0, atol=1e-10)
    npt.assert_allclose(f.properties['h'], 0.6711)
    npt.assert_allclose(f.properties['omegaM0'], 0.3)
    npt.assert_allclose(f.properties['omegaL0'], 0.7)
    npt.assert_allclose(f.properties['omegaB0'], 0.049)

    # the box is 25 Mpc/h on a side; the two files express that in different units, so the
    # boxsize must come back with units that match the coordinates it will be compared against
    npt.assert_allclose(f.properties['boxsize'].in_units("Mpc a", **f.conversion_context()),
                        25.0 / 0.6711, rtol=1e-5)
    npt.assert_allclose(
        f.properties['boxsize'].in_units(f['pos'].units, **f.conversion_context()),
        f['pos'].units.ratio("Mpc a", **f.conversion_context()) ** -1 * 25.0 / 0.6711,
        rtol=1e-5
    )


def test_gusteau_eagle_properties():
    f = pynbody.load(EAGLE)
    npt.assert_allclose(f.properties['omegaC0'], 0.251)
    npt.assert_allclose(f.properties['omegaNu0'], 0.0)
    # /Header.Time is the age of the universe, in internal time units
    npt.assert_allclose(f.properties['time'].in_units("Gyr"), 14.04692, rtol=1e-5)


def test_gusteau_eagle_arrays():
    """The EAGLE-sourced file follows swift's conventions: comoving lengths, no Hubble scalings"""
    f = pynbody.load(EAGLE)

    npt.assert_allclose(f['pos'].units.ratio("Mpc a", **f.conversion_context()), 1.0)
    npt.assert_allclose(f['vel'].units.ratio("km s^-1", **f.conversion_context()), 1.0)
    npt.assert_allclose(f['mass'].units.ratio("1e10 Msol", **f.conversion_context()), 1.0,
                        rtol=1e-3) # pynbody's Msol differs slightly from swift's

    npt.assert_allclose(f['pos'], [[0.57651471, 0.37188471, 0.18664471],
                                   [0.06105295, 0.25139295, 0.38215295],
                                   [0.41816579, 1.41555579, 13.90284578],
                                   [1.61998931, 0.58377931, 3.95751931]])
    npt.assert_allclose(f['vel'], [[-12.622681, 80.07732, 122.977325],
                                   [-23.69934, 82.30066, 119.20065],
                                   [-166.78284, -52.68284, -88.28284],
                                   [-143.49731, 81.80269, 123.10269]], rtol=1e-5)
    npt.assert_allclose(f['mass'], [0.00188725, 0.00966734, 0.00107235, 0.00188725], rtol=1e-5)
    npt.assert_equal(f['iord'], [516053, 252886, 5905141, 3794951])

    # gas quantities live in nested subgroups (e.g. Gas/Thermal/Temperatures), which must be
    # found and mapped onto pynbody names
    npt.assert_allclose(f.gas['rho'].units.ratio("1e10 Msol Mpc^-3 a^-3", **f.conversion_context()),
                        1.0, rtol=1e-3)
    npt.assert_allclose(f.gas['rho'], [0.35546875], rtol=1e-5)
    npt.assert_allclose(f.gas['temp'].in_units("K"), [5120.0], rtol=1e-5)
    npt.assert_allclose(f.gas['smooth'].in_units("Mpc a", **f.conversion_context()),
                        [0.21537267], rtol=1e-5)
    npt.assert_allclose(f.gas['u'].in_units("km^2 s^-2", **f.conversion_context()),
                        [106.919266], rtol=1e-5)
    assert f.gas['p'].units.dimensional_project(['Msol', 'Mpc', 's', 'a']) is not None


def test_gusteau_tng_arrays():
    """The TNG-sourced file retains gadget's Hubble scalings and sqrt(a) velocities"""
    f = pynbody.load(TNG)

    npt.assert_allclose(f['pos'].units.ratio("kpc a h^-1", **f.conversion_context()), 1.0, rtol=1e-6)
    npt.assert_allclose(f['vel'].units.ratio("km s^-1 a^1/2", **f.conversion_context()), 1.0)
    npt.assert_allclose(f['mass'].units.ratio("1e10 Msol h^-1", **f.conversion_context()), 1.0,
                        rtol=1e-3)

    npt.assert_allclose(f['pos'], [[18246.16, 5682.4814, 5337.1475],
                                   [18243.215, 5679.1577, 5336.367],
                                   [18243.018, 5679.5933, 5337.2017],
                                   [18243.018, 5679.5933, 5337.2017]], rtol=1e-6)
    npt.assert_allclose(f['vel'], [[142.7404, -4.2689576, -304.56458],
                                   [45.53827, 19.348883, 126.18352],
                                   [32.804043, -92.23529, 81.38516],
                                   [-6.695522, -54.019855, -23.656105]], rtol=1e-5)
    npt.assert_allclose(f['mass'], [1.4481847e-03, 6.4880629e-03, 4.8935722e-04, 8.3045864e-01],
                        rtol=1e-5)
    npt.assert_equal(f['iord'], [37250375, 17169954, 43981835, 62879555])

    npt.assert_allclose(f.gas['rho'].units.ratio("1e10 Msol kpc^-3 a^-3 h^2",
                                                 **f.conversion_context()), 1.0, rtol=1e-3)
    npt.assert_allclose(f.gas['rho'], [2.5119982e-05], rtol=1e-5)
    npt.assert_allclose(f.gas['u'].in_units("km^2 s^-2", **f.conversion_context()),
                        [1.0539128e+06], rtol=1e-5)


@pytest.mark.parametrize("filename", [EAGLE, TNG])
def test_gusteau_loadable_keys(filename):
    f = pynbody.load(filename)
    for key in ('pos', 'vel', 'mass', 'iord', 'phi'):
        assert key in f.loadable_keys()

    # untranslated names retain their path within the particle group
    assert 'Subgrid/InitialMasses' in f.star.loadable_keys()


def test_gusteau_halos_from_fof_group_ids():
    f = pynbody.load(EAGLE)
    h = f.halos()
    assert isinstance(h, pynbody.halo.number_array.HaloNumberCatalogue)
    # the 'no group' value comes from the transcoded FOF parameters, and must be excluded
    ignored = int(f._hdf_files.get_parameter_attrs()['FOF_group_id_default'])
    assert ignored not in h.keys()


@pytest.mark.parametrize("filename", [EAGLE, TNG])
def test_gusteau_region_selection_not_implemented(filename):
    """These samples have no spatial index, and pynbody cannot read gusteau's anyway"""
    with pytest.raises(NotImplementedError):
        pynbody.load(filename, take_region=pynbody.filt.Sphere("1 Mpc"))


@pytest.mark.parametrize("filename", [EAGLE, TNG])
def test_gusteau_write_array_not_implemented(filename):
    f = pynbody.load(filename)
    f['test'] = 1.0
    with pytest.raises(NotImplementedError):
        f.write_array('test')


def test_gusteau_boxsize_takes_the_coordinate_scalings():
    """The bounding box shares the cosmological scalings of the Coordinates datasets

    The gusteau spec fixes only the bounding box's unit conversion, leaving the a- and h-scalings
    implicit (see GusteauSnap._get_boxsize). The two samples are transcodings of the same 25 Mpc/h
    CAMELS box but disagree about Hubble scalings -- EAGLE stores 37.25 in units of Mpc, TNG
    stores 25000 in units of ckpc/h -- so they agree on the physical box only under this reading.
    """
    for filename in (EAGLE, TNG):
        f = pynbody.load(filename)
        context = f.conversion_context()
        boxsize = f.properties['boxsize']

        npt.assert_allclose(boxsize.in_units("Mpc a h^-1", **context), 25.0, rtol=1e-6)

        # the box must be usable against the positions: they lie within it, and wrapping (which
        # compares the two directly) brings everything into [-L/2, L/2]
        side = float(boxsize.in_units(f['pos'].units, **context))
        assert (np.asarray(f['pos']) >= 0).all() and (np.asarray(f['pos']) <= side).all()
        f.wrap()
        assert np.abs(np.asarray(f['pos'])).max() <= 0.5 * side


def test_gusteau_bounding_box_is_origin_and_widths(tmp_path):
    """/Header.Bounding_box is [x, y, z, dx, dy, dz], not a pair of opposite corners"""
    shifted = tmp_path / "shifted.hdf5"
    shutil.copy(EAGLE, shifted)
    with h5py.File(shifted, "r+") as h:
        box = np.array(h["Header"].attrs["Bounding_box"])
        box[:3] = -0.5 * box[3:]  # recentre the volume on the origin, leaving the widths alone
        h["Header"].attrs["Bounding_box"] = box

    # moving the origin must not change the box size; reading the array as two opposite
    # corners instead would give 1.5 times the true side length here
    npt.assert_allclose(pynbody.load(shifted).properties['boxsize'].in_units("Mpc a"),
                        pynbody.load(EAGLE).properties['boxsize'].in_units("Mpc a"))


def test_gusteau_negative_time_means_scalefactor():
    """A code with no absolute time may store -a instead, in which case we use the cosmology

    The TNG sample does exactly this, so its age must come out equal to that of the EAGLE
    sample, which is a transcoding of the same box at the same redshift and does supply a time.
    """
    tng = pynbody.load(TNG)
    assert tng._hdf_files[0]['Header'].attrs['Time'] < 0

    npt.assert_allclose(tng.properties['time'].in_units("Gyr"),
                        pynbody.load(EAGLE).properties['time'].in_units("Gyr"), rtol=1e-4)


def test_gusteau_ignores_source_file_count(tmp_path):
    """Num_files_per_snapshot describes the source snapshot; gusteau itself uses virtual datasets"""
    spanned = tmp_path / "spanned.hdf5"
    shutil.copy(EAGLE, spanned)
    with h5py.File(spanned, "r+") as h:
        h["Header"].attrs["Num_files_per_snapshot"] = 4

    f = pynbody.load(spanned)
    assert len(f) == 4
    assert not f.is_partially_loaded()


def test_gusteau_unknown_cosmology_parameters_are_omitted(tmp_path):
    """The 'best effort' cosmology attributes use -1 to mean 'not known'"""
    unknown = tmp_path / "unknown.hdf5"
    shutil.copy(EAGLE, unknown)
    with h5py.File(unknown, "r+") as h:
        h["Cosmology"].attrs["Omega_baryon"] = -1.0

    assert 'omegaB0' not in pynbody.load(unknown).properties.keys()
    assert 'omegaB0' in pynbody.load(EAGLE).properties.keys()
