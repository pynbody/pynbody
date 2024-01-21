import glob
import os.path
import shutil
import stat
import subprocess

import numpy as np
import numpy.testing as npt
import pytest
import warnings

import pynbody

@pytest.fixture
def cleanup_fpos_file():
    if os.path.exists("testdata/g15784.lr.01024.AHF_fpos"):
        os.remove("testdata/g15784.lr.01024.AHF_fpos")
    yield


def test_load_ahf_catalogue():
    f = pynbody.load("testdata/g15784.lr.01024")
    h = pynbody.halo.AHFCatalogue(f)
    assert len(h)==1411

@pytest.mark.parametrize("do_load_all", [True, False])
def test_ahf_particles(do_load_all, cleanup_fpos_file):
    f = pynbody.load("testdata/g15784.lr.01024")
    h = pynbody.halo.AHFCatalogue(f)

    if do_load_all:
        h.load_all()

    assert len(h[1])==502300
    assert (h[1]['iord'][::10000]==[57, 27875, 54094, 82969, 112002, 140143, 173567, 205840, 264606,
           301694, 333383, 358730, 374767, 402300, 430180, 456015, 479885, 496606,
           519824, 539971, 555195, 575204, 596047, 617669, 652724, 1533992, 1544021,
           1554045, 1564080, 1574107, 1584130, 1594158, 1604204, 1614257, 1624308, 1634376,
           1644485, 1654580, 1664698, 1674831, 1685054, 1695252, 1705513, 1715722, 1725900,
           1736070, 1746235, 1756400, 1766584, 1776754, 1786886]).all()
    assert len(h[20])==3272
    assert(h[20]['iord'][::1000] == [232964, 341019, 752354, 793468]).all()


@pytest.mark.parametrize("do_load_all", [True, False])
def test_load_ahf_catalogue_non_gzipped(do_load_all):
    for extension in ["halos", "particles", "substructure"]:
        subprocess.call(["gunzip",f"testdata/g15784.lr.01024.z0.000.AHF_{extension}.gz"])
    try:
        f = pynbody.load("testdata/g15784.lr.01024")
        h = pynbody.halo.AHFCatalogue(f)
        if do_load_all:
            h.load_all()
        assert len(h)==1411
    finally:
        for extension in ["halos", "particles", "substructure"]:
            subprocess.call(["gzip", f"testdata/g15784.lr.01024.z0.000.AHF_{extension}"])


def test_ahf_properties():
    f = pynbody.load("testdata/g15784.lr.01024")
    h = pynbody.halo.AHFCatalogue(f)
    assert np.allclose(h[1].properties['Mvir'], 1.69639e+12)
    assert np.allclose(h[2].properties['Ekin'],6.4911e+17)
    assert np.allclose(h[2].properties['Mvir'], 1.19684e+13)


@pytest.fixture
def setup_unwritable_ahf_situation():
    if os.path.exists("testdata/test_unwritable"):
        os.chmod("testdata/test_unwritable", stat.S_IRUSR | stat.S_IXUSR | stat.S_IWUSR)
        shutil.rmtree("testdata/test_unwritable")
    os.mkdir("testdata/test_unwritable/")
    for fname in glob.glob("testdata/g15784*"):
        if "AHF_fpos" not in fname:
            os.symlink("../"+fname[9:], "testdata/test_unwritable/"+fname[9:])
    os.chmod("testdata/test_unwritable", stat.S_IRUSR | stat.S_IXUSR)


def test_ahf_unwritable(setup_unwritable_ahf_situation):
    f = pynbody.load("testdata/test_unwritable/g15784.lr.01024")

    # check we can still get a halo even without ability to write fpos file, but a warning is issued
    with pytest.warns(UserWarning, match="Unable to write AHF_fpos file;.*"):
        h = f.halos()
        _ = h[1]

    # check no subsequent warning issued
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = h[2]

    # check if we use load_all there is no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h = f.halos()
        h.load_all()
        _ = h[1]


def test_detecting_ahf_catalogues_with_without_trailing_slash():
    # Test small fixes in #688 to detect AHF catalogues with and wihtout trailing slashes in directories
    for name in (
        "testdata/ramses_new_format_cosmo_with_ahf_output_00110",
        "testdata/ramses_new_format_cosmo_with_ahf_output_00110/"
    ):
        f = pynbody.load(name)
        _halos = pynbody.halo.AHFCatalogue(f)


def test_ramses_ahf_family_mapping_with_new_format():
    # Test Issue 691 where family mapping of AHF catalogues with Ramses new particle formats would go wrong
    f = pynbody.load("testdata/ramses_new_format_cosmo_with_ahf_output_00110")
    halos = pynbody.halo.AHFCatalogue(f)

    assert len(halos) == 149    # 150 lines in AHF halos file

    # Load halos and check that stars, DM and gas are correctly mapped by pynbody
    # Halo 1 is the main halo and has all three families, while other are random picks
    halo_numbers = [1, 10, 15]
    for halo_number in halo_numbers:
        halo = halos[halo_number]

        # There should not be any extra families in the halo particles
        assert(all(fam in [pynbody.family.dm, pynbody.family.star, pynbody.family.gas] for fam in halo.families()))

        # Check we now have the same number of particles assigned to the halo
        # than its AHF header, family by family
        assert halo.properties['npart'] == len(halo)
        assert halo.properties['n_star'] == len(halo.st)
        assert halo.properties['n_gas'] == len(halo.g)
        ndm = halo.properties['npart'] - halo.properties['n_star'] - halo.properties['n_gas']
        assert ndm == len(halo.d)

        # Derive some masses to check that we are identifying the right particles, in addition to their right numbers
        dm_mass = halo.properties['Mhalo'] - halo.properties['M_star'] - halo.properties['M_gas']
        gas_mass = halo.properties['M_gas']

        rtol = 1e-2 # We are not precise to per cent level with unit conversion through the different steps
        hubble = f.properties['h']      # AHF internal units are Msol/h and need to be manually corrected, which has not been done on this test output
        npt.assert_allclose(dm_mass / hubble, halo.d['mass'].sum().in_units("Msol"), rtol=rtol)
        npt.assert_allclose(gas_mass / hubble, halo.g['mass'].sum().in_units("Msol"), rtol=rtol)
        npt.assert_allclose(halo.properties['Mhalo'] / hubble, halo['mass'].sum().in_units("Msol"), rtol=rtol)

def test_ahf_substructure():
    # TODO
    assert False