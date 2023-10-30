import glob
import os.path
import shutil
import stat
import subprocess

import numpy as np

import pynbody


def test_load_ahf_catalogue():
    f = pynbody.load("testdata/g15784.lr.01024")
    h = pynbody.halo.AHFCatalogue(f)
    assert len(h)==1411

def test_load_ahf_catalogue_non_gzipped():
    subprocess.call(["gunzip","testdata/g15784.lr.01024.z0.000.AHF_halos.gz"])
    try:
        f = pynbody.load("testdata/g15784.lr.01024")
        h = pynbody.halo.AHFCatalogue(f)
        assert len(h)==1411
    finally:
        subprocess.call(["gzip","testdata/g15784.lr.01024.z0.000.AHF_halos"])

def test_ahf_properties():
    f = pynbody.load("testdata/g15784.lr.01024")
    h = pynbody.halo.AHFCatalogue(f)
    assert h[1].properties['children']==[]
    assert h[1].properties['fstart']==23


def _setup_unwritable_ahf_situation():
    if os.path.exists("testdata/test_unwritable"):
        os.chmod("testdata/test_unwritable", stat.S_IRUSR | stat.S_IXUSR | stat.S_IWUSR)
        shutil.rmtree("testdata/test_unwritable")
    os.mkdir("testdata/test_unwritable/")
    for fname in glob.glob("testdata/g15784*"):
        if "AHF_fpos" not in fname:
            os.symlink("../"+fname[9:], "testdata/test_unwritable/"+fname[9:])
    os.chmod("testdata/test_unwritable", stat.S_IRUSR | stat.S_IXUSR)


def test_ahf_unwritable():
    _setup_unwritable_ahf_situation()
    f = pynbody.load("testdata/test_unwritable/g15784.lr.01024")
    h = f.halos()
    assert len(h)==1411


def test_detecting_ahf_catalogues_with_without_trailing_slash():
    # Test small fixes in #688 to detect AHF catalogues with and wihtout trailing slashes in directories
    for name in ["testdata/ramses_new_format_cosmo_with_ahf_output_00110", "testdata/ramses_new_format_cosmo_with_ahf_output_00110/"]:
        f = pynbody.load(name)
        halos = pynbody.halo.AHFCatalogue(f)


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
        assert(halo.properties['npart'] == len(halo))
        assert(halo.properties['n_star'] == len(halo.st))
        assert(halo.properties['n_gas'] == len(halo.g))
        ndm = halo.properties['npart'] - halo.properties['n_star'] - halo.properties['n_gas']
        assert(ndm == len(halo.d))

        # Derive some masses to check that we are identifying the right particles, in addition to their right numbers
        dm_mass = halo.properties['Mhalo'] - halo.properties['M_star'] - halo.properties['M_gas']
        gas_mass = halo.properties['M_gas']

        import numpy.testing as npt
        rtol = 1e-2 # We are not precise to per cent level with unit conversion through the different steps
        hubble = f.properties['h']      # AHF internal units are Msol/h and need to be manually corrected, which has not been done on this test output
        npt.assert_allclose(dm_mass / hubble, halo.d['mass'].sum().in_units("Msol"), rtol=rtol)
        npt.assert_allclose(gas_mass / hubble, halo.g['mass'].sum().in_units("Msol"), rtol=rtol)
        npt.assert_allclose(halo.properties['Mhalo'] / hubble, halo['mass'].sum().in_units("Msol"), rtol=rtol)
