import glob
import os.path
import pathlib
import shutil
import stat
import subprocess
import warnings

import numpy as np
import numpy.testing as npt
import pytest

import pynbody


@pytest.fixture
def cleanup_fpos_file():
    if os.path.exists("testdata/gasoline_ahf/g15784.lr.01024.AHF_fpos"):
        os.remove("testdata/gasoline_ahf/g15784.lr.01024.AHF_fpos")
    yield


def test_load_ahf_catalogue():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    h = pynbody.halo.ahf.AHFCatalogue(f)
    assert len(h)==1411

h0_sample_iords = [57, 27875, 54094, 82969, 112002, 140143, 173567, 205840, 264606,
           301694, 333383, 358730, 374767, 402300, 430180, 456015, 479885, 496606,
           519824, 539971, 555195, 575204, 596047, 617669, 652724, 1533992, 1544021,
           1554045, 1564080, 1574107, 1584130, 1594158, 1604204, 1614257, 1624308, 1634376,
           1644485, 1654580, 1664698, 1674831, 1685054, 1695252, 1705513, 1715722, 1725900,
           1736070, 1746235, 1756400, 1766584, 1776754, 1786886]

@pytest.mark.parametrize("do_load_all", [True, False])
def test_ahf_particles(do_load_all, cleanup_fpos_file):
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    h = pynbody.halo.ahf.AHFCatalogue(f)

    if do_load_all:
        h.load_all()

    assert len(h[0])==502300
    assert h[0].ancestor is f
    assert (h[0]['iord'][::10000]==h0_sample_iords).all()
    assert len(h[19])==3272
    assert(h[19]['iord'][::1000] == [232964, 341019, 752354, 793468]).all()

def test_load_copy():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    h = pynbody.halo.ahf.AHFCatalogue(f)
    hcopy = h.load_copy(0)
    assert (hcopy['iord'][::10000] == h0_sample_iords).all()
    assert hcopy.ancestor is not f

@pytest.mark.parametrize("do_load_all", [True, False])
def test_load_ahf_catalogue_non_gzipped(do_load_all):
    for extension in ["halos", "particles", "substructure"]:
        subprocess.call(["gunzip",f"testdata/gasoline_ahf/g15784.lr.01024.z0.000.AHF_{extension}.gz"])
    try:
        f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
        h = pynbody.halo.ahf.AHFCatalogue(f)
        if do_load_all:
            h.load_all()
        assert len(h)==1411
    finally:
        for extension in ["halos", "particles", "substructure"]:
            subprocess.call(["gzip", f"testdata/gasoline_ahf/g15784.lr.01024.z0.000.AHF_{extension}"])


def test_ahf_properties():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    h = pynbody.halo.ahf.AHFCatalogue(f)
    assert np.allclose(h[0].properties['Mvir'], 1.69639e+12)
    assert np.allclose(h[1].properties['Ekin'],6.4911e+17)
    assert np.allclose(h[1].properties['Mvir'], 1.19684e+13)

def test_ahf_all_properties():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    h = pynbody.halo.ahf.AHFCatalogue(f)

    # nb AHF reader currently doesn't infer units anyway, so with_units will make no difference
    properties = h.get_properties_all_halos(with_units=False)
    assert np.allclose(properties['Mvir'][0], 1.69639e+12)
    assert np.allclose(properties['Ekin'][1],6.4911e+17)
    assert np.allclose(properties['Mvir'][1], 1.19684e+13)

@pytest.fixture
def snap_in_unwritable_folder():
    if os.path.exists("testdata/test_unwritable"):
        os.chmod("testdata/test_unwritable", stat.S_IRUSR | stat.S_IXUSR | stat.S_IWUSR)
        shutil.rmtree("testdata/test_unwritable")
    os.mkdir("testdata/test_unwritable/")
    for fname in glob.glob("testdata/gasoline_ahf/*"):
        if "AHF_fpos" not in fname:
            os.symlink("../"+fname[9:], "testdata/test_unwritable/"+fname[22:])
    os.chmod("testdata/test_unwritable", stat.S_IRUSR | stat.S_IXUSR)

    yield "testdata/test_unwritable/g15784.lr.01024"

    os.chmod("testdata/test_unwritable", stat.S_IRUSR | stat.S_IXUSR | stat.S_IWUSR)
    shutil.rmtree("testdata/test_unwritable")



def test_ahf_unwritable(snap_in_unwritable_folder):
    f = pynbody.load(snap_in_unwritable_folder)

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
        _halos = pynbody.halo.ahf.AHFCatalogue(f)


def test_ramses_ahf_family_mapping_with_new_format():
    # Test Issue 691 where family mapping of AHF catalogues with Ramses new particle formats would go wrong
    f = pynbody.load("testdata/ramses_new_format_cosmo_with_ahf_output_00110")
    halos = pynbody.halo.ahf.AHFCatalogue(f)

    assert len(halos) == 149    # 150 lines in AHF halos file

    # Load halos and check that stars, DM and gas are correctly mapped by pynbody
    # Halo 0 is the main halo and has all three families, while other are random picks
    halo_numbers = [0, 9, 14]
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

def test_ahf_corrupt_substructure():
    """For some reason lost in history, the AHF substructure file for gasoline_ahf/g15784 is corrupt.
    Use that to test we can see the exception"""
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    with pytest.raises(KeyError):
        _ = pynbody.halo.ahf.AHFCatalogue(f, ignore_missing_substructure=False)


def test_ahf_substructure():
    f = pynbody.load("testdata/ramses_new_format_cosmo_with_ahf_output_00110")
    halos = pynbody.halo.ahf.AHFCatalogue(f)
    halos.load_all()

    check_parents = [0,1]
    check_children = [[72, 74, 138, 89], [115, 124,  20,  96]]

    for parent, children in zip(check_parents, check_children):
        halo = halos[parent]
        assert len(halo.properties['children']) == len(children)
        assert (halo.properties['children'] == children).all()
        assert halo.properties['hostHalo'] == -1
        #assert 'parent' not in halo.properties.keys()
        for child in children:
            assert halos[child].properties['parent'] == parent # from substructure file
            assert halos[child].properties['hostHalo'] == parent # from halos file

        assert len(halo.subhalos) == len(halo.properties['children'])
        for child_halo, child_number in zip(halo.subhalos, children):
            assert (child_halo.dm['iord'] == halos[child_number].dm['iord']).all()

@pytest.fixture
def snap_with_non_sequential_halos():
    base_folder = pathlib.Path("testdata/ahf_with_non_sequential_ids").absolute()
    if not base_folder.exists():
        base_folder.mkdir()
        for fname in base_folder.parent.glob("gasoline_ahf/*"):
            if "AHF" not in fname.name:
                (base_folder/fname.name).symlink_to(fname.absolute())

        # copy AHF_halos line by line, incorporating random IDs
        np.random.seed(0)
        my_random_ids = np.random.choice(1000000, 1411, replace=False)
        my_file_order = np.arange(1411)

        # shuffle the file ordering too
        np.random.shuffle(my_file_order)

        # my_file_order[:20] = np.arange(19,-1,-1) # reverse order of first 20

        with open(base_folder/"g15784.lr.01024.z0.000.AHF_halos", "w") as f:
            with pynbody.util.open_("testdata/gasoline_ahf/g15784.lr.01024.z0.000.AHF_halos", "rt") as f2:
                header = f2.readline()
                f.write("#ID(0) "+header)

                remaining_lines = f2.readlines()
                remaining_lines = [remaining_lines[i] for i in my_file_order]
                for line, id_ in zip(remaining_lines, my_random_ids[my_file_order]):
                    f.write(f"{id_} {line}")

        particle_file_per_halo = []
        with pynbody.util.open_("testdata/gasoline_ahf/g15784.lr.01024.z0.000.AHF_particles", "rt") as f2:
            assert f2.readline().strip() == "1411"
            for i in range(1411):
                lines = [f2.readline()]
                n_to_read = int(lines[0])
                for _ in range(n_to_read):
                    lines.append(f2.readline())
                particle_file_per_halo.append(lines)

        with open(base_folder/"g15784.lr.01024.z0.000.AHF_particles", "w") as f:
            f.write("1411\r\n")
            for o in my_file_order:
                f.writelines(particle_file_per_halo[o])

    return pynbody.load("testdata/ahf_with_non_sequential_ids/g15784.lr.01024")

@pytest.mark.parametrize("halo_numbering_mode, halo_ids",
                         [('ahf', (157105, 608171)),
                          ('v1', (761, 944)),
                          ('file-order', (760, 943)),
                          ('length-order-v1', (1,20)),
                          ('length-order', (0, 19))])
def test_ahf_non_sequential_ids(snap_with_non_sequential_halos,
                                halo_numbering_mode,
                                halo_ids):
    f = snap_with_non_sequential_halos
    h = pynbody.halo.ahf.AHFCatalogue(f, halo_numbers=halo_numbering_mode)

    assert len(h)==1411
    assert len(h[halo_ids[0]])==502300

    assert halo_ids[0] in h

    assert (h[halo_ids[0]]['iord'][::10000]==np.array([57, 27875, 54094, 82969, 112002, 140143, 173567, 205840, 264606,
           301694, 333383, 358730, 374767, 402300, 430180, 456015, 479885, 496606,
           519824, 539971, 555195, 575204, 596047, 617669, 652724, 1533992, 1544021,
           1554045, 1564080, 1574107, 1584130, 1594158, 1604204, 1614257, 1624308, 1634376,
           1644485, 1654580, 1664698, 1674831, 1685054, 1695252, 1705513, 1715722, 1725900,
           1736070, 1746235, 1756400, 1766584, 1776754, 1786886], dtype=np.int32)).all()

    assert len(h[halo_ids[1]])==3272
    assert(h[halo_ids[1]]['iord'][::1000] == [232964, 341019, 752354, 793468]).all()

def test_ahf_dosort_kwarg(snap_with_non_sequential_halos):
    # checks for compatibility with v1 behaviour
    f = snap_with_non_sequential_halos
    with pytest.warns(DeprecationWarning):
        h = pynbody.halo.ahf.AHFCatalogue(f, dosort=True)
    assert len(h)==1411
    assert len(h[1])==502300
    assert len(h[20])==3272
