import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gadget", "gasoline_ahf", "lpicola")

@pytest.fixture
def snap():
    return pynbody.load("testdata/gadget2/test_g2_snap")

def test_construct(snap):
    """Check the basic properties of the snapshot"""
    assert np.size(snap._files) == 2
    assert snap.header.num_files == 2
    assert snap.filename == "testdata/gadget2/test_g2_snap" or snap.filename == r"testdata\gadget2\test_g2_snap"
    assert snap._num_particles == 8192
    for f in snap._files:
        assert f.format2
        assert f.endian == "="

def test_properties(snap):
    assert "time" in snap.properties


def test_loadable(snap):
    """Check we have found all the blocks that should be in the snapshot"""
    blocks = snap.loadable_keys()
    expected_gas = ['nhp', 'smooth', 'nhe', 'u', 'sfr', 'pos',
                    'vel', 'iord', 'mass', 'nh', 'rho', 'nheq', 'nhep']
    expected_all = ['pos', 'vel', 'iord', 'mass']

    # Check that they have the right families
    assert(set(snap.gas.loadable_keys()) == set(expected_gas))
    assert(set(snap.dm.loadable_keys()) == set(expected_all))
    assert(set(snap.star.loadable_keys()) == set(expected_all))
    assert(set(snap.loadable_keys()) == set(expected_all))
    assert(snap.neutrino.loadable_keys() == [])


def test_standard_arrays(snap):
    """Check we can actually load some of these arrays"""
    snap.dm['pos']
    snap.gas['pos']
    snap.star['pos']
    snap['pos']
    snap['mass']
    # Load a second time to check that family_arrays still work
    snap.dm['pos']
    snap['vel']
    snap['iord']
    snap.gas['rho']
    snap.gas['u']
    snap.star['mass']


def test_array_sizes(snap):
    """Check we have the right sizes for the arrays"""
    assert(np.shape(snap.dm['pos']) == (4096, 3))
    assert(np.shape(snap['vel']) == (8192, 3))
    assert(np.shape(snap.gas['rho']) == (4039,))
    assert(snap.gas['u'].dtype == np.float32)
    assert(snap.gas['iord'].dtype == np.int32)


def test_fam_sim():
    """Check that an array loaded as families is the same as one loaded as a simulation array"""
    snap2 = pynbody.load("testdata/gadget2/test_g2_snap")
    snap3 = pynbody.load("testdata/gadget2/test_g2_snap")
    snap3.gas["pos"]
    snap3.dm["pos"]
    snap3.star["pos"]
    assert((snap3["pos"] == snap2["pos"]).all())


def test_array_contents(snap):
    """Check some array elements"""
    assert(np.max(snap["iord"]) == 8192)
    assert(np.min(snap["iord"]) == 1)
    assert(np.mean(snap["iord"]) == 4096.5)

    # 10/11/13 - AP - suspect the following tests are incorrect
    # because ordering of file did not agree with pynbody ordering

    assert(abs(np.mean(snap["pos"]) - 1434.664) < 0.004)
    assert(abs(snap["pos"][52][1] - 456.69678) < 0.001)
    assert(abs(snap.gas["u"][100] - 438.39496) < 0.001)
    assert(abs(snap.dm["mass"][5] - 0.04061608) < 0.001)


def test_header(snap):
    """Check some header properties"""
    assert(abs(snap.header.BoxSize - 3000.0) < 0.001)
    assert(abs(snap.header.HubbleParam - 0.710) < 0.001)
    assert(abs(snap.header.Omega0 - 0.2669) < 0.001)
    assert(snap.header.flag_cooling == 1)
    assert(snap.header.flag_metals == 0)


def test_g1_load():
    """Check we can load gadget-1 files also"""
    with pytest.warns(RuntimeWarning, match=r"Run out of block names in the config file. Using fallbacks: UNK\*"):
        snap2 = pynbody.load("testdata/gadget1.snap")


def test_write(snap):
    """Check that we can write a new snapshot and read it again,
    and the written and the read are the same."""

    # note that only loaded blocks are written. This is by design, but possibly a flawed
    # design. It was uncovered when improving the test structure in June 2024, and since it is
    # a long-standing "feature" we will leave it as is for now. As a result, we need to trigger
    # a bunch of loads to make sure the written file has the right fields
    for x in snap.loadable_keys():
        _ = snap[x]
    for x in snap.gas.loadable_keys():
        _ = snap.gas[x]

    snap.write(filename='testdata/test_gadget_write')

    snap3 = pynbody.load('testdata/test_gadget_write')
    assert set(snap.loadable_keys()) == set(snap3.loadable_keys())
    npt.assert_equal(snap3["pos"].view(np.ndarray), snap["pos"])
    npt.assert_equal(snap3.gas["rho"].view(np.ndarray), snap.gas["rho"])
    assert snap3.check_headers(snap.header, snap3.header)


def test_conversion():
    """Check that we can convert a file from tipsy format and load it again"""
    snap4 = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    snap4.write(fmt=pynbody.snapshot.gadget.GadgetSnap,
                filename="testdata/test_conversion.gadget")
    snap5 = pynbody.load("testdata/test_conversion.gadget")


def test_write_single_array(snap):
    """Check that we can write a single array and read it back"""
    snap["pos"].write(overwrite=True)
    snap6 = pynbody.load("testdata/gadget2/test_g2_snap")
    assert((snap6["pos"] == snap["pos"]).all())


def test_no_mass_block():
    f = pynbody.load("testdata/gadget_no_mass")
    f['mass']  # should succeed


def test_unit_persistence():
    f = pynbody.load("testdata/gadget2/test_g2_snap")

    # f2 is the comparison case - just load the whole
    # position array and convert it, simple
    f2 = pynbody.load("testdata/gadget2/test_g2_snap")
    f2['pos']
    f2.physical_units()

    f.gas['pos']
    f.physical_units()
    assert (f.gas['pos'] == f2.gas['pos']).all()

    # the following lazy-loads should lead to the data being
    # auto-converted
    f.dm['pos']
    assert (f.gas['pos'] == f2.gas['pos']).all()
    assert (f.dm['pos'] == f2.dm['pos']).all()

    # the final one is the tricky one because this will trigger
    # an array promotion and hence internally inconsistent units
    f.star['pos']

    assert (f.star['pos'] == f2.star['pos']).all()

    # also check it hasn't messed up the other bits of the array!
    assert (f.gas['pos'] == f2.gas['pos']).all()
    assert (f.dm['pos'] == f2.dm['pos']).all()

    assert (f['pos'] == f2['pos']).all()


def test_per_particle_loading():
    """Tests that loading one family at a time results in the
    same final array as loading all at once. There are a number of
    subtelties in the gadget handler that could mess this up by loading
    the wrong data."""

    f_all = pynbody.load("testdata/gadget2/test_g2_snap")
    f_part = pynbody.load("testdata/gadget2/test_g2_snap")

    f_part.dm['pos']
    f_part.star['pos']
    f_part.gas['pos']

    assert (f_all['pos'] == f_part['pos']).all()

def test_issue321():
    """L-PICOLA outputs single-precision with no mass block, which causes problems
    with testing kd-trees"""

    f = pynbody.load("testdata/lpicola/lpicola_z0p000.0")
    assert f['pos'].dtype==np.dtype('float32')
    assert f['mass'].dtype==np.dtype('float32')


def test_units_override():
    f = pynbody.load("testdata/gadget2/test_g2_snap.0")
    assert f.filename == "testdata/gadget2/test_g2_snap" or f.filename == r"testdata\gadget2\test_g2_snap"
    assert f['pos'].units == "kpc a h^-1"

    # In this case the unit override system is not effective because
    # the final ".1" is not stripped away in the filename:
    # the file `gadget2/test_g2_snap.units` is not used
    f_no_unit_override = pynbody.load("testdata/gadget2/test_g2_snap.1")
    assert f_no_unit_override.filename == "testdata/gadget2/test_g2_snap.1" or f_no_unit_override.filename == r"testdata\gadget2\test_g2_snap.1"
    assert f_no_unit_override['pos'].units == "Mpc a h^-1"  # from default_config.ini


def test_ignore_cosmology():
    f = pynbody.load("testdata/gadget2/test_g2_snap.1")
    f.physical_units()
    np.testing.assert_allclose(f.properties['time'].in_units('Gyr'), 2.57689526)
    f_no_cosmo = pynbody.load("testdata/gadget2/test_g2_snap.1", ignore_cosmo=True)
    np.testing.assert_allclose(f_no_cosmo.properties['time'].in_units('Gyr'), 271.608952)
