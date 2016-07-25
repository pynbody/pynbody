import pynbody
import numpy as np


def setup():
    global f
    f = pynbody.load("testdata/ramses_partial_output_00250")


def test_lengths():
    assert len(f.gas) == 152667
    assert len(f.star) == 2655
    assert len(f.dm) == 51887


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

