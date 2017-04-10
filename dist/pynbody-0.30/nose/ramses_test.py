import pynbody
import numpy as np


def setup():
    global f
    f = pynbody.load("testdata/ramses_partial_output_00250")


def test_lengths():
    assert len(f.gas) == 152574
    assert len(f.star) == 2655
    assert len(f.dm) == 51887


def close_enough(x, y):
    return abs(x - y) < 1.e-5


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
