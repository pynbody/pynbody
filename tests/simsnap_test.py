"""Note that most simsnap tests are in more specific test files."""

import pathlib

import numpy as np
import pytest

import pynbody


def test_load_empty_folder():

    pathlib.Path("testdata").joinpath('empty_folder_0001').mkdir(exist_ok=True)

    with pytest.raises(IOError) as excinfo:
        f = pynbody.load("testdata/empty_folder_0001")
    assert "Is a directory" in str(excinfo.value)

@pytest.mark.parametrize("family_array", [False, True], ids=['all', 'family'])
@pytest.mark.parametrize("dimensionality", [1, 3], ids=['1D', '3D'])
def test_change_dtype(family_array, dimensionality):
    f = pynbody.new(dm=50, gas=50)
    if family_array:
        f = f.dm
    if dimensionality == 1:
        f['testarray'] = np.arange(len(f), dtype=np.float64)
    else:
        f['testarray'] = np.arange(3*len(f), dtype=np.float64).reshape(len(f), 3)
    assert f['testarray'].dtype == np.float64
    f['testarray'].units="Msol"
    f.set_array_dtype('testarray', np.float32)
    assert f['testarray'].dtype == np.float32
    assert f['testarray'].units == "Msol"
    if dimensionality == 1:
        assert np.allclose(f['testarray'], np.arange(len(f), dtype=np.float32))
    else:
        assert np.allclose(f['testarray'], np.arange(3*len(f), dtype=np.float32).reshape(len(f), 3))


def test_change_dtype_nonexistent_array():
    f = pynbody.new(dm=50, gas=50)
    with pytest.raises(KeyError) as excinfo:
        f.set_array_dtype('nonexistent_array', np.float32)
    assert "nonexistent_array" in str(excinfo.value)
