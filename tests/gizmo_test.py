import os

import numpy as np
import pytest

import pynbody
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_gizmo_data():
    pynbody.test_utils.ensure_test_data_available("gizmo")

@pytest.mark.filterwarnings("ignore:Unable to find cosmological factors")
def test_load_gizmo_file():
    f = pynbody.load("testdata/gizmo/snapshot_000.hdf5")
    assert isinstance(f, pynbody.snapshot.gadgethdf.GadgetHDFSnap)
    assert len(f) == 12432742
    assert len(f.dm) == 7840638
    assert len(f.gas) == 4592104
    assert np.allclose(f.properties['z'], 99.0)

@pytest.mark.filterwarnings("ignore:Unable to find cosmological factors")
def test_issue_943():
    # Need to create a test snapshot_000.hdf5.units file that 
    # violates the conventions
    unitfile_name = "testdata/gizmo/snapshot_000.hdf5.units"
    with open(unitfile_name,"w") as unitfile:
        unitfile.write("pos; kpc a h^-1\n") # deliberate typo
    
    try:
        # Load the gizmo file
        f = pynbody.load("testdata/gizmo/snapshot_000.hdf5")
        # ensure the error would actually occur
        with pytest.raises(TypeError):
            f.filename + ".units" 
        # Note: Gizmo snapshots do not call _override_units_system by default
        # so we call it manually
        # This should produce an OSError because the file is formatted incorrectly
        # not a TypeError because it's trying to concatenate a PosixPath and a str
        with pytest.raises(OSError):
            f._override_units_system()

    finally:
        # remove unitfile
        os.remove(unitfile_name)

@pytest.mark.filterwarnings("ignore:Unable to find cosmological factors")
def test_units():
    f = pynbody.load("testdata/gizmo/snapshot_000.hdf5")
    assert np.allclose(f['pos'].units.in_units("4.5377647058823524e21 cm a"), 1.0)
    assert f['vel'].units == "km a^1/2 s^-1"
    assert f['mass'].units == "2.925e43 g"

@pytest.mark.filterwarnings("ignore:Unable to find cosmological factors")
def test_allow_blank_lines():
    # Need to create a test snapshot_000.hdf5.units file that 
    # violates the previous conventions
    unitfile_name = "testdata/gizmo/snapshot_000.hdf5.units"
    with open(unitfile_name,"w") as unitfile:
        unitfile.write("# This is a comment\n") 
        unitfile.write("\n \t") # empty line with whitespace wasn't allowed
        unitfile.write("pos: kpc a h^-1\n")

    try:
        # Load the gizmo file
        f = pynbody.load("testdata/gizmo/snapshot_000.hdf5")
        # this should not raise an error due to whitespace
        f._override_units_system()
        assert f['pos'].units == "kpc a h^-1"
        
        # Now add a deliberate typo to check it still errors on incorrect formats
        with open(unitfile_name,"a") as unitfile:
            unitfile.write("\n")
            unitfile.write("vel; km s^-1\n") # deliberate typo

        f = pynbody.load("testdata/gizmo/snapshot_000.hdf5")
        with pytest.raises(OSError):
            f._override_units_system()
    finally:
        # remove unitfile
        os.remove(unitfile_name)
        
@pytest.fixture(scope='module', autouse=True)
def get_fire_data():
    pynbody.test_utils.ensure_test_data_available("tiny_FIRE")
    
@pytest.mark.filterwarnings("ignore:Unable to find cosmological factors",
                            "ignore:No unit information found either")
def test_load_file_fire():
    f = pynbody.load("testdata/tiny_FIRE/output/m11i_res7100_truncated_1000.hdf5")
    assert isinstance(f, pynbody.snapshot.gadgethdf.GadgetHDFSnap)
    assert len(f) == 4000
    assert len(f.dm) == 2000
    assert len(f.gas) == 1000
    assert np.allclose(f.properties['z'], 0)
    
@pytest.mark.filterwarnings("ignore:Unable to find cosmological factors",
                            "ignore:No unit information found either")
def test_units_fire():
    f = pynbody.load("testdata/tiny_FIRE/output/m11i_res7100_truncated_1000.hdf5")
    assert np.allclose(f['pos'].units.in_units("3.085678e21 cm a h^-1"), 1.0)
    assert f['vel'].units == "km a^1/2 s^-1"
    assert np.allclose(f['mass'].units.in_units("1.989e+43 g h^-1"), 1.0)
    
    
@pytest.mark.filterwarnings("ignore:Unable to find cosmological factors",
                            "ignore:No unit information found either")
def test_units_fire_specify_param_file_name():
    os.rename('testdata/tiny_FIRE/gizmo_parameters.txt', 'testdata/tiny_FIRE/foo.txt')
    try:
        f = pynbody.load("testdata/tiny_FIRE/output/m11i_res7100_truncated_1000.hdf5", param_filename = 'testdata/tiny_FIRE/foo.txt')
    finally:
        os.rename('testdata/tiny_FIRE/foo.txt', 'testdata/tiny_FIRE/gizmo_parameters.txt')
    assert np.allclose(f['pos'].units.in_units("3.085678e21 cm a h^-1"), 1.0)
    assert f['vel'].units == "km a^1/2 s^-1"
    assert np.allclose(f['mass'].units.in_units("1.989e+43 g h^-1"), 1.0)
    
@pytest.mark.filterwarnings("ignore:Unable to find cosmological factors",
                            "ignore:No unit information found either")
def test_units_fire_no_param_file():
    os.rename('testdata/tiny_FIRE/gizmo_parameters.txt', 'testdata/tiny_FIRE/foo.txt')
    f = pynbody.load("testdata/tiny_FIRE/output/m11i_res7100_truncated_1000.hdf5")
    os.rename('testdata/tiny_FIRE/foo.txt', 'testdata/tiny_FIRE/gizmo_parameters.txt')
    assert f['pos'].units == "kpc a h^-1"
    assert f['vel'].units == "km a^1/2 s^-1"
    assert f['mass'].units == "1e10 Msol h^-1"

@pytest.mark.filterwarnings("ignore:Unable to find cosmological factors",
                            "ignore:No unit information found either")
def test_derived_arrays():
    f = pynbody.load("testdata/tiny_FIRE/output/m11i_res7100_truncated_1000.hdf5")
    new_derived_arrays = ['metals_list', 'H', 'He', 'C', 'N', 'O',
                          'Mg', 'Si', 'S', 'Ca', 'Fe', 'metals', 'rprocess']
                          
    for arr in new_derived_arrays:
        f.star[arr]
        f.gas[arr]
