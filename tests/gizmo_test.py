import os

import numpy as np
import pytest

import pynbody
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_gizmo_data():
    pynbody.test_utils.ensure_test_data_available("gizmo")

@pytest.mark.filterwarnings("ignore:No unit information", "ignore:Assuming default value")
def test_load_gizmo_file():
    f = pynbody.load("testdata/gizmo/snapshot_000.hdf5")
    assert isinstance(f, pynbody.snapshot.gadgethdf.GadgetHDFSnap)
    assert len(f) == 12432742
    assert len(f.dm) == 7840638
    assert len(f.gas) == 4592104
    assert np.allclose(f.properties['z'], 99.0)

@pytest.mark.filterwarnings("ignore:No unit information", "ignore:Assuming default value")
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

@pytest.mark.filterwarnings("ignore:No unit information", "ignore:Assuming default value", "ignore:Unable to infer units")
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
        assert f['pos'].units == "Mpc a h^-1"
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

