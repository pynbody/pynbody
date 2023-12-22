import numpy as np
import pynbody
import pynbody.snapshot.swift

def test_load_swift():
    f = pynbody.snapshot.swift.SwiftSnap("testdata/swiftsnap.hdf5")

def test_load_identifies_swift():
    f = pynbody.load("testdata/swiftsnap.hdf5")
    assert isinstance(f, pynbody.snapshot.swift.SwiftSnap)

def test_swift_properties():
    f = pynbody.load("testdata/swiftsnap.hdf5")

    assert f.properties['a'] == 1.0
    assert f.properties['z'] == 0.0
    assert np.allclose(f.properties['h'], 0.681)
    assert np.allclose(f.properties['boxsize'].in_units("Mpc a"), 200.)
    assert np.allclose(f.properties['OmegaM0'], 0.304611)
    assert np.allclose(f.properties['OmegaL0'], 0.693922)
    assert np.allclose(f.properties['OmegaNu0'], 0.0013891)

def test_swift_arrays():
    f = pynbody.load("testdata/swiftsnap.hdf5")
    print(f['pos'], f['pos'].units)
    assert False # TODO: implement test

