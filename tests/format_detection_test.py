import builtins
import os
import pathlib
import warnings

import h5py
import numpy as np
import pytest

import pynbody
import pynbody.test_utils
from pynbody import config, halo
from pynbody.snapshot import SimSnap, gadgethdf, tipsy


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gasoline_ahf", "swift", "ramses", "gadget", "arepo",
                                                  "tng_subfind", "subfind", "hbt", "nchilada", "grafic",
                                                  "pkdgrav3", "gizmo", "lpicola")


def _snap_class(path):
    """Return the name of the class that pynbody.load would use for path, without loading it"""
    with gadgethdf._share_files_during_detection():
        for c in SimSnap.iter_subclasses_with_priority(config['snap-class-priority']):
            if c._can_load(pathlib.Path(path)):
                return c.__name__
    return None


def _halo_class(sim):
    """Return the name of the class that sim.halos() would use, without loading it"""
    for c in halo.HaloCatalogue.iter_subclasses_with_priority(config['halo-class-priority']):
        try:
            if c._can_load(sim):
                return c.__name__
        except TypeError:
            pass
    return None


@pytest.mark.parametrize("path, expected", [
    ("gasoline_ahf/g15784.lr.01024", "TipsySnap"),
    ("gasoline_ahf/g15784.lr.01024.gz", "TipsySnap"),
    ("SWIFT/snap_0150.hdf5", "SwiftSnap"),
    ("SWIFT/multifile_without_vds/snap_0000", "SwiftSnap"),
    ("SWIFT/multifile_with_vds/snap_0000.hdf5", "SwiftSnap"),
    ("ramses/output_00080", "RamsesSnap"),
    ("gadget2/test_g2_snap", "GadgetSnap"),
    ("gadget2/test_g2_snap.1", "GadgetSnap"),
    ("gadget1.snap", "GadgetSnap"),
    ("gizmo/snapshot_000.hdf5", "GizmoHDFSnap"),
    ("gadget3/data/subhalos_103/subhalo_103", "SubFindHDFSnap"),
    ("gadget3/snap_028_z000p000.0.hdf5", "EagleLikeHDFSnap"),
    ("gadget3/data/snapshot_103/snap_103.hdf5", "GadgetHDFSnap"),
    ("nchilada_test/12M.00001", "NchiladaSnap"),
    ("gadget4_subfind_HBT/snapshot_034.hdf5", "GadgetHDFSnap"),
    ("grafic_test/", "GrafICSnap"),
    ("arepo/cosmobox_015.hdf5", "ArepoHDFSnap"),
    ("arepo/tng/snapdir_261/snap_261", "ArepoHDFSnap"),
    ("subfind/snapshot_019", "GadgetSnap"),
    ("pkdgrav3/cosmoSF.00010", "PkdgravHDFSnap"),
    ("lpicola/lpicola_z0p000.0", "GadgetSnap"),
    ("nonexistent_file", None),
])
def test_snapshot_class_detection(path, expected):
    assert _snap_class("testdata/" + path) == expected


@pytest.mark.parametrize("path, expected", [
    ("gasoline_ahf/g15784.lr.01024", "AHFCatalogue"),
    ("SWIFT/snap_0150.hdf5", "HaloNumberCatalogue"),
    ("ramses/output_00080", "AdaptaHOPCatalogue"),
    ("gadget3/data/subhalos_103/subhalo_103", "SubFindHDFHaloCatalogue"),
    ("gadget4_subfind_HBT/snapshot_034.hdf5", "Gadget4SubfindHDFCatalogue"),
    ("arepo/cosmobox_015.hdf5", "ArepoSubfindHDFCatalogue"),
    ("arepo/tng/snapdir_261/snap_261", "TNGSubfindHDFCatalogue"),
    ("subfind/snapshot_019", "SubfindCatalogue"),
    ("gadget2/test_g2_snap", None),
])
def test_halo_class_detection(path, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = pynbody.load("testdata/" + path)
    assert _halo_class(f) == expected


@pytest.fixture
def op_counts(monkeypatch):
    counts = {"h5py.File": 0, "is_hdf5": 0, "open": 0, "listdir": 0}

    class CountingFile(h5py.File):
        def __init__(self, *args, **kwargs):
            counts["h5py.File"] += 1
            super().__init__(*args, **kwargs)

    def counting(name, fn):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return fn(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(h5py, "File", CountingFile)
    monkeypatch.setattr(h5py, "is_hdf5", counting("is_hdf5", h5py.is_hdf5))
    monkeypatch.setattr(builtins, "open", counting("open", builtins.open))
    monkeypatch.setattr(os, "scandir", counting("listdir", os.scandir))
    monkeypatch.setattr(os, "listdir", counting("listdir", os.listdir))
    return counts


def test_gadgethdf_detection_opens_hdf5_once(op_counts):
    assert _snap_class("testdata/gadget3/data/snapshot_103/snap_103.hdf5") == "GadgetHDFSnap"
    assert op_counts["h5py.File"] <= 1
    assert op_counts["is_hdf5"] <= 1


def test_tipsy_detection_io(op_counts):
    assert _snap_class("testdata/gasoline_ahf/g15784.lr.01024") == "TipsySnap"
    assert op_counts["open"] <= 1
    assert op_counts["listdir"] == 0


def test_tipsy_halo_detection_io(op_counts):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    op_counts["open"] = 0
    assert _halo_class(f) == "AHFCatalogue"
    # only the candidate grp and amiga.grp files should be inspected, not every auxiliary array
    assert op_counts["open"] <= 3


@pytest.mark.filterwarnings("ignore")
def test_tipsy_has_loadable_key(tmp_path):
    filename = str(tmp_path / "test.tipsy")
    f = pynbody.new(dm=10, gas=5, star=3)
    f['pos'] = np.random.uniform(size=(18, 3))
    f.write(fmt=tipsy.TipsySnap, filename=filename)

    f = pynbody.load(filename)
    f['grp'] = np.arange(18, dtype=np.int32)
    f['grp'].write()
    f.gas['gasonly'] = np.arange(5, dtype=np.float32)
    f.gas['gasonly'].write()
    with open(filename + ".badlength", "w") as fd:
        fd.write("3\n1\n2\n3\n")

    names = ["grp", "amiga.grp", "pos", "mass", "tform", "gasonly", "badlength", "nonexistent"]

    f = pynbody.load(filename)
    cheap = {n: f._has_loadable_key(n) for n in names}
    assert len(f._loadable_keys_registry) == 0  # cheap calls must not build the registry
    assert _halo_class(f) == "HaloNumberCatalogue"
    assert len(f._loadable_keys_registry) == 0

    loadable = f.loadable_keys()
    assert cheap == {n: n in loadable for n in names}
    assert cheap["grp"] and not cheap["gasonly"] and not cheap["badlength"]

    # once the registry exists, the answers come from it
    assert {n: f._has_loadable_key(n) for n in names} == cheap


@pytest.mark.filterwarnings("ignore")
def test_hbt_halos_load_twice():
    # guards against HDF5 handles opened during detection being left open with inconsistent locking flags
    f = pynbody.load("testdata/gadget4_subfind_HBT/snapshot_034.hdf5")
    f.halos(priority=["HBTPlusCatalogue"])
    f.halos(priority=["HBTPlusCatalogue"])
