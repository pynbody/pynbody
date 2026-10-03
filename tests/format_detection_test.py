import builtins
import gc
import gzip
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
    c = pynbody.snapshot.identify(path)
    return None if c is None else c.__name__


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
    gadgethdf._cached_hdf5_inspection.clear()
    assert _snap_class("testdata/gadget3/data/snapshot_103/snap_103.hdf5") == "GadgetHDFSnap"
    assert op_counts["h5py.File"] <= 1
    assert op_counts["is_hdf5"] <= 1


def test_tipsy_detection_io(op_counts):
    assert _snap_class("testdata/gasoline_ahf/g15784.lr.01024") == "TipsySnap"
    # this test file exists only gzipped, so GadgetSnap (which reads the first 4 bytes of an uncompressed file)
    # does not open it, and only TipsySnap's own header read remains
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
    # an array whose name on disk is translated by tipsy-name-mapping (tempEff on disk is Tinc in pynbody)
    with open(filename + ".tempEff", "w") as fd:
        fd.write("18\n" + "\n".join(str(i) for i in range(18)) + "\n")
    # a gzipped auxiliary array
    with gzip.open(filename + ".zipped.gz", "wt") as fd:
        fd.write("18\n" + "\n".join(str(i) for i in range(18)) + "\n")

    names = ["grp", "amiga.grp", "pos", "mass", "tform", "gasonly", "badlength", "nonexistent",
             "Tinc", "tempEff", "zipped"]

    f = pynbody.load(filename)
    cheap = {n: f._has_loadable_key(n) for n in names}
    assert len(f._loadable_keys_registry) == 0  # cheap calls must not build the registry
    assert _halo_class(f) == "HaloNumberCatalogue"
    assert len(f._loadable_keys_registry) == 0

    loadable = f.loadable_keys()
    assert cheap == {n: n in loadable for n in names}
    assert cheap["grp"] and cheap["Tinc"] and cheap["zipped"]
    assert not cheap["gasonly"] and not cheap["badlength"]

    # once the registry exists, the answers come from it
    assert {n: f._has_loadable_key(n) for n in names} == cheap


@pytest.mark.filterwarnings("ignore")
def test_hbt_halos_load_twice():
    # guards against HDF5 handles opened during detection being left open with inconsistent locking flags
    f = pynbody.load("testdata/gadget4_subfind_HBT/snapshot_034.hdf5")
    f.halos(priority=["HBTPlusCatalogue"])
    f.halos(priority=["HBTPlusCatalogue"])


def test_detection_closes_shared_hdf5_files(monkeypatch):
    opened = []

    class RecordingFile(h5py.File):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

    monkeypatch.setattr(h5py, "File", RecordingFile)
    gadgethdf._cached_hdf5_inspection.clear()
    assert _snap_class("testdata/gadget3/data/snapshot_103/snap_103.hdf5") == "GadgetHDFSnap"
    assert _snap_class("testdata/gasoline_ahf/g15784.lr.01024") == "TipsySnap"
    assert len(opened) > 0
    assert not any(f.id.valid for f in opened)


def test_other_classes_can_open_hdf5_with_different_locking(tmp_path):
    """A class outside the GadgetHDFSnap family may open an HDF5 file with locking=False (as e.g. HBT+ catalogues
    do); HDF5 refuses this if the file is still open from GadgetHDFSnap's checks, so they must not leave it open."""
    filename = tmp_path / "unlocked_format.hdf5"
    with h5py.File(filename, "w") as f:
        f.create_group("MyFormatHeader")

    class UnlockedHDF5Snap(SimSnap):
        @classmethod
        def _can_load(cls, f):
            if f.name != "unlocked_format.hdf5":
                return False
            with h5py.File(f, "r", locking=False) as h5:
                return "MyFormatHeader" in h5

    try:
        assert pynbody.snapshot.identify(filename) is UnlockedHDF5Snap
    finally:
        del UnlockedHDF5Snap
        gc.collect()


def test_unidentified_file_errors(tmp_path):
    with pytest.raises(OSError, match="path does not exist"):
        pynbody.load(tmp_path / "nonexistent")
    with pytest.raises(OSError, match="path is a directory"):
        pynbody.load(tmp_path)

    # a file too short to hold even GadgetSnap's 4-byte header used to raise struct.error
    (tmp_path / "short").write_bytes(b"ab")
    with pytest.raises(OSError, match="not recognised"):
        pynbody.load(tmp_path / "short")

    with h5py.File(tmp_path / "other.hdf5", "w") as f:
        f.create_group("SomethingElse")
    with pytest.raises(OSError, match="HDF5 with top-level entries: SomethingElse"):
        pynbody.load(tmp_path / "other.hdf5")


def test_gadgethdf_detection_notices_changed_file(tmp_path, monkeypatch):
    # within the recheck interval, a file is assumed not to have changed; here, check it every time
    monkeypatch.setattr(gadgethdf._CachedHDF5Inspection, "recheck_interval", 0.0)
    filename = tmp_path / "changing.hdf5"
    with h5py.File(filename, "w") as f:
        f.create_group("Unrelated")
    assert pynbody.snapshot.identify(filename) is None

    with h5py.File(filename, "w") as f:
        f.create_group("Header")
        f.create_group("PartType1")
        f["PartType1"].create_dataset("Coordinates", data=np.zeros((2, 3)))
    assert pynbody.snapshot.identify(filename) is gadgethdf.GadgetHDFSnap


def test_gadgethdf_detection_stats_once_per_file(monkeypatch):
    stats = []
    original_stat = os.stat

    def counting_stat(path, *args, **kwargs):
        stats.append(os.fspath(path))
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", counting_stat)
    hdf_classes = [gadgethdf.GadgetHDFSnap, *gadgethdf.GadgetHDFSnap.iter_subclasses()]
    for path in ["testdata/gadget3/data/snapshot_103/snap_103.hdf5", "testdata/gasoline_ahf/g15784.lr.01024"]:
        gadgethdf._cached_hdf5_inspection.clear()
        stats.clear()
        for c in hdf_classes:
            c._can_load(pathlib.Path(path))
        # all the classes together should check the file, and its ".0.hdf5" alternative, at most once each
        guess = str(pathlib.Path(path).with_suffix(".0.hdf5"))
        assert stats.count(path) <= 1
        assert stats.count(guess) <= 1


def test_corrupt_gzip_not_identified(tmp_path):
    # a valid gzip header followed by a corrupt deflate stream raises zlib.error, not OSError, when read
    data = bytearray(gzip.compress(b"x" * 1000))
    data[10:20] = b"\xff" * 10
    (tmp_path / "corrupt.gz").write_bytes(bytes(data))
    with pytest.raises(OSError, match="format not understood"):
        pynbody.load(tmp_path / "corrupt.gz")


def test_unidentified_multifile_stem_error(tmp_path):
    with h5py.File(tmp_path / "snap.0.hdf5", "w") as f:
        f.create_group("SomethingElse")
    with pytest.raises(OSError, match="snap.0.hdf5' does: file is HDF5 with top-level entries: SomethingElse"):
        pynbody.load(tmp_path / "snap")


def test_gadgethdf_subclass_with_staticmethod_test(tmp_path):
    class StaticTestHDFSnap(gadgethdf.GadgetHDFSnap):
        @staticmethod
        def _test_for_hdf5_key(f, method):
            return False

    try:
        # previously, looking up __func__ on the staticmethod raised AttributeError whenever this class was asked,
        # which happens for any file that no class before it in the priority order recognises
        assert _snap_class("testdata/nonexistent_file") is None
        assert _snap_class("testdata/gadget3/data/snapshot_103/snap_103.hdf5") == "GadgetHDFSnap"
    finally:
        del StaticTestHDFSnap
        gc.collect()


def test_gadgethdf_detection_distrusts_recent_mtime(tmp_path, monkeypatch):
    """A file rewritten within one tick of a coarse mtime may keep the same stat signature, so a signature taken
    just after a modification must not be trusted to show that the file is unchanged"""
    monkeypatch.setattr(gadgethdf._CachedHDF5Inspection, "recheck_interval", 0.0)
    filename = tmp_path / "rewritten.hdf5"
    with h5py.File(filename, "w") as f:
        f.create_group("Unrelated")
    original_stat = os.stat(filename)
    assert pynbody.snapshot.identify(filename) is None

    with h5py.File(filename, "w") as f:
        f.create_group("Header")
        f.create_group("PartType1")

    # simulate a filesystem on which the rewrite left the stat signature unchanged
    real_stat = os.stat
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: original_stat if os.fspath(p) == str(filename)
                        else real_stat(p, *a, **k))
    assert pynbody.snapshot.identify(filename) is gadgethdf.GadgetHDFSnap


class _FakeRemoteDir:
    """Stands in for an hdfstream remote directory, serving local HDF5 files under other names"""
    def __init__(self, files):
        self._files = files

    def is_hdf5(self, filename):
        return pathlib.PurePath(filename).as_posix() in self._files

    def File(self, filename, mode="r"):
        return h5py.File(self._files[pathlib.PurePath(filename).as_posix()], mode)


def test_identify_passes_relevant_kwargs():
    remote_dir = _FakeRemoteDir({"remote/snap_103.hdf5": "testdata/gadget3/data/snapshot_103/snap_103.hdf5"})
    # remote_dir is named by GadgetHDFSnap._can_load, so reaches it; no class can find this path locally
    assert pynbody.snapshot.identify("remote/snap_103.hdf5", remote_dir=remote_dir) is gadgethdf.GadgetHDFSnap
    assert pynbody.snapshot.identify("remote/snap_103.hdf5") is None

    # keyword arguments that no _can_load names, such as take, are not passed and do not affect identification
    assert _snap_class("testdata/gasoline_ahf/g15784.lr.01024") == \
           pynbody.snapshot.identify("testdata/gasoline_ahf/g15784.lr.01024", take=[1, 2, 3]).__name__


def test_can_load_receives_only_kwargs_it_accepts(tmp_path):
    received = {}

    class NamesOneKwarg(SimSnap):
        @classmethod
        def _can_load(cls, f, special_option=None):
            received["NamesOneKwarg"] = special_option
            return False

    class TakesAllKwargs(SimSnap):
        @classmethod
        def _can_load(cls, f, **kwargs):
            received["TakesAllKwargs"] = kwargs
            return False

    class LegacySignature(SimSnap):
        @classmethod
        def _can_load(cls, f):
            received["LegacySignature"] = True
            return False

    try:
        assert pynbody.snapshot.identify(tmp_path / "nonexistent", special_option=1, take=[1]) is None
        assert received == {"NamesOneKwarg": 1, "TakesAllKwargs": {"special_option": 1, "take": [1]},
                            "LegacySignature": True}
    finally:
        del NamesOneKwarg, TakesAllKwargs, LegacySignature
        gc.collect()
