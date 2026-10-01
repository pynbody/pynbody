import gc
import glob
import os
import pathlib

import h5py
import numpy as np
import pytest

import pynbody
import pynbody.test_utils
from pynbody.util import file_probe


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gasoline_ahf", "swift", "ramses", "gadget", "arepo",
                                                  "tng_subfind", "subfind", "hbt", "nchilada", "grafic",
                                                  "pkdgrav3", "gizmo", "lpicola")


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning", "ignore::RuntimeWarning")


class _OperationCounter:
    """Counts filesystem operations, so that tests can check that format identification stays cheap"""

    def __init__(self, monkeypatch):
        self.counts = {'stat': 0, 'open': 0, 'scandir': 0, 'hdf5_open': 0, 'is_hdf5': 0}

        def counting(name, original):
            def wrapper(*args, **kwargs):
                self.counts[name] += 1
                return original(*args, **kwargs)
            return wrapper

        monkeypatch.setattr(os, "stat", counting('stat', os.stat))
        monkeypatch.setattr(os, "scandir", counting('scandir', os.scandir))
        monkeypatch.setattr(os, "listdir", counting('scandir', os.listdir))
        monkeypatch.setattr("builtins.open", counting('open', open))
        monkeypatch.setattr(h5py, "is_hdf5", counting('is_hdf5', h5py.is_hdf5))

        original_file = h5py.File
        counter = self

        class CountingFile(original_file):
            def __init__(self, *args, **kwargs):
                counter.counts['hdf5_open'] += 1
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(h5py, "File", CountingFile)


@pytest.fixture
def counter(monkeypatch):
    return _OperationCounter(monkeypatch)


# Expected identification of test data. This pins down the behaviour of format identification, so that any change
# to the way files are recognised is deliberate.
_expected_snapshot_classes = {
    "testdata/gasoline_ahf/g15784.lr.01024": "TipsySnap",
    "testdata/gasoline_ahf/g15784.lr.01024.gz": "TipsySnap",
    "testdata/SWIFT/snap_0150.hdf5": "SwiftSnap",
    "testdata/SWIFT/multifile_without_vds/snap_0000": "SwiftSnap",
    "testdata/SWIFT/multifile_with_vds/snap_0000.hdf5": "SwiftSnap",
    "testdata/ramses/output_00080": "RamsesSnap",
    "testdata/gadget2/test_g2_snap": "GadgetSnap",
    "testdata/gadget2/test_g2_snap.1": "GadgetSnap",
    "testdata/gadget1.snap": "GadgetSnap",
    "testdata/gizmo/snapshot_000.hdf5": "GizmoHDFSnap",
    "testdata/gadget3/data/subhalos_103/subhalo_103": "SubFindHDFSnap",
    "testdata/gadget3/snap_028_z000p000.0.hdf5": "EagleLikeHDFSnap",
    "testdata/gadget3/data/snapshot_103/snap_103.hdf5": "GadgetHDFSnap",
    "testdata/nchilada_test/12M.00001": "NchiladaSnap",
    "testdata/gadget4_subfind_HBT/snapshot_034.hdf5": "GadgetHDFSnap",
    "testdata/grafic_test/": "GrafICSnap",
    "testdata/arepo/cosmobox_015.hdf5": "ArepoHDFSnap",
    "testdata/arepo/tng/snapdir_261/snap_261": "ArepoHDFSnap",
    "testdata/subfind/snapshot_019": "GadgetSnap",
    "testdata/pkdgrav3/cosmoSF.00010": "PkdgravHDFSnap",
    "testdata/lpicola/lpicola_z0p000.0": "GadgetSnap",
    "testdata/nonexistent_file": None,
    "testdata/gadget2": None,
}

_expected_halo_classes = {
    "testdata/gasoline_ahf/g15784.lr.01024": "AHFCatalogue",
    "testdata/SWIFT/snap_0150.hdf5": "HaloNumberCatalogue",
    "testdata/ramses/output_00080": "AdaptaHOPCatalogue",
    "testdata/gadget3/data/subhalos_103/subhalo_103": "SubFindHDFHaloCatalogue",
    "testdata/gadget4_subfind_HBT/snapshot_034.hdf5": "Gadget4SubfindHDFCatalogue",
    "testdata/arepo/cosmobox_015.hdf5": "ArepoSubfindHDFCatalogue",
    "testdata/arepo/tng/snapdir_261/snap_261": "TNGSubfindHDFCatalogue",
    "testdata/subfind/snapshot_019": "SubfindCatalogue",
    "testdata/gadget2/test_g2_snap": None,
}


@pytest.mark.parametrize("path, expected", _expected_snapshot_classes.items())
def test_identify_snapshot(path, expected):
    result = pynbody.snapshot.identify(path)
    assert (result.__name__ if result is not None else None) == expected


def _identify_halo_class(sim):
    from pynbody import config, halo
    with file_probe.ProbeCache() as probes:
        for c in halo.HaloCatalogue.iter_subclasses_with_priority(config['halo-class-priority']):
            try:
                if c._can_load_with_dispatch(sim, probes):
                    return c.__name__
            except TypeError:
                # as in SimSnap.halos, a TypeError indicates the class can't be used
                pass
    return None


@pytest.mark.parametrize("path, expected", _expected_halo_classes.items())
def test_identify_halos(path, expected):
    assert _identify_halo_class(pynbody.load(path)) == expected


@pytest.mark.parametrize("path", ["testdata/gasoline_ahf/g15784.lr.01024", "testdata/arepo/cosmobox_015.hdf5"])
def test_legacy_can_load_still_works(path):
    """Calling _can_load directly on a converted class should still give the right answer"""
    snap_class = pynbody.snapshot.identify(path)
    assert snap_class._can_load(pathlib.Path(path))
    assert not pynbody.snapshot.ramses.RamsesSnap._can_load(pathlib.Path(path))

    f = pynbody.load(path)
    halo_class = getattr(pynbody.halo, _expected_halo_classes[path], None) or \
        {c.__name__: c for c in pynbody.halo.HaloCatalogue.iter_subclasses()}[_expected_halo_classes[path]]
    assert halo_class._can_load(f)
    assert not pynbody.halo.rockstar.RockstarCatalogue._can_load(f)


def test_legacy_subclass_overrides_can_load():
    """A subclass written against the old API, overriding only _can_load, must have that override respected
    even though its parent now implements _can_load_from_probe"""

    class LegacySubclass(pynbody.snapshot.gadgethdf.GadgetHDFSnap):
        @classmethod
        def _can_load(cls, f):
            return f.name == "legacy_sentinel_name" or (f.name == "snap_103.hdf5" and super()._can_load(f))

    class LegacyHaloSubclass(pynbody.halo.HaloCatalogue):
        @classmethod
        def _can_load(cls, sim, special=False):
            return special

    try:
        with file_probe.ProbeCache() as probes:
            assert LegacySubclass._can_load_with_dispatch(probes.probe("nowhere/legacy_sentinel_name"))
            # in the old API, the subclass could call up to the parent; it should still be able to do so
            assert LegacySubclass._can_load_with_dispatch(probes.probe("testdata/gadget3/data/snapshot_103/snap_103.hdf5"))
            assert not LegacySubclass._can_load_with_dispatch(probes.probe("testdata/gizmo/snapshot_000.hdf5"))
            assert not LegacySubclass._can_load_with_dispatch(probes.probe("testdata/gasoline_ahf/g15784.lr.01024"))

            assert LegacyHaloSubclass._can_load_with_dispatch(None, probes, special=True)
            assert not LegacyHaloSubclass._can_load_with_dispatch(None, probes)
    finally:
        del LegacySubclass, LegacyHaloSubclass
        gc.collect()


def test_load_error_message_describes_file():
    with pytest.raises(OSError, match="does not exist"):
        pynbody.load("testdata/nonexistent_file")
    with pytest.raises(OSError, match="directory"):
        pynbody.load("testdata/gadget2")


@pytest.mark.parametrize("path, max_counts", [
    # (max stats, max python-level opens, max directory listings, max HDF5 opens)
    ("testdata/gadget3/data/snapshot_103/snap_103.hdf5", (4, 1, 0, 1)),
    ("testdata/arepo/tng/snapdir_261/snap_261", (6, 1, 0, 1)),
    ("testdata/gasoline_ahf/g15784.lr.01024", (6, 1, 0, 0)),
    ("testdata/gadget2/test_g2_snap", (4, 1, 0, 0)),
    ("testdata/ramses/output_00080", (4, 0, 0, 0)),
])
def test_identify_snapshot_is_cheap(path, max_counts, counter):
    pynbody.snapshot.identify(path)
    c = counter.counts
    assert all(a <= b for a, b in zip((c['stat'], c['open'], c['scandir'], c['hdf5_open']), max_counts)), c
    assert c['is_hdf5'] == 0


def test_hdf5_snapshot_opened_once(counter):
    f = pynbody.load("testdata/gadget3/data/snapshot_103/snap_103.hdf5")
    # identification opens the file, and GadgetHDFSnap takes it over rather than reopening it
    assert counter.counts['hdf5_open'] == 1
    assert counter.counts['is_hdf5'] == 0
    assert len(f.dm['pos']) > 0


def test_halo_identification_is_cheap(counter):
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    counter.counts.update({k: 0 for k in counter.counts})
    assert _identify_halo_class(f) == "AHFCatalogue"
    c = counter.counts
    # previously, this opened every one of the snapshot's auxiliary files, and listed the directory repeatedly
    assert c['open'] <= 1, c
    # the parent and grandparent directories are listed, and the absent VR subdirectories are probed
    assert c['scandir'] <= 4, c
    assert c['stat'] <= 10, c


@pytest.fixture
def tipsy_with_aux_arrays(tmp_path):
    f = pynbody.new(dm=10, gas=5, star=3)
    f['pos'] = np.random.uniform(size=(18, 3))
    f.write(fmt=pynbody.snapshot.tipsy.TipsySnap, filename=str(tmp_path / "test.tipsy"))
    f = pynbody.load(tmp_path / "test.tipsy")
    f['grp'] = np.arange(18, dtype=np.int32)
    f['grp'].write()
    f.gas['gasonly'] = np.arange(5, dtype=float)
    f.gas['gasonly'].write()
    (tmp_path / "test.tipsy.badlength").write_text("3\n1\n2\n3\n")
    return tmp_path / "test.tipsy"


def test_tipsy_has_loadable_key_matches_loadable_keys(tipsy_with_aux_arrays):
    names = ["grp", "amiga.grp", "pos", "mass", "tform", "gasonly", "badlength", "nonexistent"]
    f = pynbody.load(tipsy_with_aux_arrays)
    cheap = {n: f._has_loadable_key(n) for n in names}
    assert len(f._loadable_keys_registry) == 0  # i.e. we didn't build the full registry
    expensive = {n: n in f.loadable_keys() for n in names}
    assert cheap == expensive
    assert cheap["grp"] and cheap["pos"]
    assert not cheap["gasonly"] and not cheap["badlength"] and not cheap["nonexistent"]

    # once the registry exists, it is used directly
    assert f._has_loadable_key("grp")

    # and the halo catalogue identification uses it
    assert _identify_halo_class(pynbody.load(tipsy_with_aux_arrays)) == "HaloNumberCatalogue"


def test_hbt_halos_reloadable_while_open():
    """HBT+ catalogues are opened with locking=False, so probes must cope with that file already being open"""
    f = pynbody.load("testdata/gadget4_subfind_HBT/snapshot_034.hdf5")
    h1 = f.halos(priority=["HBTPlusCatalogue"])
    h2 = f.halos(priority=["HBTPlusCatalogue"])
    assert type(h1) is type(h2) is pynbody.halo.hbtplus.HBTPlusCatalogue
    assert len(h1) == len(h2)


# Unit tests for the probe machinery itself

def test_probe_caches_stat(tmp_path, counter):
    (tmp_path / "a").write_bytes(b"hello")
    with file_probe.ProbeCache() as probes:
        p = probes.probe(tmp_path / "a")
        for _ in range(3):
            assert p.exists() and p.is_file() and not p.is_dir() and p.size() == 5
        assert probes.probe(tmp_path / "a") is p
        assert not probes.exists(tmp_path / "b")
        assert not probes.exists(tmp_path / "b")
    assert counter.counts['stat'] == 2


def test_probe_head(tmp_path, counter):
    data = bytes(range(256)) * 40
    (tmp_path / "a").write_bytes(data)
    with file_probe.ProbeCache() as probes:
        p = probes.probe(tmp_path / "a")
        assert p.head(4) == data[:4]
        assert p.head(100) == data[:100]
        assert counter.counts['open'] == 1
        assert p.head(8000) == data[:8000]
        assert counter.counts['open'] == 2
        assert p.head(20000) == data
        assert counter.counts['open'] == 3
        # now we know the whole file is in memory, so no more reads are needed
        assert p.head(30000) == data
        assert counter.counts['open'] == 3

        assert probes.probe(tmp_path / "nonexistent").head() == b""
        assert probes.probe(tmp_path).head() == b""


def test_probe_head_decompressed(tmp_path):
    import gzip
    with gzip.open(tmp_path / "a.gz", "wb") as f:
        f.write(b"uncompressed content")
    with file_probe.ProbeCache() as probes:
        p = probes.probe(tmp_path / "a.gz")
        assert p.head_decompressed(12) == b"uncompressed"
        assert p.head(2) == b"\x1f\x8b"


def test_probe_hdf5(tmp_path, counter):
    with h5py.File(tmp_path / "test.hdf5", "w") as f:
        f.create_group("Header")
    with h5py.File(tmp_path / "userblock.hdf5", "w", userblock_size=8192) as f:
        f.create_group("Other")
    (tmp_path / "plain").write_bytes(b"x" * 10000)
    counter.counts['hdf5_open'] = 0

    with file_probe.ProbeCache() as probes:
        p = probes.probe(tmp_path / "test.hdf5")
        assert p.is_hdf5()
        assert "Header" in p.hdf5()
        assert p.hdf5() is p.hdf5()
        assert counter.counts['hdf5_open'] == 1
        assert counter.counts['is_hdf5'] == 0

        assert not probes.probe(tmp_path / "plain").is_hdf5()
        assert probes.probe(tmp_path / "plain").hdf5() is None
        assert not probes.probe(tmp_path / "nonexistent").is_hdf5()

        # a large user block means the signature is beyond the bytes we read, so falls back to h5py
        p_userblock = probes.probe(tmp_path / "userblock.hdf5")
        assert p_userblock.is_hdf5()
        assert "Other" in p_userblock.hdf5()

        f = p.hdf5()
        g = p_userblock.hdf5()

    # leaving the context closes the files
    assert not f.id.valid
    assert not g.id.valid


def test_probe_adopt_hdf5(tmp_path):
    with h5py.File(tmp_path / "test.hdf5", "w") as f:
        f.create_group("Header")
    with file_probe.ProbeCache() as probes:
        assert probes.probe(tmp_path / "test.hdf5").hdf5() is not None
        assert probes.adopt_hdf5(tmp_path / "test.hdf5", mode="r+") is None
        adopted = probes.adopt_hdf5(tmp_path / "test.hdf5")
        assert adopted is not None
        assert probes.adopt_hdf5(tmp_path / "test.hdf5") is None
    assert adopted.id.valid
    assert "Header" in adopted
    adopted.close()


@pytest.mark.parametrize("pattern", ["*", "*.txt", "a*", ".*", "?.txt", "[ab].txt", "nomatch*", "b.txt",
                                     "subdir/*", "*/*.txt", "nonexistent_dir/*"])
def test_probe_glob_matches_glob(tmp_path, pattern):
    for name in ["a.txt", "b.txt", "abc", ".hidden.txt"]:
        (tmp_path / name).write_text("")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "c.txt").write_text("")

    full_pattern = os.path.join(str(tmp_path), pattern)
    with file_probe.ProbeCache() as probes:
        assert sorted(probes.glob(full_pattern)) == sorted(glob.glob(full_pattern))

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with file_probe.ProbeCache() as probes:
            assert sorted(probes.glob(pattern)) == sorted(glob.glob(pattern))
    finally:
        os.chdir(cwd)


def test_probe_glob_shares_listing(tmp_path, counter):
    for name in ["a.txt", "b.bin"]:
        (tmp_path / name).write_text("")
    with file_probe.ProbeCache() as probes:
        probes.glob(str(tmp_path / "*.txt"))
        probes.glob(str(tmp_path / "*.bin"))
        probes.listdir(tmp_path)
        assert probes.probe(tmp_path).listdir()['a.txt'].is_file()
    assert counter.counts['scandir'] == 1


def test_probe_listdir(tmp_path):
    (tmp_path / "a").write_text("")
    (tmp_path / "d").mkdir()
    with file_probe.ProbeCache() as probes:
        assert sorted(probes.listdir(tmp_path)) == ["a", "d"]
        assert probes.probe(tmp_path).listdir()["d"].is_dir()
        assert probes.probe(tmp_path / "a").listdir() is None
        with pytest.raises(OSError):
            probes.listdir(tmp_path / "a")
