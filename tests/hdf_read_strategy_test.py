import pytest

from pynbody.util import hdf_read_strategy
from pynbody.util.hdf_read_strategy import ReadSummary, choose_read_strategy


@pytest.fixture(autouse=True)
def plenty_of_cpus(monkeypatch):
    monkeypatch.setattr(hdf_read_strategy, "available_cpus", lambda: 64)
    monkeypatch.setitem(hdf_read_strategy.config, "bulk-read-threads", "auto")
    monkeypatch.setitem(hdf_read_strategy.config, "parallel-filesystem-threads", 4)
    monkeypatch.setitem(hdf_read_strategy.config, "compressed-data-threads", 3)


def on(monkeypatch, fs_type):
    monkeypatch.setattr(hdf_read_strategy, "filesystem_type", lambda path: fs_type)


def summary(num_reads=100, num_files=16, all_direct=True, compressed=True):
    return ReadSummary(paths=("/data/snap.0.hdf5",), num_reads=num_reads, num_files=num_files, all_direct=all_direct,
                       compressed=compressed)


def test_parallel_filesystem_uses_one_thread_per_file(monkeypatch):
    on(monkeypatch, "lustre")
    strategy = choose_read_strategy(summary())
    assert (strategy.threads, strategy.per_file) == (4, True)
    assert "parallel filesystem" in strategy.reason
    # compression makes no difference there
    assert choose_read_strategy(summary(compressed=False)).threads == 4


def test_parallel_filesystem_never_more_threads_than_files(monkeypatch):
    on(monkeypatch, "lustre")
    assert choose_read_strategy(summary(num_files=2)).threads == 2
    assert choose_read_strategy(summary(num_files=1)).threads == 1  # a single file is read serially


def test_local_compressed_shares_files(monkeypatch):
    on(monkeypatch, "ext4")
    strategy = choose_read_strategy(summary(num_files=1))
    assert (strategy.threads, strategy.per_file) == (3, False)
    assert "compressed" in strategy.reason


def test_local_uncompressed_is_serial(monkeypatch):
    on(monkeypatch, "xfs")
    assert choose_read_strategy(summary(compressed=False)).threads == 1


def test_unknown_filesystem_counts_as_local(monkeypatch):
    on(monkeypatch, None)
    assert choose_read_strategy(summary(compressed=False)).threads == 1
    assert choose_read_strategy(summary()).per_file is False


def test_h5py_reads_are_serial(monkeypatch):
    on(monkeypatch, "lustre")
    strategy = choose_read_strategy(summary(all_direct=False))
    assert strategy.threads == 1
    assert "h5py" in strategy.reason


def test_single_read_is_serial(monkeypatch):
    on(monkeypatch, "lustre")
    assert choose_read_strategy(summary(num_reads=1, num_files=1)).threads == 1


@pytest.mark.parametrize("fs_type, per_file", [("lustre", True), ("ext4", False)])
def test_fixed_number_of_threads(monkeypatch, fs_type, per_file):
    on(monkeypatch, fs_type)
    monkeypatch.setitem(hdf_read_strategy.config, "bulk-read-threads", "8")
    strategy = choose_read_strategy(summary(compressed=False))
    assert (strategy.threads, strategy.per_file) == (8, per_file)
    monkeypatch.setitem(hdf_read_strategy.config, "bulk-read-threads", "1")
    assert choose_read_strategy(summary()).threads == 1
    # but still reads through h5py are serial whatever is configured
    monkeypatch.setitem(hdf_read_strategy.config, "bulk-read-threads", "8")
    assert choose_read_strategy(summary(all_direct=False)).threads == 1


def test_limited_by_cpus(monkeypatch):
    on(monkeypatch, "ext4")
    monkeypatch.setattr(hdf_read_strategy, "available_cpus", lambda: 2)
    strategy = choose_read_strategy(summary())
    assert strategy.threads == 2
    assert "limited" in strategy.reason


def test_filesystem_type_from_mount_table(monkeypatch):
    table = (("/", "ext4"), ("/cosma8", "lustre"), ("/cosma8/data/local", "xfs"), ("/scratch space", "gpfs"))
    monkeypatch.setattr(hdf_read_strategy, "_mount_table", lambda: table)
    monkeypatch.setattr(hdf_read_strategy.os.path, "realpath", lambda p: p)
    assert hdf_read_strategy.filesystem_type("/cosma8/data/dp004/snap.0.hdf5") == "lustre"
    assert hdf_read_strategy.filesystem_type("/cosma8/data/local/snap.hdf5") == "xfs"  # longest mount point wins
    assert hdf_read_strategy.filesystem_type("/cosma8x/snap.hdf5") == "ext4"  # not a prefix at a separator
    assert hdf_read_strategy.filesystem_type("/scratch space/a.hdf5") == "gpfs"
    monkeypatch.setattr(hdf_read_strategy, "_mount_table", lambda: ())
    assert hdf_read_strategy.filesystem_type("/anything") is None


def test_real_mount_table_parses():
    table = hdf_read_strategy._mount_table()
    assert all(isinstance(mount_point, str) and isinstance(fs_type, str) for mount_point, fs_type in table)
    hdf_read_strategy.filesystem_type(__file__)  # does not raise
