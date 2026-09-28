import pytest

from pynbody.util.hdf_bulk_read import strategy as hdf_read_strategy
from pynbody.util.hdf_bulk_read.strategy import ReadSummary, choose_read_strategy

MB = 1024 ** 2


@pytest.fixture(autouse=True)
def plenty_of_cpus(monkeypatch):
    monkeypatch.setattr(hdf_read_strategy, "available_cpus", lambda: 64)
    monkeypatch.setitem(hdf_read_strategy.config, "io-threads", "auto")
    monkeypatch.setitem(hdf_read_strategy.config, "parallel-filesystem-io-threads", 4)
    monkeypatch.setitem(hdf_read_strategy.config, "decode-threads", "auto")
    monkeypatch.setitem(hdf_read_strategy.config, "max-decode-threads", 6)
    monkeypatch.setitem(hdf_read_strategy.config, "decode-memory", 2 * 1024 ** 3)


def on(monkeypatch, fs_type):
    monkeypatch.setattr(hdf_read_strategy, "filesystem_type", lambda path: fs_type)


def summary(num_reads=100, num_files=16, all_direct=True, compressed=True, max_chunk_nbytes=4 * MB, num_chunks=200):
    return ReadSummary(paths=("/data/snap.0.hdf5",), num_reads=num_reads, num_files=num_files, all_direct=all_direct,
                       compressed=compressed, max_chunk_nbytes=max_chunk_nbytes, num_chunks=num_chunks)


def threads(strategy):
    return strategy.io_threads, strategy.decode_threads


def test_parallel_filesystem_reads_several_files_at_once(monkeypatch):
    on(monkeypatch, "lustre")
    strategy = choose_read_strategy(summary())
    assert threads(strategy) == (4, 6)
    assert "parallel filesystem" in strategy.reason
    # uncompressed data have nothing to decode, and the input threads put them in place themselves
    assert threads(choose_read_strategy(summary(compressed=False))) == (4, 0)


def test_input_threads_never_outnumber_files(monkeypatch):
    on(monkeypatch, "lustre")
    assert threads(choose_read_strategy(summary(num_files=2))) == (2, 6)
    # a single file is read by one thread, and decoded by several
    assert threads(choose_read_strategy(summary(num_files=1))) == (1, 6)


def test_local_filesystem_has_one_reader(monkeypatch):
    on(monkeypatch, "ext4")
    strategy = choose_read_strategy(summary())
    assert threads(strategy) == (1, 6)
    assert "compressed" in strategy.reason
    uncompressed = choose_read_strategy(summary(compressed=False))
    assert threads(uncompressed) == (1, 0) and uncompressed.serial


def test_unknown_filesystem_counts_as_local(monkeypatch):
    on(monkeypatch, None)
    assert threads(choose_read_strategy(summary())) == (1, 6)


def test_h5py_reads_are_serial(monkeypatch):
    on(monkeypatch, "lustre")
    strategy = choose_read_strategy(summary(all_direct=False))
    assert strategy.serial
    assert "h5py" in strategy.reason


def test_nothing_or_one_chunk_is_read_serially(monkeypatch):
    on(monkeypatch, "lustre")
    assert choose_read_strategy(summary(num_reads=0, num_files=0, num_chunks=0)).serial
    assert choose_read_strategy(summary(num_reads=1, num_files=1, num_chunks=1)).serial


def test_decode_threads_never_outnumber_chunks(monkeypatch):
    on(monkeypatch, "ext4")
    assert threads(choose_read_strategy(summary(num_chunks=3))) == (1, 3)


@pytest.mark.parametrize("fs_type", ["lustre", "ext4"])
def test_fixed_numbers_of_threads(monkeypatch, fs_type):
    on(monkeypatch, fs_type)
    monkeypatch.setitem(hdf_read_strategy.config, "io-threads", "8")
    monkeypatch.setitem(hdf_read_strategy.config, "decode-threads", "3")
    assert threads(choose_read_strategy(summary())) == (8, 3)
    monkeypatch.setitem(hdf_read_strategy.config, "decode-threads", "0")  # (input threads decode for themselves)
    assert threads(choose_read_strategy(summary())) == (8, 0)
    monkeypatch.setitem(hdf_read_strategy.config, "io-threads", 1)
    assert choose_read_strategy(summary()).serial
    # but still reads through h5py are serial whatever is configured
    monkeypatch.setitem(hdf_read_strategy.config, "io-threads", "8")
    monkeypatch.setitem(hdf_read_strategy.config, "decode-threads", "8")
    assert choose_read_strategy(summary(all_direct=False)).serial


def test_decode_threads_limited_by_cpus(monkeypatch):
    on(monkeypatch, "ext4")
    monkeypatch.setattr(hdf_read_strategy, "available_cpus", lambda: 2)
    strategy = choose_read_strategy(summary())
    assert threads(strategy) == (1, 2)
    assert "2 CPUs" in strategy.reason


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


@pytest.mark.parametrize("chunk_mb, decode_threads", [(1, 6), (10, 6), (100, 4), (300, 1), (5000, 0)])
def test_decode_threads_limited_by_memory(monkeypatch, chunk_mb, decode_threads):
    """Each decode thread, with what waits for it, needs about five times the size of its chunks, which decode-memory
    limits"""
    on(monkeypatch, "lustre")
    strategy = choose_read_strategy(summary(max_chunk_nbytes=chunk_mb * MB))
    assert strategy.decode_threads == decode_threads
    assert ("decode-memory" in strategy.reason) == (chunk_mb >= 100)
    # what may wait to be decoded is two chunks per decode thread; with no decode threads, each input thread decodes
    # a chunk itself, as many at once as decode-memory allows (but at least one)
    inflight_chunks = 2 * decode_threads if decode_threads else 1
    assert strategy.inflight_nbytes == inflight_chunks * chunk_mb * MB


def test_input_threads_decoding_for_themselves_are_limited_by_memory(monkeypatch):
    on(monkeypatch, "lustre")
    monkeypatch.setitem(hdf_read_strategy.config, "decode-threads", "0")
    strategy = choose_read_strategy(summary(max_chunk_nbytes=10 * MB))
    assert threads(strategy) == (4, 0) and strategy.inflight_nbytes == 4 * 10 * MB  # (one chunk per input thread)
    strategy = choose_read_strategy(summary(max_chunk_nbytes=200 * MB))
    assert threads(strategy) == (4, 0) and strategy.inflight_nbytes == 2 * 200 * MB  # (2 GiB / (5 x 200 MB))
    assert "decode-memory" in strategy.reason
    strategy = choose_read_strategy(summary(max_chunk_nbytes=10 * MB, compressed=False))
    assert threads(strategy) == (4, 0) and strategy.inflight_nbytes == 4 * 10 * MB


@pytest.mark.parametrize("option", ["io-threads", "decode-threads"])
@pytest.mark.parametrize("value", ["many", "-3", "2.5"])
def test_nonsensical_thread_counts_are_ignored(monkeypatch, option, value):
    on(monkeypatch, "lustre")
    monkeypatch.setitem(hdf_read_strategy.config, option, value)
    strategy = choose_read_strategy(summary())
    if value == "-3":
        assert threads(strategy) == ((1, 6) if option == "io-threads" else (4, 0))  # the least allowed
    else:
        assert threads(strategy) == (4, 6)  # as for 'auto'


def test_bad_configuration_values_fall_back_to_defaults(monkeypatch, caplog):
    values = {"io-threads": "lots", "parallel-filesystem-io-threads": "sixteen", "decode-threads": "-1",
              "max-decode-threads": "0", "decode-memory": "2GB"}
    monkeypatch.setattr(hdf_read_strategy.config_parser, "get",
                        lambda section, name, fallback=None: values.get(name, fallback))
    with caplog.at_level("WARNING", logger="pynbody.util.hdf_bulk_read.strategy"):
        config = hdf_read_strategy._read_config()
    assert config == {"io-threads": "auto", "parallel-filesystem-io-threads": 16, "decode-threads": "auto",
                      "max-decode-threads": 16, "decode-memory": 2 * 1024 ** 3}
    assert len([r for r in caplog.records if "Ignoring the value" in r.getMessage()]) == 5


def test_nfs_is_not_a_parallel_filesystem(monkeypatch):
    on(monkeypatch, "nfs4")
    assert threads(choose_read_strategy(summary())) == (1, 6)  # treated like a local disk


def test_options_cannot_be_misspelt():
    with pytest.raises(KeyError, match="not an option"):
        hdf_read_strategy.config["threads"] = "1"
    hdf_read_strategy.config["io-threads"] = hdf_read_strategy.config["io-threads"]  # (existing options can be changed)


@pytest.mark.parametrize("section, name", [("gadgethdf", "bulk-read-threads"), ("hdf-bulk-read", "threads")])
def test_old_option_names_are_reported(monkeypatch, caplog, section, name):
    monkeypatch.setattr(hdf_read_strategy.config_parser, "has_option", lambda s, n: (s, n) == (section, name))
    with caplog.at_level("WARNING", logger="pynbody.util.hdf_bulk_read.strategy"):
        hdf_read_strategy._read_config()
    assert any(name in r.getMessage() and f"[{section}]" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("fs_type", ["lustre", "ext4"])
def test_small_chunks_are_decoded_by_input_threads(monkeypatch, fs_type):
    """Decompressing chunks much smaller than 128 kB takes no longer than the Python work around it, which threads
    cannot share, so decode threads would only slow reading down"""
    on(monkeypatch, fs_type)
    strategy = choose_read_strategy(summary(max_chunk_nbytes=64 * 1024, num_chunks=10000))
    assert strategy.decode_threads == 0 and "too small" in strategy.reason
    assert strategy.io_threads == (4 if fs_type == "lustre" else 1)
    monkeypatch.setitem(hdf_read_strategy.config, "decode-threads", "3")  # (unless asked for)
    assert choose_read_strategy(summary(max_chunk_nbytes=64 * 1024, num_chunks=10000)).decode_threads == 3


def test_budget_is_counted_in_jobs(monkeypatch):
    """Small chunks decoded several to a job are budgeted for as jobs"""
    on(monkeypatch, "ext4")
    s = ReadSummary(paths=("/data/snap.0.hdf5",), num_reads=10, num_files=1, all_direct=True, compressed=True,
                    max_chunk_nbytes=512 * 1024, max_job_nbytes=MB, num_chunks=100)
    strategy = choose_read_strategy(s)
    assert threads(strategy) == (1, 6)
    assert strategy.inflight_nbytes == 2 * 6 * MB
