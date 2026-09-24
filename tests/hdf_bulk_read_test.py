import os

import h5py
import numpy as np
import pytest

from pynbody.util import hdf_bulk_read

pytestmark = pytest.mark.skipif(not hdf_bulk_read.pyfive_available(), reason="pyfive is not installed")


def _make_chunked(filename, data, chunks, filters=(), fillvalue=None, name="x"):
    """Write *data* as a chunked dataset with the given filters, applied on writing in the given order.

    Filters are HDF5 ids: 1 deflate, 2 shuffle, 3 fletcher32. The low-level API is used because h5py's
    create_dataset always puts shuffle first, whereas e.g. SWIFT applies fletcher32 first."""
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk(chunks)
    for filter_id in filters:
        if filter_id == 1:
            dcpl.set_deflate(4)
        elif filter_id == 2:
            dcpl.set_shuffle()
        elif filter_id == 3:
            dcpl.set_fletcher32()
    if fillvalue is not None:
        dcpl.set_fill_value(np.array(fillvalue, dtype=data.dtype))
    with h5py.File(filename, "a") as f:
        space = h5py.h5s.create_simple(data.shape)
        dataset_id = h5py.h5d.create(f.id, name.encode(), h5py.h5t.py_create(data.dtype), space, dcpl=dcpl)
        dataset_id.write(h5py.h5s.ALL, h5py.h5s.ALL, np.ascontiguousarray(data))


def _row_selections(n):
    """Row ranges exercising whole reads, reads within one chunk, across chunk boundaries, and empty reads."""
    return [slice(None), slice(0, n), slice(0, 1), slice(n - 1, n), slice(3, 17), slice(n // 3, 2 * n // 3),
            slice(5, 5), slice(n, n)]


def _check_reader_matches_h5py(filename, name="x", expected_type=None):
    reader = hdf_bulk_read.BulkReader()
    with h5py.File(filename, "r") as f:
        dataset = f[name]
        wrapped = reader.open(dataset)
        if expected_type is not None:
            assert isinstance(wrapped, expected_type)
        for sel in _row_selections(dataset.shape[0]):
            expected = dataset[sel]
            np.testing.assert_array_equal(wrapped[sel], expected)

            target = np.empty_like(expected)
            wrapped.read_direct(target, source_sel=None if sel == slice(None) else sel)
            np.testing.assert_array_equal(target, expected)
    reader.close()


@pytest.mark.parametrize("filters", [(), (1,), (2, 1), (3, 2, 1), (2, 1, 3), (3,), (1, 3)],
                         ids=lambda f: "filters-" + "-".join(map(str, f)) if f else "unfiltered")
@pytest.mark.parametrize("dtype", ["<f8", "<f4", "<i8", ">f8", "u1"])
@pytest.mark.parametrize("shape, chunks", [((1000,), (64,)), ((1000,), (1000,)), ((300, 3), (64, 3)),
                                           ((300, 3), (50, 1)), ((301, 3), (300, 2))])
def test_chunked_matches_h5py(tmp_path, filters, dtype, shape, chunks):
    rng = np.random.default_rng(1234)
    data = (rng.random(shape) * 200).astype(dtype)
    filename = tmp_path / "chunked.h5"
    _make_chunked(filename, data, chunks, filters)
    _check_reader_matches_h5py(filename, expected_type=hdf_bulk_read._ChunkedDataset)


def test_contiguous_matches_h5py(tmp_path):
    filename = tmp_path / "contiguous.h5"
    with h5py.File(filename, "w") as f:
        f["x"] = np.arange(3000, dtype=np.float64).reshape(1000, 3)
    _check_reader_matches_h5py(filename, expected_type=hdf_bulk_read._ContiguousDataset)


@pytest.mark.parametrize("words", [[0xFFFF], [0xFFFF, 0x0000], [0x1234, 0xEDCB], [0, 0, 0]])
def test_fletcher32_end_around_carry(tmp_path, words):
    """HDF5 keeps each fletcher32 sum in [1, 0xFFFF] by end-around carry, so a nonzero sum divisible by 65535 is
    stored as 0xFFFF, not 0. pyfive 1.2.1 reduces modulo 65535 instead, and so rejects such chunks as corrupt."""
    data = np.array(words, dtype=">u2")
    filename = tmp_path / "fletcher.h5"
    _make_chunked(filename, data, (len(words),), (3,))
    _check_reader_matches_h5py(filename)


@pytest.mark.parametrize("nbytes", [1, 2, 3, 7, 360 * 2 + 1, 100001])
def test_fletcher32_matches_hdf5(tmp_path, nbytes):
    rng = np.random.default_rng(nbytes)
    data = rng.integers(0, 256, nbytes).astype(np.uint8)
    filename = tmp_path / "fletcher.h5"
    _make_chunked(filename, data, (nbytes,), (3,))
    _check_reader_matches_h5py(filename)


def test_corrupt_checksum_is_detected(tmp_path):
    filename = tmp_path / "corrupt.h5"
    data = np.arange(100, dtype=np.float64)
    _make_chunked(filename, data, (100,), (3,))
    with h5py.File(filename, "r") as f:
        offset = f["x"].id.get_chunk_info(0).byte_offset
    with open(filename, "r+b") as f:
        f.seek(offset + 10)
        byte = f.read(1)
        f.seek(offset + 10)
        f.write(bytes([byte[0] ^ 0xFF]))

    with h5py.File(filename, "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        with pytest.raises(OSError, match="fletcher32"):
            wrapped[:]


def test_unallocated_chunks_read_as_fillvalue(tmp_path):
    filename = tmp_path / "sparse.h5"
    with h5py.File(filename, "w") as f:
        dataset = f.create_dataset("x", shape=(1000,), dtype=np.int32, chunks=(100,), fillvalue=-7,
                                   compression="gzip")
        dataset[250:420] = np.arange(170)
    _check_reader_matches_h5py(filename, expected_type=hdf_bulk_read._ChunkedDataset)


def test_chunk_cache_avoids_repeated_decoding(tmp_path, monkeypatch):
    filename = tmp_path / "bigchunk.h5"
    data = np.arange(10000, dtype=np.float64)
    _make_chunked(filename, data, (10000,), (2, 1))

    decodes = []
    original_decode = hdf_bulk_read.decode_chunk
    monkeypatch.setattr(hdf_bulk_read, "decode_chunk", lambda *args: decodes.append(1) or original_decode(*args))

    with h5py.File(filename, "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        result = np.concatenate([wrapped[i:i + 700] for i in range(0, 10000, 700)])
    np.testing.assert_array_equal(result, data)
    assert len(decodes) == 1

    decodes.clear()
    with h5py.File(filename, "r") as f:
        wrapped = hdf_bulk_read.BulkReader(cache_nbytes=1000).open(f["x"])  # too small to hold the chunk
        wrapped[0:700]
        wrapped[700:1400]
    assert len(decodes) == 2


def _make_file_with_datasets_pyfive_should_not_read(filename, external_filename):
    with h5py.File(filename, "w") as f:
        f.create_dataset("compound", data=np.zeros(10, dtype=[("a", "f4"), ("b", "i4")]))
        f.create_dataset("scaleoffset", data=np.arange(100, dtype=np.int32), chunks=(10,), scaleoffset=0)
        f.create_dataset("external", shape=(10,), dtype=np.float64, external=[(str(external_filename), 0, 80)])
        f["external"][:] = np.arange(10)
        f.create_dataset("never_written", shape=(10,), dtype=np.float32, fillvalue=3.0)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_layout(h5py.h5d.COMPACT)
        compact = h5py.h5d.create(f.id, b"compact", h5py.h5t.NATIVE_INT32, h5py.h5s.create_simple((4,)), dcpl=dcpl)
        compact.write(h5py.h5s.ALL, h5py.h5s.ALL, np.arange(4, dtype=np.int32))
        f["scalar"] = 1.5
        f["fine"] = np.arange(10.0)


@pytest.mark.parametrize("name", ["compound", "scaleoffset", "external", "never_written", "compact", "scalar"])
def test_falls_back_to_h5py(tmp_path, name):
    filename = tmp_path / "fallback.h5"
    _make_file_with_datasets_pyfive_should_not_read(filename, tmp_path / "external.bin")
    with h5py.File(filename, "r") as f:
        assert isinstance(hdf_bulk_read.BulkReader().open(f[name]), h5py.Dataset)
        assert not isinstance(hdf_bulk_read.BulkReader().open(f["fine"]), h5py.Dataset)


def test_disabled_reader_uses_h5py(tmp_path):
    filename = tmp_path / "fine.h5"
    with h5py.File(filename, "w") as f:
        f["x"] = np.arange(10)
    with h5py.File(filename, "r") as f:
        reader = hdf_bulk_read.BulkReader(use_pyfive=False)
        assert not reader.uses_pyfive
        assert isinstance(reader.open(f["x"]), h5py.Dataset)


def test_reader_reopens_after_close(tmp_path):
    filename = tmp_path / "rewritten.h5"
    with h5py.File(filename, "w") as f:
        f["x"] = np.arange(10)
    reader = hdf_bulk_read.BulkReader()
    with h5py.File(filename, "r") as f:
        np.testing.assert_array_equal(reader.open(f["x"])[:], np.arange(10))
    reader.close()
    with h5py.File(filename, "a") as f:
        f["y"] = np.arange(5) * 2
    with h5py.File(filename, "r") as f:
        np.testing.assert_array_equal(reader.open(f["y"])[:], np.arange(5) * 2)


# ---------------------------------------------------------------------------------------------------------------------
# Virtual datasets


def _make_sources(directory, row_counts, trailing=(3,), dtype=np.float64, compress=True):
    """Write one source file per entry of row_counts, each holding dataset 'x', and return their filenames and data"""
    filenames, arrays = [], []
    offset = 0
    for i, n in enumerate(row_counts):
        data = (np.arange(n * int(np.prod(trailing))) + offset).reshape((n,) + trailing).astype(dtype)
        offset += data.size
        filename = os.path.join(directory, f"source.{i}.h5")
        with h5py.File(filename, "w") as f:
            if compress and n > 0:
                f.create_dataset("x", data=data, chunks=(max(n // 3, 1),) + trailing, compression="gzip",
                                 shuffle=True, fletcher32=True)
            else:
                f["x"] = data
        filenames.append(filename)
        arrays.append(data)
    return filenames, arrays


def _make_vds(filename, sources, row_counts, trailing=(3,), dtype=np.float64, relative=True, gap=0,
              fillvalue=None):
    """Write a virtual dataset 'x' concatenating the sources along axis 0, leaving `gap` unmapped rows between them"""
    total = sum(row_counts) + gap * (len(row_counts) - 1)
    layout = h5py.VirtualLayout(shape=(total,) + trailing, dtype=dtype)
    row = 0
    for source, n in zip(sources, row_counts):
        name = os.path.basename(source) if relative else source
        layout[row:row + n] = h5py.VirtualSource(name, "x", shape=(n,) + trailing)
        row += n + gap
    with h5py.File(filename, "a") as f:
        f.create_virtual_dataset("x", layout, fillvalue=fillvalue)


def _check_vds(filename, expect_pyfive=True):
    reader = hdf_bulk_read.BulkReader()
    with h5py.File(filename, "r") as f:
        dataset = f["x"]
        wrapped = reader.open(dataset)
        if expect_pyfive:
            assert isinstance(wrapped, hdf_bulk_read._VirtualDataset)
        else:
            assert isinstance(wrapped, h5py.Dataset)
        for sel in _row_selections(dataset.shape[0]) + [slice(0, 45), slice(44, 46), slice(40, 90)]:
            np.testing.assert_array_equal(wrapped[sel], dataset[sel])
    reader.close()


@pytest.mark.parametrize("relative", [True, False])
@pytest.mark.parametrize("trailing", [(3,), ()])
def test_vds_matches_h5py(tmp_path, relative, trailing):
    row_counts = [45, 0, 30, 70]
    sources, _ = _make_sources(tmp_path, row_counts, trailing)
    _make_vds(tmp_path / "virtual.h5", sources, row_counts, trailing, relative=relative)
    _check_vds(tmp_path / "virtual.h5")


def test_vds_sources_are_read_through_pyfive(tmp_path):
    row_counts = [45, 30, 70]
    sources, arrays = _make_sources(tmp_path, row_counts)
    _make_vds(tmp_path / "virtual.h5", sources, row_counts)
    with h5py.File(tmp_path / "virtual.h5", "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        np.testing.assert_array_equal(wrapped[50:60], arrays[1][5:15])
        opened = [b for b in wrapped._blocks if b.reader_resolved]
        assert len(opened) == 1  # only the source that was needed has been opened
        assert isinstance(opened[0].reader, hdf_bulk_read._ChunkedDataset)


def test_vds_found_from_another_directory(tmp_path, monkeypatch):
    row_counts = [10, 20]
    sources, _ = _make_sources(tmp_path, row_counts)
    _make_vds(tmp_path / "virtual.h5", sources, row_counts, relative=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    _check_vds(tmp_path / "virtual.h5")


def test_vds_gaps_read_as_fillvalue(tmp_path):
    row_counts = [10, 20, 15]
    sources, _ = _make_sources(tmp_path, row_counts)
    _make_vds(tmp_path / "virtual.h5", sources, row_counts, gap=4, fillvalue=-1.0)
    _check_vds(tmp_path / "virtual.h5")


def test_vds_missing_source(tmp_path):
    row_counts = [10, 20, 15]
    sources, _ = _make_sources(tmp_path, row_counts)
    _make_vds(tmp_path / "virtual.h5", sources, row_counts, fillvalue=-2.0)
    os.remove(sources[1])
    _check_vds(tmp_path / "virtual.h5")


def test_vds_converts_dtype(tmp_path):
    row_counts = [10, 20]
    sources, _ = _make_sources(tmp_path, row_counts, dtype=np.float32)
    _make_vds(tmp_path / "virtual.h5", sources, row_counts, dtype=np.float64)
    _check_vds(tmp_path / "virtual.h5")


def test_vds_of_part_of_a_source_and_same_file(tmp_path):
    filename = tmp_path / "virtual.h5"
    big_source = np.arange(300.).reshape(100, 3)
    local = np.arange(30.).reshape(10, 3) * -1
    with h5py.File(tmp_path / "big.h5", "w") as f:
        f["x"] = big_source
    with h5py.File(filename, "w") as f:
        f["local"] = local
        layout = h5py.VirtualLayout(shape=(35, 3), dtype=np.float64)
        layout[0:25] = h5py.VirtualSource("big.h5", "x", shape=(100, 3))[40:65]
        layout[25:35] = h5py.VirtualSource(".", "local", shape=(10, 3))
        f.create_virtual_dataset("x", layout)
    _check_vds(filename)


@pytest.mark.parametrize("layout_kind", ["overlapping", "columns", "strided"])
def test_unsupported_vds_layouts_fall_back_to_h5py(tmp_path, layout_kind):
    sources, _ = _make_sources(tmp_path, [20, 20])
    filename = tmp_path / "virtual.h5"
    with h5py.File(filename, "w") as f:
        if layout_kind == "overlapping":
            layout = h5py.VirtualLayout(shape=(30, 3), dtype=np.float64)
            layout[0:20] = h5py.VirtualSource("source.0.h5", "x", shape=(20, 3))
            layout[10:30] = h5py.VirtualSource("source.1.h5", "x", shape=(20, 3))
        elif layout_kind == "columns":
            layout = h5py.VirtualLayout(shape=(20, 6), dtype=np.float64)
            layout[:, 0:3] = h5py.VirtualSource("source.0.h5", "x", shape=(20, 3))
            layout[:, 3:6] = h5py.VirtualSource("source.1.h5", "x", shape=(20, 3))
        else:
            layout = h5py.VirtualLayout(shape=(40, 3), dtype=np.float64)
            layout[0:40:2] = h5py.VirtualSource("source.0.h5", "x", shape=(20, 3))
            layout[1:40:2] = h5py.VirtualSource("source.1.h5", "x", shape=(20, 3))
        f.create_virtual_dataset("x", layout)
    _check_vds(filename, expect_pyfive=False)


def test_resolve_virtual_source_filename(tmp_path, monkeypatch):
    (tmp_path / "vds").mkdir()
    (tmp_path / "prefix").mkdir()
    virtual = str(tmp_path / "vds" / "virtual.h5")
    for path in [tmp_path / "vds" / "a.h5", tmp_path / "prefix" / "a.h5", tmp_path / "vds" / "b.h5"]:
        path.touch()

    resolve = hdf_bulk_read._resolve_virtual_source_filename
    monkeypatch.delenv("HDF5_VDS_PREFIX", raising=False)
    assert resolve(virtual, ".") == virtual
    assert resolve(virtual, "a.h5") == str(tmp_path / "vds" / "a.h5")
    assert resolve(virtual, "missing.h5") is None
    # an absolute name that does not exist is looked for by its final component
    assert resolve(virtual, "/no/such/directory/b.h5") == str(tmp_path / "vds" / "b.h5")

    monkeypatch.setenv("HDF5_VDS_PREFIX", str(tmp_path / "prefix"))
    assert resolve(virtual, "a.h5") == str(tmp_path / "prefix" / "a.h5")
    monkeypatch.setenv("HDF5_VDS_PREFIX", "${ORIGIN}")
    assert resolve(virtual, "a.h5") == str(tmp_path / "vds" / "a.h5")
