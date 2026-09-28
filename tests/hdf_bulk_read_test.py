import os
import shutil
import zlib

import h5py
import numpy as np
import pytest

from pynbody.util import hdf_bulk_read


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
@pytest.mark.parametrize("dtype", ["<f8", "<f4", "<i8", ">f8", "u1", "<i2", ">u2"])
@pytest.mark.parametrize("shape, chunks", [((1000,), (64,)), ((1000,), (1000,)), ((300, 3), (64, 3)),
                                           ((300, 3), (50, 1)), ((301, 3), (300, 2))])
def test_chunked_matches_h5py(tmp_path, filters, dtype, shape, chunks):
    rng = np.random.default_rng(1234)
    data = (rng.random(shape) * 200).astype(dtype)
    filename = tmp_path / "chunked.h5"
    _make_chunked(filename, data, chunks, filters)
    _check_reader_matches_h5py(filename, expected_type=hdf_bulk_read.datasets._ChunkedReader)


@pytest.mark.parametrize("pipeline", [(2, 1, 3), (3, 2, 1), (2, 1)], ids=lambda f: "filters-" + "-".join(map(str, f)))
def test_chunks_skipping_filters(tmp_path, pipeline):
    """Chunks written with some of the pipeline's filters skipped (as recorded in each chunk's filter mask)"""
    rng = np.random.default_rng(7)
    data = rng.random((800, 3)).astype("<f4")
    filename = tmp_path / "masked.h5"
    _make_chunked(filename, np.zeros_like(data), (100, 3), pipeline)
    with h5py.File(filename, "a") as f:
        for i, row in enumerate(range(0, 800, 100)):
            mask = i % (1 << len(pipeline))  # every combination of skipped filters
            chunk = data[row:row + 100].tobytes()
            for position, filter_id in enumerate(pipeline):  # apply, in order, the filters not skipped
                if mask & (1 << position):
                    continue
                if filter_id == 1:
                    chunk = zlib.compress(chunk)
                elif filter_id == 2:
                    n = len(chunk) // 4
                    chunk = np.frombuffer(chunk[:n * 4], np.uint8).reshape(n, 4).T.tobytes() + chunk[n * 4:]
                elif filter_id == 3:
                    chunk += hdf_bulk_read.fletcher32(chunk).to_bytes(4, "little")
            f["x"].id.write_direct_chunk((row, 0), chunk, filter_mask=mask)
    with h5py.File(filename, "r") as f:
        np.testing.assert_array_equal(f["x"][:], data)  # (the chunks are as HDF5 would have written them)
    _check_reader_matches_h5py(filename, expected_type=hdf_bulk_read.datasets._ChunkedReader)


def test_contiguous_matches_h5py(tmp_path):
    filename = tmp_path / "contiguous.h5"
    with h5py.File(filename, "w") as f:
        f["x"] = np.arange(3000, dtype=np.float64).reshape(1000, 3)
    _check_reader_matches_h5py(filename, expected_type=hdf_bulk_read.datasets._ContiguousReader)


@pytest.mark.parametrize("words", [[0xFFFF], [0xFFFF, 0x0000], [0x1234, 0xEDCB], [0, 0, 0]])
def test_fletcher32_end_around_carry(tmp_path, words):
    """HDF5 keeps each fletcher32 sum in [1, 0xFFFF] by end-around carry, so a nonzero sum divisible by 65535 is
    stored as 0xFFFF, not 0. Reducing modulo 65535 instead (as pyfive 1.2.1 does) wrongly rejects such chunks."""
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
        # pynbody notices, and leaves it to HDF5 to decide what to do, which is to raise an error
        with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="fletcher32"):
            with pytest.raises(OSError):
                wrapped[:]


@pytest.mark.parametrize("element_size", [2, 4, 8])
@pytest.mark.parametrize("kind", ["random", "zeros", "ones", "one-nonzero", "multiple-of-65535"])
@pytest.mark.parametrize("num_elements", [1, 2, 511, 1024, 5000])
def test_fletcher32_of_planes_matches_unshuffled(element_size, kind, num_elements):
    rng = np.random.default_rng(num_elements)
    nbytes = element_size * num_elements
    if kind == "random":
        data = rng.integers(0, 256, nbytes, dtype=np.uint8)
    elif kind == "zeros":
        data = np.zeros(nbytes, dtype=np.uint8)
    elif kind == "ones":
        data = np.full(nbytes, 0xFF, dtype=np.uint8)
    elif kind == "one-nonzero":
        data = np.zeros(nbytes, dtype=np.uint8)
        data[nbytes // 2] = 7
    else:  # the words 0x0001, 0xfffe, ... sum to multiples of 65535
        data = np.tile(np.array([0x00, 0x01, 0xFF, 0xFE], dtype=np.uint8), nbytes // 4 + 1)[:nbytes]
    planes = np.ascontiguousarray(data.reshape(num_elements, element_size).T)
    assert hdf_bulk_read.decode._fletcher32_of_planes(planes) == hdf_bulk_read.fletcher32(data)


@pytest.mark.parametrize("block_rows", [1, 3, 65536])
def test_index_weighted_sums_are_exact(monkeypatch, block_rows):
    monkeypatch.setattr(hdf_bulk_read.decode, "_column_sum_block_rows", block_rows)
    rng = np.random.default_rng(1)
    for values in [rng.integers(0, 65536, 10 * 1024 + 17, dtype=np.uint16), np.full(5000, 0xFFFE, dtype=np.uint16),
                   rng.integers(0, 256, 3000, dtype=np.uint8), np.zeros(0, dtype=np.uint16)]:
        exact = values.astype(np.int64)
        index = np.arange(len(values), dtype=np.int64)
        total, index_weighted = hdf_bulk_read.decode._sum_and_index_weighted_sum(values)
        assert total == int(exact.sum())
        assert index_weighted == int((index * exact).sum()) % 65535


@pytest.mark.parametrize("dtype", ["<i2", "<f4", "<f8"])
@pytest.mark.parametrize("where", ["data", "checksum"])
def test_corrupt_checksum_is_detected_before_shuffle(tmp_path, dtype, where):
    """With fletcher32 applied before the shuffle (as SWIFT sometimes writes), the checksum is verified on the
    shuffled data; corrupting either the data or the checksum itself is noticed"""
    filename = tmp_path / "corrupt.h5"
    data = np.arange(1000).astype(dtype)
    _make_chunked(filename, data, (1000,), (3, 2))  # no compression, so that bytes can be corrupted in place
    with h5py.File(filename, "r") as f:
        info = f["x"].id.get_chunk_info(0)
    position = info.byte_offset + (10 if where == "data" else info.size - 2)
    with open(filename, "r+b") as f:
        f.seek(position)
        byte = f.read(1)
        f.seek(position)
        f.write(bytes([byte[0] ^ 0x5A]))
    with h5py.File(filename, "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="fletcher32"):
            with pytest.raises(OSError):
                wrapped[:]


def test_unallocated_chunks_read_as_fillvalue(tmp_path):
    filename = tmp_path / "sparse.h5"
    with h5py.File(filename, "w") as f:
        dataset = f.create_dataset("x", shape=(1000,), dtype=np.int32, chunks=(100,), fillvalue=-7,
                                   compression="gzip")
        dataset[250:420] = np.arange(170)
    _check_reader_matches_h5py(filename, expected_type=hdf_bulk_read.datasets._ChunkedReader)


def test_chunk_cache_avoids_repeated_decoding(tmp_path, monkeypatch):
    filename = tmp_path / "bigchunk.h5"
    data = np.arange(10000, dtype=np.float64)
    _make_chunked(filename, data, (10000,), (2, 1))

    decodes = []
    original_decode = hdf_bulk_read.decode.decode_chunk
    monkeypatch.setattr(hdf_bulk_read.decode, "decode_chunk", lambda *args, **kwargs: decodes.append(1) or original_decode(*args, **kwargs))

    with h5py.File(filename, "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        result = np.concatenate([wrapped[i:i + 700] for i in range(0, 10000, 700)])
    np.testing.assert_array_equal(result, data)
    assert len(decodes) == 1

    decodes.clear()
    with h5py.File(filename, "r") as f:
        wrapped = hdf_bulk_read.BulkReader(cache_nbytes=1000).open(f["x"])  # too small to hold the chunk...
        wrapped[0:700]
        wrapped[700:1400]
    assert len(decodes) == 1  # ...which is kept all the same, alone, for the next read


def test_chunk_cache_empties_as_reads_consume_chunks(tmp_path):
    """Reads in increasing order leave nothing in the cache once they are past a chunk"""
    filename = tmp_path / "chunks.h5"
    data = np.arange(10000, dtype=np.float64)
    _make_chunked(filename, data, (1000,), (2, 1))
    with h5py.File(filename, "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        for i in range(0, 10000, 700):
            np.testing.assert_array_equal(wrapped[i:i + 700], data[i:i + 700])
            assert len(wrapped._cache) <= 1
        assert len(wrapped._cache) == 0


@pytest.mark.parametrize("lookups_before_indexing", [0, 8, 10 ** 9], ids=["index", "hybrid", "one-by-one"])
@pytest.mark.parametrize("prepared", [False, True])
def test_chunk_locations(tmp_path, monkeypatch, lookups_before_indexing, prepared):
    """Chunks are found alike whether looked up one by one or through an index of all of them, including chunks
    never written, and along trailing axes"""
    monkeypatch.setattr(hdf_bulk_read.datasets, "_chunk_lookups_before_indexing", lookups_before_indexing)
    filename = tmp_path / "chunks.h5"
    with h5py.File(filename, "w") as f:
        dataset = f.create_dataset("x", shape=(1000, 5), dtype="f4", chunks=(30, 2), compression="gzip",
                                   fillvalue=-1)
        dataset[0:400] = np.arange(2000, dtype="f4").reshape(400, 5)
        dataset[700:1000, 2:4] = 7  # leaving rows 400 to 700 unwritten, and parts of rows 700 onwards
    with h5py.File(filename, "r") as f:
        expected = f["x"][:]
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        pieces = [(0, 5), (5, 333), (333, 690), (690, 1000)]
        if prepared:
            for start, stop in pieces:
                wrapped.prepare(start, stop)
        for start, stop in pieces:
            np.testing.assert_array_equal(wrapped[start:stop], expected[start:stop])
        assert bool(wrapped._chunk_index) == (lookups_before_indexing < 10 ** 9)


def test_chunk_locations_without_chunk_iter(tmp_path, monkeypatch):
    """Where h5py or HDF5 cannot iterate over chunks, they are looked up one by one"""
    monkeypatch.setattr(hdf_bulk_read.datasets, "_chunk_lookups_before_indexing", 0)
    filename = tmp_path / "chunks.h5"
    data = np.arange(3000.0)
    _make_chunked(filename, data, (100,), (2, 1))

    class WithoutChunkIter:
        def __init__(self, dataset_id):
            self._id = dataset_id

        def __getattr__(self, name):
            if name == "chunk_iter":
                raise AttributeError(name)
            return getattr(self._id, name)

    class Dataset:
        def __init__(self, dataset):
            self._dataset, self.id = dataset, WithoutChunkIter(dataset.id)

        def __getattr__(self, name):
            return getattr(self._dataset, name)

    with h5py.File(filename, "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        wrapped._dataset = Dataset(f["x"])
        np.testing.assert_array_equal(wrapped[:], data)
        assert wrapped._chunk_index is False and len(wrapped._chunk_info) == 30


def _make_file_with_datasets_to_read_through_h5py(filename, external_filename):
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


@pytest.mark.parametrize("name, warns", [("compound", False), ("scaleoffset", True), ("external", True),
                                         ("never_written", False), ("compact", False), ("scalar", False)])
def test_falls_back_to_h5py(tmp_path, name, warns, recwarn):
    filename = tmp_path / "fallback.h5"
    _make_file_with_datasets_to_read_through_h5py(filename, tmp_path / "external.bin")
    with h5py.File(filename, "r") as f:
        assert isinstance(hdf_bulk_read.BulkReader().open(f[name]), h5py.Dataset)
        assert not isinstance(hdf_bulk_read.BulkReader().open(f["fine"]), h5py.Dataset)
    fallback_warnings = [w for w in recwarn if issubclass(w.category, hdf_bulk_read.BulkReadFallbackWarning)]
    assert len(fallback_warnings) == (1 if warns else 0)


def test_fallback_warns_once_per_reason(tmp_path):
    filename = tmp_path / "lzf.h5"
    with h5py.File(filename, "w") as f:
        for name in "abc":
            f.create_dataset(name, data=np.arange(100.0), chunks=(10,), compression="lzf")
    reader = hdf_bulk_read.BulkReader()
    with h5py.File(filename, "r") as f:
        with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="lzf") as record:
            for name in "abc":
                np.testing.assert_array_equal(reader.open(f[name])[:], np.arange(100.0))
    assert len(record) == 1


def test_disabled_reader_uses_h5py(tmp_path):
    filename = tmp_path / "fine.h5"
    with h5py.File(filename, "w") as f:
        f["x"] = np.arange(10)
    with h5py.File(filename, "r") as f:
        reader = hdf_bulk_read.BulkReader(enabled=False)
        assert not reader.enabled
        assert isinstance(reader.open(f["x"]), h5py.Dataset)


def test_file_open_for_writing_uses_h5py(tmp_path, recwarn):
    filename = tmp_path / "fine.h5"
    with h5py.File(filename, "w") as f:
        f["x"] = np.arange(10)
    with h5py.File(filename, "r+") as f:
        assert isinstance(hdf_bulk_read.BulkReader().open(f["x"]), h5py.Dataset)
    assert len(recwarn) == 0


def test_other_driver_uses_h5py(tmp_path):
    filename = tmp_path / "fine.h5"
    with h5py.File(filename, "w") as f:
        f["x"] = np.arange(10)
    with h5py.File(filename, "r", driver="core") as f:
        with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="core"):
            assert isinstance(hdf_bulk_read.BulkReader().open(f["x"]), h5py.Dataset)


@pytest.mark.skipif(os.name == "nt", reason="a file that is open cannot be replaced on Windows")
def test_replaced_file_uses_h5py(tmp_path):
    filename = tmp_path / "original.h5"
    with h5py.File(filename, "w") as f:
        f["x"] = np.arange(10.0)
    with h5py.File(tmp_path / "impostor.h5", "w") as f:
        f["x"] = np.arange(10.0) * -1
    with h5py.File(filename, "r") as f:
        os.replace(tmp_path / "impostor.h5", filename)
        with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="not the one HDF5 has open"):
            wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        np.testing.assert_array_equal(wrapped[:], np.arange(10.0))


def test_nonstandard_datatype_uses_h5py(tmp_path):
    """A 32-bit integer type with only 24 bits of precision is read by h5py as int32, but its bytes are not those
    of an int32 (HDF5 masks the padding bits), so must not be read directly"""
    filename = tmp_path / "odd.h5"
    odd_type = h5py.h5t.STD_I32LE.copy()
    odd_type.set_precision(24)
    with h5py.File(filename, "w") as f:
        dataset_id = h5py.h5d.create(f.id, b"x", odd_type, h5py.h5s.create_simple((10,)))
        dataset_id.write(h5py.h5s.ALL, h5py.h5s.ALL, np.arange(10, dtype=np.int32))
    with h5py.File(filename, "r") as f:
        assert f["x"].dtype == np.int32
        with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="not stored in the standard way"):
            assert isinstance(hdf_bulk_read.BulkReader().open(f["x"]), h5py.Dataset)


def test_userblock(tmp_path):
    filename = tmp_path / "userblock.h5"
    with h5py.File(filename, "w", userblock_size=1024) as f:
        f["x"] = np.arange(300.0).reshape(100, 3)
        f.create_dataset("y", data=np.arange(300.0).reshape(100, 3), chunks=(10, 3), compression="gzip")
    _check_reader_matches_h5py(filename, "x", hdf_bulk_read.datasets._ContiguousReader)
    _check_reader_matches_h5py(filename, "y", hdf_bulk_read.datasets._ChunkedReader)


def test_unexpected_data_falls_back_to_h5py(tmp_path, monkeypatch):
    filename = tmp_path / "fine.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("x", data=np.arange(1000.0), chunks=(100,), compression="gzip")

    def short_read(*args, **kwargs):
        raise hdf_bulk_read.common._UnexpectedData("the file supplied too few bytes")

    with h5py.File(filename, "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        np.testing.assert_array_equal(wrapped[0:10], np.arange(10.0))
        monkeypatch.setattr(hdf_bulk_read.files, "_read_bytes", short_read)
        with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="too few bytes"):
            np.testing.assert_array_equal(wrapped[500:700], np.arange(500.0, 700.0))
        # and from then on the dataset is read through h5py, without further warnings
        np.testing.assert_array_equal(wrapped[800:900], np.arange(800.0, 900.0))


@pytest.mark.parametrize("chunked", [False, True])
def test_narrowing_conversion_is_left_to_hdf5(tmp_path, chunked):
    """HDF5 clips out-of-range integers when converting, where numpy would wrap them round"""
    filename = tmp_path / "wide.h5"
    data = np.array([1, 2 ** 40, -2 ** 40, 3], dtype=np.int64)
    with h5py.File(filename, "w") as f:
        f.create_dataset("x", data=data, chunks=(2,) if chunked else None)
    with h5py.File(filename, "r") as f:
        expected = np.empty(4, dtype=np.int32)
        f["x"].read_direct(expected)
        got = np.empty(4, dtype=np.int32)
        hdf_bulk_read.BulkReader().open(f["x"]).read_direct(got)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("chunked", [False, True])
def test_widening_conversion(tmp_path, chunked):
    filename = tmp_path / "narrow.h5"
    data = np.linspace(0, 1, 100, dtype=np.float32)
    with h5py.File(filename, "w") as f:
        f.create_dataset("x", data=data, chunks=(30,) if chunked else None)
    with h5py.File(filename, "r") as f:
        got = np.empty(50, dtype=np.float64)
        hdf_bulk_read.BulkReader().open(f["x"]).read_direct(got, source_sel=np.s_[20:70])
    np.testing.assert_array_equal(got, data[20:70].astype(np.float64))


@pytest.mark.parametrize("filters", [(2, 1), (3, 2, 1), (1,)], ids=lambda f: "filters-" + "-".join(map(str, f)))
@pytest.mark.parametrize("chunks", [(64, 3), (50, 2)])
def test_chunks_into_converting_or_strided_destinations(tmp_path, filters, chunks):
    """Rows copied out of a chunk (still shuffled or not) into destinations that are not laid out like the chunk:
    another dtype, or a strided view"""
    rng = np.random.default_rng(5)
    data = (rng.random((300, 3)) * 200).astype("<f4")
    filename = tmp_path / "chunked.h5"
    _make_chunked(filename, data, chunks, filters)
    with h5py.File(filename, "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        widened = np.empty((190, 3), dtype=np.float64)
        wrapped.read_direct(widened, source_sel=np.s_[33:223])
        np.testing.assert_array_equal(widened, data[33:223].astype(np.float64))
        backing = np.zeros((190, 6), dtype="<f4")
        strided = backing[:, ::2]
        wrapped.read_direct(strided, source_sel=np.s_[33:223])
        np.testing.assert_array_equal(strided, data[33:223])
        assert not backing[:, 1::2].any()


def test_contiguous_conversion_is_done_in_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(hdf_bulk_read.datasets, "_conversion_block_nbytes", 100)
    data = np.arange(3000, dtype="<f4").reshape(1000, 3)
    filename = tmp_path / "contiguous.h5"
    with h5py.File(filename, "w") as f:
        f["x"] = data
    reads = []
    original_read = hdf_bulk_read.files._read_bytes
    monkeypatch.setattr(hdf_bulk_read.files, "_read_bytes",
                        lambda file, offset, nbytes, into=None: reads.append(nbytes) or
                        original_read(file, offset, nbytes, into))
    with h5py.File(filename, "r") as f:
        got = np.empty((990, 3), dtype=np.float64)
        hdf_bulk_read.BulkReader().open(f["x"]).read_direct(got, source_sel=np.s_[7:997])
    np.testing.assert_array_equal(got, data[7:997])
    assert max(reads) <= 100 and sum(reads) == 990 * 12


@pytest.mark.parametrize("nbytes", [None, 10, 8000, 8001, 10 ** 6])
def test_decode_chunk_size_hint_is_only_a_hint(nbytes):
    raw = zlib.compress(np.arange(1000, dtype=np.float64).tobytes())
    decoded = hdf_bulk_read.decode.decode_chunk(raw, 0, [{'filter_id': 1}], 8, nbytes=nbytes)
    np.testing.assert_array_equal(decoded.view(np.float64), np.arange(1000.0))


@pytest.mark.parametrize("filters", [(2, 1), (3, 2, 1), (2, 1, 3)], ids=lambda f: "filters-" + "-".join(map(str, f)))
def test_decoding_memory_is_bounded(tmp_path, filters):
    """Decoding a chunk needs little more memory than the chunk takes compressed and decompressed: in particular,
    no buffer grown by doubling, and no second copy of the whole chunk for unshuffling"""
    import tracemalloc
    rng = np.random.default_rng(2)
    data = rng.random(4_000_000)  # 32 MB in one chunk, compressing poorly
    filename = tmp_path / "bigchunk.h5"
    _make_chunked(filename, data, data.shape, filters)
    with h5py.File(filename, "r") as f:
        compressed = f["x"].id.get_storage_size()
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        out = np.empty_like(data)
        tracemalloc.start()
        try:
            wrapped.read_direct(out)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    np.testing.assert_array_equal(out, data)
    assert peak < compressed + data.nbytes + 2 * 1024 * 1024


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


def _check_vds(filename, expect_direct=True):
    reader = hdf_bulk_read.BulkReader()
    with h5py.File(filename, "r") as f:
        dataset = f["x"]
        wrapped = reader.open(dataset)
        if expect_direct:
            assert isinstance(wrapped, hdf_bulk_read.virtual._VirtualReader)
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


def test_vds_sources_are_read_directly(tmp_path):
    row_counts = [45, 30, 70]
    sources, arrays = _make_sources(tmp_path, row_counts)
    _make_vds(tmp_path / "virtual.h5", sources, row_counts)
    with h5py.File(tmp_path / "virtual.h5", "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        np.testing.assert_array_equal(wrapped[50:60], arrays[1][5:15])
        opened = [b for b in wrapped._blocks if b.reader_resolved]
        assert len(opened) == 1  # only the source that was needed has been opened
        assert isinstance(opened[0].reader, hdf_bulk_read.datasets._ChunkedReader)


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


def test_vds_divide(tmp_path):
    """A virtual dataset divides reading its rows into parts read from each source (with rows counted in the source),
    from nothing for unmapped rows (which read as the fill value), and through h5py for sources that cannot be read
    directly"""
    row_counts = [10, 20, 15]
    sources, _ = _make_sources(tmp_path, row_counts)
    layout = h5py.VirtualLayout(shape=(60, 3), dtype=np.float64)
    layout[0:10] = h5py.VirtualSource("source.0.h5", "x", shape=(10, 3))
    layout[10:20] = h5py.VirtualSource("source.1.h5", "x", shape=(20, 3))[0:10]
    layout[20:30] = h5py.VirtualSource("source.1.h5", "x", shape=(20, 3))[10:20]
    # rows 30 to 35 are unmapped
    layout[35:50] = h5py.VirtualSource("source.2.h5", "x", shape=(15, 3))
    layout[50:60] = h5py.VirtualSource("missing.h5", "x", shape=(10, 3))
    with h5py.File(tmp_path / "virtual.h5", "w") as f:
        f.create_virtual_dataset("x", layout, fillvalue=-1.0)
    with h5py.File(tmp_path / "virtual.h5", "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        assert isinstance(wrapped, hdf_bulk_read.virtual._VirtualReader)
        destination = np.zeros((60, 3))
        parts = wrapped.divide(slice(0, 60), destination)
        kinds = [type(source).__name__ for source, _, _, _ in parts]
        assert kinds == ["_ChunkedReader", "_ChunkedReader", "_ChunkedReader", "_Fill", "_ChunkedReader", "Dataset"]
        assert [rows for _, rows, _, _ in parts] == [slice(0, 10), slice(0, 10), slice(10, 20), slice(30, 35),
                                                     slice(0, 15), slice(50, 60)]
        assert [os.path.basename(name) for _, _, _, name in parts] == \
            ["source.0.h5", "source.1.h5", "source.1.h5", "virtual.h5", "source.2.h5", "virtual.h5"]
        assert [len(part) for _, _, part, _ in parts] == [10, 10, 10, 5, 15, 10]
        for _, _, part, _ in parts:
            assert np.shares_memory(part, destination)
        # index selections are divided alike, keeping only the parts with rows selected
        parts = wrapped.divide(np.array([3, 12, 31, 36, 37]), np.zeros((5, 3)))
        assert [(type(source).__name__, list(rows), len(part)) for source, rows, part, _ in parts] == \
            [("_ChunkedReader", [3], 1), ("_ChunkedReader", [2], 1), ("_Fill", [31], 1), ("_ChunkedReader", [1, 2], 2)]
        expected = f["x"][:]
        hdf_bulk_read.BulkReader().read([hdf_bulk_read.ReadRequest(f["x"], slice(0, 60), destination)])
    np.testing.assert_array_equal(destination, expected)


def test_vds_missing_source(tmp_path):
    row_counts = [10, 20, 15]
    sources, _ = _make_sources(tmp_path, row_counts)
    _make_vds(tmp_path / "virtual.h5", sources, row_counts, fillvalue=-2.0)
    os.remove(sources[1])
    _check_vds(tmp_path / "virtual.h5")


def test_vds_needing_conversion_reads_sources_through_h5py(tmp_path):
    row_counts = [10, 20]
    sources, _ = _make_sources(tmp_path, row_counts, dtype=np.float32)
    _make_vds(tmp_path / "virtual.h5", sources, row_counts, dtype=np.float64)
    with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="different datatype"):
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
    with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="virtual dataset layout"):
        _check_vds(filename, expect_direct=False)


def test_hdf5_absolute_names(monkeypatch):
    """Absolute source names are recognised by HDF5's rule, whatever os.path.isabs says on this Python version"""
    monkeypatch.setattr(hdf_bulk_read.files.os, "name", "nt")
    assert hdf_bulk_read.virtual._hdf5_considers_absolute("C:\\data\\x.h5")
    assert hdf_bulk_read.virtual._hdf5_considers_absolute("d:/data/x.h5")
    assert not hdf_bulk_read.virtual._hdf5_considers_absolute("/data/x.h5")
    assert not hdf_bulk_read.virtual._hdf5_considers_absolute("C:x.h5")
    assert not hdf_bulk_read.virtual._hdf5_considers_absolute("x.h5")
    monkeypatch.setattr(hdf_bulk_read.files.os, "name", "posix")
    assert hdf_bulk_read.virtual._hdf5_considers_absolute("/data/x.h5")
    assert not hdf_bulk_read.virtual._hdf5_considers_absolute("C:/data/x.h5")
    assert not hdf_bulk_read.virtual._hdf5_considers_absolute("x.h5")


def test_resolve_virtual_source_filename(tmp_path, monkeypatch):
    (tmp_path / "vds").mkdir()
    (tmp_path / "elsewhere").mkdir()
    virtual = str(tmp_path / "vds" / "virtual.h5")
    for path in [tmp_path / "vds" / "a.h5", tmp_path / "vds" / "b.h5", tmp_path / "elsewhere" / "c.h5"]:
        path.touch()

    resolve = hdf_bulk_read.virtual._resolve_virtual_source_filename
    assert resolve(virtual, ".") == virtual
    assert resolve(virtual, "a.h5") == str(tmp_path / "vds" / "a.h5")
    assert resolve(virtual, "missing.h5") is None
    # an absolute name (by HDF5's rules) that does not exist is looked for by its final component
    missing_absolute = "C:/no/such/directory/b.h5" if os.name == "nt" else "/no/such/directory/b.h5"
    assert resolve(virtual, missing_absolute) == str(tmp_path / "vds" / "b.h5")
    if os.name == "nt":
        # names HDF5 does not consider absolute, but Windows might reinterpret, are left to HDF5
        for ambiguous in ["/no/such/directory/b.h5", "\\\\server\\share\\b.h5", "C:b.h5"]:
            with pytest.raises(hdf_bulk_read.common._CannotReadDirectly):
                resolve(virtual, ambiguous)
    # and, last of all, a name relative to the current directory, which is returned as an absolute path
    monkeypatch.chdir(tmp_path / "elsewhere")
    assert resolve(virtual, "c.h5") == str(tmp_path / "elsewhere" / "c.h5")


# ---------------------------------------------------------------------------------------------------------------------
# Regressions found by adversarial review


def _libhdf5():
    """The libhdf5 h5py was built against, for properties h5py cannot set, or None if it cannot be found"""
    import ctypes
    import glob
    candidates = glob.glob(os.path.join(os.path.dirname(h5py.__file__) + ".libs", "libhdf5-*.so*"))
    return ctypes.CDLL(candidates[0]) if candidates else None


needs_libhdf5 = pytest.mark.skipif(_libhdf5() is None, reason="cannot locate libhdf5 to set properties h5py cannot")


@needs_libhdf5
@pytest.mark.parametrize("filters", [(2,), (1,), (3,), (2, 1)])
def test_unfiltered_partial_edge_chunks(tmp_path, recwarn, filters):
    """HDF5 can store partial edge chunks unfiltered (H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) while reporting a filter
    mask of 0, and h5py cannot tell us so. With shuffle alone, decoding such a chunk gives plausible garbage."""
    import ctypes
    lib = _libhdf5()
    lib.H5Pset_chunk_opts.argtypes = [ctypes.c_int64, ctypes.c_uint]
    data = np.arange(103, dtype="<i4") * 1000003
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((10,))
    for filter_id in filters:
        {1: lambda: dcpl.set_deflate(4), 2: dcpl.set_shuffle, 3: dcpl.set_fletcher32}[filter_id]()
    assert lib.H5Pset_chunk_opts(dcpl.id, 0x0002) >= 0
    filename = tmp_path / "edge.h5"
    with h5py.File(filename, "w") as f:
        dataset_id = h5py.h5d.create(f.id, b"x", h5py.h5t.STD_I32LE, h5py.h5s.create_simple((103,)), dcpl=dcpl)
        dataset_id.write(h5py.h5s.ALL, h5py.h5s.ALL, data)
    _check_reader_matches_h5py(filename, expected_type=hdf_bulk_read.datasets._ChunkedReader)
    assert len(recwarn) == 0  # the edge chunk is read through h5py, and the rest directly, without complaint


@pytest.mark.skipif(os.name == "nt", reason="a file that is open cannot be replaced on Windows")
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("kept_open", [True, False])
def test_file_replaced_after_planning(tmp_path, monkeypatch, recwarn, chunked, kept_open):
    """Replacing a file on disk after planning never changes what is read: a file kept open is still the file HDF5
    has open, and a file opened for each read is checked to be it"""
    if not kept_open:
        monkeypatch.setattr(hdf_bulk_read.files, "_max_files_kept_open", lambda: 0)
    filename = tmp_path / "original.h5"
    for name, sign in [("original.h5", 1), ("impostor.h5", -1)]:
        with h5py.File(tmp_path / name, "w") as f:
            f.create_dataset("x", data=np.arange(100.0) * sign, chunks=(10,) if chunked else None)
    with h5py.File(filename, "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        os.replace(tmp_path / "impostor.h5", filename)
        np.testing.assert_array_equal(wrapped[10:14], np.arange(10.0, 14.0))
    fallbacks = [w for w in recwarn if issubclass(w.category, hdf_bulk_read.BulkReadFallbackWarning)]
    if kept_open:
        assert fallbacks == []
    else:
        assert len(fallbacks) == 1 and "has been replaced" in str(fallbacks[0].message)


@pytest.mark.parametrize("failures", ["first", "all"])
def test_files_that_cannot_be_opened(tmp_path, monkeypatch, recwarn, failures):
    """If pynbody cannot open a file itself (for instance, because the process has too many files open), it opens
    the file for each read instead, or failing that reads through h5py; it never gives up on the read"""
    import errno
    data = np.arange(3000.0).reshape(1000, 3)
    filename = tmp_path / "x.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("x", data=data, chunks=(100, 3), compression="gzip")
    original_open = os.open
    calls = []

    def failing_open(*args, **kwargs):
        calls.append(args[0])
        if failures == "all" or len(calls) == 1:
            raise OSError(errno.EMFILE, "Too many open files")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(hdf_bulk_read.files.os, "open", failing_open)
    with h5py.File(filename, "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        np.testing.assert_array_equal(wrapped[:], data)
    fallbacks = [w for w in recwarn if issubclass(w.category, hdf_bulk_read.BulkReadFallbackWarning)]
    # (without os.pread, files are opened only as reads need them, so a failure is met while reading, and the
    # dataset is then read through h5py)
    assert len(fallbacks) == (1 if failures == "all" or not hasattr(os, "pread") else 0)


def test_files_are_opened_once(tmp_path, monkeypatch):
    """Where positioned reads are available, each file is opened once however many reads are made"""
    if not hasattr(os, "pread"):
        pytest.skip("no os.pread on this platform")
    filename = tmp_path / "many_chunks.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("x", data=np.arange(10000.0), chunks=(100,), compression="gzip")
        f["y"] = np.arange(5000.0)
    opens = []
    original_open = os.open
    monkeypatch.setattr(hdf_bulk_read.files.os, "open", lambda *args, **kwargs: opens.append(args[0]) or
                        original_open(*args, **kwargs))
    reader = hdf_bulk_read.BulkReader()
    with h5py.File(filename, "r") as f:
        x, y = reader.open(f["x"]), reader.open(f["y"])
        np.testing.assert_array_equal(np.concatenate([x[i:i + 700] for i in range(0, 10000, 700)]),
                                      np.arange(10000.0))
        np.testing.assert_array_equal(y[:], np.arange(5000.0))
    assert len(opens) == 1  # one file, shared by both datasets and all 100 chunks
    reader.close()


def test_too_many_files_are_opened_per_read(tmp_path, monkeypatch):
    monkeypatch.setattr(hdf_bulk_read.files, "_max_files_kept_open", lambda: 1)
    reader = hdf_bulk_read.BulkReader()
    handles = []
    for i in range(3):
        with h5py.File(tmp_path / f"f{i}.h5", "w") as f:
            f["x"] = np.arange(10.0) + i
    files = [h5py.File(tmp_path / f"f{i}.h5", "r") for i in range(3)]
    try:
        for i, f in enumerate(files):
            wrapped = reader.open(f["x"])
            handles.append(wrapped._file)
            np.testing.assert_array_equal(wrapped[:], np.arange(10.0) + i)
        assert [h._keep_open for h in handles] == [True, False, False]
    finally:
        reader.close()
        for f in files:
            f.close()


def test_vds_opened_by_relative_path_then_chdir(tmp_path, monkeypatch):
    """Sources are relative to the virtual file's directory, which must not be taken from a stale relative path"""
    (tmp_path / "data").mkdir()
    (tmp_path / "elsewhere").mkdir()
    sources, _ = _make_sources(tmp_path / "data", [5, 5], trailing=())
    _make_vds(tmp_path / "data" / "virtual.h5", sources, [5, 5], trailing=())
    # a decoy with the same relative path, but from the other directory
    _make_sources(tmp_path / "elsewhere", [5, 5], trailing=())
    with h5py.File(tmp_path / "elsewhere" / "source.0.h5", "a") as f:
        f["x"][...] *= -1
    shutil.copy(tmp_path / "data" / "virtual.h5", tmp_path / "elsewhere" / "virtual.h5")

    monkeypatch.chdir(tmp_path / "data")
    with h5py.File("virtual.h5", "r") as f:
        monkeypatch.chdir(tmp_path / "elsewhere")
        with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="not the one HDF5 has open"):
            wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        np.testing.assert_array_equal(wrapped[:], f["x"][:])


@pytest.mark.parametrize("prefix", ["${ORIGIN}/sub", "${ORIGIN}", "absolute"])
def test_vds_prefix_is_left_to_hdf5(tmp_path, monkeypatch, prefix):
    (tmp_path / "sub").mkdir()
    sources, _ = _make_sources(tmp_path, [5], trailing=())
    _make_sources(tmp_path / "sub", [5], trailing=(), dtype=np.float64)
    _make_vds(tmp_path / "virtual.h5", sources, [5], trailing=())
    monkeypatch.setenv("HDF5_VDS_PREFIX", str(tmp_path / "sub") if prefix == "absolute" else prefix)
    with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="search path"):
        _check_vds(tmp_path / "virtual.h5", expect_direct=False)


@needs_libhdf5
def test_undefined_fill_value(tmp_path):
    import ctypes
    lib = _libhdf5()
    lib.H5Pset_fill_value.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((10,))
    assert lib.H5Pset_fill_value(dcpl.id, h5py.h5t.NATIVE_FLOAT.id, None) >= 0
    filename = tmp_path / "undefined_fill.h5"
    with h5py.File(filename, "w") as f:
        h5py.h5d.create(f.id, b"x", h5py.h5t.IEEE_F32LE, h5py.h5s.create_simple((100,)), dcpl=dcpl)
        f["x"][:] = np.arange(100)
    with h5py.File(filename, "r") as f:
        with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="no defined value"):
            wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        np.testing.assert_array_equal(wrapped[:], np.arange(100.0))


def test_fill_time_never(tmp_path):
    """HDF5 leaves the destination untouched for unallocated chunks, rather than writing a fill value"""
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((10,))
    dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
    dcpl.set_fill_value(np.array(-3.0))
    filename = tmp_path / "never.h5"
    with h5py.File(filename, "w") as f:
        h5py.h5d.create(f.id, b"x", h5py.h5t.IEEE_F64LE, h5py.h5s.create_simple((100,)), dcpl=dcpl)
        f["x"][0:10] = np.arange(10.0)
    with h5py.File(filename, "r") as f:
        with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="no defined value"):
            wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        expected = np.full(10, 7.0)
        f["x"].read_direct(expected, source_sel=np.s_[50:60])
        got = np.full(10, 7.0)
        wrapped.read_direct(got, source_sel=np.s_[50:60])
    np.testing.assert_array_equal(got, expected)


def test_vds_with_unlimited_mapping(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for i in range(3):
        with h5py.File(f"s{i}.h5", "w") as f:
            f["d"] = np.arange(10.0) + 100 * i
    unlimited = h5py.h5s.UNLIMITED
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    virtual_space = h5py.h5s.create_simple((30,), (unlimited,))
    virtual_space.select_hyperslab((0,), (unlimited,), (10,), (10,))
    dcpl.set_virtual(virtual_space, b"s%b.h5", b"d", h5py.h5s.create_simple((10,)))
    with h5py.File("unlimited.h5", "w", libver="latest") as f:
        h5py.h5d.create(f.id, b"x", h5py.h5t.IEEE_F64LE, h5py.h5s.create_simple((30,), (unlimited,)), dcpl=dcpl)
    with h5py.File("unlimited.h5", "r") as f:
        with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="virtual dataset layout"):
            wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        assert isinstance(wrapped, h5py.Dataset)
        np.testing.assert_array_equal(wrapped[8:12], [8.0, 9.0, 100.0, 101.0])


def test_extended_precision_is_left_to_hdf5(tmp_path):
    filename = tmp_path / "longdouble.h5"
    with h5py.File(filename, "w") as f:
        f["x"] = np.arange(10, dtype=np.longdouble)
    with h5py.File(filename, "r") as f:
        if f["x"].dtype.itemsize not in (1, 2, 4, 8):
            with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="extended-precision"):
                wrapped = hdf_bulk_read.BulkReader().open(f["x"])
            np.testing.assert_array_equal(wrapped[2:5], f["x"][2:5])


@pytest.mark.parametrize("from_dtype, to_dtype", [(">f4", "<f8"), ("<f4", ">f8"), ("<f2", "<f4"), ("<f2", "<f8"),
                                                  ("<f4", "<f8"), (">i2", "<i8"), ("<i4", "<f8")])
@pytest.mark.parametrize("chunked", [False, True])
def test_widening_conversions_match_hdf5_bitwise(tmp_path, from_dtype, to_dtype, chunked):
    """HDF5 replaces NaNs with a canonical NaN when it converts byte-swapped or half-precision floats"""
    if np.dtype(from_dtype).kind == "f":
        values = np.array([1.5, np.nan, -np.inf, 0.0, -0.0, 3.0], dtype=from_dtype)
        values.view(f"u{values.itemsize}")[1] |= 1  # a NaN with a nonzero payload
    else:
        values = np.array([1, -2, 3, 32767, -32768, 0], dtype=from_dtype)
    filename = tmp_path / "convert.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("x", data=values, chunks=(2,) if chunked else None)
    with h5py.File(filename, "r") as f:
        expected = np.zeros(6, dtype=to_dtype)
        f["x"].read_direct(expected)
        got = np.zeros(6, dtype=to_dtype)
        with np.errstate(invalid="raise"):
            hdf_bulk_read.BulkReader().open(f["x"]).read_direct(got)
    assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("from_dtype, to_dtype, exact", [("<i4", "<f8", True), ("<u2", "<f4", True),
                                                        ("<i2", "<f4", True), ("<i8", "<f8", False),
                                                        ("<u8", "<f8", False), ("<i4", "<f4", False),
                                                        ("<i4", "<i8", True), ("<f4", "<f8", True)])
def test_conversion_is_exact(from_dtype, to_dtype, exact):
    """Conversions numpy does itself must be exact for every value: integers too large for the float's mantissa
    are left to HDF5"""
    assert hdf_bulk_read.datasets._conversion_is_exact(np.dtype(from_dtype), np.dtype(to_dtype)) == exact


def test_unshuffle_writes_in_place():
    """Unshuffled bytes land in the destination given, even a strided one (never in a copy of it)"""
    elements = np.arange(40, dtype=np.uint8).reshape(10, 4)
    planes = np.ascontiguousarray(elements.T)
    backing = np.zeros(80, dtype=np.uint8)
    hdf_bulk_read.decode._unshuffle_planes_into(planes, backing[::2])
    np.testing.assert_array_equal(backing[::2], elements.reshape(-1))
    assert not backing[1::2].any()


def test_one_reader_shared_between_threads(tmp_path):
    import concurrent.futures
    filename = tmp_path / "shared.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("x", data=np.arange(20000.0), chunks=(100,))
    with h5py.File(filename, "r") as f:
        wrapped = hdf_bulk_read.BulkReader(cache_nbytes=3 * 800).open(f["x"])

        def job(seed):
            rng = np.random.default_rng(seed)
            for _ in range(500):
                start = int(rng.integers(0, 50)) * 100 + 50
                np.testing.assert_array_equal(wrapped[start:start + 20], np.arange(start, start + 20.0))

        with concurrent.futures.ThreadPoolExecutor(8) as executor:
            list(executor.map(job, range(8)))


def test_swmr_is_left_to_hdf5(tmp_path, recwarn):
    filename = tmp_path / "swmr.h5"
    with h5py.File(filename, "w", libver="latest") as f:
        f.create_dataset("x", data=np.arange(50.0), maxshape=(None,), chunks=(64,))
    with h5py.File(filename, "r", swmr=True) as f:
        assert isinstance(hdf_bulk_read.BulkReader().open(f["x"]), h5py.Dataset)
    assert len(recwarn) == 0


@pytest.mark.parametrize("how", ["fileobj", "stdio", "core"])
def test_vds_in_file_opened_other_ways(tmp_path, how):
    """The virtual dataset's own file must be located before its sources can be; if the file is open other than
    through the sec2 driver, pynbody cannot confirm where it is, so leaves the virtual dataset to HDF5"""
    sources, _ = _make_sources(tmp_path, [5, 5], trailing=())
    _make_vds(tmp_path / "virtual.h5", sources, [5, 5], trailing=())
    filename = str(tmp_path / "virtual.h5")
    with open(filename, "rb") as raw:
        f = h5py.File(raw, "r") if how == "fileobj" else h5py.File(filename, "r", driver=how)
        with f:
            with pytest.warns(hdf_bulk_read.BulkReadFallbackWarning, match="driver"):
                wrapped = hdf_bulk_read.BulkReader().open(f["x"])
            assert isinstance(wrapped, h5py.Dataset)
            if how != "fileobj":  # reading a virtual dataset through a Python file object crashes h5py 3.16 itself
                np.testing.assert_array_equal(wrapped[:], np.arange(10.0))


def test_prepared_reads_make_no_hdf5_lookups(tmp_path, monkeypatch):
    """After prepare(), reading a virtual dataset opens no files through h5py and looks up no chunks, so that reads
    from several threads do not queue for h5py's lock"""
    row_counts = [30] * 12
    sources, arrays = _make_sources(tmp_path, row_counts)
    _make_vds(tmp_path / "virtual.h5", sources, row_counts)
    expected = np.concatenate(arrays)
    with h5py.File(tmp_path / "virtual.h5", "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        wrapped.prepare(40, 200)

        def forbidden(*args, **kwargs):
            raise AssertionError("an HDF5 lookup was made after prepare()")

        class Forbidden:
            def __getattr__(self, name):
                forbidden()

        monkeypatch.setattr(h5py, "File", forbidden)
        prepared = [b.reader for b in wrapped._blocks if b.reader_resolved]
        assert len(prepared) == 6  # the blocks overlapping rows 40 to 200, and only those
        for reader in prepared:
            reader._dataset = Forbidden()  # through which any chunk lookup would have to go
        np.testing.assert_array_equal(wrapped[40:200], expected[40:200])
        np.testing.assert_array_equal(wrapped[100:150], expected[100:150])


def test_many_source_blocks(tmp_path):
    """Reads touch only the blocks they overlap (checked on a virtual dataset with many small sources)"""
    row_counts = [3] * 400
    sources, arrays = _make_sources(tmp_path, row_counts, trailing=(), compress=False)
    _make_vds(tmp_path / "virtual.h5", sources, row_counts, trailing=())
    expected = np.concatenate(arrays)
    with h5py.File(tmp_path / "virtual.h5", "r") as f:
        wrapped = hdf_bulk_read.BulkReader().open(f["x"])
        np.testing.assert_array_equal(wrapped[601:607], expected[601:607])
        assert sum(b.reader_resolved for b in wrapped._blocks) == 3
        np.testing.assert_array_equal(wrapped[:], expected)


def test_reading_without_pread(tmp_path, monkeypatch):
    """Where os.pread is unavailable (Windows), each thread reads through a handle of its own"""
    import concurrent.futures
    monkeypatch.delattr(hdf_bulk_read.files.os, "pread", raising=False)
    monkeypatch.delattr(hdf_bulk_read.files.os, "preadv", raising=False)
    filename = tmp_path / "no_pread.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("x", data=np.arange(20000.0), chunks=(100,), compression="gzip")
        f["y"] = np.arange(20000.0)
    reader = hdf_bulk_read.BulkReader()
    with h5py.File(filename, "r") as f:
        for name in "xy":
            wrapped = reader.open(f[name])

            def job(start):
                np.testing.assert_array_equal(wrapped[start:start + 1000], np.arange(start, start + 1000.0))

            for _ in range(3):  # (a new pool of threads for each array, as the loader uses)
                with concurrent.futures.ThreadPoolExecutor(4) as executor:
                    list(executor.map(job, range(0, 19000, 500)))
        # both datasets share one handle, which never has more files open than there were reads at once
        assert 1 <= len(wrapped._file._all_thread_files) <= 4
    reader.close()
    assert wrapped._file._all_thread_files == []


# --- Requests (BulkReader.read), planning and grouping


def _make_mixed_file(filename):
    rng = np.random.default_rng(3)
    data = {"chunked": rng.random((5000, 3)), "contiguous": rng.random(4000), "compound": np.zeros(100, "f4,i4")}
    with h5py.File(filename, "w") as f:
        f.create_dataset("chunked", data=data["chunked"], chunks=(700, 3), compression="gzip", shuffle=True)
        f["contiguous"] = data["contiguous"]
        f["compound"] = data["compound"]  # (read through h5py)
    return data


@pytest.mark.parametrize("block_nbytes", [16 * 1024 * 1024, 100])
@pytest.mark.parametrize("name", ["chunked", "contiguous"])
@pytest.mark.parametrize("direct", [True, False])
def test_read_requests(tmp_path, monkeypatch, block_nbytes, name, direct):
    """Requests of slices and of index arrays (read a block at a time) fill their destinations, converting dtype"""
    monkeypatch.setattr(hdf_bulk_read.plan, "_gather_block_nbytes", block_nbytes)
    data = _make_mixed_file(tmp_path / "x.h5")[name]
    rng = np.random.default_rng(4)
    selections = [slice(10, 2000), np.sort(rng.choice(len(data), 300, replace=False)), np.arange(50, 120),
                  np.array([7]), slice(3, 3), np.array([], dtype=int)]
    with h5py.File(tmp_path / "x.h5", "r") as f:
        destinations = []
        requests = []
        for rows in selections:
            expected = data[rows]
            for dtype in (data.dtype, np.float32):
                destination = np.zeros(expected.shape, dtype=dtype)
                requests.append(hdf_bulk_read.ReadRequest(f[name], rows, destination))
                destinations.append((destination, expected.astype(dtype)))
        strategy = hdf_bulk_read.BulkReader(enabled=direct).read(iter(requests))
    for destination, expected in destinations:
        np.testing.assert_array_equal(destination, expected)
    assert strategy.threads >= 1


def test_read_requests_through_h5py_and_directly_together(tmp_path, recwarn):
    data = _make_mixed_file(tmp_path / "x.h5")
    with h5py.File(tmp_path / "x.h5", "r") as f:
        a = np.zeros(10, "f4,i4")
        b = np.zeros((20, 3))
        strategy = hdf_bulk_read.BulkReader().read([hdf_bulk_read.ReadRequest(f["compound"], np.arange(0, 20, 2), a),
                                                     hdf_bulk_read.ReadRequest(f["chunked"], slice(100, 120), b)])
    np.testing.assert_array_equal(a, data["compound"][0:20:2])
    np.testing.assert_array_equal(b, data["chunked"][100:120])
    assert strategy.threads == 1 and "h5py" in strategy.reason


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("rows, destination", [
    (slice(0, 10), np.zeros(9)), (np.arange(5), np.zeros(6)), (slice(0, 10, 2), np.zeros(5)),
    (np.array([0, 2, 1, 3]), np.zeros(4)), (np.array([50, 10, 20]), np.zeros(3)), (np.array([1, 1, 2]), np.zeros(3)),
    (np.array([1.7, 2.2]), np.zeros(2)), (np.array([-1, 2]), np.zeros(2)), (np.array([3998, 4000]), np.zeros(2)),
    (np.arange(4).reshape(2, 2), np.zeros(4)), (slice(0, 10), np.zeros(20)[::2]), (slice(0, 10), np.zeros((10, 3))),
    (slice(0, 10), np.frombuffer(bytes(80))), (slice(0, 10), [0.0] * 10)],
    ids=["too-few", "too-many", "step", "unsorted", "decreasing", "repeated", "floats", "negative", "beyond-end",
         "2d-rows", "strided-destination", "destination-shape", "read-only-destination", "list-destination"])
def test_bad_read_requests_are_refused(tmp_path, rows, destination, direct):
    """Requests that cannot be read as they say are refused, whether the dataset would be read directly or not"""
    _make_mixed_file(tmp_path / "x.h5")
    with h5py.File(tmp_path / "x.h5", "r") as f:
        request = hdf_bulk_read.ReadRequest(f["contiguous"], rows, destination)
        with pytest.raises((ValueError, TypeError)):
            hdf_bulk_read.BulkReader(enabled=direct).read([request])


def test_read_request_slices_follow_python_conventions(tmp_path):
    data = _make_mixed_file(tmp_path / "x.h5")["contiguous"]
    with h5py.File(tmp_path / "x.h5", "r") as f:
        a, b = np.zeros(10), np.zeros(len(data) - 3990)
        hdf_bulk_read.BulkReader().read([hdf_bulk_read.ReadRequest(f["contiguous"], slice(None, 10), a),
                                         hdf_bulk_read.ReadRequest(f["contiguous"], slice(3990, None), b)])
    np.testing.assert_array_equal(a, data[:10])
    np.testing.assert_array_equal(b, data[3990:])


def test_one_dataset_through_several_objects_is_opened_once(tmp_path, monkeypatch):
    _make_mixed_file(tmp_path / "x.h5")
    opened = []
    original_open = hdf_bulk_read.BulkReader.open
    monkeypatch.setattr(hdf_bulk_read.BulkReader, "open", lambda self, d: opened.append(d) or original_open(self, d))
    with h5py.File(tmp_path / "x.h5", "r") as f:
        requests = [hdf_bulk_read.ReadRequest(f["chunked"], slice(i, i + 10), np.zeros((10, 3))) for i in (0, 10, 20)]
        hdf_bulk_read.BulkReader().read(requests)
    assert len(opened) == 1


def test_empty_read(tmp_path):
    strategy = hdf_bulk_read.BulkReader().read([])
    assert strategy.threads == 1 and "nothing" in strategy.reason


@pytest.mark.parametrize("kind", ["slice", "indices"])
def test_virtual_dataset_requests_are_split_at_sources(tmp_path, kind):
    row_counts = [10, 20, 15]
    sources, arrays = _make_sources(tmp_path, row_counts)
    _make_vds(tmp_path / "virtual.h5", sources, row_counts, gap=4, fillvalue=-1.0)
    rows = slice(5, 50) if kind == "slice" else np.array([0, 3, 12, 29, 30, 31, 33, 40, 52])
    with h5py.File(tmp_path / "virtual.h5", "r") as f:
        expected = f["x"][rows]
        reader = hdf_bulk_read.BulkReader()
        destination = np.zeros_like(expected)
        works = hdf_bulk_read.plan.plan([hdf_bulk_read.ReadRequest(f["x"], rows, destination)], reader.open)
        names = [os.path.basename(work.filename) for work in works]
        # the sources supply rows 0-10, 14-34 and 38-53, the gaps between them being unmapped
        if kind == "slice":
            assert names == ["source.0.h5", "virtual.h5", "source.1.h5", "virtual.h5", "source.2.h5"]
        else:
            assert names == ["source.0.h5", "virtual.h5", "source.1.h5", "source.2.h5"]
        assert sum(len(work.destination) for work in works) == len(expected)
        for work in works:
            assert np.shares_memory(work.destination, destination)
        hdf_bulk_read.execute.perform(works, hdf_bulk_read.strategy.ReadStrategy(1, True, "test"))
    np.testing.assert_array_equal(destination, expected)


def test_group():
    class Work:
        def __init__(self, filename, n, chunks=()):
            self.filename, self.n, self.chunks = filename, n, chunks

        def shares_chunk_with(self, following):
            if self.filename != following.filename or not self.chunks or not following.chunks:
                return False
            return self.chunks[-1] == following.chunks[0]

    works = [Work(*w) for w in [("a", 1), ("b", 2), ("a", 3), ("v", 4, (0, 1)), ("v", 5, (1,)), ("v", 6, (2,)),
                                ("b", 7)]]
    grouped = hdf_bulk_read.plan.group(works, per_file=True)
    assert [[w.n for w in task] for task in grouped] == [[1, 3], [2, 7], [4, 5, 6]]
    # one task per unit of work, except that consecutive units needing the same chunk go together
    ungrouped = hdf_bulk_read.plan.group(works, per_file=False)
    assert [[w.n for w in task] for task in ungrouped] == [[1], [2], [3], [4, 5], [6], [7]]


@pytest.mark.parametrize("direct", [True, False])
def test_selections_convert_as_whole_reads_do(tmp_path, direct):
    """Rows read by index are converted to the destination's dtype exactly as a whole read converts them (by HDF5 for a
    narrowing conversion), including values beyond the range of the destination"""
    data = np.array([1.0, 3.4e38, 3.5e38, -3.5e39, 1e-50, np.nan, 2.0 ** 60, -0.0] * 50)
    with h5py.File(tmp_path / "x.h5", "w") as f:
        f.create_dataset("x", data=data, chunks=(64,), compression="gzip")
    rows = np.arange(0, len(data), 3)
    with h5py.File(tmp_path / "x.h5", "r") as f:
        whole = np.zeros(len(data), dtype=np.float32)
        part = np.zeros(len(rows), dtype=np.float32)
        hdf_bulk_read.BulkReader(enabled=direct).read([hdf_bulk_read.ReadRequest(f["x"], slice(0, len(data)), whole),
                                                        hdf_bulk_read.ReadRequest(f["x"], rows, part)])
    np.testing.assert_array_equal(part.view(np.uint32), whole[rows].view(np.uint32))
