"""Property assembly shared by Gadget3, Gadget4, Arepo and TNG catalogues."""

import tracemalloc
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from pynbody.halo.subfindhdf import SubFindHDFHaloCatalogue
from pynbody.snapshot.gadgethdf import _GadgetHdfMultiFileManager


class _RemoteDataset:
    """An hdfstream-style interface without __array__, backed by a local dataset."""

    def __init__(self, dataset):
        self._dataset = dataset
        self.shape, self.size, self.dtype = dataset.shape, dataset.size, dataset.dtype
        self.read_count = 0

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, key):
        values = self._dataset[key]
        self.read_count += 1
        return values

    def read_direct(self, target):
        self._dataset.read_direct(target)
        self.read_count += 1


def _read_properties(tmp_path, chunks, ignored=(), remote=False):
    stem = tmp_path / 'catalogue'
    for i, properties in enumerate(chunks):
        with h5py.File(f'{stem}.{i}.hdf5', 'w') as f:
            f.create_group('Header').attrs['NumFilesPerSnapshot'] = len(chunks)
            group = f.create_group('Subhalo')
            for name, values in properties.items():
                group[name] = values

    catalogue = object.__new__(SubFindHDFHaloCatalogue)
    catalogue.fof_ignore = ignored
    files = _GadgetHdfMultiFileManager(stem)
    if remote:
        roots = [{'Subhalo': {key: _RemoteDataset(dataset) for key, dataset in h['Subhalo'].items()}}
                 for h in files.iterroot()]
        files = SimpleNamespace(get_file0_root=lambda: roots[0], iterroot=lambda: iter(roots),
                                local_files=files)
    return catalogue, files


@pytest.mark.parametrize('chunks, expected', [
    ([np.arange(6, dtype=np.float32).reshape(2, 3)],
     np.arange(6, dtype=np.float32).reshape(2, 3)),
    ([np.arange(6, dtype=np.float32).reshape(2, 3),
      np.arange(6, 9, dtype=np.float32).reshape(1, 3)],
     np.arange(9, dtype=np.float32)),
    ([np.empty((0, 3), dtype=np.float32),
      np.arange(6, dtype=np.float32).reshape(2, 3),
      np.empty((0, 3), dtype=np.float32)],
     np.arange(6, dtype=np.float32)),
    ([np.empty((0, 3), dtype=np.float32)], np.empty((0, 3), dtype=np.float32)),
    ([np.empty((0, 3), dtype=np.float32)] * 2, np.empty(0, dtype=np.float32)),
    ([np.empty((2, 0), dtype=np.float32)] * 2, np.empty(0, dtype=np.float32)),
    ([np.array(7, dtype=np.int64)], np.array(7, dtype=np.int64)),
    ([np.array(7, dtype=np.int64), np.array(8, dtype=np.int64)],
     np.array([7, 8], dtype=np.int64)),
    ([np.array([1], dtype=np.int16), np.array([40000], dtype=np.int32)],
     np.array([1, 40000], dtype=np.int32)),
    ([np.array([1.5], dtype=np.float32), np.array([1.00000000001], dtype=np.float64)],
     np.array([1.5, 1.00000000001], dtype=np.float64)),
    ([np.array([1], dtype=np.int8), np.array([2], dtype=np.uint8),
      np.array([0.5], dtype=np.float16)], np.array([1, 2, 0.5], dtype=np.float16)),
    ([np.array([1], dtype=np.int16), np.empty(0, dtype=np.float64)],
     np.array([1], dtype=np.float64)),
    ([np.array(7, dtype='>i4')], np.array(7, dtype='>i4')),
    ([np.array([1], dtype='>i4')], np.array([1], dtype='>i4')),
    ([np.array([1], dtype='>i4'), np.array([2], dtype='>i4')],
     np.array([1, 2], dtype=np.int32)),
    ([np.arange(8, dtype=np.float64).reshape(2, 2, 2),
      np.arange(8, 12, dtype=np.float64).reshape(1, 2, 2)],
     np.arange(12, dtype=np.float64)),
])
def test_property_values_shapes_and_dtypes(tmp_path, chunks, expected):
    catalogue, files = _read_properties(tmp_path, [{'Value': chunk} for chunk in chunks])
    actual = catalogue._get_properties_from_multifile(files, 'Subhalo')['Value']
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, expected, strict=True)


@pytest.mark.parametrize('values', [
    np.arange(6, dtype=np.float32).reshape(2, 3),
    np.empty((0, 3), dtype=np.float32),
    np.array(7, dtype='>i4'),
], ids=['multirow', 'empty', 'big-endian-scalar'])
def test_remote_property_bulk_read(tmp_path, values):
    catalogue, files = _read_properties(tmp_path, [{'Value': values}], remote=True)
    actual = catalogue._get_properties_from_multifile(files, 'Subhalo')['Value']
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, values, strict=True)
    # Falling back to Python's sequence protocol makes one network request per row.
    assert files.get_file0_root()['Subhalo']['Value'].read_count <= 1


def test_missing_and_ignored_properties(tmp_path):
    catalogue, files = _read_properties(tmp_path, [
        {'Mass': [1, 2], 'Position': [[1., 2., 3.]], 'Ignored': [9]},
        {'Mass': [3], 'OnlyInLaterFile': [42]},
        {'Mass': [4, 5], 'Position': [[4., 5., 6.]], 'Ignored': [10]},
    ], ignored=('Ignored',))
    actual = catalogue._get_properties_from_multifile(files, 'Subhalo')
    assert list(actual) == ['Mass', 'Position']
    np.testing.assert_array_equal(actual['Mass'], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(actual['Position'], [1., 2., 3., 4., 5., 6.])


def test_property_in_only_one_file_keeps_its_shape(tmp_path):
    position = np.arange(6, dtype=np.float32).reshape(2, 3)
    catalogue, files = _read_properties(tmp_path, [{'Position': position}, {}, {}])
    actual = catalogue._get_properties_from_multifile(files, 'Subhalo')['Position']
    np.testing.assert_array_equal(actual, position, strict=True)


@pytest.mark.parametrize('widths', [[1] * 20, [1] * 19 + [8]], ids=['equal', 'largest-last'])
def test_multifile_property_assembly_bounds_temporary_memory(tmp_path, widths):
    """Temporary storage should scale with one property, not the whole catalogue."""
    chunks = {f'Property{i:02d}': np.arange(8192 * width, dtype=np.float64)
              for i, width in enumerate(widths)}
    catalogue, files = _read_properties(tmp_path, [chunks] * 16)
    tracemalloc.start()
    try:
        actual = catalogue._get_properties_from_multifile(files, 'Subhalo')
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    for key, chunk in chunks.items():
        np.testing.assert_array_equal(actual[key].reshape(16, -1), np.broadcast_to(chunk, (16, len(chunk))))
    output_bytes = sum(values.nbytes for values in actual.values())
    largest_property_bytes = max(values.nbytes for values in actual.values())
    # Allow the largest property's pieces plus headroom for HDF5/Python metadata.
    limit = output_bytes + 1.25 * largest_property_bytes + 1024**2
    assert peak < limit, f'{peak=}, {output_bytes=}, {largest_property_bytes=}'
