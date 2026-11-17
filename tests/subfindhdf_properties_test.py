"""Property assembly shared by Gadget3, Gadget4, Arepo and TNG catalogues."""

import tracemalloc

import h5py
import numpy as np
import pytest

from pynbody.halo.subfindhdf import SubFindHDFHaloCatalogue
from pynbody.snapshot.gadgethdf import _GadgetHdfMultiFileManager


def _read_properties(tmp_path, chunks, ignored=()):
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
    np.testing.assert_array_equal(actual, expected, strict=True)


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


def test_property_dtype_promotion_preserves_intermediate_rounding(tmp_path):
    # Sequential append first converts the large integer to float64. A later
    # promotion to longdouble must not recover precision that was already lost.
    chunks = [np.array([2**63 + 1], dtype=np.uint64),
              np.array([-1], dtype=np.int64), np.array([0.5], dtype=np.longdouble)]
    catalogue, files = _read_properties(tmp_path, [{'Value': chunk} for chunk in chunks])
    actual = catalogue._get_properties_from_multifile(files, 'Subhalo')['Value']
    expected = np.array([float(2**63 + 1), -1., 0.5], dtype=np.longdouble)
    np.testing.assert_array_equal(actual, expected, strict=True)


def test_multifile_property_assembly_bounds_temporary_memory(tmp_path):
    """Assembling many pieces should not need another output-sized array."""
    chunk = np.arange(65536, dtype=np.float64)
    catalogue, files = _read_properties(tmp_path, [{'Value': chunk}] * 32)
    tracemalloc.start()
    try:
        actual = catalogue._get_properties_from_multifile(files, 'Subhalo')['Value']
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(actual.reshape(32, -1), np.broadcast_to(chunk, (32, len(chunk))))
    assert peak < 1.5 * actual.nbytes, f'{peak=}, output bytes={actual.nbytes}'
