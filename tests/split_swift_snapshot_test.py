from pathlib import Path

import h5py
import numpy as np
import pytest

import pynbody
from pynbody.test_utils.split_swift_snapshot import (
    hash_swift_cell_coordinates,
    split_swift_snapshot,
)


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("swift")


@pytest.mark.parametrize("nr_files", [1,2,3,4])
@pytest.mark.parametrize("write_zero_size_datasets", [True, False])
def test_split_snapshot(tmp_path, nr_files, write_zero_size_datasets):

    # Split one of the test data snapshots across multiple files
    input_snapshot = "./testdata/SWIFT/snap_0150.hdf5"
    output_snapshot = tmp_path / "split_snap_0150.0.hdf5"
    rng = np.random.default_rng(0)
    cell_file_index = rng.integers(nr_files, size=512) # randomly assign cells to files
    split_swift_snapshot(input_snapshot, nr_files, cell_file_index, output_snapshot,
                         write_zero_size_datasets=write_zero_size_datasets)

    # Check that each cell still contains the same particle coordinates
    assert hash_swift_cell_coordinates(input_snapshot) == hash_swift_cell_coordinates(output_snapshot)


@pytest.mark.parametrize("write_zero_size_datasets", [True, False])
def test_split_snapshot_with_mask(tmp_path, write_zero_size_datasets):

    cell_to_keep = 123

    # Create mask with cells to keep
    cell_mask = {
        "PartType0" : np.zeros(512, dtype=bool),
        "PartType1" : np.ones(512, dtype=bool),
    }
    cell_mask["PartType0"][cell_to_keep] = True # discard gas in all cells but one

    # Generate the multi file snapshot
    nr_files = 8
    input_snapshot = "./testdata/SWIFT/snap_0150.hdf5"
    output_snapshot = tmp_path / "split_snap_0150.0.hdf5"
    rng = np.random.default_rng(0)
    cell_file_index = rng.integers(nr_files, size=512) # randomly assign cells to files
    split_swift_snapshot(input_snapshot, nr_files, cell_file_index, output_snapshot,
                         cell_mask=cell_mask, write_zero_size_datasets=write_zero_size_datasets)

    # Check that we're only omitting datasets where expected
    file_with_gas = cell_file_index[cell_to_keep]
    for file_nr in range(nr_files):
        with h5py.File(tmp_path / f"split_snap_0150.{file_nr}.hdf5", "r") as f:
            assert "PartType1/Coordinates" in f
            if file_nr != file_with_gas and not write_zero_size_datasets:
                assert "PartType0/Coordinates" not in f
            else:
                assert "PartType0/Coordinates" in f
