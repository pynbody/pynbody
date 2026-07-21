from pathlib import Path

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
def test_split_snapshot(tmp_path, nr_files):

    # Split one of the test data snapshots across multiple files
    input_snapshot = "./testdata/SWIFT/snap_0150.hdf5"
    output_snapshot = tmp_path / "split_snap_0150.0.hdf5"
    rng = np.random.default_rng(0)
    cell_file_index = rng.integers(nr_files, size=512) # randomly assign cells to files
    split_swift_snapshot(input_snapshot, nr_files, cell_file_index, output_snapshot)

    # Check that each cell still contains the same particle coordinates
    assert hash_swift_cell_coordinates(input_snapshot) == hash_swift_cell_coordinates(output_snapshot)
