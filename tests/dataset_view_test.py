import io
import h5py
import numpy as np
import pytest

import pynbody.test_utils
from pynbody.util import dataset_view


@pytest.fixture(scope='module')
def test_file():
    pynbody.test_utils.ensure_test_data_available("swift")
    with h5py.File("testdata/SWIFT/snap_0150.hdf5", "r") as tf:
        yield tf


def test_dataset_view_single_slice_start(test_file):
    """
    Basic test applying different slices to two groups
    """

    # Choose dataset slices to include in the view
    slices = {
        "PartType0" : [slice(0,10)],
        "PartType1" : [slice(0,20)],
        }

    # Create the view
    group_view = dataset_view.GroupView(test_file, slices=slices)
    # Check the first dataset
    dataset0_view = group_view["PartType0"]["ParticleIDs"]
    assert len(dataset0_view) == 10
    assert dataset0_view.shape == (10,)
    assert dataset0_view.size == 10
    assert np.all(dataset0_view[...] == test_file["PartType0"]["ParticleIDs"][:10])
    # Check the second dataset
    dataset1_view = group_view["PartType1"]["ParticleIDs"]
    assert len(dataset1_view) == 20
    assert dataset1_view.shape == (20,)
    assert dataset1_view.size == 20
    assert np.all(dataset1_view[...] == test_file["PartType1"]["ParticleIDs"][:20])


def test_dataset_view_no_slice(test_file):
    """
    If no slices are specified for a group, all elements should be returned
    """

    # Choose dataset slices to include in the view
    slices = {
        "PartType0" : [slice(0,10)],
        # No slices specified for PartType1, so should return all elements
        }

    # Create the view
    group_view = dataset_view.GroupView(test_file, slices=slices)
    # Check the first dataset
    dataset0_view = group_view["PartType0"]["ParticleIDs"]
    assert len(dataset0_view) == 10
    assert dataset0_view.shape == (10,)
    assert dataset0_view.size == 10
    assert np.all(dataset0_view[...] == test_file["PartType0"]["ParticleIDs"][:10])
    # Check the second dataset
    dset1 = test_file["PartType1"]["ParticleIDs"]
    dataset1_view = group_view["PartType1"]["ParticleIDs"]
    assert len(dataset1_view) == len(dset1)
    assert dataset1_view.shape == dset1.shape
    assert dataset1_view.size == dset1.size
    assert np.all(dataset1_view[...] == dset1[...])


def test_dataset_view_slice_none(test_file):
    """
    Setting slices to None should also return all elements
    """

    # Choose dataset slices to include in the view
    slices = {
        "PartType0" : [slice(0,10)],
        "PartType1" : None, # Should return all elements
    }

    # Create the view
    group_view = dataset_view.GroupView(test_file, slices=slices)
    # Check the first dataset
    dataset0_view = group_view["PartType0"]["ParticleIDs"]
    assert len(dataset0_view) == 10
    assert dataset0_view.shape == (10,)
    assert dataset0_view.size == 10
    assert np.all(dataset0_view[...] == test_file["PartType0"]["ParticleIDs"][:10])
    # Check the second dataset
    dset1 = test_file["PartType1"]["ParticleIDs"]
    dataset1_view = group_view["PartType1"]["ParticleIDs"]
    assert len(dataset1_view) == len(dset1)
    assert dataset1_view.shape == dset1.shape
    assert dataset1_view.size == dset1.size
    assert np.all(dataset1_view[...] == dset1[...])


def test_dataset_view_slice_empty_list(test_file):
    """
    An empty list of slices indicates a zero element view
    """

    # Choose dataset slices to include in the view
    slices = {
        "PartType0" : [slice(0,10)],
        "PartType1" : [] # zero size selection
    }

    # Create the view
    group_view = dataset_view.GroupView(test_file, slices=slices)
    # Check the first dataset
    dataset0_view = group_view["PartType0"]["ParticleIDs"]
    assert len(dataset0_view) == 10
    assert dataset0_view.shape == (10,)
    assert dataset0_view.size == 10
    assert np.all(dataset0_view[...] == test_file["PartType0"]["ParticleIDs"][:10])
    # Check the second dataset
    dset1 = test_file["PartType1"]["ParticleIDs"]
    dataset1_view = group_view["PartType1"]["ParticleIDs"]
    assert len(dataset1_view) == 0
    assert dataset1_view.shape == (0,)
    assert dataset1_view.size == 0
    assert dataset1_view[...].shape == (0,)


def slicing_test(test_file, slices):

    # Create the view
    file_view = dataset_view.GroupView(test_file, slices=slices)

    # Check that the view contains the expected data
    for group_name in slices.keys():

        group_view = file_view[group_name]
        assert isinstance(group_view, dataset_view.GroupView) # should not be a h5py.Group

        for dataset_name in ("ParticleIDs", "Coordinates"):

            # Find the real HDF5 dataset
            dset = test_file[group_name][dataset_name]

            # Find the sliced view of the same dataset
            dset_view = file_view[group_name][dataset_name]
            assert isinstance(dset_view, dataset_view.SlicedDatasetView) # should not be a h5py.Dataset
            assert dset.dtype == dset_view.dtype

            # Find the expected number of elements in the view
            ntot = sum([s.stop-s.start for s in slices[group_name]])
            assert dset_view.shape[0] == ntot
            assert dset_view.shape[1:] == dset.shape[1:]
            for s in dset.shape[1:]:
                ntot *= s
            assert dset_view.size == ntot

            # Check that the dataset elements have the expected values
            offset = 0
            for sl in slices[group_name]:
                n = sl.stop-sl.start
                real_data = dset[sl,...]
                view_data = dset_view[offset:offset+n,...]
                assert np.all(real_data==view_data)
                offset += n

def test_dataset_view_contiguous(test_file):

    slices = {
        "PartType0" : [slice(0,100), slice(100,200)],
        "PartType1" : [slice(5000,5100), slice(5100,5200)],
        }
    slicing_test(test_file, slices)


def test_dataset_view_noncontiguous(test_file):

    slices = {
        "PartType0" : [slice(0,100), slice(1100,1200)],
        "PartType1" : [slice(5000,5100), slice(6100,6200)],
        }
    slicing_test(test_file, slices)


def test_dataset_view_all(test_file):

    slices = {
        "PartType0" : [slice(i*1000,(i+1)*1000) for i in range(10)],
        "PartType1" : [slice(i*100,(i+1)*100) for i in range(100)]
        }
    slicing_test(test_file, slices)
