import gc

import numpy as np
import pytest

import pynbody
import pynbody.test_utils

# PENDING: find/generate smaller test data, upload to zenodo
"""
@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("pkdgrav3")

"""


@pytest.fixture
def snap():
    f = pynbody.load('testdata/m11i_res7100/output/snapshot_600.hdf5')
    yield f
    del f
    gc.collect()


@pytest.fixture
def multi_snap():
    f = pynbody.load('testdata/m10q_res30/output/snapdir_600/snapshot_600')
    yield f
    del f
    gc.collect()
