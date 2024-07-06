import numpy as np
import pytest

import pynbody
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_gizmo_data():
    pynbody.test_utils.ensure_test_data_available("gizmo")

@pytest.mark.filterwarnings("ignore:No unit information", "ignore:Assuming default value")
def test_load_gizmo_file():
    f = pynbody.load("testdata/gizmo/snapshot_000.hdf5")
    assert isinstance(f, pynbody.snapshot.gadgethdf.GadgetHDFSnap)
    assert len(f) == 12432742
    assert len(f.dm) == 7840638
    assert len(f.gas) == 4592104
    assert np.allclose(f.properties['z'], 99.0)
