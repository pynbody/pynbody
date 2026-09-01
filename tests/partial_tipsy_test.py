import numpy as np
import pytest

import pynbody
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gasoline_ahf")

def test_indexing():
    f1 = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")

    np.random.seed(1)
    for test_len in [100, 10000, 20000]:
        for i in range(5):
            subindex = np.random.permutation(np.arange(0, len(f1)))[:test_len]
            subindex.sort()
            f2 = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024", take=subindex)

            assert (f2['x'] == f1[subindex]['x']).all()
            assert (f2['iord'] == f1[subindex]['iord']).all()

@pytest.fixture(scope='module')
def full_snapshot():
    return pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")


@pytest.mark.parametrize("take", [slice(100), slice(0, 100), slice(0, 100, 2),
                                  slice(-100, None), slice(None, None, 20000), slice(None)],
                         ids=["stop-only", "start-stop", "step", "negative-start", "step-only", "everything"])
def test_take_slice(full_snapshot, take):
    """A slice selects particles as they lie on disk, following the usual python conventions.

    The omitted parts of an everyday slice are None, so the slice has to be resolved against the number of
    particles on disk rather than handed to np.arange as it stands."""
    partial = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024", take=take)

    expected = np.arange(*take.indices(len(full_snapshot)))
    assert len(partial) == len(expected)
    assert (partial['iord'] == full_snapshot['iord'][expected]).all()
    assert (partial['x'] == full_snapshot['x'][expected]).all()


@pytest.mark.parametrize("take", [[50, 3, 10], [3, 3, 10], [10, 3]],
                         ids=["unsorted", "repeated", "descending"])
def test_take_array_must_be_strictly_ascending(take):
    """Ids index the particles in disk order, and readers walk the file forwards.

    Out-of-order ids raise an IndexError from inside the gadgethdf chunk handling while tipsy happens to
    tolerate them, and a repeated id yields a snapshot holding the same particle twice; neither is worth
    supporting, so both are refused where the problem can still be explained."""
    with pytest.raises(ValueError, match="strictly ascending"):
        pynbody.load("testdata/gasoline_ahf/g15784.lr.01024", take=np.array(take))


@pytest.mark.parametrize("take", [[7], []], ids=["single", "empty"])
def test_take_array_accepts_degenerate_cases(take):
    """Too short to be out of order, and so not something to complain about"""
    partial = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024", take=np.array(take, dtype=int))
    assert len(partial) == len(take)


def test_take_slice_rejects_negative_step():
    """Readers require their ids in disk order, so a reversed slice cannot be honoured"""
    with pytest.raises(ValueError, match="negative step"):
        pynbody.load("testdata/gasoline_ahf/g15784.lr.01024", take=slice(None, None, -1))


def test_load_copy():
    f1 = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")

    subview = f1[::5]

    f_subview = subview.load_copy()

    assert (subview['x']==f_subview['x']).all()

    # sanity check that the loaded copy is not linked to the original:
    subview['x'][0]=0
    f_subview['x'][0]=1
    assert subview['x'][0]==0

    with pytest.raises(NotImplementedError):
        f_subview[:5].load_copy()

def test_grp_load_copy():
    f1 = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    h = f1.halos()
    h1_copy = h[1].load_copy()
    assert (h1_copy['x']==h[1]['x']).all()
    assert h1_copy.ancestor is h1_copy
