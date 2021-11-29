import pynbody
import numpy as np


def test_indexing():
    f1 = pynbody.load("testdata/g15784.lr.01024")

    np.random.seed(1)
    for test_len in [100, 10000, 20000]:
        for i in range(5):
            subindex = np.random.permutation(np.arange(0, len(f1)))[:test_len]
            subindex.sort()
            f2 = pynbody.load("testdata/g15784.lr.01024", take=subindex)

            assert (f2['x'] == f1[subindex]['x']).all()
            assert (f2['iord'] == f1[subindex]['iord']).all()

def test_load_copy():
    f1 = pynbody.load("testdata/g15784.lr.01024")

    subview = f1[::5]

    f_subview = subview.load_copy()

    assert (subview['x']==f_subview['x']).all()

    # sanity check that the loaded copy is not linked to the original:
    subview['x'][0]=0
    f_subview['x'][0]=1
    assert subview['x'][0]==0

    with np.testing.assert_raises(NotImplementedError):
        f_subview[:5].load_copy()

def test_grp_load_copy():
    f1 = pynbody.load("testdata/g15784.lr.01024")
    h = f1.halos()
    h1_copy = h[1].load_copy()
    assert (h1_copy['x']==h[1]['x']).all()
    assert h1_copy.ancestor is h1_copy
