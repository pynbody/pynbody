import pynbody
import numpy as np

def test_pickle():
    # Regression test for issue 80
    import pickle
    assert pickle.loads(pickle.dumps(pynbody.family.gas)) is pynbody.family.gas

def test_family_array_dtype() :
    # test for issue #186
    f = pynbody.load('testdata/g15784.lr.01024.gz')
    f.g['rho'] = np.zeros(len(f.g),dtype=np.float32)
    f.s['rho']

def test_family_array_null_slice():
    """Regression test for issue where getting a family array for an IndexedSubSnap containing no members of that family
    - would erroneously return the entire family array"""

    test = pynbody.new(dm=10, star=10, order='dm,star')
    test.star['TestFamilyArray'] = 1.0
    assert len(test[[1,3,5,7]].star)==0 # this always succeeded
    assert len(test[[1,3,5,7]].star['mass'])==0 # this always succeeded
    assert len(test[1:9:2].star['TestFamilyArray'])==0 # this always succeeded
    assert len(test[[1, 3, 5, 11,13]].star['TestFamilyArray']) == 2  # this always succeeded
    assert len(test[[1,3,5,7]].star['TestFamilyArray'])==0 # this would fail
