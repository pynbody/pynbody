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
