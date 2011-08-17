import pynbody


def test_pickle() :
    # Regression test for issue 80
    import pickle
    assert pickle.loads(pickle.dumps(pynbody.family.gas)) is pynbody.family.gas
    
