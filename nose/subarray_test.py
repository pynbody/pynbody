import pynbody
import numpy as np
import pickle

def test_pickle() :
    import pickle
    f = pynbody.new(10)
    f['blob']=np.arange(10)
    s = f[[3,6,7]]
    assert (s['blob']==[3,6,7]).all(), "Preliminary check to testing pickle failed!"

    reloaded = pickle.loads(pickle.dumps(s['blob']))
    assert (reloaded==[3,6,7]).all(), "Unpickled array had incorrect contents"
