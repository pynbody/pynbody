import pynbody
import numpy as np
import pickle
import gc

def test_pickle() :
    import pickle
    f = pynbody.new(10)
    f['blob']=np.arange(10)
    s = f[[3,6,7]]
    assert (s['blob']==[3,6,7]).all(), "Preliminary check to testing pickle failed!"

    reloaded = pickle.loads(pickle.dumps(s['blob']))
    assert (reloaded==[3,6,7]).all(), "Unpickled array had incorrect contents"

def test_sim_propagation() :
    f = pynbody.new(10)
    f['blob']=0

    # check arrays remember their simulation
    assert f['blob'].sim is f
    assert f[::2]['blob'].sim is f

    assert f[[1,2,5]]['blob'].sim.ancestor is f
    assert f[[1,2,5]]['blob'].sim is not f

    # if we do anything that constructs a literal SimArray, the simulation
    # reference jumps up to be the main snapshot since we can't keep track
    # of what SubSnap it came from in generality
    assert pynbody.array.SimArray(f[[1,2,5]]['blob']).sim is f
    assert (f[[1,2,5]]['blob']).sum().sim is f

    # the reference should be weak: check
    X = f['blob']
    assert X.sim is f
    del f
    gc.collect()
    assert X.sim is None
    
    
