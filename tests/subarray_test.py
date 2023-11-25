import gc
import pickle

import numpy as np

import pynbody
import pynbody.snapshot.simsnap
import pynbody.snapshot.subsnap


def test_pickle():
    import pickle
    f = pynbody.new(10)
    f['blob'] = np.arange(10)
    s = f[[3, 6, 7]]
    assert (s['blob'] == [3, 6, 7]).all(
    ), "Preliminary check to testing pickle failed!"

    reloaded = pickle.loads(pickle.dumps(s['blob']))
    assert (reloaded == [3, 6, 7]).all(
    ), "Unpickled array had incorrect contents"


def test_sim_propagation():
    f = pynbody.new(10)
    f['blob'] = 0

    # check arrays remember their simulation
    assert f['blob'].sim is f
    assert f[::2]['blob'].sim is f

    assert f[[1, 2, 5]]['blob'].sim.ancestor is f
    assert f[[1, 2, 5]]['blob'].sim is not f

    # if we do anything that constructs a literal SimArray, the simulation
    # reference jumps up to be the main snapshot since we can't keep track
    # of what SubSnap it came from in generality
    assert pynbody.array.SimArray(f[[1, 2, 5]]['blob']).sim is f
    assert (f[[1, 2, 5]]['blob']).sum().sim is f

    # the reference should be weak: check
    X = f['blob']
    assert X.sim is f
    del f
    gc.collect()
    assert X.sim is None

def test_base_correctness():
    f = pynbody.new(10)
    f_sub = f[[2,3,4]]
    assert f_sub.ancestor is f

    f_sub = f[::2]
    assert f_sub.ancestor is f

    f_sub = pynbody.snapshot.subsnap.CopyOnAccessIndexedSubSnap(f, [2,3,4])
    assert f_sub.ancestor is f_sub
    assert not hasattr(f_sub, 'base')

def test_copy_on_access_subsnap_data_isolation():
    # Test the copy_on_access subsnap, which only gets data from the underlying rather than pointing back to it
    # This is used by tangos to implement shared memory server mode
    f = pynbody.new(10)
    f['blob'] = np.arange(10)

    f_sub = pynbody.snapshot.subsnap.CopyOnAccessIndexedSubSnap(f, [2, 3, 4])
    # doesn't simply copy everything in to start with:
    assert 'blob' not in f_sub.keys()

    # can get the underlying data lazily
    assert (f_sub['blob'] == [2,3,4]).all()

    f_sub['blob'] = [100,101,102]
    assert (f_sub['blob'] == [100, 101, 102]).all()

    # copy_on_access: shouldn't have updated the underlying
    assert (f['blob'] == np.arange(10)).all()

class TestSnap(pynbody.snapshot.simsnap.SimSnap):
    pass

@TestSnap.derived_quantity
def foo(sim):
    return sim['blob']+5

def test_copy_on_access_subsnap_emulating_class():
    f = pynbody.new(10, class_=TestSnap)
    f['blob'] = np.arange(10)

    f_sub = pynbody.snapshot.subsnap.CopyOnAccessIndexedSubSnap(f, [2, 3, 4])
    assert (f_sub['foo'] == [7, 8, 9]).all()

    assert 'foo' not in f.keys()

def test_copy_on_access_subsnap_family_array():
    f = pynbody.new(dm=10,star=10)
    f.dm['dm_only'] = np.arange(10)
    f.st['star_only'] = np.arange(10,20)

    f_sub = pynbody.snapshot.subsnap.CopyOnAccessIndexedSubSnap(f, np.arange(0,20,2))
    assert (f_sub.dm['dm_only'] == np.arange(0,10,2)).all()
    assert (f_sub.star['star_only'] == np.arange(10, 20, 2)).all()
    assert 'dm_only' not in f_sub.keys()
    assert 'star_only' not in f_sub.keys()

def test_ndim_issue_399():
    f = pynbody.new(10)
    f_sub = f[[1,2,3,6]]
    f['blob'] = np.arange(10)
    assert f['blob'].ndim==1
    assert f_sub['blob'].ndim==1

    f['blob_3d'] = np.zeros((10,3))
    assert f['blob_3d'].ndim==2
    assert f['blob_3d'].shape==(10,3)
    assert f_sub['blob_3d'].ndim==2
    assert f_sub['blob_3d'].shape==(4,3)
