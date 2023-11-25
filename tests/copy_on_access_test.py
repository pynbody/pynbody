import numpy as np

import pynbody
import pynbody.snapshot


def test_copy_on_access_subsnap_data_isolation():
    # Test the copy_on_access subsnap, which only gets data from the underlying rather than pointing back to it
    # This is used by tangos to implement shared memory server mode
    f = pynbody.new(10)
    f['blob'] = np.arange(10)

    f_sub = f[[2, 3, 4]].get_copy_on_access_view()
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

    f_sub = f[[2, 3, 4]].get_copy_on_access_view()
    assert (f_sub['foo'] == [7, 8, 9]).all()

    assert 'foo' not in f.keys()


def test_copy_on_access_subsnap_family_array():
    f = pynbody.new(dm=10,star=10)
    f.dm['dm_only'] = np.arange(10)
    f.st['star_only'] = np.arange(10,20)

    f_sub = f[np.arange(0,20,2)].get_copy_on_access_view()
    assert (f_sub.dm['dm_only'] == np.arange(0,10,2)).all()
    assert (f_sub.star['star_only'] == np.arange(10, 20, 2)).all()
    assert 'dm_only' not in f_sub.keys()
    assert 'star_only' not in f_sub.keys()

def test_base_correctness():
    f = pynbody.new(10)
    f_sub = f[[2,3,4]]
    assert f_sub.ancestor is f

    f_sub = f[::2]
    assert f_sub.ancestor is f

    f_sub = f[[2,3,4]].get_copy_on_access_view()
    assert f_sub.ancestor is f_sub
    assert not hasattr(f_sub, 'base')
