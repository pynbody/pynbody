import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.snapshot
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gasoline_ahf")


def test_copy_on_access_subsnap_data_isolation():
    # Test the copy_on_access subsnap, which only gets data from the underlying rather than pointing back to it
    # This is used by tangos to implement shared memory server mode
    f = pynbody.new(10)
    f['blob'] = np.arange(10)

    for subscript in ([2, 3, 4], slice(2,5)):

        f_sub = f[subscript].get_copy_on_access_simsnap()
        # doesn't simply copy everything in to start with:
        assert 'blob' not in f_sub.keys()

        # can get the underlying data lazily
        assert (f_sub['blob'] == [2,3,4]).all()

        f_sub['blob'] = [100,101,102]
        assert (f_sub['blob'] == [100, 101, 102]).all()

        # copy_on_access: shouldn't have updated the underlying
        assert (f['blob'] == np.arange(10)).all()


class ExampleSnap(pynbody.snapshot.simsnap.SimSnap):
    pass


@ExampleSnap.derived_array
def foo(sim):
    return sim['blob']+5


def test_copy_on_access_subsnap_emulating_class():
    f = pynbody.new(10, class_=ExampleSnap)
    f['blob'] = np.arange(10)

    f_sub = f[[2, 3, 4]].get_copy_on_access_simsnap()
    assert (f_sub['foo'] == [7, 8, 9]).all()

    assert 'foo' not in f.keys()

def test_copy_on_access_subsnap_emulating_class_two_layers_down():
    f = pynbody.new(10, class_=ExampleSnap)
    f['blob'] = np.arange(10)

    f_sub_copy = f[[2, 3, 4]].get_copy_on_access_simsnap()

    f_sub_copy_copy = f_sub_copy.get_copy_on_access_simsnap()


    assert (f_sub_copy_copy['foo'] == [7, 8, 9]).all()

    assert 'foo' not in f_sub_copy.keys()
    assert 'foo' not in f.keys()



def test_copy_on_access_subsnap_family_array():
    f = pynbody.new(dm=10,star=10)
    f.dm['dm_only'] = np.arange(10)
    f.st['star_only'] = np.arange(10,20)

    f_sub = f[np.arange(0,20,2)].get_copy_on_access_simsnap()
    assert (f_sub.dm['dm_only'] == np.arange(0,10,2)).all()
    assert (f_sub.star['star_only'] == np.arange(10, 20, 2)).all()
    assert 'dm_only' not in f_sub.keys()
    assert 'star_only' not in f_sub.keys()

def test_copy_on_access_mixing_derived_and_nonderived():
    f = pynbody.new(dm=10, star=10, gas=10, class_=ExampleSnap)
    f.dm['blob'] = np.arange(10)
    f.star['foo'] = np.arange(10, 20)

    f_copy = f.get_copy_on_access_simsnap()
    assert (f_copy.dm['foo'] == np.arange(5, 15)).all()
    assert f_copy.dm['foo'].derived

    assert (f_copy.star['foo'] == np.arange(10, 20)).all()

    assert f_copy.dm['foo'].derived
    assert not f_copy.star['foo'].derived

    f_copy.dm['blob'] += 1

    assert (f_copy.dm['blob'] == np.arange(1, 11)).all()
    assert (f_copy.dm['foo'] == np.arange(6, 16)).all()
    assert (f_copy.star['foo'] == np.arange(10, 20)).all()
    assert (f.dm['blob'] == np.arange(10)).all()
def test_base_correctness():
    f = pynbody.new(10)
    f_sub = f[[2,3,4]]
    assert f_sub.ancestor is f

    f_sub = f[::2]
    assert f_sub.ancestor is f

    f_sub = f[[2,3,4]].get_copy_on_access_simsnap()
    assert f_sub.ancestor is f_sub
    assert not hasattr(f_sub, 'base')

def test_properties():
    f = pynbody.new(10)
    f.properties['test_property'] = 101
    f_c = f.get_copy_on_access_simsnap()
    # should have been copied in:
    assert f_c.properties['test_property'] == 101

    # should not reflect back to parent:
    f_c.properties['test_property'] = 100
    assert f_c.properties['test_property'] == 100
    assert f.properties['test_property'] == 101

def test_repr():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    f_c = f.get_copy_on_access_simsnap()

    assert repr(f_c) == '<CopyOnAccessSimSnap "testdata/gasoline_ahf/g15784.lr.01024:copied_on_access" len=1717156>' or \
           repr(f_c) == r'<CopyOnAccessSimSnap "testdata\gasoline_ahf\g15784.lr.01024:copied_on_access" len=1717156>'

def test_loadable_keys():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    f['pos'] # noqa
    f.dm['new_array'] = np.empty(len(f.dm))

    f_c = f.get_copy_on_access_simsnap()
    # anything loadable in the base is loadable in the copy
    assert 'pos' in f_c.loadable_keys()
    assert 'HI' in f_c.gas.loadable_keys()

    # check HI will actually load:
    f_c.gas['HI'] # noqa

    assert 'new_array' not in f_c.dm.keys()
    # it's in the parent keys, but not yet copied across
    assert 'new_array' in f_c.dm.loadable_keys()

def test_all_keys():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    f_c = f.get_copy_on_access_simsnap()
    assert 'pos' in f_c.all_keys()


def test_only_try_loading_once():
    f = pynbody.new(10)
    f_c = f.get_copy_on_access_simsnap()


    with pytest.raises(OSError, match="Not found"):
        f_c.dm._load_array('nonexistent')

    with pytest.raises(OSError, match="Previously tried"):
        f_c.dm._load_array('nonexistent')


def test_loadable_keys_reports_nd_arrays_not_their_slices():
    """The 1D slices that accompany an in-memory ND array must not be advertised as loadable.

    A file-backed snapshot offers the blocks it stores, i.e. 'vel' but never 'vz', and code that
    inspects loadable_keys() to decide what a snapshot can provide relies on that.
    """
    f = pynbody.new(10)
    f['pos'] = np.random.normal(size=(10, 3))
    f['vel'] = np.random.normal(size=(10, 3))
    f['mass'] = np.ones(10)

    f_c = f.get_copy_on_access_simsnap()

    assert set(f_c.loadable_keys()) == {'pos', 'vel', 'mass'}


def test_slices_of_nd_arrays_still_load():
    """Hiding the slice names from loadable_keys must not stop them being accessible."""
    f = pynbody.new(10)
    f['vel'] = np.random.normal(size=(10, 3))

    f_c = f.get_copy_on_access_simsnap()
    assert 'vz' not in f_c.loadable_keys()
    assert 'vz' not in f_c.keys()

    npt.assert_allclose(f_c['vz'], f['vel'][:, 2])
    # asking for the slice brings in the whole parent array
    assert 'vel' in f_c.keys()


def test_loaded_slices_are_views_onto_the_nd_array():
    """A slice must be a view onto the copy's own ND array, not an independently stored copy.

    Loading is triggered by the slice name here, which is the case most at risk of ending up with
    'vx' and 'vel' as two unrelated arrays.
    """
    f = pynbody.new(10)
    f['vel'] = np.random.normal(size=(10, 3))
    original_vel = np.array(f['vel'])

    f_c = f.get_copy_on_access_simsnap()
    f_c['vx']  # noqa - triggers the load via the slice name rather than the parent

    for i, name in enumerate(['vx', 'vy', 'vz']):
        assert np.shares_memory(f_c[name], f_c['vel'][:, i])

    # ...and the consequence that actually matters: writes are seen from both directions
    f_c['vel'][:, 0] = np.arange(10)
    npt.assert_allclose(f_c['vx'], np.arange(10))

    f_c['vy'] = np.arange(10, 20)
    npt.assert_allclose(f_c['vel'][:, 1], np.arange(10, 20))

    # the copy remains isolated from the snapshot it copied from, which is the whole point of
    # copy-on-access: none of those writes may reach the original
    assert not np.shares_memory(f_c['vel'], f['vel'])
    npt.assert_allclose(f['vel'], original_vel)
