"""Tests of pynbody.ProcessPool, which shares snapshots, halo catalogues and trees with worker processes.

The functions run by the pools are at module level, so that worker processes started by 'spawn' (the only method
available on all platforms) can find them."""

import gc
import pathlib
import subprocess
import sys
import types

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils
from pynbody.array import shared


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gasoline_ahf")


@pytest.fixture(scope='module')
def snap_and_halos():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    f.enable_shared_arrays()
    f.physical_units()
    f['pos']
    f['mass']
    h = f.halos()
    f.build_tree()
    yield f, h
    del f, h
    gc.collect()


def _centre_and_profile(f, h, halo_number):
    halo = h[halo_number]
    centre = pynbody.analysis.halo.shrink_sphere_center(halo)
    local = halo.get_copy_on_access_simsnap()
    pynbody.analysis.center(local)
    profile = pynbody.analysis.Profile(local.dm, ndim=3, rmin='0.5 kpc', rmax='50 kpc', nbins=10)
    num_in_sphere = len(f[pynbody.filt.Sphere('20 kpc', centre)])
    return centre, profile['density'], num_in_sphere


def test_pool_analyses_halos(snap_and_halos):
    f, h = snap_and_halos
    with pynbody.ProcessPool(f, h, processes=2, start_method='spawn') as pool:
        results = pool.map(_centre_and_profile, [1, 2, 3, 4])

    for halo_number, (centre, density, num_in_sphere) in zip([1, 2, 3, 4], results):
        expected_centre, expected_density, expected_num = _centre_and_profile(f, h, halo_number)
        npt.assert_allclose(centre, expected_centre)
        npt.assert_allclose(density, expected_density)
        assert num_in_sphere == expected_num

    # the original snapshot was not moved by the analysis
    npt.assert_allclose(pynbody.analysis.halo.shrink_sphere_center(h[1]), results[0][0])


def _describe_worker(f, h, _):
    halo = h[1]
    return {'snapshot_type': type(f).__name__,
            'catalogue_type': type(h).__name__,
            'halo_base_is_f': h.base is f,
            'pos_writeable': f['pos'].flags['WRITEABLE'],
            'has_tree': hasattr(f, 'kdtree'),
            'thread_limit': pynbody.util.threads.current_limit(),
            'halo_length': len(halo),
            'properties': h.get_properties_one_halo(1)['Mvir']}


def test_pool_worker_state(snap_and_halos):
    f, h = snap_and_halos
    with pynbody.ProcessPool(f, h, processes=1, start_method='spawn', threads_per_process=2) as pool:
        description = pool.map(_describe_worker, [None])[0]
    assert description['snapshot_type'] == 'PortableSimSnap'
    assert description['catalogue_type'] == 'PortableHaloCatalogue'
    assert description['halo_base_is_f']
    assert not description['pos_writeable']
    assert description['has_tree']
    assert description['thread_limit'] == 2
    assert description['halo_length'] == len(h[1])
    assert description['properties'] == h.get_properties_one_halo(1)['Mvir']


def _load_unloaded(f, h, name):
    return f[name]


def _write(f, h, name):
    f[name][0] = 0


def _centre_in_place(f, h, halo_number):
    pynbody.analysis.center(h[halo_number])


def test_pool_errors(snap_and_halos):
    f, h = snap_and_halos
    with pynbody.ProcessPool(f, h, processes=1, start_method='spawn') as pool:
        with pytest.raises(KeyError, match="'iord' was not loaded"):
            pool.map(_load_unloaded, ['iord'])
        with pytest.raises(ValueError, match="read-only"):
            pool.map(_write, ['pos'])
        with pytest.raises(ValueError, match="get_copy_on_access_simsnap"):
            pool.submit(_centre_in_place, 1).result()
        # the pool remains usable
        assert len(pool.map(_load_unloaded, ['mass'])[0]) == len(f)


def _halo_mass(h, halo_number):
    return float(h[halo_number]['mass'].sum())


def test_pool_shares_halos_alone(snap_and_halos):
    """A halo catalogue brings its snapshot with it"""
    f, h = snap_and_halos
    with pynbody.ProcessPool(h, processes=2, start_method='spawn') as pool:
        masses = pool.map(_halo_mass, [1, 2, 3])
    npt.assert_allclose(masses, [float(h[i]['mass'].sum()) for i in (1, 2, 3)])


def _use_other_objects(tree, array, obj, multiplier, offset=0):
    return (len(tree.particles_in_sphere([0.5, 0.5, 0.5], 0.2)), float(array.sum()) * multiplier + offset,
            obj['key'], array.flags['WRITEABLE'])


def test_pool_shares_other_objects():
    np.random.seed(2)
    f = pynbody.new(dm=20000)
    f['pos'] = np.random.uniform(size=(20000, 3))
    f['mass'] = np.ones(20000)
    f.build_tree()
    array = np.arange(100000.0)

    owned_before = shared.get_num_shared_arrays_owned()
    pool = pynbody.ProcessPool(f.kdtree, array, {'key': 'value'}, processes=1, start_method='spawn')
    # the tree's arrays and the array have been copied to shared memory (but not the small mass array)
    assert shared.get_num_shared_arrays_owned() > owned_before
    with pool:
        result = pool.map(_use_other_objects, [2.0])[0]
        result_from_submit = pool.submit(_use_other_objects, 2.0, offset=1).result()
    assert result == (len(f.kdtree.particles_in_sphere([0.5, 0.5, 0.5], 0.2)), array.sum() * 2, 'value', False)
    assert result_from_submit[1] == array.sum() * 2 + 1

    del pool
    gc.collect()
    assert shared.get_num_shared_arrays_owned() == owned_before


def _get_array(f, name):
    return f[name]


def test_pool_copies_unshared_snapshot():
    f = pynbody.new(dm=20000)
    f['pos'] = np.random.uniform(size=(20000, 3))
    owned_before = shared.get_num_shared_arrays_owned()
    with pytest.warns(RuntimeWarning, match="enable_shared_arrays"):
        pool = pynbody.ProcessPool(f, processes=1, start_method='spawn')
    with pool:
        npt.assert_array_equal(pool.map(_get_array, ['pos'])[0], f['pos'])
    assert shared.get_num_shared_arrays_owned() == owned_before


def _number_of_particles(f, item):
    return len(f), item


def test_pool_refuses_view(snap_and_halos):
    f, h = snap_and_halos
    with pytest.raises(TypeError, match="whole snapshot"):
        pynbody.ProcessPool(f.dm, processes=1)


def test_pool_refuses_interactive_function(snap_and_halos, monkeypatch):
    f, h = snap_and_halos
    monkeypatch.setitem(sys.modules, '__main__', types.ModuleType('__main__'))

    def interactive(f, item):
        return item
    interactive.__module__ = '__main__'

    with pynbody.ProcessPool(f, processes=1, start_method='spawn') as pool:
        with pytest.raises(ValueError, match="defined interactively"):
            pool.map(interactive, [1])


@pytest.mark.skipif(sys.platform == 'win32', reason="fork is not available on Windows")
def test_pool_with_fork(snap_and_halos):
    f, h = snap_and_halos
    with pynbody.ProcessPool(f, processes=2, start_method='fork') as pool:
        assert pool.map(_number_of_particles, [1, 2], chunksize=2) == [(len(f), 1), (len(f), 2)]


def test_documentation_example():
    """The template in the parallelism tutorial runs as a script"""
    tests_dir = pathlib.Path(__file__).parent
    script = tests_dir.parent / "docs" / "tutorials" / "example_code" / "process_pool_halos.py"
    if not script.exists():
        pytest.skip("documentation is not present")
    result = subprocess.run([sys.executable, str(script)], cwd=tests_dir, capture_output=True, text=True,
                            timeout=600)
    assert result.returncode == 0, result.stderr
    assert "Halo 9:" in result.stdout
