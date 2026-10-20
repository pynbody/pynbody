"""Tests of pynbody.ProcessPool, which shares snapshots, halo catalogues and trees with worker processes.

The functions run by the pools are at module level, so that worker processes started by 'spawn' (the only method
available on all platforms) can find them."""

import gc
import os
import pathlib
import subprocess
import sys
import textwrap
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


def test_packer_does_not_confuse_arrays():
    """Arrays in a state may be temporaries, so one may be freed and its memory reused by another"""
    packer = pynbody.util.processes._SharedMemoryPacker()
    first = packer(np.full(20000, 1.0))
    second = packer(np.full(20000, 2.0))
    assert shared.from_shared_reference(second)[0] == 2.0
    assert shared.from_shared_reference(first)[0] == 1.0
    packer.release()


def _halo_lengths(h, halo_number):
    return len(h[halo_number]), h.base.families()


def test_pool_shares_family_halos():
    f = pynbody.new(dm=1000, star=500)
    f.dm['grp'] = np.arange(1000) % 3
    dm = f.dm  # the catalogue refers to its base only weakly, so it must be kept alive
    h = pynbody.halo.number_array.HaloNumberCatalogue(dm, array='grp')
    with pynbody.ProcessPool(h, processes=1) as pool:
        lengths = pool.map(_halo_lengths, [1, 2])
    assert lengths == [(len(h[1]), [pynbody.family.dm]), (len(h[2]), [pynbody.family.dm])]


def test_pool_warns_about_bare_number(snap_and_halos):
    f, h = snap_and_halos
    with pytest.warns(UserWarning, match="processes=4"):
        pynbody.ProcessPool(f, 4, processes=1).shutdown()


_crash_script = textwrap.dedent("""
    import os, concurrent.futures
    import numpy as np
    import pynbody

    def crash(f, item):
        os._exit(1)

    if __name__ == '__main__':
        f = pynbody.new(dm=100000)
        f.enable_shared_arrays()
        f['pos'] = np.zeros((100000, 3))
        with pynbody.ProcessPool(f, processes=2, start_method=START_METHOD) as pool:
            try:
                pool.map(crash, [1, 2, 3, 4])
            except concurrent.futures.process.BrokenProcessPool:
                print("broken pool reported")
""")


@pytest.mark.parametrize('start_method', ['spawn', pytest.param('fork', marks=pytest.mark.skipif(
    sys.platform == 'win32', reason="fork is not available on Windows"))])
def test_pool_survives_worker_death(start_method, tmp_path):
    """A worker that dies is reported, rather than hanging the pool (as it did with fork, when forked workers
    inherited a SIGTERM handler that ignored the pool's attempts to stop them)"""
    script = tmp_path / "crash.py"
    script.write_text(_crash_script.replace("START_METHOD", repr(start_method)))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert "broken pool reported" in result.stdout, result.stderr
    assert result.returncode == 0


@pytest.mark.skipif(sys.platform == 'win32', reason="SIGTERM cannot be caught on Windows")
def test_sigterm_terminates_after_cleanup(tmp_path):
    script = tmp_path / "sleeper.py"
    script.write_text(textwrap.dedent("""
        import sys, time
        import pynbody
        ar = pynbody.array.shared.make_shared_array(10, float)
        print(ar._shared_fname, flush=True)
        time.sleep(120)
    """))
    with subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True) as process:
        name = process.stdout.readline().strip()
        process.terminate()
        assert process.wait(timeout=30) != 0
    if os.path.isdir("/dev/shm"):
        assert not os.path.exists("/dev/shm/" + name)


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
