"""Tests of pynbody.ProcessPool, which shares snapshots, halo catalogues and trees with worker processes.

The functions run by the pools are at module level, so that worker processes started by 'spawn' (the only method
available on all platforms) can find them."""

import gc
import os
import pathlib
import subprocess
import sys
import textwrap
import time
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


@pytest.mark.parametrize('serve', [True, False])
def test_pool_worker_state(snap_and_halos, serve):
    f, h = snap_and_halos
    with pynbody.ProcessPool(f, h, processes=1, start_method='spawn', threads_per_process=2, serve=serve) as pool:
        description = pool.map(_describe_worker, [None])[0]
    assert description['snapshot_type'] == ('ServedSimSnap' if serve else 'PortableSimSnap')
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
        with pytest.raises(KeyError):
            pool.map(_load_unloaded, ['nonexistent_array'])
        with pytest.raises(ValueError, match="read-only"):
            pool.map(_write, ['pos'])
        with pytest.raises(ValueError, match="get_copy_on_access_simsnap"):
            pool.submit(_centre_in_place, 1).result()
        # the pool remains usable
        assert len(pool.map(_load_unloaded, ['mass'])[0]) == len(f)


def test_frozen_pool_refuses_unloaded_arrays(snap_and_halos):
    f, h = snap_and_halos
    with pynbody.ProcessPool(f, h, processes=1, start_method='spawn', serve=False) as pool:
        with pytest.raises(KeyError, match="'iord' was not loaded"):
            pool.map(_load_unloaded, ['iord'])
    assert 'iord' not in f.keys()


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
    packer = pynbody.parallel.transport.SharedMemoryTransport()
    first = packer.encode(np.full(20000, 1.0))
    second = packer.encode(np.full(20000, 2.0))
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


# Tests of serving: the main process provides what the workers' snapshots lack


@pytest.fixture
def fresh_snap_and_halos():
    """A snapshot with nothing loaded, as a user would have before creating a pool"""
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    f.physical_units()
    h = f.halos()
    yield f, h
    del f, h
    gc.collect()


def _served_analysis(f, h, halo_number):
    halo = h[halo_number]
    f.build_tree()
    centre = pynbody.analysis.halo.shrink_sphere_center(halo)
    local = halo.get_copy_on_access_simsnap()
    pynbody.analysis.center(local)
    return {'iord_sum': int(halo['iord'].sum()),
            'rxy_sum': float(halo['rxy'].sum()),
            'num_in_sphere': len(f[pynbody.filt.Sphere('20 kpc', centre)]),
            'local_vel_centre': np.asarray(local['vel'][:5]),
            'gas_HI': np.asarray(halo.gas['HI'][:5])}


@pytest.mark.parametrize('start_method', ['spawn', pytest.param('fork', marks=pytest.mark.skipif(
    sys.platform == 'win32', reason="fork is not available on Windows"))])
def test_pool_serves_arrays_trees_and_derived_arrays(fresh_snap_and_halos, start_method):
    f, h = fresh_snap_and_halos
    assert 'pos' not in f.keys()
    with pynbody.ProcessPool(f, h, processes=2, start_method=start_method) as pool:
        results = pool.map(_served_analysis, [1, 2, 3])

    # The main process now holds what the workers asked for, in shared memory
    assert isinstance(f['iord'], shared.SharedMemorySimArray)
    assert f.is_derived_array('rxy')
    assert hasattr(f, 'kdtree')

    for halo_number, result in zip([1, 2, 3], results):
        expected = _served_analysis(f, h, halo_number)
        assert result['iord_sum'] == expected['iord_sum']
        npt.assert_allclose(result['rxy_sum'], expected['rxy_sum'])
        assert result['num_in_sphere'] == expected['num_in_sphere']
        npt.assert_allclose(result['local_vel_centre'], expected['local_vel_centre'])
        npt.assert_allclose(result['gas_HI'], expected['gas_HI'])


def _iord_sum(f, _):
    return int(f['iord'].sum())


def test_pool_serves_concurrent_requests(fresh_snap_and_halos):
    f, h = fresh_snap_and_halos
    with pynbody.ProcessPool(f, processes=3) as pool:
        sums = pool.map(_iord_sum, range(6))
    assert sums == [int(f['iord'].sum())] * 6


@pynbody.derived_array
def _process_pool_test_doubled(sim):
    return sim['_process_pool_test_worker_array'] * 2


def _derive_from_worker_array(f, _):
    f['_process_pool_test_worker_array'] = np.arange(len(f), dtype=float)
    return np.asarray(f['_process_pool_test_doubled'][:3])


def test_pool_derives_locally_what_server_cannot():
    f = pynbody.new(dm=100)
    with pynbody.ProcessPool(f, processes=1) as pool:
        assert list(pool.map(_derive_from_worker_array, [None])[0]) == [0., 2., 4.]
    assert '_process_pool_test_doubled' not in f.keys()


_worker_states_seen_by_server = []


def _process_states_once_settled(pids, stopped):
    """Process states, after waiting briefly for signals (which are delivered asynchronously) to take effect"""
    for _ in range(100):
        states = _process_states(pids)
        if all((state == 'T') == stopped for state in states):
            break
        time.sleep(0.01)
    return states


def _process_states(pids):
    states = []
    for pid in pids:
        with open(f"/proc/{pid}/stat") as stat:
            # the state follows the command name, which is in brackets
            states.append(stat.read().rsplit(")", 1)[1].split()[0])
    return states


@pynbody.derived_array
def _process_pool_test_pause_check(sim):
    _worker_states_seen_by_server.extend(
        _process_states_once_settled(_pool_being_tested._listener.worker_pids(), stopped=True))
    return np.zeros(len(sim))


def _ask_for_pause_check(f, _):
    return float(f['_process_pool_test_pause_check'].sum())


def _sleep(f, seconds):
    time.sleep(seconds)


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason="process states are read from /proc")
def test_pool_pauses_workers():
    global _pool_being_tested
    f = pynbody.new(dm=100)
    with pynbody.ProcessPool(f, processes=2) as pool:
        _pool_being_tested = pool
        # two tasks that overlap, so that both processes are started
        pool.map(_sleep, [0.5, 0.5])
        pids = pool._listener.worker_pids()
        assert len(pids) == 2

        # deriving an array in the main process pauses all the workers, including the one that asked for it
        _worker_states_seen_by_server.clear()
        assert pool.map(_ask_for_pause_check, [None]) == [0.0]
        assert _worker_states_seen_by_server == ['T', 'T']
        assert 'T' not in _process_states_once_settled(pids, stopped=False)

        # the workers resume even if the work done while they are paused fails
        with pytest.raises(ZeroDivisionError):
            with pool._paused_workers():
                assert _process_states_once_settled(pids, stopped=True) == ['T', 'T']
                1 / 0
        assert 'T' not in _process_states_once_settled(pids, stopped=False)


class _DirectChannel:
    """Stands in for a channel between processes, by calling the server directly"""
    def __init__(self, server):
        self._server = server

    def request(self, name, *args):
        return self._server.handle(name, args)


def test_server_with_copy_transport(fresh_snap_and_halos):
    """The server and served snapshot don't depend on shared memory, or on processes"""
    from pynbody.parallel.served import ServedSimSnap
    from pynbody.parallel.server import SnapshotServer
    from pynbody.parallel.transport import CopyTransport

    f, h = fresh_snap_and_halos
    transport = CopyTransport()
    server = SnapshotServer([f], transport)
    state = transport.decode_state(transport.encode_state(f.get_portable_state()))
    served = ServedSimSnap(state, _DirectChannel(server), (0, None), transport)

    npt.assert_array_equal(served['iord'], f['iord'])
    assert not np.shares_memory(served['iord'], f['iord'])
    assert not served['iord'].flags['WRITEABLE']
    npt.assert_array_equal(served.gas['HI'], f.gas['HI'])
    npt.assert_allclose(served['rxy'], f['rxy'])
    assert served.is_derived_array('rxy')
    served.build_tree()
    sphere = pynbody.filt.Sphere('200 kpc', f['pos'][1000])
    npt.assert_array_equal(served[sphere].get_index_list(served), f[sphere].get_index_list(f))
    with pytest.raises(KeyError):
        served['nonexistent_array']
