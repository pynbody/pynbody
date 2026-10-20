import concurrent.futures
import threading
import time

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
from pynbody.sph import renderers
from pynbody.util import threads
from pynbody.util.bulk_read import pipeline


@pytest.fixture
def four_threads():
    """Configure pynbody to use four threads, so that limits below that can be seen to take effect"""
    saved = pynbody.config['number_of_threads']
    pynbody.config['number_of_threads'] = 4
    yield
    pynbody.config['number_of_threads'] = saved


@pytest.fixture
def snap():
    np.random.seed(1)
    f = pynbody.new(gas=5000)
    f['pos'] = np.random.uniform(-0.5, 0.5, size=(5000, 3))
    f['mass'] = np.ones(5000)
    f['vel'] = np.random.normal(size=(5000, 3))
    return f


def test_thread_limit(four_threads):
    assert threads.current_limit() is None
    assert threads.compute_threads() == 4
    with pynbody.thread_limit() as limit:
        assert limit.n == 1
        assert threads.compute_threads() == 1
        assert threads.compute_threads(8) == 1
        with pynbody.thread_limit(3):
            # the smallest limit applies
            assert threads.current_limit() == 1
    assert threads.current_limit() is None

    with pynbody.thread_limit(2):
        assert threads.compute_threads() == 2
        assert threads.compute_threads(1) == 1  # (never more threads than asked for)
    assert threads.compute_threads() == 4


@pytest.mark.parametrize("n", [0, -1, 1.5])
def test_thread_limit_rejects_nonsense(n):
    with pytest.raises(ValueError):
        pynbody.thread_limit(n)


def test_thread_limit_applies_to_other_threads(four_threads):
    with pynbody.thread_limit(2):
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            assert pool.submit(threads.compute_threads).result() == 2

    # a limit entered in another thread applies here too, until that thread leaves it
    entered, leave = threading.Event(), threading.Event()

    def limited():
        with pynbody.thread_limit(3):
            entered.set()
            assert leave.wait(timeout=30)

    thread = threading.Thread(target=limited)
    thread.start()
    assert entered.wait(timeout=30)
    assert threads.compute_threads() == 3
    leave.set()
    thread.join()
    assert threads.compute_threads() == 4


def test_kdtree_within_limit(four_threads, snap, monkeypatch):
    snap['rho']  # (builds the tree, and gives it smoothing lengths and densities)
    kd = snap.kdtree
    rho_free = kd.sph_mean(snap['vel'], 32).copy()

    built_with = []
    original_build = pynbody.kdtree.kdmain.build
    monkeypatch.setattr(pynbody.kdtree.kdmain, "build",
                        lambda *args: (built_with.append(args[-1]), original_build(*args))[1])
    mapped_over = []
    original_map = pynbody.util.thread_map
    monkeypatch.setattr(pynbody.util, "thread_map",
                        lambda func, *args: (mapped_over.append(len(args[0])), original_map(func, *args))[1])

    # a tree built before the block is entered still respects it
    with pynbody.thread_limit():
        rho_limited = kd.sph_mean(snap['vel'], 32)
        pynbody.kdtree.KDTree(snap['pos'], snap['mass'])
    assert mapped_over == [] and built_with == [1]
    npt.assert_allclose(rho_limited, rho_free)

    with pynbody.thread_limit(2):
        kd.sph_mean(snap['vel'], 32)
        list(kd.pair_blocks(blocksize=1000))
    assert mapped_over and all(n <= 2 for n in mapped_over)


def test_threaded_renderer_within_limit(four_threads, snap, monkeypatch):
    def render():
        return renderers.make_render_pipeline(snap, quantity='rho', width=1.0, resolution=50,
                                              threaded=True).render()

    image_free = render()

    pools = []

    class RecordingPool(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, max_workers=None, **kwargs):
            pools.append(max_workers)
            super().__init__(max_workers, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", RecordingPool)
    with pynbody.thread_limit():
        image_limited = render()
    assert pools == []
    npt.assert_allclose(image_limited, image_free, rtol=1e-5)

    with pynbody.thread_limit(2):
        render()
    assert pools == [2]


def test_openmp_routines_consult_limit(four_threads, snap, monkeypatch):
    consulted = []
    monkeypatch.setattr(threads, "current_limit", lambda: (consulted.append(True), 1)[1])

    ar = np.random.uniform(size=1000)
    npt.assert_allclose(pynbody.util._util.sum(ar), ar.sum())
    assert len(consulted) == 1

    selected = snap[pynbody.filt.Sphere(0.3)]
    assert (np.linalg.norm(selected['pos'], axis=1) < 0.3).all()
    assert len(consulted) == 2

    pynbody.analysis.halo.shrink_sphere_center(snap)
    assert len(consulted) == 3

    pynbody.gravity.direct(snap, np.zeros((1, 3)), eps=0.01)
    assert len(consulted) == 4

    sorted_ar = np.arange(100)
    found = pynbody.util._util.binary_search(sorted_ar[10:20], sorted_ar, np.arange(100))
    npt.assert_array_equal(found, np.arange(10, 20))
    assert len(consulted) == 5


@pytest.mark.parametrize("strategy, limit, decode_threads", [
    ((1, 0), 1, 0),  # already serial
    ((1, 4), 1, 1),  # the calling thread reads, one thread decodes
    ((1, 4), 2, 2),
    ((1, 4), 8, 4),  # (never more than the strategy chose)
    ((4, 0), 4, 0),  # input threads decode, and there are few enough of them
    ((4, 0), 2, 2),  # too many input threads to decode, so two decode threads do it instead
    ((4, 0), 1, 1),
    ((4, 8), 3, 3),
    ((16, 2), 1, 1),  # (the input threads are not limited)
])
def test_read_strategy_within_limit(strategy, limit, decode_threads):
    io_threads, original_decode_threads = strategy
    original = pipeline.ReadStrategy(io_threads, original_decode_threads, "test", inflight_nbytes=1000)
    assert pipeline.apply_thread_limit(original) is original
    with pynbody.thread_limit(limit):
        limited = pipeline.apply_thread_limit(original)
    assert limited.io_threads == io_threads
    assert limited.decode_threads == decode_threads
    assert limited.inflight_nbytes == original.inflight_nbytes


@pytest.mark.parametrize("decode_threads", [0, 4])
def test_pipeline_decodes_within_limit(decode_threads):
    """Under a limit of 2, input threads only read, and no more than two threads decode at once"""
    lock = threading.Lock()
    fetching_threads, decoding_threads = set(), set()
    decoding_now = [0, 0]  # (now, most at once)

    def make_job():
        def fetch():
            with lock:
                fetching_threads.add(threading.current_thread().name)
            return True

        def process(fetched):
            with lock:
                decoding_threads.add(threading.current_thread().name)
                decoding_now[0] += 1
                decoding_now[1] = max(decoding_now)
            time.sleep(0.002)
            with lock:
                decoding_now[0] -= 1

        return pipeline.Job(fetch, process, 1)

    tasks = [[make_job() for _ in range(10)] for _ in range(4)]
    with pynbody.thread_limit(2):
        pipeline.perform_tasks(tasks, pipeline.ReadStrategy(4, decode_threads, "test", inflight_nbytes=1000))

    assert len(fetching_threads) > 1 and all(name.startswith("pynbody-input") for name in fetching_threads)
    assert all(name.startswith("pynbody-decode") for name in decoding_threads)
    assert decoding_now[1] <= 2


def test_threadpoolctl_limits_pynbody(four_threads):
    threadpoolctl = pytest.importorskip("threadpoolctl")
    if not hasattr(threadpoolctl, "register"):
        pytest.skip("threadpoolctl too old to control pynbody")

    info = [lib for lib in threadpoolctl.threadpool_info() if lib['user_api'] == 'pynbody']
    assert len(info) == 1 and info[0]['num_threads'] == 4

    with threadpoolctl.threadpool_limits(2):
        assert threads.compute_threads() == 2
        with pynbody.thread_limit(1):
            assert threads.compute_threads() == 1
        assert threads.compute_threads() == 2
    assert threads.current_limit() is None

    # entering threadpool_limits inside a thread_limit block must not leave a limit behind once both are left
    with pynbody.thread_limit(1):
        with threadpoolctl.threadpool_limits(limits=3, user_api='pynbody'):
            assert threads.compute_threads() == 1
    assert threads.current_limit() is None
    assert threads.compute_threads() == 4
