import pynbody
import pynbody.array as pyn_array
import pynbody.array.shared as shared
import pynbody.test_utils
import pynbody.units as units

SA = pynbody.array.SimArray
import gc
import os
import platform
import signal
import sys
import time

import numpy as np
import numpy.testing as npt
import pytest


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gadget")

class _FakeSim:
    """Minimal SimSnap-like object that returns conversion_context."""
    def __init__(self, *args, **kwargs):
        self._conversion_context = dict(*args, **kwargs)

    def conversion_context(self):
        return self._conversion_context

    @property
    def ancestor(self):
        return self

    def __getitem__(self, key):
        # For family routing in SimArray.sim
        return self

def test_pickle():
    import pickle
    x = SA([1, 2, 3, 4], units='kpc')
    assert str(x.units) == 'kpc'
    y = pickle.loads(pickle.dumps(x))
    assert y[3] == 4
    assert str(y.units) == 'kpc'

def test_issue255() :
    r = SA(0.5, 'au')
    assert isinstance(r**2, SA)
    assert float(r**2)==0.25
    assert str((r**2).units)=="au**2"

def test_sim_attachment():
    f = pynbody.new(dm=10)
    f['test'] = SA(np.arange(10), 'kpc')
    f['test2'] = SA(np.arange(10), 'kpc')
    assert f['test'].sim is f
    assert f['test2'].sim is f

    # sim ownership should be propagated through arithmetic operations
    assert (f['test'] + f['test2']).sim is f

    f['pos'] = SA(np.random.rand(10, 3), 'kpc')
    assert f['pos'].sim is f
    assert np.linalg.norm(f['pos'], axis=1).sim is f

    f2 = pynbody.new(dm=10)
    f2['test_other'] = SA(np.arange(10), 'kpc')
    assert f2['test_other'].sim is f2

    assert (f['test'] + f2['test_other']).sim is f


def test_return_types():

    x = SA([1, 2, 3, 4])
    y = SA([2, 3, 4, 5])

    assert type(x) is SA
    assert type(x ** 2) is SA
    assert type(x + y) is SA
    assert type(x * y) is SA
    assert type(x ** y) is SA
    assert type(2 ** x) is SA
    assert type(x + 2) is SA
    assert type(x[:2]) is SA

    x2d = SA([[1, 2, 3, 4], [5, 6, 7, 8]])

    assert type(x2d.sum(axis=1)) is SA

def test_add_iop_to_plain_array():
    x = np.array([1,2,3])
    y = SA([1,2,3], "kpc")
    x+=y
    assert (x == [2,4,6]).all()

def test_add_iop_with_conversion_context():
    f = pynbody.new(dm=10)
    f.properties['a'] = 0.5
    f['pos'] = SA(np.random.rand(10, 3), 'kpc a')
    f['pos2'] = SA(np.random.rand(10, 3), 'kpc')

    pos3 = f['pos'] + f['pos2']
    assert pos3.units == 'kpc a'
    npt.assert_allclose(pos3, f['pos'] + f['pos2'].in_units('kpc a'))

def test_unit_tracking():

    x = SA([1, 2, 3, 4])
    x.units = "kpc"

    y = SA([5, 6, 7, 8])
    y.units = "Mpc"

    assert abs((x * y).units.ratio("kpc Mpc") - 1.0) < 1.e-9

    assert ((x ** 2).units.ratio("kpc**2") - 1.0) < 1.e-9 # ... translates to np.square

    assert ((x ** (3,2)).units.ratio("kpc**3/2") - 1.0) < 1.e-9  # ... translates to np.power

    npt.assert_allclose(x**(3,2), x.view(np.ndarray)**1.5) # double check the right power was taken

    assert ((x / y).units.ratio("") - 1.e-3) < 1.e-12

    if hasattr(np.mean(x), 'units'):
        assert np.var(x).units == "kpc**2"
        assert np.std(x).units == "kpc"
        assert np.mean(x).units == "kpc"


def test_iop_units():
    x = SA([1, 2, 3, 4], dtype=float)
    x.units = 'kpc'

    y = SA([2, 3, 4, 5], dtype=float)
    y.units = 'km s^-1'

    z = SA([1000, 2000, 3000, 4000], dtype=float)
    z.units = 'm s^-1'

    assert repr(x) == "SimArray([1., 2., 3., 4.], 'kpc')"

    with pytest.raises(pynbody.units.UnitsException):
        x += y

    x *= pynbody.units.Unit('K')

    assert x.units == 'K kpc'

    x.units = 'kpc'

    x *= y

    assert x.units == 'km s^-1 kpc'
    npt.assert_allclose(x, [2, 6, 12, 20])

    y += z
    assert y.units == 'km s^-1'

    npt.assert_allclose(y, [3, 5, 7, 9])


def test_iop_sanity():
    x = SA([1.0, 2.0, 3.0, 4.0])
    x_id = id(x)
    x += 1
    assert id(x) == x_id
    x -= 1
    assert id(x) == x_id
    x *= 2
    assert id(x) == x_id
    x /= 2
    assert id(x) == x_id
    
def test_in_units_normal_ratio():
    x = SA([1.0, 2.0, 3.0], 'a kpc')
    y = x.in_units('pc', a=0.001)
    assert y.units == 'pc'
    npt.assert_allclose(y, np.array([1.0, 2.0, 3.0]), rtol=1e-12, atol=0)
    
    y = x.in_units('pc', a = np.array([0.001]))
    npt.assert_allclose(y, np.array([1.0, 2.0, 3.0]), rtol=1e-12, atol=0)
    
    y = x.in_units('pc', a = SA(np.array([0.001])))
    npt.assert_allclose(y, np.array([1.0, 2.0, 3.0]), rtol=1e-12, atol=0)

def test_in_units_array1d_ratio1d():
    x = SA([1.0, 2.0, 3.0], 'a kpc')
    a = np.array([0.2, 0.3, 0.4])
    y = x.in_units('kpc', a=a)
    assert y.units == 'kpc'
    npt.assert_allclose(y, np.array([0.2, 0.6, 1.2]), rtol=1e-12, atol=0)

def test_in_units_array3d_ratio1d():
    N = 5
    x = SA(np.arange(1, N*3 + 1, dtype=float).reshape(N, 3), 'a kpc')
    a = np.linspace(0.1, 0.5, N)
    y = x.in_units('kpc', a=a)
    expected = x.view(np.ndarray) * a.reshape(N, 1)
    assert y.units == 'kpc'
    npt.assert_allclose(y, expected, rtol=1e-12, atol=0)
    
def test_in_units_array_ratio_mismatch_raises():
    x = SA(np.ones(5), 'a kpc')
    with pytest.raises(ValueError):
        _ = x.in_units('kpc', a=np.ones(4))
        _ = x.in_units('kpc', a=np.ones((5,2)))
        
    x = SA(np.ones(1), 'a kpc')
    with pytest.raises(ValueError):
        _ = x.in_units('kpc', a=np.ones(2))

def test_add_with_array_ratio_context_broadcast():
    a = np.array([0.2, 0.3, 0.4, 0.5])
    x = SA(np.ones((4, 3)), 'a kpc')
    y = SA(2.0 * np.ones((4, 3)), 'kpc')

    fake = _FakeSim(a=a)
    x.sim = fake
    y.sim = fake

    z = x + y
    expected = x.view(np.ndarray) + y.view(np.ndarray) * (1.0 / a).reshape(4, 1)
    assert z.units == 'a kpc'
    npt.assert_allclose(z, expected, rtol=1e-12, atol=0)


def test_unit_array_interaction():
    """Test for issue 113 and related"""
    x = pynbody.units.Unit('1 Mpc')
    y = SA(np.ones(10), 'kpc')
    npt.assert_allclose(x + y, SA([1.001] * 10, 'Mpc'))
    npt.assert_allclose(x - y, SA([0.999] * 10, 'Mpc'))

    assert (x + y).units == 'Mpc'

    npt.assert_allclose(y + x, SA([1001] * 10, 'kpc'))
    npt.assert_allclose(y - x, SA([-999.] * 10, 'kpc'))

def test_norm_units():
    x = SA(np.ones((10, 3) ), "kpc")
    result = np.linalg.norm(x, axis=1)
    npt.assert_allclose(result, np.ones(10) * np.sqrt(3), rtol=1.e-5)
    assert result.units == "kpc"

def test_gradient_units():

    # general use
    x = SA([1.0, 2.0, 3.0, 4.0], "kpc")
    result = np.gradient(x)
    assert result.units == "kpc"
    npt.assert_allclose(result, np.ones(4), rtol=1.e-5)

    # two SimArray inputs
    t = SA(np.arange(4)*2,"Gyr")
    result = np.gradient(x, t)
    assert result.units == "kpc Gyr**-1"
    npt.assert_allclose(result, np.ones(4)*0.5, rtol=1.e-5)

    # ndim > 1, tuple of SimArray, (or list for numpy 1.26.4)
    xg = np.gradient(SA(np.array([[1, 2.,3.],
                               [1, 2.,3.]]),"kpc"),)
    assert all(isinstance(gi, SA) for gi in xg)
    assert all(gi.units == "kpc" for gi in xg)

    # different units along each axis
    v = SA([[1, 2, 3], [4, 5, 6]], 'K')
    y = SA([1, 2], 'km')
    t = SA([1, 2, 3], 's')
    xg = np.gradient(v, y, t)
    assert xg[0].units == 'K km**-1'
    assert xg[1].units == 'K s**-1'


def test_dimensionful_comparison():
    # check that dimensionful units compare correctly
    # see issue 130
    a1 = SA(np.ones(2), 'kpc')
    a2 = SA(np.ones(2) * 2, 'pc')
    assert (a2 < a1).all()
    assert not (a2 > a1).any()
    a2 = SA(np.ones(2) * 1000, 'pc')
    assert (a1 == a2).all()
    assert (a2 <= a2).all()

    a2 = SA(np.ones(2), 'Msol')
    with pytest.raises(pynbody.units.UnitsException):
        a2 < a1

    a2 = SA(np.ones(2))
    with pytest.raises(pynbody.units.UnitsException):
        a2 < a1

    assert (a1 < pynbody.units.Unit("0.5 Mpc")).all()
    assert (a1 > pynbody.units.Unit("400 pc")).all()

    # now check with subarrays

    x = pynbody.new(10)
    x['a'] = SA(np.ones(10), 'kpc')
    x['b'] = SA(2 * np.ones(10), 'pc')

    y = x[[1, 2, 5]]

    assert (y['b'] < y['a']).all()
    assert not (y['b'] > y['a']).any()

def test_squeeze_units():
    x = SA([[1.0, 2.0, 3.0]], "kpc")
    assert np.squeeze(x).units == "kpc"

def test_issue_485_1():
    s = pynbody.load("testdata/gadget2/test_g2_snap.1")
    stars = s.s
    indexed_arr = stars[1,2]
    np.testing.assert_almost_equal(np.sum(indexed_arr['vz'].in_units('km s^-1')), -20.13701057434082031250)
    np.testing.assert_almost_equal(np.std(indexed_arr['vz'].in_units('km s^-1')), 11.09318065643310546875)

def test_issue_485_2():
    # Adaptation of examples/vdisp.py
    s = pynbody.load("testdata/gadget2/test_g2_snap.1")

    stars = s.s
    rxyhist, rxybins = np.histogram(stars['rxy'], bins=20)
    rxyinds = np.digitize(stars['rxy'], rxybins)
    nrbins = len(np.unique(rxyinds))
    sigvz = np.zeros(nrbins)
    sigvr = np.zeros(nrbins)
    sigvt = np.zeros(nrbins)
    rxy = np.zeros(nrbins)

    assert len(np.unique(rxyinds)) == 3
    for i, ind in enumerate(np.unique(rxyinds)):
        bininds = np.where(rxyinds == ind)
        sigvz[i] = np.std(stars[bininds]['vz'].in_units('km s^-1'))
        sigvr[i] = np.std(stars[bininds]['vr'].in_units('km s^-1'))
        sigvt[i] = np.std(stars[bininds]['vt'].in_units('km s^-1'))
        rxy[i] = np.mean(stars[bininds]['rxy'].in_units('kpc'))


    np.testing.assert_allclose(sigvz, np.array([19.68325233, 29.49512482,  0.]), rtol=1e-6)
    np.testing.assert_allclose(sigvr, np.array([25.64306641, 26.01454544,  0.]), rtol=1e-6)
    np.testing.assert_allclose(sigvt, np.array([28.49997711, 18.84262276,  0.]), rtol=1e-6)
    np.testing.assert_allclose(rxy, np.array([1136892.125, 1606893.625, 1610494.75]), rtol=1e-6)

def _test_and_alter_shared_value(array_info):
    array = pynbody.array.shared.unpack(array_info)
    assert (array[:] == np.arange(3)[: , np.newaxis] * np.arange(5)[np.newaxis, :]).all()
    array[:] = np.arange(3)[:, np.newaxis]

def test_shared_arrays():

    gc.collect() # this is to start with a clean slate, get rid of any shared arrays that might be hanging around
    baseline_num_shared_arrays = pyn_array.shared.get_num_shared_arrays_owned() # hopefully zero, but we can't guarantee that

    ar = pyn_array.array_factory((3, 5), dtype=np.float32, zeros=True, shared=True)

    assert ar.shape == (3,5)
    assert (ar == 0.0).all()

    ar[:] = np.arange(3)[: , np.newaxis] * np.arange(5)[np.newaxis, :]

    # now let's see if we can transfer it to another process:

    import multiprocessing as mp
    context = mp.get_context('spawn')
    p = context.Process(target=_test_and_alter_shared_value, args=(pyn_array.shared.pack(ar),))
    p.start()
    p.join()

    # check that the other process has successfully changed our value:
    assert (ar[:] ==  np.arange(3)[:, np.newaxis] ).all()

    assert pyn_array.shared.get_num_shared_arrays_owned() == 1 + baseline_num_shared_arrays

    ar2 = pyn_array.array_factory((3, 5), dtype=np.float32, zeros=True, shared=True)
    assert pyn_array.shared.get_num_shared_arrays_owned() == 2 + baseline_num_shared_arrays

    del ar, ar2
    gc.collect()

    assert pyn_array.shared.get_num_shared_arrays_owned() == baseline_num_shared_arrays

def test_shared_array_with_stride():
    ar = pyn_array.array_factory((9, 3), dtype=np.float32, zeros=True, shared=True)

    decon = shared._recursive_shared_array_deconstruct(ar[::2])
    recon = shared._recursive_shared_array_reconstruct(decon)
    assert recon.shape == (5, 3)

    recon[0, :] = 1.0
    recon[1, :] = 2.0
    assert (ar[0, :] == 1.0).all()
    assert (ar[2, :] == 2.0).all()
    assert (ar[1, :] == 0.0).all()
    

def test_shared_array_ownership():
    """Test that we can have two copies of a shared array in a process, but that only the 'owner' cleans up the memory"""

    import pynbody.array as pyn_array

    baseline_num_shared_arrays = pyn_array.shared.get_num_shared_arrays_owned()  # hopefully zero, but we can't guarantee that
    ar = pyn_array.array_factory((10,), int, True, True)
    assert pyn_array.shared.get_num_shared_arrays_owned() == 1 + baseline_num_shared_arrays

    array_info = pyn_array.shared.pack(ar)
    ar2 = pynbody.array.shared.unpack(array_info)
    del ar2

    gc.collect()

    # shouldn't have been deleted!
    assert pyn_array.shared.get_num_shared_arrays_owned() == 1 + baseline_num_shared_arrays



@pytest.fixture
def clean_up_test_protection():
    import platform
    if platform.system() == 'Windows':
        # On Windows, shared memory is automatically cleaned up when all handles are closed
        # We don't need to explicitly unlink like on POSIX systems
        yield
    else:
        from pynbody.array.shared.posix_detail import unlink_shared_memory
        unlink_shared_memory("pynbody-test-cleanup")
        yield
        unlink_shared_memory("pynbody-test-cleanup")
 
def _test_shared_arrays_cleaned_on_exit():
    global ar
    ar = shared.make_shared_array((10,), dtype=np.int32, zeros=True, fname="pynbody-test-cleanup")
    # intentionally don't delete it, to see if it gets cleaned up on exit

def test_shared_arrays_cleaned_on_exit(clean_up_test_protection):
    _run_function_externally("_test_shared_arrays_cleaned_on_exit")

    _assert_shared_memory_cleaned_up()


def _test_shared_arrays_cleaned_on_terminate():
    # designed to run in a completely separate python process (i.e. not a subprocess of the test process)
    ar = shared.make_shared_array((10,), dtype=np.int32, zeros=True, fname="pynbody-test-cleanup")

    # send SIGTERM to ourselves:
    os.kill(os.getpid(), signal.SIGTERM)

    # wait to die...
    time.sleep(2.0)


def test_shared_arrays_cleaned_on_kill(clean_up_test_protection):
    stderr = _run_function_externally("_test_shared_arrays_cleaned_on_terminate")
    _assert_shared_memory_cleaned_up()


def _assert_shared_memory_cleaned_up():
    with pytest.raises(pynbody.array.shared.SharedArrayNotFound):
        _ = shared.make_shared_array((10,), dtype=np.int32, zeros=False,
                                     fname="pynbody-test-cleanup", create=False)


def _run_function_externally(function_name):
    pwd = os.path.dirname(__file__)
    python = sys.executable
    import subprocess
    process = subprocess.Popen([python, "-c",
                                f"from array_test import {function_name}; {function_name}()"]
                               , cwd=pwd)
    process.wait()


def test_ufunc_multi_input():
    # Test for any of the following creating an infinite recursion. See #844
    #
    # Note that arguably these should look at their units, but at the moment they don't. That's a secondary
    # issue and at least we don't hit infinite recursion any more.
    np.concatenate([SA([1, 2, 3]), SA([4, 5, 6])])
    np.vstack([SA([1, 2, 3]), SA([4, 5, 6])])
    np.hstack([SA([1, 2, 3]), SA([4, 5, 6])])


def test_shared_name_collision_raises(clean_up_test_protection):
    myar = shared.make_shared_array((10,), dtype=np.int32, zeros=True, fname="pynbody-test-cleanup")
    with pytest.raises(OSError):
        shared.make_shared_array((10,), dtype=np.int32, zeros=True, fname="pynbody-test-cleanup")

def _create_shared_array_with_random_name():
    """Create a shared array with a random name to avoid name collisions"""
    remote_ar = shared.make_shared_array((10,), dtype=np.int32, zeros=True)

@pytest.mark.skipif(platform.system() == 'Windows', reason="Windows does not support fork")
def test_shared_name_accidental_rng_collision():
    """Check that if the rng collides (this can happen after a fork made by multiprocessing),
    make_shared_array still succeeds"""

    # create a shared array to initialise the underlying rng
    first_local_ar = shared.make_shared_array((10,), dtype=np.int32)

    import multiprocessing as mp
    context = mp.get_context('fork')
    p = context.Process(target=_create_shared_array_with_random_name)
    p.start()

    # this will collide with the other process's shared array name if the rng is not
    # properly reinitialised
    second_local_ar = shared.make_shared_array((10,), dtype=np.int32)

    p.join()

    assert p.exitcode == 0, "Child process did not exit cleanly"
