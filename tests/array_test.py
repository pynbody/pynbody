import pynbody
import pynbody.array as pyn_array
import pynbody.array.shared as shared
import pynbody.units as units

SA = pynbody.array.SimArray
import gc
import os
import signal
import sys
import time

import numpy as np
import numpy.testing as npt
import pytest


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
    array = pynbody.array.shared._shared_array_reconstruct(array_info)
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
    p = context.Process(target=_test_and_alter_shared_value, args=(pyn_array.shared._shared_array_deconstruct(ar),))
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

def test_shared_array_ownership():
    """Test that we can have two copies of a shared array in a process, but that only the 'owner' cleans up the memory"""

    import pynbody.array as pyn_array

    baseline_num_shared_arrays = pyn_array.shared.get_num_shared_arrays_owned()  # hopefully zero, but we can't guarantee that
    ar = pyn_array.array_factory((10,), int, True, True)
    assert pyn_array.shared.get_num_shared_arrays_owned() == 1 + baseline_num_shared_arrays

    array_info = pyn_array.shared._shared_array_deconstruct(ar)
    ar2 = pynbody.array.shared._shared_array_reconstruct(array_info)
    del ar2

    gc.collect()

    # shouldn't have been deleted!
    assert pyn_array.shared.get_num_shared_arrays_owned() == 1 + baseline_num_shared_arrays



@pytest.fixture
def clean_up_test_protection():
    import posix_ipc
    try:
        posix_ipc.unlink_shared_memory("pynbody-test-cleanup")
    except posix_ipc.ExistentialError:
        pass
    yield
    try:
        posix_ipc.unlink_shared_memory("pynbody-test-cleanup")
    except posix_ipc.ExistentialError:
        pass

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
