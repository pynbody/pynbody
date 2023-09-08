import copy
import gc

import numpy as np
import numpy.testing as npt
import pytest

import pynbody


def setup_module():
    global f, original

    f = pynbody.new(dm=1000)
    f['pos'] = np.random.normal(scale=1.0, size=f['pos'].shape)
    f['vel'] = np.random.normal(scale=1.0, size=f['vel'].shape)
    f['mass'] = np.random.uniform(1.0, 10.0, size=f['mass'].shape)

    original = copy.deepcopy(f)


def test_translate():
    global f, original

    with pynbody.transformation.translate(f, [1, 0, 0]):
        npt.assert_almost_equal(f['pos'], original['pos'] + [1, 0, 0])

    # check moved back
    npt.assert_almost_equal(f['pos'], original['pos'])

    # try again with with abnormal exit
    try:
        with pynbody.transformation.translate(f, [1, 0, 0]):
            npt.assert_almost_equal(f['pos'], original['pos'] + [1, 0, 0])
            raise RuntimeError
    except RuntimeError:
        pass

    npt.assert_almost_equal(f['pos'], original['pos'])


def test_v_translate():
    global f, original

    with pynbody.transformation.v_translate(f, [1, 0, 0]):
        npt.assert_almost_equal(f['vel'], original['vel'] + [1, 0, 0])

    # check moved back
    npt.assert_almost_equal(f['vel'], original['vel'])

    # try again with with abnormal exit
    try:
        with pynbody.transformation.v_translate(f, [1, 0, 0]):
            npt.assert_almost_equal(f['vel'], original['vel'] + [1, 0, 0])
            raise RuntimeError
    except RuntimeError:
        pass

    npt.assert_almost_equal(f['vel'], original['vel'])


def test_vp_translate():
    global f, original

    with pynbody.transformation.xv_translate(f, [1, 0, 0], [2, 0, 0]):
        npt.assert_almost_equal(f['vel'], original['vel'] + [2, 0, 0])
        npt.assert_almost_equal(f['pos'], original['pos'] + [1, 0, 0])

    # check moved back
    npt.assert_almost_equal(f['vel'], original['vel'])
    npt.assert_almost_equal(f['pos'], original['pos'])

    # try again with with abnormal exit
    try:
        with pynbody.transformation.xv_translate(f, [1, 0, 0], [2, 0, 0]):
            npt.assert_almost_equal(f['vel'], original['vel'] + [2, 0, 0])
            npt.assert_almost_equal(f['pos'], original['pos'] + [1, 0, 0])
            raise RuntimeError
    except RuntimeError:
        pass

    npt.assert_almost_equal(f['vel'], original['vel'])
    npt.assert_almost_equal(f['pos'], original['pos'])


def test_rotate():
    global f, original

    with f.rotate_x(90):
        npt.assert_almost_equal(f['y'], -original['z'])
        npt.assert_almost_equal(f['z'], original['y'])

    npt.assert_almost_equal(f['pos'], f['pos'])

def test_family_rotate():
    """Test that rotating a family works as expected

    See issue #728"""
    f = pynbody.new(dm=10, gas=20, bh=10)
    f['pos'] = np.zeros((40, 3))
    f['pos'][:, 0] = 1.0

    del f['vel']

    f.bh['vel'] = np.zeros((10, 3))
    f.bh['vel'][:, 0] = 1.0

    f.rotate_z(90)

    npt.assert_almost_equal(f['pos'][:, 1], 1.0)
    npt.assert_almost_equal(f.bh['vel'][:, 1], 1.0)

def test_chaining():
    with pynbody.transformation.translate(f.rotate_x(90), [0, 1, 0]):
        npt.assert_almost_equal(f['y'], 1.0 - original['z'])
        npt.assert_almost_equal(f['z'], original['y'])

    npt.assert_almost_equal(f['pos'], original['pos'])


def test_halo_managers():
    with pynbody.analysis.angmom.sideon(f, disk_size=1, cen_size=1):
        pass

    npt.assert_almost_equal(f['pos'], original['pos'])


def test_weakref():
    global f
    tx1 = f.rotate_y(90)
    tx2 = pynbody.transformation.translate(f.rotate_x(90), [0, 1, 0])
    assert tx1.sim is not None
    assert tx2.sim is not None
    del f
    gc.collect()
    assert tx1.sim is None
    assert tx2.sim is None
