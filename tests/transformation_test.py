import copy
import gc

import numpy as np
import numpy.testing as npt
import pytest

import pynbody


@pytest.fixture
def test_simulation_with_copy():
    s = pynbody.new(dm=1000)
    s['pos'] = np.random.normal(scale=1.0, size=s['pos'].shape)
    s['vel'] = np.random.normal(scale=1.0, size=s['vel'].shape)
    s['mass'] = np.random.uniform(1.0, 10.0, size=s['mass'].shape)
    return s, copy.deepcopy(s)


def test_translate(test_simulation_with_copy):
    f, original = test_simulation_with_copy

    with f.translate([1, 0, 0]):
        npt.assert_almost_equal(f['pos'], original['pos'] + [1, 0, 0])

    # check moved back
    npt.assert_almost_equal(f['pos'], original['pos'])

    # try again with with abnormal exit
    try:
        with f.translate([1, 0, 0]):
            npt.assert_almost_equal(f['pos'], original['pos'] + [1, 0, 0])
            raise RuntimeError
    except RuntimeError:
        pass

    npt.assert_almost_equal(f['pos'], original['pos'])


def test_v_translate(test_simulation_with_copy):
    f, original = test_simulation_with_copy

    with f.offset_velocity([1, 0, 0]):
        npt.assert_almost_equal(f['vel'], original['vel'] + [1, 0, 0])

    # check moved back
    npt.assert_almost_equal(f['vel'], original['vel'])

    # try again with with abnormal exit
    try:
        with f.offset_velocity([1, 0, 0]):
            npt.assert_almost_equal(f['vel'], original['vel'] + [1, 0, 0])
            raise RuntimeError
    except RuntimeError:
        pass

    npt.assert_almost_equal(f['vel'], original['vel'])


def test_vp_translate(test_simulation_with_copy):
    f, original = test_simulation_with_copy

    with f.translate([1, 0, 0]).offset_velocity([2,0,0]):
        npt.assert_almost_equal(f['vel'], original['vel'] + [2, 0, 0])
        npt.assert_almost_equal(f['pos'], original['pos'] + [1, 0, 0])

    # check moved back
    npt.assert_almost_equal(f['vel'], original['vel'])
    npt.assert_almost_equal(f['pos'], original['pos'])

    # try again with with abnormal exit
    try:
        with f.translate([1, 0, 0]).offset_velocity([2,0,0]):
            npt.assert_almost_equal(f['vel'], original['vel'] + [2, 0, 0])
            npt.assert_almost_equal(f['pos'], original['pos'] + [1, 0, 0])
            raise RuntimeError
    except RuntimeError:
        pass

    npt.assert_almost_equal(f['vel'], original['vel'])
    npt.assert_almost_equal(f['pos'], original['pos'])


def test_rotate(test_simulation_with_copy):
    f, original = test_simulation_with_copy

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

def test_chaining(test_simulation_with_copy):
    f, original = test_simulation_with_copy

    with f.rotate_x(90).translate([0, 1, 0]):
        npt.assert_almost_equal(f['y'], 1.0 - original['z'])
        npt.assert_almost_equal(f['z'], original['y'])

    npt.assert_almost_equal(f['pos'], original['pos'])


def test_halo_managers(test_simulation_with_copy):
    f, original = test_simulation_with_copy

    with pynbody.analysis.angmom.sideon(f, disk_size=1.0):
        pass

    npt.assert_almost_equal(f['pos'], original['pos'])

def test_repr(test_simulation_with_copy):
    f, _ = test_simulation_with_copy

    with f.rotate_x(45).translate([0, 1, 0]).offset_velocity([0, 0, 1]) as tx:
        assert str(tx) == "rotate_x(45), translate, offset_velocity"
        assert repr(tx) == "<Transformation rotate_x(45), translate, offset_velocity>"

def test_null(test_simulation_with_copy):
    f, original = test_simulation_with_copy

    with pynbody.transformation.NullTransformation(f) as tx:
        npt.assert_almost_equal(f['pos'], original['pos'])
        assert tx.sim is not None

    with tx.rotate_x(90) as tx2:
        assert tx2.next_transformation is None
        assert str(tx2) == "rotate_x(90)"
        npt.assert_almost_equal(f['y'], -original['z'])


    npt.assert_almost_equal(f['pos'], original['pos'])

def test_weakref():
    f = pynbody.new(dm=10)

    tx1 = f.rotate_y(90)
    tx2 = tx1.rotate_x(90).translate([0, 1, 0])
    assert tx1.sim is not None
    assert tx2.sim is not None
    del f
    gc.collect()
    assert tx1.sim is None
    assert tx2.sim is None

@pytest.mark.parametrize("family", [True, False])
def test_derived_3d_array(family):
    """Test for a bug where transformations would try to rotate derived arrays, raising an error"""
    f = pynbody.new(dm=5, gas=5)

    @pynbody.derived_array
    def test_derived_3d_array_transformation(sim):
        return np.random.normal(size=(len(sim), 3))

    if family:
        f = f.dm

    _ = f['test_derived_3d_array_transformation']

    with f.ancestor.rotate_y(90):
        _ = f['test_derived_3d_array_transformation']
