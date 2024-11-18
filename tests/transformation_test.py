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
        assert tx2._previous_transformation is None
        assert str(tx2) == "rotate_x(90)"
        npt.assert_almost_equal(f['y'], -original['z'])


    npt.assert_almost_equal(f['pos'], original['pos'])

@pytest.mark.skip(reason="Simulations are no longer stored as weakrefs")
def test_weakref():
    """Simulations used to be stored as weakrefs in a transformation object, to avoid problems
    with garbage collection. However this caused issues with reverting transformations reliably,
    esp if a user did something like

    with f.dm.rotate_x(90):
        ...

    f.dm would have been GCed by the time the block exited.

    This test is therefore disabled for now
    """
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

@pytest.fixture
def loadable_3d_arrays():
    source = pynbody.new(dm=5, star=5)
    source['pos'] = np.zeros((10, 3))
    source['pos'][:, 0] = np.ones(10)
    source['another3d'] = np.zeros((10, 3))
    source['another3d'][:, 0] = np.ones(10)
    source.st['star_only_3d'] = np.zeros((5,3))
    source.st['star_only_3d'][:, 0] = np.ones(5)

    destination = source.get_copy_on_access_simsnap()
    return destination

def test_persistent_transform(loadable_3d_arrays):
    f = loadable_3d_arrays
    f['pos'] # noqa - force 'load'

    with f.rotate_z(90):
        npt.assert_allclose(f['pos'][:,1], np.ones(10))

        # test that the array that was unloaded at the time of the transformation is  transformed
        npt.assert_allclose(f['another3d'][:, 1], np.ones(10))

        npt.assert_allclose(f.st['star_only_3d'][:, 1], np.ones(5))

    # check everything is transformed back
    npt.assert_allclose(f['pos'][:, 0], np.ones(10))
    npt.assert_allclose(f['another3d'][:, 0], np.ones(10))
    npt.assert_allclose(f.st['star_only_3d'][:, 0], np.ones(5))

def test_persistent_chained_transform(loadable_3d_arrays):
    f = loadable_3d_arrays

    with f.translate([1,0,0]).rotate_z(90):
        npt.assert_allclose(f['pos'][:, 1], np.repeat(2.0, 10))
        npt.assert_allclose(f['pos'][:, [0,2]], 0, atol=1e-10)

        # test that the array that was unloaded at the time of the transformation is  transformed
        npt.assert_allclose(f['another3d'][:, 1], np.ones(10))

    # check everything is transformed back
    npt.assert_allclose(f['pos'][:, 0], np.ones(10))
    npt.assert_allclose(f['another3d'][:, 0], np.ones(10))

    npt.assert_allclose(f['pos'][:, 1], np.zeros(10), atol=1e-10)

def test_persistent_nested_transform(loadable_3d_arrays):
    f = loadable_3d_arrays

    with f.translate([1, 0, 0]):
        with f.rotate_z(90):
            npt.assert_allclose(f['pos'][:, 1], np.repeat(2.0, 10))
            npt.assert_allclose(f['pos'][:, [0, 2]], 0, atol=1e-10)
            npt.assert_allclose(f['another3d'][:, 1], np.ones(10))
        npt.assert_allclose(f['pos'][:, 0], np.repeat(2.0, 10))
        npt.assert_allclose(f['another3d'][:, 0], np.repeat(1.0, 10))

    # check everything is transformed back
    npt.assert_allclose(f['pos'][:, 0], np.ones(10))
    npt.assert_allclose(f['another3d'][:, 0], np.ones(10))

    npt.assert_allclose(f['pos'][:, 1], np.zeros(10), atol=1e-10)

def test_revert_transform_out_of_order(loadable_3d_arrays):
    f = loadable_3d_arrays

    translate = f.translate([1,0,0])
    rotate = f.rotate_z(90)

    with npt.assert_raises(pynbody.transformation.TransformationException):
        translate.revert()

    rotate.revert()
    translate.revert()

    npt.assert_allclose(f['pos'][:, 0], np.ones(10))

def test_persistent_array_with_derived(loadable_3d_arrays):
    f = loadable_3d_arrays

    @pynbody.derived_array
    def derived_pos(sim):
        return sim['pos']

    @pynbody.derived_array
    def derived_another3d(sim):
        return sim['another3d']

    f['pos']

    with f.rotate_z(90):
        npt.assert_allclose(f['derived_pos'][:, 1], np.ones(10))

        t = f['derived_another3d']
        npt.assert_allclose(f['derived_another3d'][:, 1], np.ones(10))

    npt.assert_allclose(f['derived_pos'][:,0], np.ones(10))

@pytest.mark.parametrize('preload', ['preload-before-rotate', 'preload-after-rotate', 'family-load'])
def test_transform_subarray(loadable_3d_arrays, preload):
    f = loadable_3d_arrays

    if preload == 'preload-before-rotate':
        f['pos'] # noqa - just force a load

    with f.dm.rotate_z(90):
        if preload == 'preload-after-rotate':
            f['pos'] # noqa - just force a load

        npt.assert_allclose(f.dm['pos'][:, 1], np.ones(5))
        npt.assert_allclose(f.dm['pos'][:, 0], np.zeros(5), atol=1e-8)

        npt.assert_allclose(f.st['pos'][:,0], np.ones(5))
        npt.assert_allclose(f.st['pos'][:,1], np.zeros(5))

    npt.assert_allclose(f['pos'][:, 0], np.ones(10))
    npt.assert_allclose(f['pos'][:, 1:], np.zeros((10, 2)))


def test_impossible_compound_transform():
    f = pynbody.new()

    rot = f.rotate_x(90)

    with npt.assert_raises(pynbody.transformation.TransformationException):
        with f.rotate_y(90):
            rot.rotate_x(90)

def test_multiple_reversions():
    f = pynbody.new()
    rot = f.rotate_x(90)
    rot.revert()

    with npt.assert_raises(pynbody.transformation.TransformationException):
        rot.revert()

    with npt.assert_raises(pynbody.transformation.TransformationException):
        with rot:
            pass

def test_apply_move_to_subsnap():
    # due to indexedsimarrays not having a name, there was a bug that stopped this from working
    f = pynbody.new(dm=50000)
    f['pos'] = np.random.normal(size=(50000, 3))
    f['vel'] = np.random.normal(size=(50000, 3))
    f['mass'] = np.ones(50000)

    subindex = np.arange(0,20000)

    # the following would fail
    tx = f[subindex].translate([1,0,0])
