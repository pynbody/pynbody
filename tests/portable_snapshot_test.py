"""Tests of the portable state of snapshots and KD-trees, within a single process.

The use of these states to share snapshots between processes is tested in process_pool_test.py."""

import gc

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils
from pynbody.snapshot.portable import ArrayNotTransferredError, PortableSimSnap, ReadOnlySnapshotError


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gasoline_ahf")


@pytest.fixture
def snap():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    f.physical_units()
    f['pos']
    f['mass']
    f.gas['rho']
    f['r']  # a derived array
    f.build_tree()
    yield f
    del f
    gc.collect()


@pytest.fixture
def portable(snap):
    return pynbody.snapshot.SimSnap.from_portable_state(snap.get_portable_state())


def test_portable_snapshot_structure(snap, portable):
    assert isinstance(portable, PortableSimSnap)
    assert len(portable) == len(snap)
    assert portable.families() == snap.families()
    for fam in snap.families():
        assert len(portable[fam]) == len(snap[fam])
    assert set(portable.keys()) == set(snap.keys())
    assert set(portable.family_keys()) == set(snap.family_keys())
    assert portable.filename == snap.filename


def test_portable_snapshot_arrays(snap, portable):
    for name in ('pos', 'x', 'mass', 'r'):
        npt.assert_array_equal(portable[name], snap[name])
        assert portable[name].units == snap[name].units
    npt.assert_array_equal(portable.gas['rho'], snap.gas['rho'])
    assert portable.gas['rho'].units == snap.gas['rho'].units
    assert portable.is_derived_array('r')
    assert not portable.is_derived_array('pos')

    # the arrays are not copies
    assert np.shares_memory(portable['pos'], snap['pos'])
    assert portable['pos'].sim is portable
    assert portable['x'].sim is portable


def test_portable_snapshot_properties_exact(snap, portable):
    assert portable.properties.keys() == snap.properties.keys()
    # units are transferred to full precision, not rounded as in their string representation
    assert portable.properties['boxsize'].ratio(snap.properties['boxsize']) == 1.0
    assert float(portable.properties['time'].in_units('Gyr')) == float(snap.properties['time'].in_units('Gyr'))
    assert portable.properties['a'] == snap.properties['a']
    for ours, theirs in zip(portable._file_units_system, snap._file_units_system):
        assert ours.ratio(theirs, a=0.5, h=0.7) == 1.0


def test_portable_snapshot_is_read_only(snap, portable):
    with pytest.raises(ValueError):
        portable['pos'][0] = 0
    with pytest.raises(ValueError):
        portable.gas['rho'][0] = 0
    # ... but the original is unaffected
    snap['mass'][0] = snap['mass'][0]


def test_portable_snapshot_refuses_transformation(portable):
    with pytest.raises(ReadOnlySnapshotError, match="get_copy_on_access_simsnap"):
        portable.translate([1, 0, 0])
    with pytest.raises(ReadOnlySnapshotError):
        pynbody.analysis.faceon(portable[:1000])


def test_portable_snapshot_unit_conversion(portable):
    # already in these units, so nothing to do
    portable.physical_units()
    with pytest.raises(ReadOnlySnapshotError, match="converted"):
        portable.physical_units(distance='Mpc')


def test_portable_snapshot_refuses_to_load(snap, portable):
    assert 'iord' in snap.loadable_keys()
    with pytest.raises(ArrayNotTransferredError, match="'iord' was not loaded"):
        portable['iord']
    with pytest.raises(KeyError):
        portable.dm['iord']

    # a family-level array
    assert 'HI' in snap.gas.loadable_keys()
    with pytest.raises(KeyError, match="f.gas\\['HI'\\]"):
        portable.gas['HI']

    assert portable.loadable_keys() == []


def test_portable_snapshot_refuses_to_load_slice():
    """A 1D slice of an array that was not loaded refers to the whole array"""
    f = pynbody.new(dm=10)
    f['pos'] = np.zeros((10, 3))
    state = f.get_portable_state()
    state['loadable_keys']['all'] = ['acc']  # as if the snapshot could have loaded it
    portable = pynbody.snapshot.SimSnap.from_portable_state(state)
    with pytest.raises(KeyError, match="f\\['acc'\\]"):
        portable['acc_x']


def test_portable_snapshot_derives(snap, portable):
    """Arrays which could not have been loaded are derived as normal, without affecting the original"""
    assert 'rxy' not in snap.keys()
    npt.assert_allclose(portable['rxy'], np.sqrt(snap['x'] ** 2 + snap['y'] ** 2))
    assert portable['rxy'].units == snap['x'].units
    assert 'rxy' not in snap.keys()

    portable['new_array'] = np.arange(len(portable))
    assert 'new_array' not in snap.keys()


def test_portable_snapshot_kdtree(snap, portable):
    assert hasattr(portable, 'kdtree')
    assert np.shares_memory(portable.kdtree.kdnodes, snap.kdtree.kdnodes)
    centre = snap['pos'][1000]
    sphere = pynbody.filt.Sphere('200 kpc', centre)
    npt.assert_array_equal(portable[sphere].get_index_list(portable), snap[sphere].get_index_list(snap))


def test_portable_snapshot_with_halos(snap, portable):
    h = snap.halos()
    h_portable = pynbody.halo.HaloCatalogue.from_portable_state(h.get_portable_state(), portable)
    npt.assert_array_equal(h_portable[1]['pos'], h[1]['pos'])
    npt.assert_allclose(pynbody.analysis.halo.shrink_sphere_center(h_portable[1]),
                        pynbody.analysis.halo.shrink_sphere_center(h[1]))


def test_portable_snapshot_copy_on_access(snap, portable):
    """The way to work on part of a portable snapshot in its own frame"""
    h = pynbody.halo.HaloCatalogue.from_portable_state(snap.halos().get_portable_state(), portable)
    local = h[1].get_copy_on_access_simsnap()
    pynbody.analysis.center(local)
    npt.assert_allclose(pynbody.analysis.halo.shrink_sphere_center(local), [0, 0, 0], atol=1e-5)

    # an array which exists on disk is not derived instead, even in a copy
    with pytest.raises(ArrayNotTransferredError):
        local['iord']


def test_portable_snapshot_of_view_refused(snap):
    with pytest.raises(TypeError, match="whole snapshot"):
        snap.dm.get_portable_state()


def test_portable_snapshot_round_trip_twice(snap, portable):
    """A portable snapshot can itself produce a portable state, which still knows what was not transferred"""
    again = pynbody.snapshot.SimSnap.from_portable_state(portable.get_portable_state())
    npt.assert_array_equal(again['pos'], snap['pos'])
    with pytest.raises(ArrayNotTransferredError):
        again['iord']


def test_portable_snapshot_derived_class_arrays():
    """Arrays derived specifically for the original's class can be derived in the portable snapshot"""
    f = pynbody.new(dm=10)
    f['pos'] = np.random.uniform(size=(10, 3))
    f['mass'] = np.ones(10)
    portable = pynbody.snapshot.SimSnap.from_portable_state(f.get_portable_state())
    assert portable._underlying_class is pynbody.snapshot.SimSnap
    npt.assert_allclose(portable['r'], f['r'])


def test_kdtree_portable_state():
    np.random.seed(1)
    f = pynbody.new(dm=5000)
    f['pos'] = np.random.uniform(size=(5000, 3))
    f['mass'] = np.ones(5000)
    f.properties['boxsize'] = 1.0
    f.build_tree()
    state = f.kdtree.get_portable_state()
    assert set(state.keys()) >= {'kdnodes', 'particle_offsets', 'pos', 'mass', 'boxsize', 'leafsize'}

    tree = pynbody.kdtree.KDTree.from_portable_state(state)
    assert tree.boxsize == f.kdtree.boxsize
    npt.assert_array_equal(np.sort(tree.particles_in_sphere([0.5, 0.5, 0.5], 0.2)),
                           np.sort(f.kdtree.particles_in_sphere([0.5, 0.5, 0.5], 0.2)))
    # periodicity is preserved
    npt.assert_array_equal(np.sort(tree.particles_in_sphere([0.0, 0.0, 0.0], 0.2)),
                           np.sort(f.kdtree.particles_in_sphere([0.0, 0.0, 0.0], 0.2)))

    # SPH smoothing in a portable snapshot uses its tree, rather than building another
    portable = pynbody.snapshot.SimSnap.from_portable_state(f.get_portable_state())
    portable_tree = portable.kdtree
    npt.assert_allclose(portable['smooth'], f['smooth'])
    npt.assert_allclose(portable['rho'], f['rho'])
    assert portable.kdtree is portable_tree

    without_arrays = f.kdtree.get_portable_state(include_particle_arrays=False)
    assert 'pos' not in without_arrays
    tree = pynbody.kdtree.KDTree.from_portable_state(without_arrays, f['pos'], f['mass'])
    npt.assert_array_equal(np.sort(tree.particles_in_sphere([0.5, 0.5, 0.5], 0.2)),
                           np.sort(f.kdtree.particles_in_sphere([0.5, 0.5, 0.5], 0.2)))


def test_build_tree_follows_shared_arrays():
    f = pynbody.new(dm=1000)
    f.enable_shared_arrays()
    f['pos'] = np.random.uniform(size=(1000, 3))
    f['mass'] = np.ones(1000)
    f.build_tree()
    assert isinstance(f.kdtree.kdnodes, pynbody.array.shared.SharedMemorySimArray)
    del f
    gc.collect()


def test_unit_exact_string():
    u = pynbody.units.Unit("kpc a h**-1") * (1.0 / 7.0) * pynbody.units.Unit("Msol")**(1, 2)
    assert pynbody.units.Unit(pynbody.units.to_exact_string(u)).ratio(u, a=0.3, h=0.7) == 1.0
    assert pynbody.units.to_exact_string(pynbody.units.Unit("kpc")) == "kpc"


def test_portable_snapshot_kdtree_array_refs():
    """The tree's own SPH operations can be used, as in the original, once smoothing lengths exist"""
    np.random.seed(3)
    f = pynbody.new(dm=5000)
    f['pos'] = np.random.uniform(size=(5000, 3))
    f['mass'] = np.random.uniform(size=5000)
    f['rho']
    portable = pynbody.snapshot.SimSnap.from_portable_state(f.get_portable_state())
    npt.assert_allclose(portable.kdtree.sph_mean(portable['mass'], 32), f.kdtree.sph_mean(f['mass'], 32))


def test_portable_snapshot_tipsy_abundances(snap):
    """Abundance ratios used to be derived by modifying their inputs in place, which a read-only snapshot forbids"""
    snap.st['FeMassFrac']
    snap.st['OxMassFrac']
    portable = pynbody.snapshot.SimSnap.from_portable_state(snap.get_portable_state())
    npt.assert_allclose(portable.st['feh'], snap.st['feh'])
    npt.assert_allclose(portable.st['ofe'], snap.st['ofe'])
    # deriving these does not alter the mass fractions
    assert (snap.st['FeMassFrac'] == 0).any()


def test_portable_properties_with_no_unit():
    f = pynbody.new(dm=10)
    f.properties['nothing'] = pynbody.units.NoUnit()
    portable = pynbody.snapshot.SimSnap.from_portable_state(f.get_portable_state())
    assert isinstance(portable.properties['nothing'], pynbody.units.NoUnit)
