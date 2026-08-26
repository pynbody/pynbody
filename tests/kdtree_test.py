import copy
import gc
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils
from pynbody.array import shared


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gasoline_ahf", "gadget")


@pytest.fixture
def snap():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    # for compatibility with original results, pretend the box
    # is not periodic
    del f.properties['boxsize']

    return f

test_folder = Path(__file__).parent
@pytest.fixture
def v_mean():
    yield np.load(test_folder / 'test_v_mean.npy')

@pytest.fixture
def v_disp():
    yield np.load(test_folder / 'test_v_disp.npy')

@pytest.fixture
def smooth():
    yield np.load(test_folder / 'test_smooth.npy')

@pytest.fixture
def rho():
    yield np.load(test_folder / 'test_rho.npy')

@pytest.fixture
def rho_W():
    yield np.load(test_folder / 'test_rho_W.npy')

@pytest.fixture
def rho_periodic():
    yield np.load(test_folder / 'test_rho_periodic.npy')

@pytest.fixture
def smooth_periodic():
    yield np.load(test_folder / 'test_smooth_periodic.npy')

@pytest.fixture
def div_curl():
    yield np.load(test_folder / 'test_div_curl.npz')

def test_smooth(v_mean, v_disp, rho, smooth, snap):
    """
    np.save('test_smooth.npy', f.dm['smooth'][::100])
    np.save('test_rho.npy', f.dm['rho'][::100])
    np.save('test_v_mean.npy',f.dm['v_mean'][::100])
    np.save('test_v_disp.npy',f.dm['v_disp'][::100])
    """

    npt.assert_allclose(snap.dm['smooth'][::100],
                        smooth, rtol=1e-8)

    npt.assert_allclose(snap.dm['rho'][::100],
                        rho, rtol=1e-8)



    npt.assert_allclose(v_mean, snap.dm['v_mean'][::100], rtol=1e-8)
    npt.assert_allclose(v_disp, snap.dm['v_disp'][::100], rtol=1e-8)

    # check 1D smooth works too
    vz_mean = snap.dm.kdtree.sph_mean(snap.dm['vz'], 32)
    npt.assert_allclose(v_mean[:,2],vz_mean[::100],rtol=1e-8)

    # check 1D dispersions
    v_disp_squared = (
            snap.dm.kdtree.sph_dispersion(snap.dm['vx'], 32) ** 2 +
            snap.dm.kdtree.sph_dispersion(snap.dm['vy'], 32) ** 2 +
            snap.dm.kdtree.sph_dispersion(snap.dm['vz'], 32) ** 2
    )

    npt.assert_allclose(v_disp**2, v_disp_squared[::100], rtol=1e-8)


def test_smooth_WendlandC2(rho_W):
    pynbody.config['sph']['kernel'] = 'WendlandC2Kernel'

    try:
        f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
        del f.properties['boxsize']
        np.save('test_rho_W2.npy', f.dm['rho'][::100])
        npt.assert_allclose(f.d['rho'][::100], rho_W, rtol=1e-6)
    finally:
        pynbody.config['sph']['kernel'] = 'CubicSplineKernel'

def test_kd_delete(snap):
    snap.dm['smooth']

    assert hasattr(snap.dm, 'kdtree')

    snap.physical_units()

    # position array has been updated - kdtree should be auto-deleted
    assert not hasattr(snap.dm, 'kdtree')


def test_kd_issue_88() :
    # number of particles less than number of smoothing neighbours
    f = pynbody.new(gas=16)
    f['pos'] = np.random.uniform(size=(16,3))
    with pytest.raises(ValueError):
        f["smooth"]

@pytest.mark.filterwarnings(r"ignore:overflow.*:RuntimeWarning")
def test_float_kd():
    f = pynbody.load("testdata/gadget2/test_g2_snap")
    del f.properties['boxsize']

    assert f.dm['mass'].dtype==f.dm['pos'].dtype==np.float32
    assert f.dm['smooth'].dtype==np.float32

    # make double copy
    g = pynbody.new(len(f.dm))
    g.dm['pos']=f.dm['pos']
    g.dm['mass']=f.dm['mass']

    assert g.dm['mass'].dtype==g.dm['pos'].dtype==g.dm['smooth'].dtype==np.float64

    # check smoothing lengths agree (they have been calculated differently
    # using floating/double routines)

    npt.assert_allclose(f.dm['smooth'],g.dm['smooth'],rtol=1e-4)
    npt.assert_allclose(f.dm['rho'],g.dm['rho'],rtol=1e-4)

    # check all combinations of float/double smoothing
    double_ar = np.ones(len(f.dm),dtype=np.float64)
    float_ar = np.ones(len(f.dm),dtype=np.float32)

    double_double = g.dm.kdtree.sph_mean(double_ar,32)
    double_float = g.dm.kdtree.sph_mean(float_ar,32)
    float_double = f.dm.kdtree.sph_mean(double_ar,32)
    float_float = f.dm.kdtree.sph_mean(float_ar,32)

    # take double-double as 'gold standard' (though of course if any of these
    # fail it could also be a problem with the double-double case)

    npt.assert_allclose(double_double,double_float,rtol=1e-4)
    npt.assert_allclose(double_double,float_double,rtol=1e-4)
    npt.assert_allclose(double_double,float_float,rtol=1e-4)

def test_periodic_smoothing(rho_periodic, smooth_periodic):
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")

    """
    np.save('test_rho_periodic.npy', f.dm['rho'][::100])
    np.save('test_smooth_periodic.npy', f.dm['smooth'][::100])
    """
    npt.assert_allclose(f.dm['rho'][::100],
                         rho_periodic,rtol=1e-5)
    npt.assert_allclose(f.dm['smooth'][::100],
                         smooth_periodic,rtol=1e-5)


@pytest.mark.filterwarnings("ignore:overflow encountered in cast:RuntimeWarning")
def test_neighbour_list():
    f = pynbody.load("testdata/gadget2/test_g2_snap")
    pynbody.sph._get_smooth_array_ensuring_compatibility(f.g)  # actual smoothing
    t = f.g.kdtree
    n_neigh = 32

    # The tree returns particles in its own internal order, which is an
    # implementation detail; pick out a specific particle so that the expected
    # values below do not depend on it.
    for position, n in enumerate(t.nn(n_neigh)):
        if n[0] == 9:
            break
    else:
        raise AssertionError("particle 9 was not returned by the neighbour list generator")

    p_idx = n[0]       # particle index in snapshot arrays
    hsml = n[1]        # smoothing length
    neigh_list = np.array(n[2])  # neighbours list
    dist2 = np.array(n[3])       # squared distances from neighbours
    assert p_idx == 9
    assert hsml == f.g['smooth'][p_idx]
    npt.assert_allclose(hsml,np.sqrt(np.max(dist2))/2, rtol=1e-6)
    assert np.allclose(hsml, 128.19053649902344)

    ordering = np.argsort(dist2)
    assert (neigh_list[ordering] == [   9,   37,    8,   10,   24,   35, 2018, 2017,   25,   41, 1998,
         20,   40,   34,   38,   11,   52, 1997,   36,   22,   19,   43,
         21,   12,   42,    7,    5, 1996,   33,   39,   23,   31]).all()
    npt.assert_allclose(dist2[ordering], [    0.        , 16521.75195312, 16879.42578125, 17574.50195312,
       20319.2890625 , 24460.67773438, 24489.19140625, 29066.84765625,
       30422.66796875, 31192.06835938, 31658.59375   , 34860.97265625,
       35900.76953125, 36082.15234375, 36883.796875  , 39369.51953125,
       41815.23046875, 45676.2578125 , 47861.25390625, 51718.3515625 ,
       52811.79296875, 54654.58984375, 56239.8828125 , 57026.3984375 ,
       58157.92578125, 58536.9765625 , 59311.27734375, 59870.70703125,
       60277.04296875, 60706.04296875, 61944.99609375, 65731.2578125 ],
                        rtol=1e-6)

    neighbour_list_all = t.all_nn(n_neigh)
    assert n == neighbour_list_all[position]
    for nl in neighbour_list_all:
        assert len(nl[2]) == n_neigh   # always find n_neigh neighbours
        idx_self = nl[2].index(nl[0])  # index of self in the neighbour list (not necessarily the first element)
        assert nl[3][idx_self] == 0.0  # distance to self

def test_div_curl_smoothing(div_curl):
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")

    """
    np.savez('test_div_curl', curl=f.g['v_curl'][::100], div=f.g['v_div'][::100])
    """
    arr = div_curl
    # print(f.g['v_curl'][::100], f.g['v_div'][::100])
    curl, div = arr['curl'], arr['div']
    npt.assert_allclose(f.g['v_curl'][::100], curl, rtol=2e-4)
    npt.assert_allclose(f.g['v_div'][::100],  div,  rtol=2e-4)
    npt.assert_equal(f.g['vorticity'], f.g['v_curl'])
    assert f.g['vorticity'].units == f.g['vel'].units/f.g['pos'].units

def test_kdtree_parallel_build():
    """Check that the parallel tree build results in an identical tree to the serial build.

    Positions here are float64 and drawn from a continuous distribution, so no two
    particles share a coordinate and the tree is fully determined. Thread counts
    that are not a power of two are included, since the build no longer rounds
    down to one.
    """
    f = pynbody.new(dm=5000)
    f['pos'] = np.random.uniform(size=(5000,3))
    f['mass'] = np.random.uniform(size=5000)

    f.build_tree(1)
    leafsize, boxsize, kdn1, poff1, kernel = f.kdtree.serialize()

    for num_threads in (2, 3, 4, 5, 8, 16):
        del f.kdtree
        f.build_tree(num_threads)
        _, _, kdn, poff, _ = f.kdtree.serialize()

        assert (kdn1['pLower'] == kdn['pLower']).all()
        assert (kdn1['pUpper'] == kdn['pUpper']).all()
        assert (kdn1['iDim'] == kdn['iDim']).all()
        npt.assert_allclose(kdn1['bnd']['fMin'], kdn['bnd']['fMin'])
        npt.assert_allclose(kdn1['bnd']['fMax'], kdn['bnd']['fMax'])
        npt.assert_allclose(kdn1['fSplit'], kdn['fSplit'])
        assert (poff1 == poff).all()


@pytest.mark.parametrize("num_threads", [0, -1])
def test_kdtree_rejects_non_positive_num_threads(num_threads):
    """A thread count below one must be rejected rather than silently doing nothing."""
    f = pynbody.new(dm=100)
    f['pos'] = np.random.uniform(size=(100, 3))
    f['mass'] = np.ones(100)

    with pytest.raises(ValueError):
        f.build_tree(num_threads)


@pytest.mark.parametrize("npart", [1, 2, 33, 200])
def test_kdtree_build_more_threads_than_work(npart):
    """Asking for more threads than the tree can use must still build it correctly."""
    f = pynbody.new(dm=npart)
    f['pos'] = np.random.uniform(size=(npart, 3))
    f['mass'] = np.ones(npart)

    f.build_tree(64)
    offsets = f.kdtree.particle_offsets
    assert sorted(offsets.tolist()) == list(range(npart))

    del f.kdtree
    f.build_tree(1)
    npt.assert_array_equal(offsets, f.kdtree.particle_offsets)


def test_kdtree_parallel_build_with_tied_coordinates():
    """A tree built from particles sharing coordinates must still be a valid tree.

    Which of a set of tied particles ends up on each side of a split is not
    defined, so the tree may differ between thread counts here; it must
    nevertheless partition every node correctly and use each particle once.
    """
    n = 20000
    pos = np.random.randint(0, 4, size=(n, 3)).astype(np.float64)  # heavily tied
    f = pynbody.new(dm=n)
    f['pos'] = pos
    f['mass'] = np.ones(n)

    for num_threads in (1, 2, 3, 4, 8):
        if hasattr(f, 'kdtree'):
            del f.kdtree
        f.build_tree(num_threads)
        kdn = f.kdtree.kdnodes
        offsets = f.kdtree.particle_offsets

        assert sorted(offsets.tolist()) == list(range(n))

        nsplit = len(kdn) // 2
        for i in range(1, nsplit):
            node = kdn[i]
            if node['iDim'] < 0 or node['pUpper'] <= node['pLower']:
                continue
            m = (node['pLower'] + node['pUpper']) // 2
            d = node['iDim']
            low = pos[offsets[node['pLower']:m + 1], d]
            high = pos[offsets[m + 1:node['pUpper'] + 1], d]
            assert low.max() <= node['fSplit']
            assert high.min() >= node['fSplit']
            # the split must halve the node, which the tree layout relies on
            assert (m - node['pLower'] + 1) - (node['pUpper'] - m) in (0, 1)


@pytest.mark.parametrize("npart", [1, 10, 100, 1000, 100000])
@pytest.mark.parametrize("offset", [0.0, 0.2, 0.5]) # checks wrapping
@pytest.mark.parametrize("radius", [0.1, 0.3, 1.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_particles_in_sphere(npart, offset, radius, dtype):
    f = pynbody.new(dm=npart)

    f._create_array('pos', 3, dtype)
    f._create_array('mass', 1, dtype)

    np.random.seed(1337)
    f['pos'] = np.random.uniform(low=-0.5, high=0.5, size=(npart,3))
    f['mass'] = np.random.uniform(size=npart)
    assert np.issubdtype(f['pos'].dtype, dtype)
    assert np.issubdtype(f['mass'].dtype, dtype)
    f.properties['boxsize'] = 1.0

    f.build_tree()
    particles = f.kdtree.particles_in_sphere([offset, 0.0, 0.0], radius)

    f['x'] -= offset
    f.wrap()
    particles_compare = np.where(f['r']<radius)[0]

    assert (np.sort(particles) == np.sort(particles_compare)).all()

def test_kdtree_from_existing_kdtree(npart=1000):
    f = _make_test_gaussian(npart)

    f_copy = copy.deepcopy(f)

    f.build_tree()
    f_copy.import_tree(f.kdtree.serialize())

    assert f_copy.kdtree is not f.kdtree

    npt.assert_allclose(f['smooth'], f_copy['smooth'], atol=1e-7)


def _make_test_gaussian(npart):
    f = pynbody.new(dm=npart)
    np.random.seed(1337)
    f['pos'] = np.random.normal(1.0, size=(npart, 3))
    f['mass'] = np.random.uniform(size=npart)
    return f


def test_kdtree_shared_mem(npart=1000):
    f = _make_test_gaussian(npart)
    gc.collect()
    n = shared.get_num_shared_arrays_owned()
    f.build_tree(shared_mem=False)
    assert shared.get_num_shared_arrays_owned() == n
    del f

    f = _make_test_gaussian(npart)
    f.build_tree(shared_mem=True)
    assert shared.get_num_shared_arrays_owned() == 2 + n
    assert f.kdtree.kdnodes._shared_fname.startswith('pynbody')
    assert f.kdtree.particle_offsets._shared_fname.startswith('pynbody')
    del f
    gc.collect()
    assert shared.get_num_shared_arrays_owned() == n

def test_boxsize_too_small():
    f = pynbody.new(dm=1000)
    f['pos'] = np.random.normal(scale=1.0, size=f['pos'].shape)
    f['vel'] = np.random.normal(scale=1.0, size=f['vel'].shape)
    f['mass'] = np.random.uniform(1.0, 10.0, size=f['mass'].shape)
    f.properties['boxsize'] = 0.1
    with pytest.warns(RuntimeWarning, match = "span a region larger than the specified boxsize"):
        _ = f['smooth']


def test_kdtree_float64_rounding(npart=1000):
    """Check boundaries are ok even if float64 is in use"""
    f = pynbody.new(dm=npart)
    f.properties['boxsize'] = np.nextafter(1.0, -1.0)  # just below 1.0

    f['pos'] = np.random.uniform(size=(npart, 3))
    f['mass'] = np.random.uniform(size=npart)

    f['pos'][0,0] = np.nextafter(1.0, -1.0)
    f['pos'][1,0] = 0.0

    f.build_tree()

    _ = f['smooth']


def _weighting_test_snapshot(npart=8000):
    f = pynbody.new(gas=npart)
    np.random.seed(2024)
    # a strong density gradient, so that rho_i and rho_j differ across a kernel
    r = 10.0 * np.random.uniform(size=npart)
    d = np.random.normal(size=(npart, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    f['pos'] = d * r[:, None]
    f['mass'] = np.random.uniform(0.5, 1.5, size=npart)
    f['vel'] = np.random.normal(size=(npart, 3))
    f['rho']
    return f


def test_mass_weighting_reproduces_a_constant_field():
    """The 'self' weighting has an exact partition of unity, 'neighbour' does not.

    rho_i is by definition sum_j m_j W_ij, so weighting each neighbour by
    m_j/rho_i makes the interpolant reproduce a constant exactly. Weighting by
    m_j/rho_j -- the usual SPH volume element -- only does so approximately.
    """
    f = _weighting_test_snapshot()
    constant = pynbody.array.SimArray(np.full(len(f), 3.25))
    inner = np.linalg.norm(np.asarray(f['pos'], dtype=np.float64), axis=1) < 8.0

    exact = np.asarray(f.kdtree.sph_mean(constant, 32, weighting='mass'))
    usual = np.asarray(f.kdtree.sph_mean(constant, 32,
                                         weighting='volume'))

    npt.assert_allclose(exact[inner], 3.25, rtol=1e-12)
    assert np.abs(usual[inner] - 3.25).max() > 1e-6, \
        "the two weightings should differ where the density varies"


@pytest.mark.parametrize("operation", ['sph_mean', 'sph_dispersion',
                                       'sph_divergence', 'sph_curl'])
def test_weighting_is_accepted_and_changes_the_result(operation):
    """Every operation that divides by a density takes the option."""
    f = _weighting_test_snapshot()
    qty = f['vel'] if operation in ('sph_divergence', 'sph_curl') else f['vx']
    call = getattr(f.kdtree, operation)

    default = np.asarray(call(qty, 32))
    neighbour = np.asarray(call(qty, 32, weighting='volume'))
    self_ = np.asarray(call(qty, 32, weighting='mass'))

    # the default must not have changed
    npt.assert_array_equal(default, neighbour)

    inner = np.linalg.norm(np.asarray(f['pos'], dtype=np.float64), axis=1) < 8.0
    assert not np.allclose(neighbour[inner], self_[inner]), \
        "%s should depend on the weighting where the density varies" % operation
    assert np.isfinite(self_).all()


def test_unknown_weighting_raises():
    f = _weighting_test_snapshot(1000)
    with pytest.raises(ValueError):
        f.kdtree.sph_mean(f['vx'], 32, weighting='not-a-weighting')


@pytest.mark.parametrize("operation", ['sph_divergence', 'sph_curl'])
def test_divergence_and_curl_use_the_periodic_minimum_image(operation):
    """Displacements must be wrapped, as the neighbour search itself is.

    Regression test: smDivQty and smCurlQty differenced the stored positions
    directly, while the ball-gather that found the neighbour works in the
    minimum image. For a pair straddling a box face the two disagree by a box
    length, so every particle within 2h of a face came out wrong -- which on
    a cosmological volume is a substantial fraction of them.
    """
    npart = 6000
    f = pynbody.new(gas=npart)
    rng = np.random.default_rng(12)
    f['pos'] = rng.uniform(0, 10.0, size=(npart, 3))
    f['pos'].units = 'kpc'
    f['mass'] = rng.uniform(0.5, 1.5, size=npart)
    f['vel'] = rng.normal(size=(npart, 3))
    f.properties['boxsize'] = pynbody.units.Unit('10 kpc')
    f['rho']

    m = np.asarray(f['mass'], dtype=np.float64)
    rho = np.asarray(f['rho'], dtype=np.float64)
    h = np.asarray(f['smooth'], dtype=np.float64)
    vel = np.asarray(f['vel'], dtype=np.float64)
    kernel = f.kdtree.kernel

    # pair_blocks supplies minimum-imaged displacements, so this is the
    # independent statement of what the C++ ought to produce
    def divergence(i, j, dx, r):
        w = (m[j] / rho[i]) * kernel.gradient(r, h[i]) / r
        if operation == 'sph_divergence':
            # smDivQty forms (x_i - x_j) . (v_j - v_i), i.e. -(dx . dv)
            return -w * np.einsum('kl,kl->k', vel[j] - vel[i], dx)
        # smCurlQty forms (x_i - x_j) x (v_j - v_i), i.e. +(dv x dx)
        return w[:, None] * np.cross(vel[j] - vel[i], dx)

    expected = f.kdtree.pair_reduce(divergence, mode='gather')
    got = np.asarray(getattr(f.kdtree, operation)(f['vel'], 32,
                                                  weighting='mass'))

    # the particles that would have been wrong are the ones near a face
    pos = np.asarray(f['pos'], dtype=np.float64)
    near_face = ((pos < 2 * h[:, None]) | (pos > 10.0 - 2 * h[:, None])).any(axis=1)
    assert near_face.mean() > 0.2, "test needs plenty of particles near a face"

    npt.assert_allclose(got, expected, rtol=1e-8, atol=1e-12)
