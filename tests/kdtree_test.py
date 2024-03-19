import copy
import gc
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
from pynbody.array import shared


def setup_module():
    global f
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")

    # for compatibility with original results, pretend the box
    # is not periodic
    del f.properties['boxsize']

def teardown_module():
    global f
    del f

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

def test_smooth(v_mean, v_disp, rho, smooth):
    global f
    """
    np.save('test_smooth.npy', f.dm['smooth'][::100])
    np.save('test_rho.npy', f.dm['rho'][::100])
    np.save('test_v_mean.npy',f.dm['v_mean'][::100])
    np.save('test_v_disp.npy',f.dm['v_disp'][::100])
    """

    npt.assert_allclose(f.dm['smooth'][::100],
                        smooth,rtol=1e-5)

    npt.assert_allclose(f.dm['rho'][::100],
                        rho,rtol=1e-5)



    npt.assert_allclose(v_mean,f.dm['v_mean'][::100],rtol=1e-3)
    npt.assert_allclose(v_disp,f.dm['v_disp'][::100],rtol=1e-3)

    # check 1D smooth works too
    vz_mean = f.dm.kdtree.sph_mean(f.dm['vz'],32)
    npt.assert_allclose(v_mean[:,2],vz_mean[::100],rtol=1e-3)

    # check 1D dispersions
    v_disp_squared = (
        f.dm.kdtree.sph_dispersion(f.dm['vx'], 32)**2+
        f.dm.kdtree.sph_dispersion(f.dm['vy'], 32)**2+
        f.dm.kdtree.sph_dispersion(f.dm['vz'], 32)**2
    )

    npt.assert_allclose(v_disp**2, v_disp_squared[::100], rtol=1e-3)


def test_smooth_WendlandC2(rho_W):

    """
        np.save('test_rho_W.npy', f.g['rho'][::100])
    """
    pynbody.config['Kernel'] = 'WendlandC2'

    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")

    npt.assert_allclose(f.g['rho'][::100],
                        rho_W,rtol=1e-5)

def test_kd_delete():
    global f
    f.dm['smooth']

    assert hasattr(f.dm,'kdtree')

    f.physical_units()

    # position array has been updated - kdtree should be auto-deleted
    assert not hasattr(f.dm,'kdtree')


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

    generator_nn = t.nn(n_neigh)
    n = next(generator_nn)
    # print(n)

    p_idx = n[0]       # particle index in snapshot arrays
    hsml = n[1]        # smoothing length
    neigh_list = n[2]  # neighbours list
    dist2 = n[3]       # squared distances from neighbours
    assert p_idx == 9
    assert hsml == f.g['smooth'][p_idx]
    npt.assert_allclose(hsml,np.sqrt(np.max(dist2))/2, rtol=1e-6)
    assert hsml == 128.19053649902344
    assert neigh_list == [9, 11, 35, 1998, 7, 12, 22, 36, 5, 20, 34, 31, 8, 19, 37, 10, 2018,
                          2017, 38, 52, 39, 41, 42, 33, 23, 1997, 43, 1996, 24, 40, 25, 21]
    npt.assert_allclose(dist2, [0.0, 39369.51953125, 24460.677734375, 31658.59375, 58536.9765625, 57026.3984375, 51718.3515625,
                         47861.25390625, 59311.27734375, 34860.97265625, 36082.15234375, 65731.2578125, 16879.42578125,
                         52811.79296875, 16521.751953125, 17574.501953125, 24489.19140625, 29066.84765625, 36883.796875,
                         41815.23046875, 60706.04296875, 31192.068359375, 58157.92578125, 60277.04296875, 61944.99609375,
                         45676.2578125, 54654.58984375, 59870.70703125, 20319.2890625, 35900.76953125, 30422.66796875, 56239.8828125],
                        rtol=1e-6)

    neighbour_list_all = t.all_nn(n_neigh)
    assert n == neighbour_list_all[0]
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
    n = shared.get_num_shared_arrays()
    f.build_tree(shared_mem=False)
    assert shared.get_num_shared_arrays() == n
    del f

    f = _make_test_gaussian(npart)
    f.build_tree(shared_mem=True)
    assert shared.get_num_shared_arrays() == 2+n
    assert f.kdtree.kdnodes._shared_fname.startswith('pynbody')
    assert f.kdtree.particle_offsets._shared_fname.startswith('pynbody')
    del f
    gc.collect()
    shared._ensure_shared_memory_clean()
    assert shared.get_num_shared_arrays() == n

def test_kdtree_mixed_dtypes(npart=1000):
    f = _make_test_gaussian(npart)
    f.set_array_dtype('pos', np.float32)
    f.set_array_dtype('mass', np.float64)
    assert f['mass'].dtype == np.float64

    with pytest.warns(RuntimeWarning, match="Converting mass array to float32"):
        f.build_tree()

    npt.assert_allclose(f['smooth'], f['smooth'], atol=1e-7)

    # kdtree should have auto-converted the dtype
    assert f['pos'].dtype == np.float32
    assert f['mass'].dtype == np.float32
