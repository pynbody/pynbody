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
                        smooth,rtol=1e-8)

    npt.assert_allclose(f.dm['rho'][::100],
                        rho,rtol=1e-8)



    npt.assert_allclose(v_mean,f.dm['v_mean'][::100],rtol=1e-8)
    npt.assert_allclose(v_disp,f.dm['v_disp'][::100],rtol=1e-8)

    # check 1D smooth works too
    vz_mean = f.dm.kdtree.sph_mean(f.dm['vz'],32)
    npt.assert_allclose(v_mean[:,2],vz_mean[::100],rtol=1e-8)

    # check 1D dispersions
    v_disp_squared = (
        f.dm.kdtree.sph_dispersion(f.dm['vx'], 32)**2+
        f.dm.kdtree.sph_dispersion(f.dm['vy'], 32)**2+
        f.dm.kdtree.sph_dispersion(f.dm['vz'], 32)**2
    )

    npt.assert_allclose(v_disp**2, v_disp_squared[::100], rtol=1e-8)


def test_smooth_WendlandC2(rho_W):
    pynbody.config['sph']['Kernel'] = 'WendlandC2'

    try:
        f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
        del f.properties['boxsize']
        np.save('test_rho_W2.npy', f.dm['rho'][::100])
        npt.assert_allclose(f.d['rho'][::100], rho_W, rtol=1e-6)
    finally:
        pynbody.config['sph']['Kernel'] = 'CubicSpline'

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

def test_kdtree_parallel_build():
    """Check that parallel tree build results in identical tree to serial build."""
    f = pynbody.new(dm=5000)
    f['pos'] = np.random.uniform(size=(5000,3))
    f['mass'] = np.random.uniform(size=5000)

    f.build_tree(1)
    result_one_thread = f.kdtree.serialize()

    _, _, kdn1, poff1, _ = result_one_thread

    del f.kdtree

    f.build_tree(4)
    result_four_threads = f.kdtree.serialize()
    _, _, kdn4, poff4, _ = result_four_threads

    assert (kdn1['pLower'] == kdn4['pLower']).all()
    assert (kdn1['pUpper'] == kdn4['pUpper']).all()
    assert (kdn1['iDim'] == kdn4['iDim']).all()
    npt.assert_allclose(kdn1['bnd']['fMin'], kdn4['bnd']['fMin'])
    npt.assert_allclose(kdn1['bnd']['fMax'], kdn4['bnd']['fMax'])
    npt.assert_allclose(kdn1['fSplit'], kdn4['fSplit'])
    assert (poff1 == poff4).all()






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
