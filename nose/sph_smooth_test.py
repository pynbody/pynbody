
import pynbody, numpy as np
import numpy.testing as npt
import pylab as p
import nose.tools

def setup():
    global f
    f = pynbody.load("testdata/g15784.lr.01024")

    # for compatibility with original results, pretend the box
    # is not periodic
    del f.properties['boxsize']

def teardown():
    global f
    del f
    
def test_smooth():
    global f

    npt.assert_allclose(f.dm['smooth'][::100],
                         np.load('test_smooth.npy'),rtol=1e-5)

    npt.assert_allclose(f.dm['rho'][::100],
                         np.load('test_rho.npy'),rtol=1e-5)



    npt.assert_allclose(np.load('test_v_mean.npy'),f.dm['v_mean'][::100],rtol=1e-3)
    npt.assert_allclose(np.load('test_v_disp.npy'),f.dm['v_disp'][::100],rtol=1e-3)

    # check 1D smooth works too
    vz_mean = f.dm.kdtree.sph_mean(f.dm['vz'],32)
    npt.assert_allclose(np.load('test_v_mean.npy')[:,2],vz_mean[::100],rtol=1e-3)

    # check 1D dispersions
    v_disp = f.dm.kdtree.sph_dispersion(f.dm['vx'],32)**2+ \
             f.dm.kdtree.sph_dispersion(f.dm['vy'],32)**2+ \
             f.dm.kdtree.sph_dispersion(f.dm['vz'],32)**2

    npt.assert_allclose(np.load('test_v_disp.npy')**2,v_disp[::100],rtol=1e-3)

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
    trigger_fn = lambda : f['smooth']
    nose.tools.assert_raises(ValueError, trigger_fn)

def test_float_kd():
    f = pynbody.load("testdata/test_g2_snap")
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

def test_periodic_smoothing():
    f = pynbody.load("testdata/g15784.lr.01024")


    npt.assert_allclose(f.dm['smooth'][::100],
                         np.load('test_smooth_periodic.npy'),rtol=1e-5)

    npt.assert_allclose(f.dm['rho'][::100],
                         np.load('test_rho_periodic.npy'),rtol=1e-5)


if __name__=="__main__":
    test_float_kd()
