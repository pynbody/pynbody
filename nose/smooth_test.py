
import pynbody, numpy as np
import numpy.testing as npt
import pylab as p
import nose.tools

f = pynbody.load("testdata/g15784.lr.01024")

def test_smooth():
    global f

    npt.assert_allclose(f.dm['smooth'][::100],
                         np.load('test_smooth.npy'))

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


if __name__=="__main__":
    test_kd_delete()
