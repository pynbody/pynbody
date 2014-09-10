
import pynbody, numpy as np
import numpy.testing as npt
import pylab as p

def test_smooth():
    global f
    f = pynbody.load("testdata/g15784.lr.01024")
    npt.assert_allclose(f.dm['smooth'][::100],
                         np.load('test_smooth.npy'))

    npt.assert_allclose(f.dm['rho'][::100],
                         np.load('test_rho.npy'),rtol=1e-5)



    npt.assert_allclose(np.load('test_v_mean.npy'),f.dm['v_mean'][::100],rtol=1e-3)
    npt.assert_allclose(np.load('test_v_disp.npy'),f.dm['v_disp'][::100],rtol=1e-3)


if __name__=="__main__":
    test_smooth()
