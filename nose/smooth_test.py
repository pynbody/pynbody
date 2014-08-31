
import pynbody, numpy as np
import numpy.testing as npt
import pylab as p

def test_smooth():
    global f
    f = pynbody.load("testdata/g15784.lr.01024")

    npt.assert_array_almost_equal(f.dm['smooth'],
                         np.load('test_smooth.npy'))

    npt.assert_array_almost_equal(f.dm['rho'],
                         np.load('test_rho.npy'))



if __name__=="__main__":
    test_smooth()
