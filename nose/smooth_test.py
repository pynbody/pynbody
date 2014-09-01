
import pynbody, numpy as np
import numpy.testing as npt
import pylab as p

def test_smooth():
    global f
    f = pynbody.load("testdata/g15784.lr.01024")

    npt.assert_allclose(f.dm['smooth'],
                         np.load('test_smooth.npy'),rtol=1e-5)

    p.plot((f.dm['rho']/np.load('test_rho.npy'))[::10])
    p.savefig('testrho.png')
    npt.assert_allclose(f.dm['rho'],
                         np.load('test_rho.npy'),rtol=1e-5)



if __name__=="__main__":
    test_smooth()
