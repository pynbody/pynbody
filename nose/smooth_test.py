
import pynbody, numpy as np
import numpy.testing as npt


def test_smooth():
    global f
    f = pynbody.load("testdata/g15784.lr.01024")
    f.dm['pos']
    import pylab as p
    p.plot(f.dm['smooth']/np.load("test_smooth.npy"))
    p.plot(f.dm['smooth'])
    p.ylim(-0.2,1.2)
    p.savefig('smtest.png')
    npt.assert_array_almost_equal(f.dm['smooth'],
                         np.load('test_smooth.npy'))



if __name__=="__main__":
    test_smooth()
