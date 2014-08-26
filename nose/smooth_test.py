
import pynbody, numpy as np
import numpy.testing as npt


def test_smooth():            
    f = pynbody.load("testdata/g15784.lr.01024")
    f.dm['pos']

    npt.assert_array_almost_equal(f.dm['smooth'],
                         np.load('test_smooth.npy'))



if __name__=="__main__":
    test_smooth()
