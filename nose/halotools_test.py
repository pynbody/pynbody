import pynbody
import numpy as np
import glob
import os

def setup() :
    global f, h
    f = pynbody.load("testdata/g15784.lr.01024")
    h = f.halos()
    

def teardown() :
    global f,h
    del f,h

def test_center() :
    global f,h
    with pynbody.analysis.halo.center(h[1]) :
        np.testing.assert_allclose(f['pos'][0],[-0.0137471,  -0.00208458, -0.04392379],atol=1e-6)

def test_align() :
    global f,h
    with pynbody.analysis.angmom.faceon(h[1]) :
        np.testing.assert_allclose(f['pos'][:2],[[-0.01071802, -0.00150358, -0.0447827 ],
                                                 [-0.01002559,  0.00244076, -0.04464959]],atol=1e-6)
        np.testing.assert_allclose(f['vel'][:2],[[ 0.01720319,  0.0184799,  -0.01985859],
                                                 [ 0.05133334,  0.02735672, -0.01030289]],atol=1e-6)


def test_virialradius() :
    global f,h
    with pynbody.analysis.halo.center(h[1]) :
        vrad = pynbody.analysis.halo.virial_radius(f)
        np.testing.assert_allclose(vrad, 0.005946911872,atol=1e-8)

