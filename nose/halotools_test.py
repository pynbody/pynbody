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
        np.testing.assert_allclose(f['pos'][0],[-0.0137471,  -0.00208458, -0.04392379],atol=1e-4)

def test_align() :
    global f,h
    with pynbody.analysis.angmom.faceon(h[1]) :
        np.testing.assert_allclose(f['pos'][:2],[[-0.03144149,  0.01132592, -0.03177311],
                                                 [-0.03079908,  0.01505963, -0.03046862]],atol=1e-4)
        
        np.testing.assert_allclose(f['vel'][:2],[[-0.00126288,  0.02604627, -0.01340912],
                                                 [ 0.03307226,  0.03773096, -0.01782989]],atol=1e-4)
                                                


def test_virialradius() :
    global f,h
    with pynbody.analysis.halo.center(h[1]) :
        vrad = pynbody.analysis.halo.virial_radius(f)
        np.testing.assert_allclose(vrad, 0.005946911872,atol=1.e-5)

