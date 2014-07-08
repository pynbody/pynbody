import pynbody, numpy as np
import pylab as p
import pickle

def test_images() :

    f = pynbody.load("testdata/g15784.lr.01024")
    h = f.halos()
    pynbody.analysis.halo.center(h[1])
    f.physical_units()

    im3d = pynbody.plot.sph.image(f.gas,width=20.0,units="m_p cm^-3",noplot=True)
    im2d = pynbody.plot.sph.image(f.gas,width=20.0,units="m_p cm^-2",noplot=True)

    compare2d, compare3d = np.load("test_im_2d.npy"), np.load("test_im_3d.npy")

    assert np.log10(im2d/compare2d).abs().mean()<0.03
    assert np.log10(im3d/compare3d).abs().mean()<0.03
    
