import pynbody
import numpy as np
import numpy.testing as npt
import copy

def setup() :
    global f, original
    
    f = pynbody.new(dm=1000)
    f['pos'] = np.random.normal(scale=1.0, size=f['pos'].shape)
    f['vel'] = np.random.normal(scale=1.0, size=f['vel'].shape)
    f['mass'] = np.random.uniform(1.0,10.0,size=f['mass'].shape)

    original = copy.deepcopy(f)
    


def test_translate():
    global f,original

    with pynbody.transformation.translate(f, [1,0,0]) :
        npt.assert_almost_equal(f['pos'],original['pos']+[1,0,0])

    # check moved back
    npt.assert_almost_equal(f['pos'],original['pos'])

    # try again with with abnormal exit
    try:
        with pynbody.transformation.translate(f, [1,0,0]) :
            npt.assert_almost_equal(f['pos'],original['pos']+[1,0,0])
            raise RuntimeError
    except RuntimeError :
        pass

    npt.assert_almost_equal(f['pos'],original['pos'])

def test_v_translate() :
    global f,original

    with pynbody.transformation.v_translate(f, [1,0,0]) :
        npt.assert_almost_equal(f['vel'],original['vel']+[1,0,0])

    # check moved back
    npt.assert_almost_equal(f['vel'],original['vel'])

    # try again with with abnormal exit
    try:
        with pynbody.transformation.v_translate(f, [1,0,0]) :
            npt.assert_almost_equal(f['vel'],original['vel']+[1,0,0])
            raise RuntimeError
    except RuntimeError :
        pass

    npt.assert_almost_equal(f['vel'],original['vel'])

def test_vp_translate() :
    global f,original

    with pynbody.transformation.xv_translate(f, [1,0,0], [2,0,0]) :
        npt.assert_almost_equal(f['vel'],original['vel']+[2,0,0])
        npt.assert_almost_equal(f['pos'],original['pos']+[1,0,0])
        
    # check moved back
    npt.assert_almost_equal(f['vel'],original['vel'])
    npt.assert_almost_equal(f['pos'],original['pos'])
    
    # try again with with abnormal exit
    try:
        with pynbody.transformation.xv_translate(f, [1,0,0], [2,0,0]) :
            npt.assert_almost_equal(f['vel'],original['vel']+[2,0,0])
            npt.assert_almost_equal(f['pos'],original['pos']+[1,0,0])
            raise RuntimeError
    except RuntimeError :
        pass

    npt.assert_almost_equal(f['vel'],original['vel'])
    npt.assert_almost_equal(f['pos'],original['pos'])
