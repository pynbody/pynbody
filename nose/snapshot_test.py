import pynbody
import numpy as np

def setup() :
    global f, h
    f = pynbody.new(dm=1000, star=500, gas=500)
    f['pos'] = np.random.normal(scale=1.0, size=f['pos'].shape)
    f['vel'] = np.random.normal(scale=1.0, size=f['vel'].shape)
    f['mass'] = np.random.uniform(1.0,10.0,size=f['mass'].shape)

def teardown() :
    global f
    del f

    
def test_subscript() :
    
    a = f[::37]
    assert(np.abs(a['pos']-f['pos'][::37]).sum()<1.e-10)

    a = f[[1,5,9,12,52,94]]
    assert len(a)==6

    
    
def test_noarray() :
    def check(z, x) :
        try :
            z[x]
        except KeyError :
            return

        # We have not got the right exception back
        assert False

    check(f, 'thisarraydoesnotexist')
    check(f.gas, 'thisarraydoesnotexist')
    check(f[::5], 'thisarraydoesnotexist')
    check(f[[1,2,7,93]], 'thisarraydoesnotexist')

def test_derived_array() :
    assert 'vr' in f.derivable_keys()
    f['vr']
    assert 'vr' in f.keys()
    assert f['vr'].derived
    assert f.gas['vr'].derived
    assert f[::4]['vr'].derived
    assert f[[2,3,7,12]]['vr'].derived
    f['pos']+=(2,0,0)
    assert 'vr' not in f.keys()
    f['pos']-=(2,0,0)
    f.gas['vr']
    assert f.gas['vr'].derived
    assert 'vr' not in f.keys()
    assert 'vr' in f.gas.keys()
    f['vr']
    assert 'vr' in f.keys()
    
    try:
        f['r'][22]+=(3,3,3)
        # Array should not be writable
        assert False
    except RuntimeError :
        pass


def test_unit_inheritance() :
    f['pos'].units = 'km'
    f['vel'].units = 'km s^-1'
