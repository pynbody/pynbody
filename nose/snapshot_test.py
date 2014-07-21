import pynbody
import numpy as np
import weakref

def setup() :
    global f, h
    f = pynbody.new(dm=1000, star=500, gas=500)
    f['pos'] = np.random.normal(scale=1.0, size=f['pos'].shape)
    f['vel'] = np.random.normal(scale=1.0, size=f['vel'].shape)
    f['mass'] = np.random.uniform(1.0,10.0,size=f['mass'].shape)
    f.gas['rho'] = np.ones(500, dtype=float)

def teardown() :
    global f
    del f



def test_array_promotion() :
    f.dm['tx'] = np.arange(0,1000)
    f.gas['tx'] = np.arange(0,500)

    # currently still a family array
    assert 'tx' in f.family_keys()
    assert 'tx' not in f.keys()
    
    try :
        f.star['tx']
        assert False # should have raised KeyError
    except KeyError :
        pass
    
    try :
        f['tx']
        assert False # should have raised KeyError
    except KeyError :
        pass

    f.star['tx'] = np.arange(0,500)

    # Should have been promoted
    assert 'tx' not in f.family_keys()
    assert 'tx' in f.keys()

    f['tx'] # should succeed
    del f['tx']


def test_one_family_promotion() :
    fx = pynbody.new(dm=10)
    fx.dm['bla'] = np.arange(10)
    # should have been made as a full-simulation array
    
    assert 'bla' in fx.keys()
    fx['bla']
    del fx

    
    

    
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
    f.dm['vr']
    assert f.dm['vr'].derived
    
    assert 'vr' not in f.keys()
    assert 'vr' in f.gas.keys()
    f['vr']
    assert 'vr' in f.keys()
    
    try:
        f['r'][22]+=3
        # Array should not be writable
        assert False
    except (RuntimeError, ValueError, TypeError) :
        pass

    f['r'].derived = False
    # Now we should be able to write to it
    f['r'][22]+=3

    f['pos']+=(2,0,0)
    assert 'r' in f.keys() # broken the link, so r shouldn't have been marked dirty
    


def test_unit_inheritance() :
    f['pos'].units = 'km'
    f['vel'].units = 'km s^-1'


def test_equality() :
    assert f==f
    assert f[[1,5,8]]!=f
    assert f[[1,6,9]]!=f[[1,5,22]]
    assert f[::5][[1,2,3]]==f[[5,10,15]]
    assert f.dm==f.dm
    assert f.dm==f[f._get_family_slice(pynbody.family.dm)]
    

def test_arraytype() :
    SA = pynbody.array.SimArray
    ISA = pynbody.array.IndexedSimArray

    assert type(f["mass"]) is SA
    assert type(f[[1,2,3]]["mass"]) is ISA
    assert type(f[::20]["mass"]) is SA
    assert type(f.dm['mass']) is SA
    assert type(f.gas['rho']) is SA
    assert type(f[[1,3,5,7,9,11]].dm['mass']) is ISA
    assert type(f[[1,2,3,8]].gas['rho']) is ISA
    
def test_persistence() :
    f.dm.kdtree = 123
    assert f.dm.kdtree==123
    f.dm[::12].kdtree = 234
    assert f.dm[::12].kdtree==234
    assert f.dm.kdtree == 123
    f.gas.kdtree=96
    assert f.gas.kdtree==96
    assert f.dm.kdtree==123
    f.kdtree = 92
    assert f.kdtree==92
    assert f.dm.kdtree==123
    assert f.dm[::12].kdtree==234
    assert f.gas.kdtree==96

def test_copy() :
    import copy
    f2 = copy.deepcopy(f)
    f2.dm['mass'][0] = 2
    f.dm['mass'][0] = 1
    assert f2.dm['mass'][0]!=f.dm['mass'][0]
    
    assert len(f2)==len(f)
    assert len(f2.dm)==len(f.dm)
    assert len(f2.gas['rho'])==500

    f2 = copy.deepcopy(f[::3])
    assert len(f2)==len(f[::3])
    assert len(f2.gas)==len(f[::3].gas)
    
    assert all(f2.gas['mass'] == f[::3].gas['mass'])
    f2.gas['mass'][0]=999
    assert not all(f2.gas['mass'] == f[::3].gas['mass'])

    f['pos'].units='kpc'
    f2 = copy.deepcopy(f[[1,2,3,4]])
    assert f2['pos'].units=='kpc'
    # this tests for a bug where units were not correctly
    # copied in from IndexedSimArrays
    
    

def test_mean_by_mass() :
    f['pos'].units = 'kpc'
    f['mass'].units = 'Msol'

    assert str(f.mean_by_mass('pos').units) == 'kpc'


def test_name_awareness() :
    assert f['pos'].name is 'pos'


def test_immediate_mode() :
    with f.immediate_mode :
        assert isinstance(f[[1,6,10]]['x'], pynbody.array.SimArray)
        test_val = f[[1,6,10]]['x']
    assert isinstance(f[[1,6,10]]['x'], pynbody.array.IndexedSimArray)
    assert (test_val==f[[1,6,10]]['x']).all()

    with f.immediate_mode :
        # check we get the same actual object two times
        fsub = f.dm[[1,6,52]]
        xa = fsub['x']
        xb = fsub['x']
        assert xa is xb
        xc = f.dm[[1,6,52]]['x']
        assert xa is xc
        wr = weakref.ref(xa)
        del xa,xb,xc
        assert wr() is not None

    # check it was deleted
    assert wr() is None

    del f['r']
    
    f.dm['r']

    # check this also works with family-level arrays
    
    with f.immediate_mode :
        assert isinstance(f.dm[[22,53,69]]['r'], pynbody.array.SimArray)
        assert isinstance(f[[600,603,670]].dm['r'], pynbody.array.SimArray)
    
def test_subsnap_by_boolean_mask() :
    print (f['x']>0).shape, len(f)
    assert (f['x'][f['x']>0]==f[f['x']>0]['x']).all()
