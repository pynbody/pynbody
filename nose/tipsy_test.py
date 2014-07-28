import pynbody
import numpy as np
import glob
import os

def setup() :
    X = glob.glob("testdata/test_out.*")
    for z in X :
        os.unlink(z)
    
    global f, h
    f = pynbody.load("testdata/g15784.lr.01024")
    h = f.halos()
    

def teardown() :
    global f
    del f


def test_get() :
    current =  f['pos'][0:100:10]
    print current

    correct = np.array([[ 0.01070931, -0.03619793, -0.16635996],
                        [ 0.01066598, -0.0328698 , -0.16544016],
                        [ 0.0080902 , -0.03409814, -0.15953901],
                        [ 0.01125323, -0.03251356, -0.14957215],
                        [ 0.01872441, -0.03908035, -0.16008312],
                        [ 0.01330984, -0.03552091, -0.14454767],
                        [ 0.01438289, -0.03428916, -0.13781759],
                        [ 0.01499815, -0.03602122, -0.13986239],
                        [ 0.0155305 , -0.0332876 , -0.14802328],
                        [ 0.01534095, -0.03525123, -0.14457548]])
    
    print "Position error of ",np.abs(current-correct).sum()
    assert (np.abs(current-correct).sum()<1.e-7)

def test_standard_arrays() :
    # just check all standard arrays are there
    with f.lazy_off :
        f['x']
        f['y']
        f['z']
        f['pos']
        f['vel']
        f['vx']
        f['vy']
        f['vz']
        f['eps']
        f['phi']
        f['mass']
        f.gas['rho']
        f.gas['temp']
        f.gas['metals']
        f.star['tform']
        f.star['metals']
    
    
def test_halo() :
    print "Length=",len(h[1])
    assert len(h[1])==502300

    

def test_loadable_array() :
    assert 'HI' in f.loadable_keys()
    f['HI']
    assert 'HI' in f.keys()
    
    assert 'HeI' in f.loadable_keys()
    f['HeI']
    assert 'HeI' in f.keys()

    assert f['HI'].dtype == np.float32
    assert f['HeI'].dtype == np.float32
    assert f['igasorder'].dtype == np.int32 

    HI_correct = np.array([  5.35406599e-08,   4.97452731e-07,   5.73000014e-01, \
         5.73000014e-01,   5.73000014e-01,   5.73000014e-01,\
         5.73000014e-01,   5.73000014e-01,   5.73000014e-01,\
         5.73000014e-01,   5.73000014e-01,   5.73000014e-01,\
         5.73000014e-01,   5.73000014e-01,   5.73000014e-01,\
         4.18154418e-01,   5.86960971e-01,   3.94545615e-01], dtype=np.float32)
    HeI_correct = np.array([  3.51669648e-12,   2.28513852e-09,   3.53999995e-03,\
         3.53999995e-03,   3.53999995e-03,   3.53999995e-03,\
         3.53999995e-03,   3.53999995e-03,   3.53999995e-03,\
         3.53999995e-03,   3.53999995e-03,   3.53999995e-03,\
         3.53999995e-03,   3.53999995e-03,   3.53999995e-03,\
         3.94968614e-02,   5.48484921e-02,   4.77905162e-02], dtype=np.float32)
    igasorder_correct = np.array([     0,      0,      0,      0,      0,      0,      0,      0,\
            0,      0,      0,      0,      0,      0,      0,  67264,\
        72514, 177485], dtype=np.int32)

    assert (f['igasorder'][::100000]==igasorder_correct).all()
    assert abs(f['HI'][::100000]-HI_correct).sum()<1.e-10
    assert abs(f['HeI'][::100000]-HeI_correct).sum()<1.e-10
    
    

def _assert_unit(u, targ, eps=0.01) :
    assert abs(u.ratio(targ)-1.0)<eps
    
def test_units() :
    print f['pos'].units
    print f['vel'].units
    print f['phi'].units
    print f.gas['rho'].units
    print f.star['tform'].units.in_units("Gyr")

    
    _assert_unit(f['pos'].units,"6.85e+04 kpc a")
    _assert_unit(f['vel'].units,"1.73e+03 km a s**-1")
    _assert_unit(f['phi'].units,"2.98e+06 km**2 a**-1 s**-2")
    _assert_unit(f.gas['rho'].units,"1.48e+02 Msol kpc**-3 a**-3")
    _assert_unit(f.star['tform'].units,"38.76 Gyr")


def test_halo_unit_conversion() :
    f.gas['rho'].convert_units('Msol kpc^-3')
    assert str(h[1].gas['rho'].units)=='Msol kpc**-3'
    
    h[1].gas['rho'].convert_units('m_p cm^-3')
    assert str(h[1].gas['rho'].units)=='m_p cm**-3'

def test_write() :
    f2 = pynbody.new(dm=10,star=10,gas=20)
    f2.dm['test_array']=np.ones(10)
    f2['x']=np.arange(0,40)
    f2['vx']=np.arange(40,80)
    f2.write(fmt=pynbody.tipsy.TipsySnap, filename="testdata/test_out.tipsy")
    
    f3 = pynbody.load("testdata/test_out.tipsy")
    assert all(f3['x']==f2['x'])
    assert all(f3['vx']==f3['vx'])
    assert all(f3.dm['test_array']==f2.dm['test_array'])

def test_array_write() :
    
    f['array_write_test'] = np.random.rand(len(f))
    f['array_write_test'].write(overwrite=True)
    f['array_read_test'] = f['array_write_test']
    del f['array_write_test']

    # will re-lazy-load
    assert all(np.abs(f['array_write_test']-f['array_read_test'])<1.e-5)

def test_isolated_read() :
    s = pynbody.load('testdata/isolated_ics.std')

def test_array_metadata() :
    f1 = pynbody.load("testdata/test_out.tipsy")
    
    f1.gas['zog'] = np.ones(len(f1.gas))
    f1.gas['zog'].units = "Msol kpc^-1"
    f1.gas['zog'].write()

    f1['banana'] = np.ones(len(f1))*0.5
    f1['banana'].units = "kpc^3 Myr^-1"
    f1['banana'].write()

    del f1

    f1 = pynbody.load("testdata/test_out.tipsy")
    assert "banana" in f1.loadable_keys()
    assert "zog" not in f1.loadable_keys()
    assert "banana" in f1.gas.loadable_keys()
    assert "zog" in f1.gas.loadable_keys()

    try:
        f1.star["zog"] # -> KeyError
        assert False # Shouldn't have been able to load gas-only array zog
    except KeyError :
        pass

    f1.gas['zog']
    assert f1.gas['zog'][0]==1.0
    assert f1.gas['zog'].units == "Msol kpc^-1"

    f1.star['banana']
    f1.gas['banana']
    f1.dm['banana']
    assert f1['banana'].units=="kpc^3 Myr^-1"

def test_array_update() : 
    f1 = pynbody.load("testdata/test_out.tipsy")
    
    f1['bla'] = np.zeros(len(f1))
    f1['bla'].units = 'km'
    f1['bla'].write()

    del(f1['bla'])

    f1['bla']

    f1.g['bla'] = 1
    f1.d['bla'] = 2
    f1.s['bla'] = 3

    # test the case where bla is a snapshot-level array

    try:
        f1.g['bla'].write()
        assert False # should not be allowed to overwrite here
    except IOError :
        pass
    
    f1.write_array( 'bla', [pynbody.family.gas,pynbody.family.dm], overwrite=True)

    del(f1['bla'])

    f1['bla']

    assert all(f1.g['bla'] == 1)
    assert all(f1.d['bla'] == 2)
    assert all(f1.s['bla'] == 0)

    # test the case where bla2 is a family-level array
    
    f1.g['bla2'] = np.zeros(len(f1.g))
    f1.g['bla2'].units = 'km'
    f1.s['bla2'] = np.ones(len(f1.s))
    
    f1.write_array('bla2',[pynbody.family.gas,pynbody.family.star])

    del(f1)

    f1 = pynbody.load("testdata/test_out.tipsy")

    assert all(f1.g['bla2'] == 0)
    assert all(f1.s['bla2'] == 1)
    

def test_snapshot_update() :
    f1 = pynbody.load("testdata/test_out.tipsy")
    f1['pos']
    f1['pos'] = np.arange(0,len(f1)*3).reshape(len(f1),3)
    
    # convert units -- the array should get written out in simulation units
    f1.g['pos'].convert_units('Mpc')

    f1['pos'].write(overwrite=True)
    f1.gas['metals'] = np.ones(len(f1.gas))*123.
    f1.star['metals'] = np.ones(len(f1.star))*345.

    f1.gas['metals'].write(overwrite=True)
    del f1

    f2 = pynbody.load("testdata/test_out.tipsy")
    assert (f2['pos']==np.arange(0,len(f2)*3).reshape(len(f2),3)).all()
    assert (f2.gas['metals']==123.).all() # should have updated gas.metals
    assert not (f2.star['metals']==345.).any() # should not have written out changes to star.metals

    # this is a completion:
    f2.dm['metals'] = np.ones(len(f2.dm))*789.1

    # should now be a simulation-level array... write it out
    f2['metals'].write(overwrite=True)

    del f2['metals']

    
    f3 = pynbody.load("testdata/test_out.tipsy")
    assert (f3.dm['metals']==789.1).all()

    

def test_unit_persistence() :
    f1 = pynbody.load("testdata/g15784.lr.01024")
    f1['pos']
    f1.physical_units()
    assert f1['pos'].units=='kpc'
    del f1['pos']
    f1['pos']
    assert f1['pos'].units=='kpc'
    del f1['pos']
    f1[[4,6,10]]['pos'] # test for fail when autoconverting on subarrays
    assert f1['pos'].units=='kpc'
    del f1['pos']
    f1.original_units()
    f1['pos']
    assert f1['pos'].units!='kpc'
