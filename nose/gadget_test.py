import pynbody
import numpy as np

def setup() :
    global snap
    snap=pynbody.load("testdata/test_g2_snap")
 
def teardown() :
    global snap
    del snap

def test_construct() :
    """Check the basic properties of the snapshot"""
    assert (np.size(snap._files)==2)
    assert(snap.header.num_files == 2)
    assert (snap.filename=="testdata/test_g2_snap")
    assert(snap._num_particles == 8192)
    for f in snap._files :
        assert(f.format2 == True)
        assert(f.endian == "=")

def test_loadable() :
    """Check we have found all the blocks that should be in the snapshot"""
    blocks=snap.loadable_keys()
    expected_gas=['nhp','smooth','nhe','u','sfr','pos','vel','iord','mass','nh','rho','nheq','nhep']
    expected_all = ['pos','vel','iord','mass']

    #Check that they have the right families
    assert(set(snap.gas.loadable_keys()) == set(expected_gas))
    assert(set(snap.dm.loadable_keys()) == set(expected_all))
    assert(set(snap.star.loadable_keys()) == set(expected_all))
    assert(set(snap.loadable_keys()) == set(expected_all))
    assert(snap.neutrino.loadable_keys() == [])

def test_standard_arrays() :
    """Check we can actually load some of these arrays"""
    snap.dm['pos']
    snap.gas['pos']
    snap.star['pos']
    snap['pos']
    snap['mass']
    #Load a second time to check that family_arrays still work
    snap.dm['pos']
    snap['vel']
    snap['iord']
    snap.gas['rho']
    snap.gas['u']
    snap.star['mass']

def test_array_sizes() :
    """Check we have the right sizes for the arrays"""
    assert(np.shape(snap.dm['pos']) == (4096,3))
    assert(np.shape(snap['vel']) == (8192,3))
    assert(np.shape(snap.gas['rho']) == (4039,))
    assert(snap.gas['u'].dtype == np.float32)
    assert(snap.gas['iord'].dtype == np.int32)

def test_fam_sim() :
    """Check that an array loaded as families is the same as one loaded as a simulation array"""
    snap2=pynbody.load("testdata/test_g2_snap")
    snap3=pynbody.load("testdata/test_g2_snap")
    snap3.gas["pos"]
    snap3.dm["pos"]
    snap3.star["pos"]
    assert((snap3["pos"] == snap2["pos"]).all())

def test_array_contents() :
    """Check some array elements"""
    assert(np.max(snap["iord"]) == 8192)
    assert(np.min(snap["iord"]) == 1)
    assert(np.mean(snap["iord"]) == 4096.5)

    # 10/11/13 - AP - suspect the following tests are incorrect
    # because ordering of file did not agree with pynbody ordering
    
    assert(abs(np.mean(snap["pos"]) - 1434.664) < 0.002)
    assert(abs(snap["pos"][52][1] - 456.69678) < 0.001)
    assert(abs(snap.gas["u"][100] - 438.39496) < 0.001)
    assert(abs(snap.dm["mass"][5] - 0.04061608) < 0.001)

def test_header() :
    """Check some header properties"""
    assert(abs(snap.header.BoxSize - 3000.0) < 0.001)
    assert(abs(snap.header.HubbleParam - 0.710) < 0.001)
    assert(abs(snap.header.Omega0 - 0.2669) < 0.001)
    assert(snap.header.flag_cooling == 1)
    assert(snap.header.flag_metals == 0)

def test_g1_load() :
    """Check we can load gadget-1 files also"""
    snap2 = pynbody.load("testdata/gadget1.snap")

def test_write() :
    """Check that we can write a new snapshot and read it again, 
    and the written and the read are the same."""
    snap.write(filename = 'testdata/test_gadget_write')
    snap3=pynbody.load('testdata/test_gadget_write')
    assert(set(snap.loadable_keys()) == set(snap3.loadable_keys()))
    assert((snap3["pos"] == snap["pos"]).all())
    assert((snap3.gas["rho"] == snap.gas["rho"]).all())
    assert(snap3.check_headers(snap.header, snap3.header))

def test_conversion() :
    """Check that we can convert a file from tipsy format and load it again"""
    snap4 = pynbody.load("testdata/g15784.lr.01024")
    snap4.write(fmt=pynbody.gadget.GadgetSnap, filename="testdata/test_conversion.gadget")
    snap5=pynbody.load("testdata/test_conversion.gadget")

def test_write_single_array():
    """Check that we can write a single array and read it back"""
    snap["pos"].write(overwrite=True)
    snap6=pynbody.load("testdata/test_g2_snap")
    assert((snap6["pos"] == snap["pos"]).all())

def test_no_mass_block() :
    f = pynbody.load("testdata/gadget_no_mass")
    f['mass'] # should succeed

def test_unit_persistence() :
    f = pynbody.load("testdata/test_g2_snap")

    # f2 is the comparison case - just load the whole
    # position array and convert it, simple    
    f2 = pynbody.load("testdata/test_g2_snap")
    f2['pos']
    f2.physical_units()

  
    f.gas['pos']
    f.physical_units()
    assert (f.gas['pos']==f2.gas['pos']).all()

    # the following lazy-loads should lead to the data being
    # auto-converted
    f.dm['pos']
    assert (f.gas['pos']==f2.gas['pos']).all()
    assert (f.dm['pos']==f2.dm['pos']).all()

    # the final one is the tricky one because this will trigger
    # an array promotion and hence internally inconsistent units
    f.star['pos']

    assert (f.star['pos']==f2.star['pos']).all()

    # also check it hasn't messed up the other bits of the array!
    assert (f.gas['pos']==f2.gas['pos']).all()
    assert (f.dm['pos']==f2.dm['pos']).all()
        
     
    assert (f['pos']==f2['pos']).all()



def test_per_particle_loading() :
    """Tests that loading one family at a time results in the
    same final array as loading all at once. There are a number of
    subtelties in the gadget handler that could mess this up by loading
    the wrong data."""

    f_all = pynbody.load("testdata/test_g2_snap")
    f_part =  pynbody.load("testdata/test_g2_snap")

    f_part.dm['pos']
    f_part.star['pos']
    f_part.gas['pos']

    assert (f_all['pos']==f_part['pos']).all()
