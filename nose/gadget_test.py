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
    expected=['nhp','hsml','nhe','u','sfr','pos','vel','id','mass','nh','rho','nheq','nhep']
    assert (blocks == expected)
    #Check that they have the right families
    assert(snap.loadable_family_keys(pynbody.family.gas) == expected)
    assert(snap.loadable_family_keys(pynbody.family.dm) == ['pos','vel','id','mass'])
    assert(snap.loadable_family_keys(pynbody.family.star) == ['pos','vel','id','mass'])
    assert(snap.loadable_family_keys(pynbody.family.neutrino) == [])

def test_standard_arrays() :
    """Check we can actually load some of these arrays"""
    snap.dm['pos']
    snap.gas['pos']
    snap.star['pos']
    snap['pos']
    #Load a second time to check that family_arrays still work
    snap.dm['pos']
    snap['vel']
    snap['id']
    snap.gas['rho']
    snap.gas['u']
    snap.star['mass']

def test_array_sizes() :
    """Check we have the right sizes for the arrays"""
    assert(np.shape(snap.dm['pos']) == (4096,3))
    assert(np.shape(snap['vel']) == (8192,3))
    assert(np.shape(snap.gas['rho']) == (4039,))
    assert(snap.gas['u'].dtype == np.float32)
    assert(snap.gas['id'].dtype == np.int32)

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
    assert(np.max(snap["id"]) == 8192)
    assert(np.min(snap["id"]) == 1)
    assert(np.mean(snap["id"]) == 4096.5)
    assert(abs(np.mean(snap["pos"]) - 1434.666) < 0.001)
    assert(abs(snap["pos"][52][1] - 456.6968) < 0.001)
    assert(abs(snap.gas["u"][100] - 438.39496) < 0.001)

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
    assert(snap.loadable_keys() == snap3.loadable_keys())
    assert((snap3["pos"] == snap["pos"]).all())
    assert((snap3.gas["rho"] == snap.gas["rho"]).all())
    assert(snap3.check_headers(snap.header, snap3.header))
