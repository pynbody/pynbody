import pynbody 
import numpy as np

def setup() : 
    global snap,subfind
    snap = pynbody.load('testdata/Test_NOSN_NOZCOOL_L010N0128/data/snapshot_103/snap_103.hdf5')
    subfind = pynbody.load('testdata/Test_NOSN_NOZCOOL_L010N0128/data/subhalos_103/subhalo_103')
    
def teardown() : 
    global snap,subfind
    del snap
    del subfind

def test_standard_arrays() : 
    """Check that the data loading works"""

    for s in [snap, subfind] : 
        s.dm['pos']
        s.gas['pos']
        s.star['pos']
        s['pos']
        s['mass']
    #Load a second time to check that family_arrays still work
        s.dm['pos']
        s['vel']
        s['iord']
        s.gas['rho']
        s.gas['u']
        s.star['mass']
        
def test_halo_loading() : 
    """ Check that halo loading works """
    h = subfind.halos()
    # check that data loading for individual fof groups works
    h[0]['pos']
    # check that loading the subhalos works
    h[0].sub[0]['pos']


    
    
    
