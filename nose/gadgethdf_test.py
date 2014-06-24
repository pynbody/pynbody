import pynbody 
import numpy as np
from itertools import chain 

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
    h[1]['pos']

    # check that loading the subhalos works
    h[0].sub[0]['pos']
    for i,halo in enumerate(h[0:10]) : 
        halo['mass'].sum()
        for fam in [halo.g, halo.d, halo.s] : 
            assert(len(fam['iord']) == subfind._hdf[0]['FOF'][subfind._my_type_map[fam.families()[0]][0]]['Length'][i])
        for s in halo.sub : 
            s['mass'].sum()
            
            
    
    # test halo catalogue slicing
    for halo in h[0:10] : pass
    for halo in h[30:40] : pass
    for sub in h[0].sub[1:5] : pass

    
    
def test_halo_values() :
    """ Check that halo values (and sizes) agree with pyread_gadget_hdf5 """
    
    filesub = 'testdata/Test_NOSN_NOZCOOL_L010N0128/data/subhalos_103/subhalo_103'

    # load Alan Duffy's module from https://bitbucket.org/astroduff/pyreadgadget
    import urllib2, sys, imp
    u = urllib2.urlopen('https://bitbucket.org/astroduff/pyreadgadget/raw/master/pyread_gadget_hdf5.py')
    code = u.read()
    module = imp.new_module('pyread_gadget_hdf5')
    exec code in module.__dict__
    pyread_gadget_hdf5 = module.pyread_gadget_hdf5    

    FoF_Mass = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'Mass', sub_dir='fof', nopanda=True, silent=True, physunits=True)
    FoF_MassType = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'MassType', sub_dir='fof', nopanda=True, silent=True, physunits=True)
    Sub_Mass = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'Mass', sub_dir='subfind', nopanda=True, silent=True, physunits=True)
    Sub_MassType = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'MassType', sub_dir='subfind', nopanda=True, silent=True, physunits=True)
    NsubPerHalo = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'NsubPerHalo', sub_dir='subfind', nopanda=True, silent=True, physunits=True)
    OffsetHalo = np.roll(NsubPerHalo.cumsum(), 1)
    OffsetHalo[0]=0 ## To start counter

    h = subfind.halos()
    subfind.physical_units()

    # Test the total mass of each component for FOF halos
    for i,halo in enumerate(h[0:10]) : 
        if np.allclose(halo.g['mass'].sum(), FoF_MassType[i,0], rtol=1e-3) == False:
            print 'FoF Gas mass failed ', halo.g['mass'].sum(), FoF_MassType[i,0]
        if np.allclose(halo.dm['mass'].sum(), FoF_MassType[i,1], rtol=1e-3) == False:
            print 'FoF DM mass failed ', halo.dm['mass'].sum(), FoF_MassType[i,1]
        if np.allclose(halo.s['mass'].sum(), FoF_MassType[i,4], rtol=1e-3) == False:
            print 'FoF Stellar mass failed ', halo.s['mass'].sum(), FoF_MassType[i,4]
        if np.allclose(halo['mass'].sum(), FoF_Mass[i], rtol=1e-3) == False:
            print 'FoF Total mass failed ', halo['mass'].sum(), FoF_Mass[i]

    # Testing masses in SubHaloes
    for i,halo in enumerate(h[0:10]) : 
        for j, s in enumerate(halo.sub) : 
            if np.allclose(s.g['mass'].sum(), Sub_MassType[OffsetHalo[i]+j,0], rtol=1e-3) == False:
                print 'Subhalo Gas mass failed ', s.g['mass'].sum(), Sub_MassType[OffsetHalo[i]+j,0]
            if np.allclose(s.dm['mass'].sum(), Sub_MassType[OffsetHalo[i]+j,1], rtol=1e-3) == False:
                print 'Subhalo DM mass failed ', s.dm['mass'].sum(), Sub_MassType[OffsetHalo[i]+j,0]
            if np.allclose(s.s['mass'].sum(), Sub_MassType[OffsetHalo[i]+j,4], rtol=1e-3) == False:
                print 'Subhalo Stellar mass failed ', s.s['mass'].sum(), Sub_MassType[OffsetHalo[i]+j,0]  
            if np.allclose(s['mass'].sum(), Sub_Mass[OffsetHalo[i]+j], rtol=1e-3) == False:
                print 'Subhalo Total mass failed ', s['mass'].sum(), Sub_Mass[OffsetHalo[i]+j]  

    FoF_Temp = pyread_gadget_hdf5(filesub+'.0.hdf5', 0, 'Temperature', sub_dir='fof', nopanda=True, silent=True, physunits=True)
    FoF_Length = pyread_gadget_hdf5(filesub+'.0.hdf5', 0, 'Length', sub_dir='fof', nopanda=True, silent=True, physunits=True)
    FoF_Offset = pyread_gadget_hdf5(filesub+'.0.hdf5', 0, 'Offset', sub_dir='fof', nopanda=True, silent=True, physunits=True)

    # Test the Particle Temperature 
    for i,halo in enumerate(h[0:10]) : 
        if np.allclose(list(halo.g['temp']), list(chain.from_iterable(FoF_Temp[np.arange(FoF_Offset[i],FoF_Offset[i]+FoF_Length[i],dtype=np.int64)])), rtol=1e-3) == False:
            print "Gas Temperature failed", i,FoF_Offset[i],FoF_Length[i]
