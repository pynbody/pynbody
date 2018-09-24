import pynbody
import numpy as np
from itertools import chain
import shutil
import h5py

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
       # s.gas['u']
        s.star['mass']


def _h5py_copy_with_key_rename(src,dest):
    shutil.copy(src,dest)
    f = h5py.File(dest)
    for tp in f:
        if "Mass" in tp:
            tp.move("Mass","Masses")
    f.close()




def test_alt_names():
    _h5py_copy_with_key_rename('testdata/Test_NOSN_NOZCOOL_L010N0128/data/snapshot_103/snap_103.hdf5',
                'testdata/Test_NOSN_NOZCOOL_L010N0128/data/snapshot_103/snap_103_altnames.hdf5')

    snap_alt = pynbody.load('testdata/Test_NOSN_NOZCOOL_L010N0128/data/snapshot_103/snap_103_altnames.hdf5')
    assert 'mass' in snap_alt.loadable_keys()
    assert all(snap_alt['mass']==snap['mass'])

def test_issue_256() :
    assert 'pos' in snap.loadable_keys()
    assert 'pos' in snap.dm.loadable_keys()
    assert 'pos' in snap.gas.loadable_keys()
    assert 'He' not in snap.loadable_keys()
    assert 'He' not in snap.dm.loadable_keys()
    assert 'He' in snap.gas.loadable_keys()

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
            assert(len(fam['iord']) == subfind._hdf_files[0][subfind._family_to_group_map[fam.families()[0]][0]]['Length'][i])
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
    from pyread_gadget_hdf5 import pyread_gadget_hdf5

    FoF_Mass = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'Mass', sub_dir='fof', nopanda=True, silent=None)
    FoF_MassType = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'MassType', sub_dir='fof', nopanda=True, silent=True)
    Sub_Mass = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'Mass', sub_dir='subfind', nopanda=True, silent=True)
    Sub_MassType = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'MassType', sub_dir='subfind', nopanda=True, silent=True)
    NsubPerHalo = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'NsubPerHalo', sub_dir='subfind', nopanda=True, silent=True)
    OffsetHalo = np.roll(NsubPerHalo.cumsum(), 1)
    OffsetHalo[0]=0 ## To start counter

    h = subfind.halos()

    FoF_CoM = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'CenterOfMass', sub_dir='fof', nopanda=True, silent=True)
    Sub_CoM = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'CenterOfMass', sub_dir='subfind', nopanda=True, silent=True)

    # Check the Halo Array values
    for i,halo in enumerate(h[0:10]) :
        assert(np.allclose(halo.properties['CenterOfMass'], FoF_CoM[i], rtol=1e-3))

        for j, s in enumerate(halo.sub) :
	        assert(np.allclose(s.properties['CenterOfMass'], Sub_CoM[OffsetHalo[i]+j], rtol=1e-3))

    ###
    # Test the Halo particle information
    ###

    # Mass of each component for FOF halos
    for i,halo in enumerate(h[0:10]) :
        assert(np.allclose(halo.g['mass'].sum(), FoF_MassType[i,0], rtol=1e-3))
        assert(np.allclose(halo.dm['mass'].sum(), FoF_MassType[i,1], rtol=1e-3))
        assert(np.allclose(halo.s['mass'].sum(), FoF_MassType[i,4], rtol=1e-3))
        assert(np.allclose(halo['mass'].sum(), FoF_Mass[i], rtol=1e-3))

    # Mass of each component for Subhalos
    for i,halo in enumerate(h[0:10]) :
        for j, s in enumerate(halo.sub) :
            assert(np.allclose(s.g['mass'].sum(), Sub_MassType[OffsetHalo[i]+j,0], rtol=1e-3))
            assert(np.allclose(s.dm['mass'].sum(), Sub_MassType[OffsetHalo[i]+j,1], rtol=1e-3))
            assert(np.allclose(s.s['mass'].sum(), Sub_MassType[OffsetHalo[i]+j,4], rtol=1e-3))
            assert(np.allclose(s['mass'].sum(), Sub_Mass[OffsetHalo[i]+j], rtol=1e-3))

    FoF_Temp = pyread_gadget_hdf5(filesub+'.0.hdf5', 0, 'Temperature', sub_dir='fof', nopanda=True, silent=True, physunits=True)
    FoF_Length = pyread_gadget_hdf5(filesub+'.0.hdf5', 0, 'Length', sub_dir='fof', nopanda=True, silent=True, physunits=True)
    FoF_Offset = pyread_gadget_hdf5(filesub+'.0.hdf5', 0, 'Offset', sub_dir='fof', nopanda=True, silent=True, physunits=True)

    # Test the Particle Temperature and implicitly the particle ordering
    for i,halo in enumerate(h[0:10]) :
        assert(np.allclose(list(halo.g['temp']), list(chain.from_iterable(FoF_Temp[np.arange(FoF_Offset[i],FoF_Offset[i]+FoF_Length[i],dtype=np.int64)])), rtol=1e-3))

def test_write():
    ar_name = 'test_array'
    snap[ar_name] = np.random.uniform(0,1,len(snap))
    snap[ar_name].write()
    snap2 = pynbody.load('testdata/Test_NOSN_NOZCOOL_L010N0128/data/snapshot_103/snap_103.hdf5')
    assert(np.allclose(snap2[ar_name], snap[ar_name]))

def test_hi_derivation():
    HI_answer = [  6.96499870e-06,   6.68348046e-06,   1.13855074e-05,
         1.10936027e-05,   1.40641633e-05,   1.67324738e-05,
         2.26228929e-05,   1.64661638e-05,   2.79337124e-05,
         3.32789555e-05,   2.38397192e-05,   5.11526743e-04,
         1.86211183e-01,   4.58309086e-02,   9.98117529e-02,
         1.76779058e-02,   8.30149935e-02,   1.08688537e-02,
         1.44146419e-07,   7.95141614e-08,   8.00568016e-05,
         5.40560080e-08,   2.73720754e-01,   2.91772885e-02,
         7.37755701e-04,   3.93431603e-02,   3.52700543e-03,
         1.46685188e-07,   4.55900305e-08,   3.85495273e-03,
         1.75020358e-07,   1.27841671e-01,   1.01551435e-07,
         3.23647121e-08,   8.22351949e-06,   1.03758201e-05,
         1.13115067e-05,   2.57878344e-05,   2.74634221e-05,
         5.34312023e-05,   2.55750061e-01,   3.83638138e-04,
         7.96613219e-03,   2.57835498e-03,   5.89219887e-08]

    assert np.allclose(subfind.halos()[0].gas['HI'][::100],HI_answer)


def test_fof_vs_sub_assignment():
    h = subfind.halos()
    assert(np.allclose(h[0].properties['Mass'],28.604694074339932))
    assert(np.allclose( h[0].properties['Halo_M_Crit200'], 29.796955896599684))
    assert(np.allclose(h[1].properties['Mass'], 8.880245794949587))
    assert(np.allclose(h[1].properties['Halo_M_Crit200'],8.116568749712314))

def test_hdf_ordering():
    # HDF files do not intrinsically specify the order in which the particle types occur
    # Because some operations may require stability, pynbody now imposes order by the particle type
    # number
    assert snap._family_slice[pynbody.family.gas] == slice(0, 2076907, None)
    assert snap._family_slice[pynbody.family.dm] == slice(2076907, 4174059, None)
    assert snap._family_slice[pynbody.family.star] == slice(4174059, 4194304, None)