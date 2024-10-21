import gc
import shutil

import h5py
import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils
from pynbody import units


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gadget", "arepo")

@pytest.fixture
def snap():
    f = pynbody.load('testdata/gadget3/data/snapshot_103/snap_103.hdf5')
    yield f
    del f
    gc.collect()

@pytest.fixture
def subfind():
    f = pynbody.load('testdata/gadget3/data/subhalos_103/subhalo_103')
    yield f
    del f
    gc.collect()

def test_standard_arrays(snap, subfind) :
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
    f = h5py.File(dest, 'r+')
    for tp in f:
        if "Mass" in tp:
            tp.move("Mass","Masses")
    f.close()




def test_alt_names(snap):
    _h5py_copy_with_key_rename('testdata/gadget3/data/snapshot_103/snap_103.hdf5',
                'testdata/gadget3/data/snapshot_103/snap_103_altnames.hdf5')

    snap_alt = pynbody.load('testdata/gadget3/data/snapshot_103/snap_103_altnames.hdf5')
    assert 'mass' in snap_alt.loadable_keys()
    assert all(snap_alt['mass']==snap['mass'])

def test_issue_256(snap) :
    assert 'pos' in snap.loadable_keys()
    assert 'pos' in snap.dm.loadable_keys()
    assert 'pos' in snap.gas.loadable_keys()
    assert 'He' not in snap.loadable_keys()
    assert 'He' not in snap.dm.loadable_keys()
    assert 'He' in snap.gas.loadable_keys()

def test_write():
    # make a copy of snap_103.hdf5 to avoid disturbing the original
    shutil.copy('testdata/gadget3/data/snapshot_103/snap_103.hdf5', 'testdata/gadget3/data/snapshot_103/snap_103_copy.hdf5')
    snap = pynbody.load('testdata/gadget3/data/snapshot_103/snap_103_copy.hdf5')
    ar_name = 'test_array'
    snap[ar_name] = np.random.uniform(0,1,len(snap))
    snap[ar_name].write()
    snap2 = pynbody.load('testdata/gadget3/data/snapshot_103/snap_103_copy.hdf5')
    v = snap[ar_name]
    with pytest.warns(UserWarning, match="Unable to infer units from HDF attributes"):
        v2 = snap2[ar_name]
    npt.assert_allclose(v, v2)

def test_hi_derivation(subfind):
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

    # interpolation routine changed so rtol increased to allow for slight deviations
    npt.assert_allclose(subfind.halos()[0].gas['HI'][::100],HI_answer,rtol=1e-6)

def test_hdf_ordering(snap):
    # HDF files do not intrinsically specify the order in which the particle types occur
    # Because some operations may require stability, pynbody now imposes order by the particle type
    # number
    assert snap._family_slice[pynbody.family.gas] == slice(0, 2076907, None)
    assert snap._family_slice[pynbody.family.dm] == slice(2076907, 4174059, None)
    assert snap._family_slice[pynbody.family.star] == slice(4174059, 4194304, None)


def test_mass_in_header():
    f = pynbody.load("testdata/gadget3/snap_028_z000p000.0.hdf5")
    f.physical_units()
    f['mass'] # load all masses
    assert np.allclose(f.dm['mass'][0], 3982880.471745421)

    f = pynbody.load("testdata/gadget3/snap_028_z000p000.0.hdf5")
    f.physical_units()
    # don't load all masses, allow it to be loaded for DM only
    assert np.allclose(f.dm['mass'][0], 3982880.471745421)

def test_gadgethdf_style_units():
    f = pynbody.load("testdata/gadget3/data/snapshot_103/snap_103.hdf5")
    npt.assert_allclose(f.st['InitialMass'].units.in_units("1.989e43 g h^-1"), 1.0,
                        rtol=1e-3)

def test_arepo_style_units():
    with pytest.warns(UserWarning, match=r"Masses are either stored in the header or have.*"):
        f = pynbody.load("testdata/arepo/agora_100.hdf5")
    npt.assert_allclose(f.st['EMP_InitialStellarMass'].units.in_units("1.989e42 g"),
                        1.0, rtol=1e-3)
    # I strongly suspect that the units in this file are wrong -- the masses are
    # in h^-1 units, so presumably these initial stellar masses should also be in
    # h^-1 units. This is backed up by checking that, numerically,
    #    (f.st['EMP_InitialStellarMass']/f.st['mass']).min() == 1.0
    # On the other hand, pynbody should just reflect back that error
    # to the user, really; we can't get involved in compensating for bugs in other codes,
    # or all hell will break loose. So, we check for the 'wrong' units.

    npt.assert_allclose(f.st['AREPOEMP_Metallicity'].units.in_units(1.0),
                        1.0, rtol=1e-5)
    # the above is a special case of a dimensionless array

    with pytest.warns(UserWarning, match="Unable to infer units from HDF attributes"):
        assert f.st['EMP_BirthTemperature'].units == units.NoUnit()
    # here is a case where no unit information is recorded in the file (who knows why)
