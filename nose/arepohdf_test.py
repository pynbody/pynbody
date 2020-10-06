import pynbody
import numpy as np
from itertools import chain
import shutil
from nose.tools import assert_equals, assert_almost_equal
import h5py

def setup():
    global cosmo, agora
    cosmo = pynbody.load('testdata/arepo/cosmobox_015.hdf5')
    agora = pynbody.load('testdata/arepo/agora_100.hdf5')

def teardown():
    global cosmo, agora
    del cosmo
    del agora

def test_standard_arrays():
    """Check that the data loading works"""

    cosmo.dm['pos']
    cosmo.gas['pos']
    cosmo.star['pos']
    cosmo['pos']
    cosmo['mass']
#Load a second time to check that family_arrays still work
    cosmo.dm['pos']
    cosmo['vel']
    cosmo['iord']
    cosmo.gas['rho']
# s.gas['u']
    cosmo.star['mass']

def test_cosmo_units():
    # Test to ensure the base cosmological units have their correct value
    assert_equals(cosmo['pos'].units, "3.085678e24 cm a h**-1")
    assert_equals(cosmo['vel'].units, "1.00e5 cm a**1/2 s**-1")
    assert_almost_equal(pynbody.units.g/pynbody.units.g, cosmo['mass'].units/(1.989e43*pynbody.units.g/pynbody.units.h))

def test_physical_units():
    # Test to ensure the base units in a non-cosmological simulation have their correct value
    assert_equals(agora['pos'].units, "3.085678e21 cm")
    assert_equals(agora['vel'].units, "1.00e5 cm s**-1")
    assert_equals(agora['mass'].units, "1.989e42 g")

def test_hdf_ordering():
    # HDF files do not intrinsically specify the order in which the particle types occur
    # Because some operations may require stability, pynbody now imposes order by the particle type
    # number
    assert cosmo._family_slice[pynbody.family.gas] == slice(0, 262101, None)
    assert cosmo._family_slice[pynbody.family.dm] == slice(262101, 524245, None)
    assert cosmo._family_slice[pynbody.family.star] == slice(524245, 524300, None)
