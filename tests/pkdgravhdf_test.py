import gc

import numpy as np
import pytest

import pynbody
import pynbody.test_utils

# @pytest.fixture(scope='module', autouse=True)
# def get_data():
#     pynbody.test_utils.ensure_test_data_available("pkdgrav3")


@pytest.fixture
def snap():
    f = pynbody.load('testdata/pkdgrav3/cosmoSF.00010')
    yield f
    del f
    gc.collect()


@pytest.fixture
def multi_snap():
    f = pynbody.load('testdata/pkdgrav3/cosmoSF.00007')
    yield f
    del f
    gc.collect()


def test_npart(multi_snap):
    print(multi_snap.properties)
    assert len(multi_snap._hdf_files) == multi_snap.properties['NumFilesPerSnapshot']

    assert len(multi_snap.g) == multi_snap.properties['NumPart_Total'][0]
    assert len(multi_snap.dm) == multi_snap.properties['NumPart_Total'][1]
    assert len(multi_snap.s) == multi_snap.properties['NumPart_Total'][4]
    assert len(multi_snap.bh) == multi_snap.properties['NumPart_Total'][5]


def test_cosmo(snap) :
    """ Check that the cosmological parameters are consistent """
    assert np.allclose(snap.properties['z'], 0.0)
    # This holds only in PKDGRAV3 default unit system!
    assert np.allclose(snap.properties['Omega0'], snap['mass'].sum())
    assert np.allclose(snap.properties['OmegaB'],
                       snap.s['mass'].sum()
                       + snap.g['mass'].sum()
                       + snap.bh['mass'].sum()
                       )


def test_standard_arrays(snap) :
    """Check that the data loading works"""

    for s in [snap] :
        s.dm['pos']
        s.gas['pos']
        s.star['pos']
        s.bh['pos']
        s['pos']
        s['mass']
        # Load a second time to check that family_arrays still work
        s.dm['pos']
        s['vel']
        s['iord']
        s.gas['rho']
        s.star['mass']
        s.bh['pos']
