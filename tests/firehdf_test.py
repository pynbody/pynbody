import gc

import numpy as np
import pytest

import pynbody
import pynbody.test_utils

@pytest.fixture
def snap():
    f = pynbody.load('testdata/FIRE/m11i_res7100/output/snapshot_600.hdf5')
    yield f
    del f
    gc.collect()
    
def test_standard_arrays(snap, multi_snap):
    """Check that the data loading works"""

    for s in [snap, multi_snap] :
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
        s.star['mass']

def test_mags(snap, multi_snap):
    """Check that magnitudes are not NaN"""
    bands = pynbody.analysis.luminosity._load_ssp_table(pynbody.analysis.luminosity._default_ssp_file[0]).bands
    for s in [snap, multi_snap] :
        for b in bands:
            mags = pynbody.analysis.luminosity.calc_mags(s.star, b)
            assert not np.any(np.isnan(mags))
