import gc

import numpy as np
import pytest

import pynbody
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("pkdgrav3")


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
    assert np.allclose(snap.dm['pos'].max(), 0.5, rtol=1e-3)
    assert np.allclose(snap.dm['pos'].min(), -0.5, rtol=1e-3)
    max_tform = snap.s['tform'].max()
    assert snap.properties['time'].in_units('Gyr') > max_tform.in_units('Gyr')


def test_units(snap):
    # From the parameter file:
    dMsolUnit = 2.747719e13
    dKpcUnit = 6e3
    assert np.allclose(snap.dm['pos'].units.in_units('a kpc'), dKpcUnit)
    assert np.allclose(snap.dm['mass'].units.in_units('Msol'), dMsolUnit)


def test_standard_arrays(snap) :
    """Check that the data loading works"""

    assert len(snap.families()) == 4
    assert len(snap.dm) == 17576
    assert len(snap.bh) == 5
    assert len(snap.gas) == 16834
    assert len(snap.star) == 737

    snap.dm['pos']
    snap.gas['pos']
    snap.star['pos']
    snap.bh['pos']
    snap['pos']
    snap['mass']
    # Load a second time to check that family_arrays still work
    snap.dm['pos']
    snap['vel']
    snap['iord']
    snap.gas['rho']
    snap.star['mass']
    snap.bh['pos']

    np.testing.assert_allclose(snap.dm['x'][::500],
                               [-0.44443448, -0.46084865, -0.46529744, -0.45955783, -0.45619511,
                                -0.14319834, 0.49613796, -0.44908009, -0.00969236, -0.44578614,
                                0.33156776, 0.45171689, -0.26648184, -0.31669561, -0.0462321 ,
                                -0.08962971, -0.47501354, -0.48073811, -0.47596492, -0.05170355,
                                0.00783169, 0.30343563, 0.1798989 , -0.06758307, -0.03830379,
                                0.07825877, 0.08893577, 0.21393484, 0.21699663, 0.19640893,
                                0.37019955, 0.44992341, 0.43195645, 0.21873396, 0.26066616,
                                0.36613503],
                               rtol=1e-5
                               )
