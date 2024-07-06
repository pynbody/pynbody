import copy
import gc

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
from pynbody.plot.stars import schmidtlaw


# Generate a fake little slab with a gas fraction of 0.5.  Size should be power of 8.
# The inner 10 kpc has stars that are 10 Myr old, and the outer 10 kpc has stars that are 90 Myr old.
def make_disc(size=8**5):
    size = int(size)
    s = pynbody.new(dm=0, g=size, s=size)
    s.properties['time'] = pynbody.units.Unit('100 Myr')
    prange = np.linspace(-1,1,int(np.round(size**(1/3))))
    s.s['mass'] = np.ones(size)
    s.g['mass'] = np.ones(size)
    s.g['x'], s.g['y'], s.g['z'] = map(np.ravel, np.meshgrid(prange,prange,0.1*prange)) # 1x1x0.1 slab
    s.s['x'], s.s['y'], s.s['z'] = map(np.ravel, np.meshgrid(prange,prange,0.1*prange)) # 1x1x0.1 slab
    s['pos'].units = 'kpc'
    s['mass'].units = 'Msol'

    # Set stellar ages
    s.s['tform'] = 90*np.ones(size)
    s.s['tform'][s.s['r'] > 0.5 ] /= 9
    s.s['tform'].units = 'Myr'

    # Scale disk
    s['x'] *= 20
    s['y'] *= 20
    s['z'] *= 3

    return s

# These should give identical surface densities if we count all stars.
def test_full_disc():
    s = make_disc()
    pg, ps = schmidtlaw(s, pretime='100 Myr')
    npt.assert_allclose(pg.in_units('Msol pc^-2'), (ps*pynbody.units.Unit('100 Myr')).in_units('Msol pc^-2'))

# These should give identical surface densities in the inner 10 kpc, and 0 for the stars in the outer 10 kpc
def test_truncated_disc():
    s = make_disc()
    pg, ps = schmidtlaw(s, pretime='50 Myr')
    npt.assert_allclose(pg.in_units('Msol pc^-2')[:4], (ps*pynbody.units.Unit('50 Myr')).in_units('Msol pc^-2')[:4])
    npt.assert_allclose(0*pg.in_units('Msol pc^-2')[5:], (ps*pynbody.units.Unit('50 Myr')).in_units('Msol pc^-2')[5:])
