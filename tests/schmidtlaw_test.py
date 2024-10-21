import numpy as np
import numpy.testing as npt
import pytest

import pynbody
from pynbody.plot.stars import schmidtlaw
from pynbody.test_utils.make_disc import make_disc


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
