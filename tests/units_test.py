import numpy.testing as npt

import pynbody
from pynbody import units


def numacc(a, b, tol=1.e-9):
    print(a, b)
    assert abs(a - b) < a * tol


def test_units_conversion():

    numacc(units.kpc.in_units(units.Mpc), 0.001)
    numacc(units.Mpc.in_units(units.kpc), 1000)
    numacc(units.yr.in_units(units.Myr), 1.e-6)
    numacc(units.au.in_units(units.pc), 4.84813681e-6)


def test_units_manipulation():
    # Just do some manipulation and check it's happy
    (units.kpc * units.yr) ** (1, 3) / units.Myr
    (units.a * units.erg) ** 9


def test_units_substitution():
    numacc((units.a / units.h).in_units(units.Unit(""), a=22, h=2), 11)


def test_units_parser():
    testunit = units.Unit("kpc a s^-2/3 Myr^2/3")
    print("Unit as parsed: ", testunit)
    testunit /= units.kpc
    testunit /= units.a
    testunit /= units.s ** (-2, 3)
    testunit /= units.Myr ** (2, 3)
    print("This should be one: ", testunit)
    assert abs(testunit.dimensionless_constant() - 1) < 1.e-10


def test_units_copy():
    # These should succeed

    import copy
    copy.copy(units.Unit("Msol kpc^-1")).ratio("kg km^-1")
    copy.deepcopy(units.Unit("Msol kpc^-1")).ratio("kg km^-1")


def test_units_pickle():
    import pickle
    pick = lambda x: pickle.loads(pickle.dumps(x))

    assert pick(units.km) is units.km  # named
    assert pick(units.m) is units.m  # irreducible
    assert pick(units.Unit("km s^-1 Msol^-5")) == units.Unit("km s^-1 Msol^-5")

def test_units_rdiv():
    assert 4.0/pynbody.units.m_p == pynbody.units.Unit("4.0 m_p^-1")

def test_dimensionless_addition():
    dimless_unit = units.Unit("0.5")
    print(dimless_unit-0.25)
    npt.assert_allclose(float(dimless_unit-0.25),0.25)
    npt.assert_allclose(float(dimless_unit+0.25),0.75)

def test_units_addition():
    _2_kpc = units.Unit("2.0 kpc")
    _3_Mpc = units.Unit("3.0 Mpc")
    npt.assert_allclose((_2_kpc + _3_Mpc).in_units("kpc"),3002)
    npt.assert_allclose((_3_Mpc - _2_kpc).in_units("kpc"), 2998)
    npt.assert_allclose((_2_kpc + 2.0).in_units("kpc"), 4.0)
    npt.assert_allclose((_3_Mpc + 2.0).in_units("kpc"), 5000)

def test_units_zero_equality():
    assert units.Unit("0.0 km s^-1") == units.Unit("0.0 cm s^-1")

def test_units_equality():
    assert units.Unit("1.0 km s^-1") == units.Unit("1e5 cm s^-1")
