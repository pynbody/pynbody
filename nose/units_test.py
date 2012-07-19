import pynbody
from pynbody import units

def numacc(a,b, tol=1.e-9) :
    print a,b
    assert abs(a-b)<a*tol
        
def test_units_conversion() :
    
        
    numacc(units.kpc.in_units(units.Mpc), 0.001)
    numacc(units.Mpc.in_units(units.kpc), 1000)
    numacc(units.yr.in_units(units.Myr),1.e-6)
    numacc(units.au.in_units(units.pc), 4.84813681e-6)


def test_units_manipulation() :
    # Just do some manipulation and check it's happy
    (units.kpc*units.yr)**(1,3)/units.Myr
    (units.a*units.erg)**9


def test_units_substitution() :
    numacc((units.a/units.h).in_units(units.Unit(""), a=22, h=2),11)
    
def test_units_parser() :
    testunit = units.Unit("kpc a s^-2/3 Myr^2/3")
    print "Unit as parsed: ",testunit
    testunit/=units.kpc
    testunit/=units.a
    testunit/=units.s**(-2,3)
    testunit/=units.Myr**(2,3)
    print "This should be one: ",testunit
    assert abs(testunit.dimensionless_constant()-1)<1.e-10
   
    
    
def test_units_copy() :
    # These should succeed

    import copy
    copy.copy(units.Unit("Msol kpc^-1")).ratio("kg km^-1")
    copy.deepcopy(units.Unit("Msol kpc^-1")).ratio("kg km^-1")

def test_units_pickle() :
    import pickle
    pick = lambda x : pickle.loads(pickle.dumps(x))
    
    assert pick(units.km) is units.km # named 
    assert pick(units.m) is units.m # irreducible
    assert pick(units.Unit("km s^-1 Msol^-5"))==units.Unit("km s^-1 Msol^-5")
    
