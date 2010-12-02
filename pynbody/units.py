"""
pynbody light-weight units module. A simple set of classes for tracking units.

Making units
============

You can make units in two ways. Either you can create a string, and
instantiate a Unit like this:

   units.Unit("Msol kpc**-3")
   units.Unit("2.1e12 m_p cm**-2/3")

Or you can do it within python, using the predefined Unit objects

   units.Msol * units.kpc**-3
   2.1e12 * units.m_p * units.cm**(-2,3)

In the last example, either a tuple describing a fraction or a
Fraction instance (from the standard python module fractions) is
acceptable.


Getting conversion ratios
=========================

To convert one unit to another, use the ``ratio`` member function:

   units.Msol.ratio(units.kg)  # ->  1.99e30
   (units.Msol / units.kpc**3).ratio(units.m_p/units.cm**3) # -> 4.04e-8

If the units cannot be converted, a UnitsException is raised:

   units.Msol.ratio(units.kpc)  # ->UnitsException

Specifying numerical values
===========================

Sometimes it's necessary to specify a numerical value in the course
of a conversion. For instance, consider a comoving distance; this
can be specified in pynbody units as follows:

   comoving_kpc = units.kpc * units.a

where units.a represents the scalefactor. We can attempt to convert
this to a physical distance as follows

   comoving_kpc.ratio(units.kpc)

but this fails, throwing a UnitsException. On the other hand we
can specify a value for the scalefactor when we request the conversion

   comoving_kpc.ratio(units.kpc, a=0.5)  # -> 0.5

and the conversion completes with the expected result. The units
module also defines units.h for the dimensionless hubble constant,
which can be used similarly. *By default, all conversions happening
within a specific simulation context should pass in values for
a and h as a matter of routine.*

Any IrreducibleUnit (see below) can have a value specified in this way,
but a and h are envisaged to be the most useful applications.

Defining new base units
=======================

The units module is fully extensible: you can define and name your own
units which then integrate with all the standard functions.

   litre = units.NamedUnit("litre",0.001*units.m**3)
   gallon = units.NamedUnit("gallon",0.004546*units.m**3)
   gallon.ratio(litre) # 4.546
   (units.pc**3).ratio(litre) # 2.94e52

You can even define completely new dimensions.

    V = units.IrreducibleUnit("V") # define a volt
    C = units.NamedUnit("C", units.J/V) # define a coulomb
    q = units.NamedUnit("q", 1.60217646e-19*C) # elementary charge
    F = units.NamedUnit("F", C/V) # Farad
    epsilon0 = 8.85418e-12 *F/units.m

    (q*V).ratio("eV") # -> 1.000...
    ((q**2)/(4*math.pi*epsilon0*units.m**2)).ratio("N") # -> 2.31e-28
    

"""


from fractions import Fraction

_registry = {}

class UnitsException(Exception) :
    pass

class UnitBase(object) :
    """Base class for units"""
    def __init__(self) :
	raise ValueError, "Cannot directly initialize abstract base class"
	pass
	
    def __pow__(self, p) :
	if isinstance(p, tuple) :
	    p = Fraction(p[0],p[1])
	return CompositeUnit(1, [self], [p]).simplify()

    def __div__(self, m) :
	if isinstance(m, UnitBase) :
	    return CompositeUnit(1, [self, m], [1, -1]).simplify()
	else :
	    return CompositeUnit(1.0/m, [self], [1]).simplify()

    def __rdiv__(self, m) :
	return CompositeUnit(m, [self], [-1]).simplify()

    def __mul__(self, m) :
	if isinstance(m, UnitBase) :
	    return CompositeUnit(1, [self, m], [1,1]).simplify()
	else :
	    return CompositeUnit(m, [self], [1]).simplify()
	
    def __rmul__(self, m) :
	return CompositeUnit(m, [self], [1]).simplify()
    
    def __repr__(self) :
	return 'Unit("'+str(self)+'")'

    def simplify(self) :	
	return self

    def is_dimensionless(self) :
	return False

    def ratio(self, other, **substitutions) :
	"""Get the conversion ratio between this Unit and another
	specified unit"""

	if isinstance(other, str) :
	    other = Unit(other)
	    
	try :
	    return (self/other).dimensionless_constant(**substitutions)
	except UnitsException :
	    raise UnitsException, "Not convertible"
	
    def irrep(self) :
	"""Return a unit equivalent to this one (may be identical) but
	expressed in terms of the currently defined IrreducibleUnit
	instances."""
	return self

    def _register_unit(self, st) :
	if st in _registry :
	    raise UnitsException, "Unit with this name already exists"
	if "**" in st or "^" in st or " " in st :
	    # will cause problems for simple string parser in Unit() factory
	    raise UnitsException, "Unit names cannot contain '**' or '^' or spaces"
	_registry[st]=self
	
class IrreducibleUnit(UnitBase) :
    def __init__(self, st) :
	self._st_rep = st
	self._register_unit(st)

	
    def __str__(self) :
	return self._st_rep

    def named_base_units(self) :
	return set([self])

    def irreducible_base_units(self) :
	return set([self])
    
class NamedUnit(UnitBase) :
    def __init__(self, st, represents) :
	self._st_rep = st
	self._represents = represents
	self._register_unit(st)
	
    def __str__(self) :
	return self._st_rep

    def irrep(self) :
	return self._represents.irrep()

class CompositeUnit(UnitBase) :
    def __init__(self, scale, bases, powers) :
	if scale==1. :
	    scale = 1

	self._scale = scale
	self._bases = bases
	self._powers = powers

    def __str__(self) :
	s=None
	if len(self._bases)==0 :
	    return "%.2e"%self._scale
	
	if self._scale!=1 :
	    s = "%.2e"%self._scale
	
	for b,p in zip(self._bases, self._powers) :
	    if s is not None :
		s+=" "+str(b)
	    else :
		s = str(b)
		
	    if p!=1 :
		s+="**"
		if isinstance(p,Fraction) :
		    s+=str(p)
		else :
		    s+=str(p)
	return s

    def _expand(self, expand_to_irrep=False) :
	"""Internal routine to expand any pointers to composite units
	into direct pointers to the base units. If expand_to_irrep is
	True, everything is expressed in irreducible units.
	A _gather will normally be necessary to sanitize the unit
	after an _expand."""
	
	trash = []

	
	
	for i,(b,p) in enumerate(zip(self._bases, self._powers)) :
	    if isinstance(b,NamedUnit) and expand_to_irrep :
		b = b._represents.irrep()		
		
	    if isinstance(b,CompositeUnit) :
		if expand_to_irrep :
		    b = b.irrep()
		    
		trash.append(i)
		self._scale*=b._scale**p
		for b_sub, p_sub in zip(b._bases, b._powers) :
		    self._bases.append(b_sub)
		    self._powers.append(p_sub*p)

	trash.sort()
	for offset,i in enumerate(trash) :
	    del self._bases[i-offset]
	    del self._powers[i-offset]


    def _gather(self) :
	"""Internal routine to gather together powers of the same base
	units, then order the base units by their power (descending)"""
	
	trash = []
	bases = list(set(self._bases))
	powers = [sum([p for bi,p in zip(self._bases, self._powers)
		       if bi is b]) \
		  for b in bases]

	bp = sorted(filter(lambda x : x[0]!=0,
			   zip(powers, bases)),reverse=True)

	if len(bp)!=0 :
	    self._powers, self._bases = map(list,zip(*bp))
	else :
	    self._powers, self._bases = [],[]




    def copy(self) :
	"""Create a copy which is 'shallow' in the sense that it
	references exactly the same underlying base units, but where
	the list of those units can be manipulated separately."""
	return CompositeUnit(self._scale, self._bases[:], self._powers[:])

    def __copy__(self) :
	"""For compatibility with python copy module"""
	return self.copy()
    
    def simplify(self) :
	self._expand()
	self._gather()
	return self

    def irrep(self) :
	x = self.copy()
	x._expand(True)
	x._gather()
	return x
    
    def is_dimensionless(self) :
	x = self.irrep()
	if len(x._powers)==0 :
	    return True

    def dimensionless_constant(self, **substitutions) :
	x = self.irrep()
	c = x._scale
	for xb, xp in zip(x._bases, x._powers) :
	    if str(xb) in substitutions :
		c*=substitutions[str(xb)]**xp
	    else :
		raise UnitsException, "Not dimensionless"
	
	return c


def Unit(s) :
    """Class factory for units. Given a string s, creates
    a Unit object.

    The string format is:
      [<scale>] [<unit_name>][**<rational_power>] [[<unit_name>] ... ]

    for example:
      "1.e30 kg"
      "kpc**2"
      "26.2 m s**-1"
    """

    x = s.split()
    try :
	scale = float(x[0])
	del x[0]
    except ValueError :
	scale = 1.0

    units = []
    powers = []


    for com in x :
	if "**" in com or "^" in com :
	    s = com.split("**" if "**" in com else "^")
	    try :
		u = _registry[s[0]]
	    except KeyError :
		raise ValueError, "Unknown unit "+s[0]
	    p = Fraction(s[1])
	    if p.denominator is 1 :
		p = p.numerator
	else :
	    u = _registry[com]
	    p = 1

	units.append(u)
	powers.append(p)
    
    return CompositeUnit(scale, units, powers)
    
m = IrreducibleUnit("m")
s = IrreducibleUnit("s")
kg = IrreducibleUnit("kg")
K = IrreducibleUnit("K")

# Cosmological quantities that can be substituted later
a = IrreducibleUnit("a")
h = IrreducibleUnit("h")


# Times
yr = NamedUnit("yr", 3.1556926e7*s)
kyr = NamedUnit("kyr", 1000*yr)
Myr = NamedUnit("Myr", 1000*kyr)
Gyr = NamedUnit("Gyr", 1000*Myr)

# Distances
cm = NamedUnit("cm", 0.01*m)
km =  NamedUnit("km", 1000*m)
au =  NamedUnit("au", 1.49598e11*m)
pc =  NamedUnit("pc", 3.08568025e16*m)
kpc = NamedUnit("kpc", 1000*pc)
Mpc = NamedUnit("Mpc", 1000*kpc)
Gpc = NamedUnit("Gpc", 1000*Mpc)


# Masses
Msol = NamedUnit("Msol", 1.98892e30*kg)
g = NamedUnit("g", 1.e-3*kg)
m_p = NamedUnit("m_p", 1.67262158e-27*kg)
m_e = NamedUnit("m_e", 9.10938188e-31*kg)

# Forces
N = NamedUnit("N", kg * m * s**-2)

# Energies
J = NamedUnit("J", N *m)
erg = NamedUnit("erg", 1.e-7 * J)
eV = NamedUnit("eV", 1.60217646e-19 * J)
keV = NamedUnit("keV", 1.e3*eV)
MeV = NamedUnit("MeV", 1.e3*keV)

# Pressures
Pa = NamedUnit("Pa", J/m)
dyn = NamedUnit("dyn", erg/cm)

# Redshift
one_plus_z = NamedUnit("(1+z)", 1/a)

# Helpful physical quantities

k = 1.3806503e-23 * J / K
c = 299792458 * m/s
G = 6.67300e-11 * m**3 * kg**-1 * s**-2
