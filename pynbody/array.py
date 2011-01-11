"""
Defines a shallow wrapper around numpy.ndarray for extra functionality like unit-tracking.

For most purposes, the differences between numpy.ndarray and
array.SimArray are not important. However, when units are specified
(by setting the ``units`` attribute), the behaviour is slightly
different. In particular,

* it becomes impossible to add or subtract arrays with incompatible
dimensions

   SimArray([1,2], "Mpc") + SimArray([1,2], "Msol")) # ->ValueError

* addition or subtraction causes auto-unit conversion. For example

   SimArray([1,2], "Mpc") + SimArray([1,2], "kpc") 

yields ``SimArray([1.001, 1.002], "Mpc")``

* Note that in this context the left value takes precedence in
specifying the return units, so that reversing the order of the
operation here would return results in kpc.

* If only one of the arrays specifies a Unit, no checking occurs and
the unit of the returned array is assumed to be the same as the
one specified input unit.

* Powers to single integer or rational powers will maintain unit
tracking.  Powers to float or other powers will not be able to do so.

   SimArray([1,2],"Msol Mpc**-3")**2 # -> SimArray([1, 4], 'Msol**2 Mpc**-6')
   SimArray([1,2],"Msol Mpc**-3")**(1,3) # -> SimArray([ 1.,1.26], 'Msol**1/3 Mpc**-1')
   # Syntax above mirrors syntax in units module, where a length-two tuple
   # can represent a rational number, in this case one third.

   SimArray([1.,2], "Msol Mpc**-3")**0.333 # -> SimArray([ 1.,1.26])
   # Lost track of units



Getting the array in specified units
====================================

Given an array, you can convert it in-place into units of your
own chosing:
  x = SimArray([1,2], "Msol")
  x.convert_units('kg)
  print x # ->  SimArray([  1.99e+30,   3.98e+30], 'kg')
  
Or you can leave the original array alone and get a *copy* in
different units, correctly converted:

   x = SimArray([1,2], "Msol")
   print x.in_units("kg") # -> SimArray([  1.99e+30,   3.98e+30], 'kg')
   print x # -> SimArray([1,2], "Msol")

If the SimArray was created by a SimSnap (which is most likely), it
has a pointer into the SimSnap's properties so that the cosmological
context is automatically fetched. For example, comoving -> physical
conversions are correctly achieved:

  f = pynbody.load("fname")
  f['pos']

  SimArray([[ 0.05419805, -0.0646539 , -0.15700017],
       [ 0.05169899, -0.06193341, -0.14475258],
       [ 0.05672406, -0.06384531, -0.15909944],
       ..., 
       [ 0.0723075 , -0.07650762, -0.07657281],
       [ 0.07166634, -0.07453796, -0.08020873],
       [ 0.07165282, -0.07468577, -0.08020587]], '2.86e+04 kpc a')

  f['pos'].convert_units('kpc')
  f['pos']

   SimArray([[ 1548.51403101, -1847.2525312 , -4485.71463308],
       [ 1477.1124212 , -1769.52421398, -4135.78377699],
       [ 1620.68592366, -1824.15000686, -4545.69387564],
       ..., 
       [ 2065.9264273 , -2185.92982874, -2187.79225915],
       [ 2047.60759667, -2129.6537339 , -2291.6758134 ],
       [ 2047.2214441 , -2133.87693163, -2291.59406997]], 'kpc')


Specifying rules for ufunc's
============================

In general, it's not possible to infer what the output units from a given
ufunc should be. While numpy built-in ufuncs should be handled OK, other
ufuncs will need their output units defined (otherwise a numpy.ndarray
will be returned instead of our custom type.)

To do this, decorate a function with SimArray.ufunc_rule(ufunc). The function
you define should take the same number of parameters as the ufunc. These will
be the input parameters of the ufunc. You should return the correct units for
the output, or raise units.UnitsException (in the latter case, the return
array will be made into a numpy.ndarray.)

For example, here is the code for the correct addition/subtraction
handler:


    @SimArray.ufunc_rule(np.add)
    @SimArray.ufunc_rule(np.subtract)
    def _consistent_units(a,b) :

	# This will be called whenever the standard numpy ufuncs np.add
	# or np.subtract are called with parameters a,b.
	
	# You should always be ready for the inputs to have no units.

	a_units = a.units if hasattr(a, 'units') else None
	b_units = b.units if hasattr(b, 'units') else None

	# Now do the logic. If we're adding incompatible units,
	# we want just to get a plain numpy array out. If we only
	# know the units of one of the arrays, we assume the output
	# is in those units.

	if a_units is not None and b_units is not None :
	    if a_units==b_units :
		return a_units
	    else :
		raise units.UnitsException("Incompatible units")

	elif a_units is not None :
	    return a_units
	else :
	    return b_units
    
"""

import numpy as np
import weakref
from . import units as units
_units = units
from .backcompat import property
from .backcompat import fractions


class SimArray(np.ndarray) :
    _ufunc_registry = {}
    
    def __new__(subtype, data, units=None, sim=None, **kwargs) :
	new = np.array(data, **kwargs).view(subtype)

        if hasattr(data, 'units') and hasattr(data, 'sim') and units is None and sim is None :
            units = data.units
            sim = data.sim
            
	if isinstance(units, str) :
	    units = _units.Unit(units)

	new._units = units
        new.sim = sim # will generate a weakref automatically

	new._name = None
	
	return new

    def __array_finalize__(self, obj) :
	if obj is None :
	    return
	elif obj is not self and hasattr(obj, 'units') :
	    self._units = obj.units
	    self._sim = obj._sim
	else :
	    self._units = None
	    self._sim = lambda : None


    def __array_wrap__(self, array, context=None) :
	if context is None :
	    return array
	
	try:
	    ufunc = context[0]
	    output_units = SimArray._ufunc_registry[ufunc](*context[1])
	    n_array = array.view(SimArray)
	    n_array.units = output_units
            n_array.sim = self.sim
	    return n_array
	except (KeyError, units.UnitsException) :
	    return array

    @staticmethod
    def ufunc_rule(for_ufunc) :
	def x(fn) :
	    SimArray._ufunc_registry[for_ufunc] = fn
	    return fn
	    
	return x
	    

    @property
    def units(self) :
	if hasattr(self.base, 'units') :
	    return self.base.units
	else :
	    return self._units

    @units.setter
    def units(self, u) :
	if isinstance(u, str):
	    u = units.Unit(u)
	    
	if hasattr(self.base, 'units') :
	    self.base.units = u
	else :
	    self._units = u


    @property
    def sim(self) :
	if self._sim() is not None :
	    return self._sim()

    @sim.setter
    def sim(self, s) :
        if s is not None :
            self._sim = weakref.ref(s)
        else :
            self._sim = lambda : None
            
    def conversion_context(self) :
	if self._sim() is not None :
	    return self._sim().conversion_context()
	else :
	    return {}
	
    def _generic_add(self, x, add_op=np.add) :
	if hasattr(x, 'units') and self.units is not None and x.units is not None :
	    # Check unit compatibility
	    try:
		cr = x.units.ratio(self.units,
				     **self.conversion_context())
	    except units.UnitsException :
		raise ValueError, "Incompatible physical dimensions"

	    
	    if cr==1.0 :
		r =  add_op(self, x)
		
	    else :
		b = np.multiply(x,cr)
		if hasattr(b,'units') :
		    b.units=None
		
		r = add_op(self, b)

	    return r
	
	else :
	    r = add_op(self, x)
	    return r

    
    def __add__(self,x) :
	return self._generic_add(x)

    def __sub__(self, x) :
	return self._generic_add(x, np.subtract)

    def __pow__(self, x) :
	numerical_x = x

	if isinstance(x, tuple) :
	    x = fractions.Fraction(x[0],x[1])
	    numerical_x = float(x)
	elif isinstance(x, fractions.Fraction) :
	    numerical_x = float(x)
	    
	# The following magic circumvents our normal unit-assignment
	# code which couldn't cope with the numerical version of x
	# in the case of fractions. All this is necessary to make the
	# magic tuple->fraction conversions work seamlessly.
	r = np.power(self.view(np.ndarray), numerical_x).view(SimArray)

	if self.units is not None and (
	    isinstance(x, fractions.Fraction) or
	    isinstance(x, int)) :
	    
	    r.units = self.units**x
	else :
	    r.units = None
	return r
    

    def __repr__(self) :
	x = np.ndarray.__repr__(self)
	if self.units is not None :
	    return x[:-1]+", '"+str(self.units)+"')"
	else :
	    return x

    def prod(self, axis=None, dtype=None, out=None) :
	x = np.ndarray.prod(self, axis, dtype, out)
	if hasattr(x, 'units') and axis is not None and self.units is not None :
	    x.units = self.units**self.shape[axis]
	return x

    def sum(self, *args, **kwargs) :
	x = np.ndarray.sum(self, *args, **kwargs)
	if hasattr(x, 'units') and self.units is not None :
	    x.units = self.units
	return x

    def set_units_like(self, new_unit) :
        """Set the units for this array by performing dimensional analysis
        on the supplied unit and referring to the units of the original
        file"""

        if self.sim is not None :
            self.units = self.sim.infer_original_units(new_unit)
        else :
            raise RuntimeError, "No link to SimSnap"
    
    def in_units(self, new_unit) :
	"""Return a copy of this array expressed relative to an alternative
	unit."""

	if self.units is not None :
	    r = self * self.units.ratio(new_unit,
					  **(self.conversion_context()))
	    r.units = new_unit
	    return r
	else :
	    raise ValueError, "Units of array unknown"

    def convert_units(self, new_unit) :
	"""Convert units of this array in-place. Note that if
	this is a sub-view, the entire base array will be converted."""

	if self.base is not None and hasattr(self.base, 'units') :
	    self.base.convert_units(new_unit)
	else :
	    self*=self.units.ratio(new_unit,
				     **(self.conversion_context()))
	    self.units = new_unit


    def recalculate(self) :
	"""If this array is derived, recalculate it using the original
	procedure.

	For instance, when one first accesses array f['r'] of a given
	snapshot f, the radius is calculated (see module derived). If
	one recentres, the radius is not automatically recalculated,
	so one needs to issue

	f['r'].recalculate()

	to recalculate the updated radius"""

	if self._sim() and self._name :
	    self._sim()._derive_array(self._name)
	else :
	    raise RuntimeError, "No link to SimSnap"
	
_u = SimArray.ufunc_rule

def _get_units_or_none(*a) :
    if len(a)==1 :
	return a[0].units
    else :
	r = []
	for x in a :
	    if hasattr(x,"units") :
		r.append(x.units)
	    else :
		r.append(None)

	return r

#
# Now we have the rules for unit outputs after numpy built-in ufuncs
#
# Note if these raise UnitsException, a standard numpy array is returned
# from the ufunc to indicate the units can't be calculated. That means
# ufuncs can do 'non-physical' things, but then return ndarrays instead
# of SimArrays.

@_u(np.sqrt)
def _sqrt_units(a) :
    if a.units is not None :
	return a.units**(1,2)
    else :
	return None
    
@_u(np.multiply)
def _mul_units(a,b) :
    a_units, b_units = _get_units_or_none(a,b)
    if a_units is not None and b_units is not None :
	return a_units * b_units
    elif a_units is not None :
	return a_units
    else :
	return b_units

@_u(np.divide)
def _div_units(a,b) :
    a_units, b_units = _get_units_or_none(a,b)
    if a_units is not None and b_units is not None :
	return a_units/b_units
    elif a_units is not None :
	return a_units
    else :
	return 1/b_units

@_u(np.add)
@_u(np.subtract)
def _consistent_units(a,b) :
    a_units, b_units = _get_units_or_none(a,b)
    if a_units is not None and b_units is not None :
	if a_units==b_units :
	    return a_units
	else :
	    raise units.UnitsException("Incompatible units")
    
    elif a_units is not None :
	return a_units
    else :
	return b_units

@_u(np.power)
def _pow_units(a,b) :
    a_units = _get_units_or_none(a)
    if a_units is not None :
	if not isinstance(b, int) and not isinstance(b, units.Fraction) :
	    raise units.UnitsException("Can't track units")
	return a_units**b
    else :
	return None



class IndexedSimArray(object) :
    def __init__(self, array, ptr) :
        self.base = array
        self._ptr = ptr

    def __array__(self) :
        return self.base[self._ptr]

    def __getitem__(self, item) :
        return self.base[self._ptr[item]]

    def __setitem__(self, item, to) :
        self.base[self._ptr[item]] = to

    def __getslice__(self, a, b) :
        return self.__getitem__(slice(a,b), to)
        
    def __setslice__(self, a, b, to) :
        self.__setitem__(slice(a,b), to)

    def __repr__(self) :
        return self.__array__().__repr__() # Could be optimized

    def __str__(self) :
        return self.__array__().__str__() # Could be optimized

    @property
    def units(self) :
        return self.base.units

    @units.setter
    def units(self, u) :
        self.base.units = u

    @property
    def sim(self) :
        return self.base.sim

    @sim.setter
    def sim(self, s) :
        self.base.sim = s

    def conversion_context(self) :
        return self.base.conversion_context()

    def set_units_like(self, new_unit) :
        self.base.set_units_like(new_unit)

    def in_units(self, new_unit) :
        return IndexedSimArray(self.base.in_units(new_unit), self._ptr)

    def convert_units(self, new_unit) :
        self.base.convert_units(new_unit)

    def recalculate(self) :
        self.base.recalculate()

    def prod(self) :
        return np.array(self).prod()


# The IndexedSimArray class is now supplemented by wrapping all the
# standard numpy methods with a generated function which extracts an
# array realization of the subview before calling the underlying
# method.

def _wrap_fn(w) :
     def q(s, *y,  **kw) :
         return w(SimArray(s), *y, **kw)
     q.__name__ = w.__name__
     return q
        
for x in np.ndarray.__dict__ :
    w = getattr(SimArray, x)
    if 'array' not in x and (not hasattr(IndexedSimArray, x)) and callable(w) :
        setattr(IndexedSimArray, x, _wrap_fn(w))
    
