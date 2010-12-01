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

* Taking a view of a 


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



"""

import numpy as np
from . import units as units
_units = units
from .backcompat import property
import fractions


class SimArray(np.ndarray) :
    def __new__(subtype, data, units=None, sim=None, **kwargs) :
	new = np.array(data, **kwargs).view(subtype)
	
	if isinstance(units, str) :
	    units = _units.Unit(units)

	new._units = units

	if sim is not None :
	    new._sim_properties = sim.properties
	else :
	    new._sim_properties = None

	return new

    def __array_finalize__(self, obj) :
	if obj is None :
	    return
	elif obj is not self and isinstance(obj, SimArray) :
	    self._units = obj.units
	    self._sim_properties = obj._sim_properties
	else :
	    self._units = None
	    self._sim_properties = None

    @property
    def units(self) :
	if isinstance(self.base, SimArray) :
	    return self.base.units
	else :
	    return self._units

    @units.setter
    def units(self, u) :
	if isinstance(u, str):
	    u = units.Unit(u)
	    
	if isinstance(self.base, SimArray) :
	    self.base.units = u
	else :
	    self._units = u

    def conversion_context(self) :
	if self._sim_properties is not None :
	    d = {}
	    wanted = ['a','h']
	    for x in wanted :
		if self._sim_properties.has_key(x) :
		    d[x] = self._sim_properties[x]
	    return d
	
	else :
	    return {}
	
    def _generic_add(self, x, add_op=np.ndarray.__add__) :
	if isinstance(x, SimArray) and self.units is not None and x.units is not None :
	    # Check unit compatibility
	    try:
		cr = x.units.convert(self.units,
				     **self.conversion_context())
	    except units.UnitsException :
		raise ValueError, "Incompatible physical dimensions"

	    if cr==1.0 :
		r =  add_op(self, x)
		
	    else :
		r = add_op(self, np.ndarray.__mul__(x, cr))

	    r.units = self.units
	    return r
	
	else :
	    r = add_op(self, x)
	    if self.units is not None :
		r.units = self.units
	    elif isinstance(x, SimArray) and x.units is not None :
		r.units = x.units
	    else :
		r.units = None
	    return r
	
    def __add__(self,x) :
	return self._generic_add(x)

    def __sub__(self, x) :
	return self._generic_add(x, np.ndarray.__sub__)

    def __radd__(self, x) :
	r = np.ndarray.__radd__(self, x)
	r.units = self.units
	return r

    def __rsub__(self, x) :
	r = np.ndarray.__rsub__(self, x)
	r.units = self.units
	return r

    def __rdiv__(self, x) :
	r = np.ndarray.__rdiv__(self, x)
	r.units = self.units
	return r

    def __rmul__(self, x) :
	r = np.ndarray.__rmul__(self, x)
	r.units = self.units
	return r

    def __pow__(self, x) :
	numerical_x = x

	if isinstance(x, tuple) :
	    x = fractions.Fraction(x[0],x[1])
	    numerical_x = float(x)
	    
	r = np.ndarray.__pow__(self, numerical_x)

	if self.units is not None and (
	    isinstance(x, fractions.Fraction) or
	    isinstance(x, int)) :
	    
	    r.units = self.units**x
	else :
	    r.units = None
	return r
    
    def __mul__(self, x) :
	r = np.ndarray.__mul__(self, x)
	if isinstance(x, SimArray) and self.units is not None and x.units is not None :
	    r.units = self.units*x.units
	else :
	    r.units = self.units
	return r
    

    def __div__(self, x) :
	r = np.ndarray.__div__(self, x)
	if isinstance(x, SimArray) and self.units is not None and x.units is not None :
	    r.units = self.units/x.units
	else :
	    r.units = self.units
	    
	return r

    def __repr__(self) :
	x = np.ndarray.__repr__(self)
	if self.units is not None :
	    return x[:-1]+", '"+str(self.units)+"')"
	else :
	    return x
	
    def in_units(self, new_unit) :
	"""Return a copy of this array expressed relative to an alternative
	unit."""

	if self.units is not None :
	    r = self * self.units.convert(new_unit,
					  **(self.conversion_context()))
	    r.units = new_unit
	    return r
	else :
	    raise ValueError, "Units of array unknown"

    def convert_units(self, new_unit) :
	"""Convert units of this array in-place. Note that if
	this is a sub-view, the entire base array will be converted."""

	if self.base is not None and isinstance(self.base, SimArray) :
	    self.base.convert_units(new_unit)
	else :
	    self*=self.units.convert(new_unit,
				     **(self.conversion_context()))
	    self.units = new_unit
