import numpy as np
from . import units

class Filter(object) :
    def __init__(self) :
	self._descriptor = "filter"
	pass

    def where(self, sim) :
	return np.where(self(sim))
	    
    def __call__(self, sim) :
	return np.ones(len(sim),dtype=bool)

    def __and__(self, f2) :
	return And(self, f2)

    def __invert__(self) :
	return Not(self)

    def __or__(self, f2) :
	return Or(self, f2)

    def __repr__(self) :
	return "Filter()"

    


class And(Filter) :
    def __init__(self, f1, f2) :
	self._descriptor = f1._descriptor+"&"+f2._descriptor
	self.f1 = f1
	self.f2 = f2

    def __call__(self, sim) :
	return self.f1(sim)*self.f2(sim)

    def __repr__(self) : 
	return "("+repr(self.f1) + " & " + repr(self.f2)+")"

class Or(Filter) :
    def __init__(self, f1, f2) :
	self._descriptor = f1._descriptor+"|"+f2._descriptor
	self.f1 = f1
	self.f2 = f2

    def __call__(self, sim) :
	return self.f1(sim)+self.f2(sim)

    def __repr__(self) : 
	return "("+repr(self.f1) + " | " + repr(self.f2)+")"

class Not(Filter) :
    def __init__(self, f) :
	self._descriptor = "~"+f._descriptor
	self.f = f

    def __call__(self, sim) :
	x = self.f(sim)
	return ~x


    def __repr__(self) :
	return "~"+repr(self.f)


class Sphere(Filter) :
    def __init__(self, radius, cen=(0,0,0)) :
	self._descriptor = "sphere"
	self.cen = np.asarray(cen)
	if self.cen.shape!=(3,) :
	    raise ValueError, "Centre must be length 3 array"

	if isinstance(radius, str) :
	    radius = units.Unit(radius)
	    
	self.radius = radius

    def __call__(self, sim) :
	radius = self.radius
	if isinstance(radius, units.UnitBase) :
	    radius = radius.ratio(sim["pos"].units,
				  **sim["pos"].conversion_context())
	distance = ((sim["pos"]-self.cen)**2).sum(axis=1)
	return distance<radius**2

    def __repr__(self) :
	if isinstance(self.radius, units.UnitBase) :
	    
	    return "Sphere('%s', %s)"%(str(self.radius), repr(self.cen))
	else :
	    return "Sphere(%.2e, %s)"%(self.radius, repr(self.cen))

def Annulus(r1, r2, cen=(0,0,0)) :
    x = Sphere(max(r1,r2),cen) & ~Sphere(min(r1,r2),cen)
    x._descriptor = "annulus"
    return x
