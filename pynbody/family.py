"""Definition of particle families for pynbody. Default types are dm,
star, and gas.

To add a new particle family, simply type

pynbody.family.Family("new_family_name")

This module might seem like over-kill (and could be removed if it
proves to be so) but exists to enable future intelligent behaviour
handling particles from different families in customizable ways."""



_registry = []

def family_names() :
    global _registry
    l = []
    for o in _registry :
	l.append(o.name)
    return l

def get_family(name) :
    try:
	return _registry[name]
    except KeyError:
	return Family(name)
    
class Family(object) :
    def __init__(self, name) :
	if name!=name.lower() :
	    raise RuntimeError, "Family names must be lower case"
	if name in family_names() :
	    raise RuntimeError, "Family name "+name+" is not unique"
	self.name = name
	_registry.append(self)

    def __repr__(self) :
	return "<Family "+self.name+">"



dm = Family("dm")
star = Family("star")
gas = Family("gas")

