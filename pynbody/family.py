"""Definition of particle families for pynbody. Default types are dm,
star, and gas.

To add a new particle family, simply type

pynbody.family.Family("new_family_name")

This module might seem like over-kill (and could be removed if it
proves to be so) but exists to enable future intelligent behaviour
handling particles from different families in customizable ways."""



_registry = []

def family_names(with_aliases=False) :
    global _registry
    l = []
    for o in _registry :
        l.append(o.name)
        if with_aliases :
            for a in o.aliases : l.append(a)
    return l

def get_family(name, create=False) :
    if isinstance(name, Family) :
        return name

    name = name.lower() # or should it check and raise rather than just convert? Not sure.
    for n in _registry :
        if n.name==name or name in n.aliases :
            return n

    if create :
        return Family(name)
    else :
        raise ValueError, name+" not a family" # is ValueError the right thing here?

class Family(object) :
    def __init__(self, name, aliases=[]) :
        if name!=name.lower() :
            raise ValueError, "Family names must be lower case"
        if name in family_names(with_aliases=True) :
            raise ValueError, "Family name "+name+" is not unique"
        for a in aliases :
            if a!=a.lower() :
                raise ValueError, "Aliases must be lower case"

        self.name = name
        self.aliases = aliases
        _registry.append(self)

    def __repr__(self) :
        return "<Family "+self.name+">"

    def __deepcopy__(self, *args) :
        # AP: 2/08/2011
        # This is a difficult decision, but it's hard to see why getting a 
        # deep copy of a Family object would ever be useful, and it makes
        # deep-copying other objects with references to Family objects
        # very hard. Therefore, I'm trying this approach, which will admittedly
        # result in very surprising behaviour for anyone who actually wants
        # to deep-copy a Family for any reason.
        return self

dm = Family("dm",["d","dark"])
star = Family("star",["stars","st","s"])
gas = Family("gas",["g"])
neutrino = Family("neutrino", ["n","neu"])
