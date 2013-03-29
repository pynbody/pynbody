"""

family
======

This module defines the Family class which represents
families of particles (e.g. dm, gas, star particles).
New Family objects are automatically registered so that
snapshots can use them in the normal syntax (snap.dm,
snap.star, etc).

In practice the easiest way to make use of the flexibility
this module provides is through adding more families of
particles in your config.ini.

"""

from . import config_parser


_registry = []


def family_names(with_aliases=False):
    """Returns a list of the names of all particle families.
    If with_aliases is True, include aliases in the list."""

    global _registry
    l = []
    for o in _registry:
        l.append(o.name)
        if with_aliases:
            for a in o.aliases:
                l.append(a)
    return l


def get_family(name, create=False):
    """Returns a family corresponding to the specified string.  If the
    family does not exist and create is False, raises ValueError. If
    the family does not exist and create is True, an appropriate
    object is instantiated, registered and returned."""

    if isinstance(name, Family):
        return name

    name = name.lower()
                      # or should it check and raise rather than just convert?
                      # Not sure.
    for n in _registry:
        if n.name == name or name in n.aliases:
            return n

    if create:
        return Family(name)
    else:
        raise ValueError(name +
                         " not a family")  # is ValueError the right thing here?


class Family(object):
    def __init__(self, name, aliases=[]):
        if name != name.lower():
            raise ValueError("Family names must be lower case")
        if name in family_names(with_aliases=True):
            raise ValueError("Family name "+name+" is not unique")
        for a in aliases:
            if a != a.lower():
                raise ValueError("Aliases must be lower case")

        self.name = name
        self.aliases = aliases
        _registry.append(self)

    def __repr__(self):
        return "<Family "+self.name+">"

    def __reduce__(self):
        return get_family, (self.name, True), {"aliases": self.aliases}

    def __iter__(self):
        # Provided so a single family can be treated as a list of families
        yield self

    def __str__(self):
        return self.name

    def __cmp__(self, other):
        return cmp(str(self), str(other))


# Instantiate the default families as specified
# by the configuration file

g = globals()
for f in config_parser.options('families'):
    aliases = config_parser.get('families', f)
    g[f] = Family(f, map(str.strip, aliases.split(",")))
