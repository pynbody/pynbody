"""

Defines the :class:`Family` class which represents families of particles
(e.g. dm, gas, star particles).

Most users will not need to use this module directly. New families of particles
can, however, be added interactively by creating new instances of the
:class:`Family` class. See the example in the class documentation.

Another way to add families, more permanently, is to use the configuration file.
See the :ref:`configuration` section for more information.
"""

import functools

from . import config_parser

_registry = []


def family_names(with_aliases=False):
    """Returns a list of the names of all particle families.

    Parameters
    ----------

    with_aliases : bool
        If True, the list will include all aliases for each family. Default is False.

    Returns
    -------

    list of str
        The list of family names.

    """

    global _registry
    l = []
    for o in _registry:
        l.append(o.name)
        if with_aliases:
            for a in o.aliases:
                l.append(a)
    return l


def get_family(name, create=False):
    """Returns a :class:`Family` object corresponding to the specified string.

    Parameters
    ----------

    name : str or :class:`Family`
        The name of the family to retrieve. If a :class:`Family` object is passed, it is returned unchanged.

    create : bool
        If True, a new family is created if the specified family does not exist. Default is False.
        If False, a ``ValueError`` is raised if the specified family does not exist.

    """

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
        raise ValueError(name + " is not a family")

@functools.total_ordering
class Family:
    """The class representing an abstract family of particles.

    Once a :class:`Family` is defined, it is automatically registered and
    can be used within :class:`~pynbody.snapshot.simsnap.SimSnap` objects. For example,

    >>> import pynbody
    >>> tachs = pynbody.family.Family("tachyon", aliases=["t", "tachyons"])
    >>> snap = pynbody.new(dm=100, tachyon=100)
    >>> snap.tachyon['pos'] # <-- get the tachyon positions
    >>> snap.t['pos'] # <-- use an alias to get the tachyon positions
    >>> snap[tachs]['pos'] # <-- another way to get the tachyon positions

    """

    def __init__(self, name, aliases=[]):
        """Create a new family with the specified name and aliases.

        Parameters
        ----------

        name : str
            The name of the family. Must be lower case and unique.

        aliases : list of str
            A list of aliases for the family. Must be lower case. These can be used
            interchangeably with the family name.
        """
        if name != name.lower():
            raise ValueError("Family names must be lower case")
        if name in family_names(with_aliases=True):
            raise ValueError("Family name " + name + " is not unique")
        for a in aliases:
            if a != a.lower():
                raise ValueError("Aliases must be lower case")

        self.name = name
        self.aliases = aliases
        _registry.append(self)

    def __repr__(self):
        return "<Family " + self.name + ">"

    def __reduce__(self):
        return get_family, (self.name, True), {"aliases": self.aliases}

    def __iter__(self):
        # Provided so a single family can be treated as a list of families
        yield self

    def __str__(self):
        return self.name

    def __cmp__(self, other):
        # for python 2.x
        return cmp(str(self), str(other))

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __hash__(self):
        return hash(str(self))

# Instantiate the default families as specified by the configuration file

g = globals()
for snap in config_parser.options('families'):
    aliases = config_parser.get('families', snap)
    g[snap] = Family(snap, list(map(str.strip, aliases.split(","))))
