"""
Defines an augmented dictionary class that manages properties of :class:`~pynbody.snapshot.simsnap.SimSnap` objects

By default, a :class:`SimDict` automatically converts between
redshift ('z') and scalefactor ('a') and implements default entries
for cosmological values listed in the [default-cosmology] section of
the `pynbody` configuration files.
"""

import warnings

from . import config

__all__ = ['SimDict']


class SimDict(dict):
    """A dictionary class for managing properties of a :class:`~pynbody.snapshot.simsnap.SimSnap` object.

    Above the standard dictionary methods, this class also provides automatic derivation of properties,
    such as converting between redshift and scalefactor, and default values for cosmological parameters.

    To add further properties use the SimDict.getter and SimDict.setter decorators.
    For instance, to add a property 'X_copy' which just reflects the value of the
    property 'X', you would use the following code:

    .. code-block:: python

     @SimDict.getter
     def X_copy(d):
         return d['X']

     @SimDict.setter
     def X_copy(d, value):
         d['X'] = value
    """
    _getters = {}
    _setters = {}

    @staticmethod
    def getter(f):
        """Define a getter function for all SimDicts

        See the class documentation for an example of usage.

        Parameters
        ----------

        f : function
            The function to be used as a getter. The function should take a single argument, the SimDict object, and
            return the value of the property. The name of the property is the name of the function.

        """
        SimDict._getters[f.__name__] = f

    @staticmethod
    def setter(f):
        """Define a setter function for all SimDicts

        See the class documentation for an example of usage.

        Parameters
        ----------

        f : function
            The function to be used as a setter. The function should take two arguments, the SimDict object and the
            value. The name of the property is the name of the function.
        """
        SimDict._setters[f.__name__] = f

    def __getitem__(self, k):
        if k in self:
            return dict.__getitem__(self, k)
        elif k in SimDict._getters:
            return SimDict._getters[k](self)
        else:
            raise KeyError(k)

    def __setitem__(self, k, v):
        if k in SimDict._setters:
            SimDict._setters[k](self, v)
        else:
            dict.__setitem__(self, k, v)


@SimDict.getter
def z(d):
    if d["a"] is None:
        return None
    try:
        return 1.0 / d["a"] - 1.0
    except ZeroDivisionError:
        return None


@SimDict.setter
def z(d, z):
    if z is None:
        d["a"] = None
    else:
        d["a"] = 1.0 / (1.0 + z)


def _default_fn(name, value):
    """Return a getter function for the default name/value pair"""
    def f(d):
        warnings.warn("Assuming default value for property '{}'={:.2e}".format(
            name, value), RuntimeWarning)
        d[name] = value
        return value
    f.__name__ = name

    return f

for k in config['default-cosmology']:
    SimDict.getter(_default_fn(k, config['default-cosmology'][k]))
