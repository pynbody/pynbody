"""
Defines arrays for simulation snapshots.

There are two distinct types of array used in pynbody.

* :class:`SimArray` defines a subclass of ``numpy.ndarray`` for extra functionality like unit tracking
* :class:`IndexedSimArray` defines a distinct type of array, which does not subclass ``numpy.ndarray`` but behaves
  like one. It points to a parent array and a set of indices, and can be used to represent a subset of elements
  within an array.

For most purposes, the differences between ``numpy.ndarray`` and
:class:`SimArray` are not important. However, when units are specified
(by setting the :attr:`SimArray.units` attribute), the behaviour is slightly
different. In particular,

* it becomes impossible to add or subtract arrays with incompatible dimensions

  >>> SimArray([1,2], "Mpc") + SimArray([1,2], "Msol"))
  ValueError

* addition or subtraction causes auto-unit conversion. For example

  >>> SimArray([1,2], "Mpc") + SimArray([1,2], "kpc")
  SimArray([1.001, 1.002], "Mpc")

* Note that in this context the left value takes precedence in
  specifying the return units, so that reversing the order of the
  operation here would return results in kpc.

* If only one of the arrays specifies a Unit, no checking occurs and
  the unit of the returned array is assumed to be the same as the one
  specified input unit.

* Powers to single integer or rational powers will maintain unit
  tracking.  Powers to float or other powers will not be able to do
  so.

  >>> SimArray([1,2],"Msol Mpc**-3")**2
  SimArray([1, 4], 'Msol**2 Mpc**-6')
  >>> SimArray([1,2],"Msol Mpc**-3")**(1,3)
  SimArray([ 1.,1.26], 'Msol**1/3 Mpc**-1')

* The syntax is similar to that used by :class:`pynbody.units.Unit`, where a length-two tuple
  can represent a rational number, in this case one third. If a floating point number is used,
  the unit tracking is lost:

  >>> SimArray([1.,2], "Msol Mpc**-3")**0.333
  SimArray([ 1.,1.26])  # Lost track of units



Getting the array in specified units
------------------------------------

Given an array, you can convert it in-place into units of your
own chosing:

>>> x = SimArray([1,2], "Msol")
>>> x.convert_units('kg')
>>> print(x)
SimArray([  1.99e+30,   3.98e+30], 'kg')

Or you can leave the original array alone and get a *copy* in
different units, correctly converted:

>>> x = SimArray([1,2], "Msol")
>>> print(x.in_units("kg"))
SimArray([  1.99e+30,   3.98e+30], 'kg')
>>> print(x)
SimArray([1,2], "Msol")

If the :class:`SimArray` (or :class:`IndexedSimArray`) was created by a SimSnap (which is most likely), it
has a pointer into the SimSnap's properties so that the cosmological
context is automatically fetched. For example, comoving -> physical
conversions are correctly achieved:

>>> f = pynbody.load("fname")
>>> f['pos']
SimArray([[ 0.05419805, -0.0646539 , -0.15700017],
         [ 0.05169899, -0.06193341, -0.14475258],
         [ 0.05672406, -0.06384531, -0.15909944],
         ...,
         [ 0.0723075 , -0.07650762, -0.07657281],
         [ 0.07166634, -0.07453796, -0.08020873],
         [ 0.07165282, -0.07468577, -0.08020587]], '2.86e+04 kpc a')

>>> f['pos'].convert_units('kpc')
>>> f['pos']
SimArray([[ 1548.51403101, -1847.2525312 , -4485.71463308],
         [ 1477.1124212 , -1769.52421398, -4135.78377699],
         [ 1620.68592366, -1824.15000686, -4545.69387564],
         ...,
         [ 2065.9264273 , -2185.92982874, -2187.79225915],
         [ 2047.60759667, -2129.6537339 , -2291.6758134 ],
         [ 2047.2214441 , -2133.87693163, -2291.59406997]], 'kpc')


Specifying rules for ufuncs
----------------------------

In general, it's not possible to infer what the output units from a given
ufunc should be. While numpy built-in ufuncs should be handled OK, other
ufuncs will need their output units defined (otherwise a `numpy.ndarray`
will be returned instead of our custom type.)

To do this, decorate a function with :meth:`SimArray.ufunc_rule`. The function
you define should take the same number of parameters as the ufunc. These will
be the input parameters of the ufunc. You should return the correct units for
the output, or raise units.UnitsException (in the latter case, the return
array will be made into a numpy.ndarray.)

For example, here is the code for the standard numpy sqrt. (You don't need this because it's already
built in, but it shows you how to do it.)

.. code-block:: python

    @SimArray.ufunc_rule(np.sqrt)
    def sqrt_units(a):
        if a.units is not None:
            return a.units ** (1, 2)
        else:
            return None

Shared memory arrays
--------------------

The array package also provides mechanisms for creating arrays that back onto shared memory. These can be easily
created using :func:`array_factory` with ``shared=True``. For implementation details see :mod:`pynbody.array.shared`.

"""

from __future__ import annotations

import fractions
import functools
import logging
import weakref
from typing import TYPE_CHECKING

import numpy as np

from .. import units

if TYPE_CHECKING:
    from .. import family, snapshot

_units = units

logger = logging.getLogger('pynbody.array')


def _copy_docstring_from(source_class):
    def wrapper(target_class):
        for attr in dir(target_class):
            if attr.startswith('_'):
                continue
            if hasattr(source_class, attr) and hasattr(getattr(source_class, attr), '__doc__'):
                docstring = getattr(source_class, attr).__doc__
                if docstring is not None:
                    setattr(getattr(target_class, attr), '__doc__', docstring)


        return target_class

    return wrapper


class SimArray(np.ndarray):
    """A shallow wrapper around numpy.ndarray for extra functionality like unit-tracking.

    .. note::

       This class inherits from ``numpy.ndarray``. Most of the methods are inherited from the parent class.
       The documentation is also inherited; in particular note that version information therefore refers to
       the version of numpy, not pynbody.

       The methods specific to this child class (and therefore documented for pynbody specifically) are:

       * :meth:`conversion_context`
       * :meth:`convert_units`
       * :meth:`in_units`
       * :meth:`ufunc_rule`

       The attributes specific to this class are:

       * :attr:`ancestor`
       * :attr:`derived`
       * :attr:`family`
       * :attr:`name`
       * :attr:`sim`
       * :attr:`units`

    """

    _ufunc_registry = {}

    __slots__ = ['_units', '_sim', '_name', '_family']

    @property
    def ancestor(self):
        """Provides the basemost SimArray that an IndexedSimArray is based on."""
        return self

    @property
    def derived(self):
        """True if this array has been derived by pynbody; False otherwise.

        For more information on derived arrays, see :ref:`derived_arrays`."""
        if self.sim and self.name:
            return self.sim.is_derived_array(self.name, getattr(self, 'family', None))
        else:
            return False

    @derived.setter
    def derived(self, value):
        if value:
            raise ValueError("Can only unlink an array. Delete an array to force a rederivation if this is the intended effect.")
        if self.derived:
            self.sim.unlink_array(self.name)

    def __reduce__(self):
        T = np.ndarray.__reduce__(self)
        T = (
            T[0], T[1], (self.units, T[2][0], T[2][1], T[2][2], T[2][3], T[2][4]))
        return T

    def __setstate__(self, args):
        self._units = args[0]
        self.sim = None
        self._name = None
        np.ndarray.__setstate__(self, args[1:])

    def __init__(self, data, units = None, sim = None, **kwargs):
        """Initialise a SimArray with the specified units and simulation context.

        Arguments
        ---------

        data : array-like
            The data to be stored in the array. If a :class:`SimArray` is passed in, it is copied and the *units* and
            *sim* arguments below are ignored.

        units : str or :class:`pynbody.units.UnitBase`
            The units of the data. If None, the data is assumed to be dimensionless. This argument is not used if
            *data* is a :class:`SimArray`.

        sim : :class:`pynbody.snapshot.simsnap.SimSnap`
            The simulation snapshot that the array belongs to. This is used to obtain context when performing
            unit conversions.  If None, the array is not associated with any simulation. This argument is not used if
            *data* is a :class:`SimArray`.

        **kwargs : dict
            Other arguments are passed to ``numpy.ndarray.__init__``.

        """

        # The actual initialisation is done by __new__ (it's actually unclear to me now why that should be the case,
        # but it's been like this for a long time and so changing it would need to be handled with care.)
        pass

    def __new__(cls, data, units=None, sim=None, **kwargs):
        new = np.array(data, **kwargs).view(cls)
        if hasattr(data, 'units') and hasattr(data, 'sim') and units is None and sim is None:
            units = data.units
            sim = data.sim

        if hasattr(data, 'family'):
            new.family = data.family

        if isinstance(units, str):
            units = _units.Unit(units)

        new._units = units

        # Always associate a SimArray with the top-level snapshot.
        # Otherwise we run into problems with how the reference should
        # behave: we don't want to lose the link to the simulation by
        # storing a weakref to a SubSnap that might be deconstructed,
        # but we also wouldn't want to store a strong ref to a SubSnap
        # since that would keep the entire simulation alive even if
        # deleted.
        #
        # So, set the sim attribute to the top-level snapshot and use
        # the normal weak-reference system.

        if sim is not None:
            new.sim = sim.ancestor
            # will generate a weakref automatically

        new._name = None

        return new

    def __array_finalize__(self, obj):
        if obj is None:
            return
        elif obj is not self and hasattr(obj, 'units'):
            self._units = obj.units
            self._sim = obj._sim
            self._name = obj._name
            if hasattr(obj, 'family'):
                self.family = obj.family
        else:
            self._units = None
            self._sim = lambda: None
            self._name = None

    @classmethod
    def _simarray_to_plain_ndarray(cls, inputs):
        """Converts any SimArrays to plain np.ndarray, for use when a ufunc or other numpy func is being called"""
        if isinstance(inputs, dict):
            return {k: cls._simarray_to_plain_ndarray(i) for k, i in inputs.items()}
        elif isinstance(inputs, list) or isinstance(inputs, tuple):
            return [cls._simarray_to_plain_ndarray(i) for i in inputs]
        elif isinstance(inputs, SimArray):
            return inputs.view(np.ndarray)
        else:
            return inputs

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out_sim = None
        for input in inputs:
            if hasattr(input, 'sim'):
                out_sim = input.sim
                break

        units_func = SimArray._ufunc_registry.get(ufunc, None)
        if units_func is None:
            out_units = units.NoUnit()
            inputs = self._simarray_to_plain_ndarray(inputs)
        else:
            result = units_func(*inputs)
            if isinstance(result, tuple):
                out_units, inputs = result
            else:
                out_units = result
                inputs = self._simarray_to_plain_ndarray(inputs)

        # convert inputs to vanilla numpy arrays for calling the underlying ufunc


        if len(inputs)==2:
            # if one of the inputs is a unit, we replace it with 1 in the ufunc call and let the
            # unit handling do the rest
            if isinstance(inputs[0], units.UnitBase):
                if not isinstance(inputs[1], np.ndarray):
                    return NotImplemented
                inputs[0] = np.array(1, dtype=inputs[1].dtype)
            elif isinstance(inputs[1], units.UnitBase):
                if not isinstance(inputs[0], np.ndarray):
                    return NotImplemented
                inputs[1] = np.array(1, dtype=inputs[0].dtype)

        out = kwargs.get('out', None)
        if out is not None:
            if len(out)!=1:
                return NotImplemented
            else:
                kwargs['out'] = (out[0].view(np.ndarray), )

        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

        if np.isscalar(result):
            # If we want to return units, numpy scalars aren't a good return type.
            # It's not ideal to directly contravene numpy convention here, but the best we can do.
            # Other options are:
            #  1. Return the scalar as numpy has returned. But then weird side-effects happen like
            #     (a+b) has no units, even if a and b are both scalar arrays with units
            #  2. Return the scalar as a UnitBase. But then some ufuncs like std() fail because they expect an
            #     object with a numpy-like interface [specifically, units have no sqrt method] and anyway it seems
            #     counterintuitive that a scalar array doesn't get mapped onto a scalar array. (This seems a weird
            #     design choice in numpy - not entirely clear why it happens.)
            #  3. Do what we're doing and wrap things up into a scalar array but ONLY if we have units to return.
            #     But this would probably just be even more baffling to a user.
            result = np.array(result)

        if out is not None:
            # checked above that there is only one output
            if isinstance(out[0], SimArray):
                out[0].units = out_units
            return out[0]
        elif isinstance(result, np.ndarray):
            # reinstate units in output
            result = result.view(SimArray)
            result.units = out_units
            result.sim = out_sim
            return result
        else:
            return result

    def __array_function__(self, func, types, args, kwargs):
        # called for non-ufuncs e.g. for numpy.linlag.norm
        ok_types = (SimArray, np.ndarray)
        if not all(issubclass(t, ok_types) for t in types):
            return NotImplemented

        args_processed = self._simarray_to_plain_ndarray(args)
        kwargs_processed = self._simarray_to_plain_ndarray(kwargs)

        result = func(*args_processed, **kwargs_processed)

        if func in SimArray._ufunc_registry:
            result = result.view(SimArray)
            if isinstance(result, SimArray): # may not be true if the result is a scalar
                sim = None
                for arg in args:
                    if isinstance(arg, SimArray):
                        sim = arg.sim
                        break
                result.units = SimArray._ufunc_registry[func](*args, **kwargs)
                result.sim = sim

        return result

    @staticmethod
    def ufunc_rule(*for_ufuncs):
        """Function decorator to mark a function as providing the output units for a given ufunc (or ufuncs).

        The function should take the same number of arguments as the ufunc, and return the output units. The function
        may optionally also return a tuple of the output units and the modified input arguments, in which case the
        input arguments will be passed to the ufunc in place of the original arguments.

        Note that while this is predominantly used for ufuncs, in fact it can also be used for things which are
        not strictly ufuncs, like numpy.linalg.norm

        """
        def x(fn):
            for for_ufunc in for_ufuncs:
                SimArray._ufunc_registry[for_ufunc] = fn
            return fn

        return x

    @property
    def units(self) -> units.UnitBase:
        """The units of the array, if known; otherwise :ref:`units.NoUnit`."""
        if hasattr(self.base, 'units'):
            return self.base.units
        else:
            if self._units is None:
                return _units.no_unit
            else:
                return self._units

    @units.setter
    def units(self, u):
        if not isinstance(u, units.UnitBase) and u is not None:
            u = units.Unit(u)

        if hasattr(self.base, 'units'):
            self.base.units = u
        else:
            if hasattr(u, "_no_unit"):
                self._units = None
            else:
                self._units = u

    @property
    def name(self) -> str | None:
        """The name of the array in the simulation snapshot, if known."""
        if hasattr(self.base, 'name'):
            return self.base.name
        return self._name

    @property
    def sim(self) -> snapshot.SimSnap | None:
        """The simulation snapshot that the array belongs to, if known."""
        if hasattr(self.base, 'sim'):
            base_sim = self.base.sim
        else:
            base_sim = self._sim()

        if self.family is not None and base_sim is not None:
            return base_sim[self.family]
        else:
            return base_sim


    @sim.setter
    def sim(self, s):
        if hasattr(self.base, 'sim'):
            self.base.sim = s
        else:
            if s is not None:
                self._sim = weakref.ref(s)
            else:
                self._sim = lambda: None

    @property
    def family(self) -> family.Family | None:
        """Returns the pynbody family that the array belongs to, if any.

        If ``family`` isn't None, an array ``a`` belongs to ``a.sim[family]`` rather than ``a.sim``.

        This doesn't necessarily mean that it is a family-level array, however, since it could be a slice into
        a simulation array.
        """
        try:
            return self._family
        except AttributeError:
            return None

    @family.setter
    def family(self, fam):
        self._family = fam

    def conversion_context(self):
        """Return a dictionary of contextual information that may be required for unit conversion.

        This is typically cosmological scalefactor and Hubble parameter."""
        if self.sim is not None:
            return self.sim.conversion_context()
        else:
            return {}

    def _generic_add(self, x, add_op=np.add):
        if hasattr(x, 'units') and not hasattr(self.units, "_no_unit") and not hasattr(x.units, "_no_unit"):
            # Check unit compatibility

            try:
                context = x.conversion_context()
            except AttributeError:
                context = {}

            # Our own contextual information overrides x's
            context.update(self.conversion_context())

            try:
                cr = x.units.ratio(self.units,
                                   **context)
            except units.UnitsException:
                raise ValueError("Incompatible physical dimensions {!r} and {!r}, context {!r}".format(
                    str(self.units), str(x.units), str(self.conversion_context())))

            if cr == 1.0:
                r = add_op(self, x)

            else:
                b = np.multiply(x, cr)
                if hasattr(b, 'units'):
                    b.units = None

                if not np.can_cast(b.dtype,self.dtype):
                    b = np.asarray(b, dtype=x.dtype)


                r = add_op(self, b)

            return r

        elif units.is_unit(x):
            x = x.in_units(self.units, **self.conversion_context())
            r = add_op(self, x)
            return r
        else:
            r = add_op(self, x)
            return r

    def __repr__(self):
        x = np.ndarray.__repr__(self)
        if not hasattr(self.units, "_no_unit"):
            return x[:-1] + ", '" + str(self.units) + "')"
        else:
            return x

    def __setitem__(self, item, to):
        if hasattr(to, "in_units") and not hasattr(self.units, "_no_unit") and not hasattr(to.units, "_no_unit"):
            np.ndarray.__setitem__(self, item, to.in_units(self.units))
        else:
            np.ndarray.__setitem__(self, item, to)

    def __setslice__(self, a, b, to):
        self.__setitem__(slice(a, b), to)

    def abs(self, *args, **kwargs):
        """Return the absolute value of the array."""
        x = np.abs(self, *args, **kwargs)
        if hasattr(x, 'units') and self.units is not None:
            x.units = self.units
        if hasattr(x, 'sim') and self.sim is not None:
            x.sim = self.sim
        return x

    def cumsum(self, axis=None, dtype=None, out=None):
        x = np.ndarray.cumsum(self, axis, dtype, out)
        if hasattr(x, 'units') and self.units is not None:
            x.units = self.units
        if hasattr(x, 'sim') and self.sim is not None:
            x.sim = self.sim
        return x

    def prod(self, axis=None, dtype=None, out=None):
        x = np.ndarray.prod(self, axis, dtype, out)
        if hasattr(x, 'units') and axis is not None and self.units is not None:
            x.units = self.units ** self.shape[axis]
        if hasattr(x, 'units') and axis is None and self.units is not None:
            x.units = self.units
        if hasattr(x, 'sim') and self.sim is not None:
            x.sim = self.sim
        return x

    def sum(self, *args, **kwargs):
        x = np.ndarray.sum(self, *args, **kwargs)
        if hasattr(x, 'units') and self.units is not None:
            x.units = self.units
        if hasattr(x, 'sim') and self.sim is not None:
            x.sim = self.sim
        return x

    def mean(self, *args, **kwargs):
        x = np.ndarray.mean(self, *args, **kwargs)
        if hasattr(x, 'units') and self.units is not None:
            x.units = self.units
        if hasattr(x, 'sim') and self.sim is not None:
            x.sim = self.sim
        return x

    def mean_by_mass(self, *args, **kwargs):
        """Removed in pynbody 2.0. Use :func:`pynbody.snapshot.simsnap.SimSnap.mean_by_mass` instead."""
        raise RuntimeError("SimArray.mean_by_mass has been removed. Use SimSnap.mean_by_mass instead.")

    def max(self, *args, **kwargs):
        x = np.ndarray.max(self, *args, **kwargs)
        if hasattr(x, 'units') and self.units is not None:
            x.units = self.units
        if hasattr(x, 'sim') and self.sim is not None:
            x.sim = self.sim
        return x

    def min(self, *args, **kwargs):
        x = np.ndarray.min(self, *args, **kwargs)
        if hasattr(x, 'units') and self.units is not None:
            x.units = self.units
        if hasattr(x, 'sim') and self.sim is not None:
            x.sim = self.sim
        return x

    def ptp(self, *args, **kwargs):
        x = np.ndarray.ptp(self, *args, **kwargs)
        if hasattr(x, 'units') and self.units is not None:
            x.units = self.units
        if hasattr(x, 'sim') and self.sim is not None:
            x.sim = self.sim
        return x

    def std(self, *args, **kwargs):
        x = np.ndarray.std(self, *args, **kwargs)
        if hasattr(x, 'units') and self.units is not None:
            x.units = self.units
        if hasattr(x, 'sim') and self.sim is not None:
            x.sim = self.sim
        return x

    def var(self, *args, **kwargs):
        x = np.ndarray.var(self, *args, **kwargs)
        if hasattr(x, 'units') and self.units is not None:
            x.units = self.units ** 2
        if hasattr(x, 'sim') and self.sim is not None:
            x.sim = self.sim
        return x

    def set_units_like(self, new_unit):
        """Set the units of this array using a guess the ``sim``'s units for a given dimensionality.

        For example, if ``sim`` has units of ``Msol`` and ``kpc``, and ``new_unit`` is ``kg m^-3``, the
        units of this array will be set to ``Msol kpc^-3``.

        The underlying code uses dimensional analysis; of course, simulation codes are free to use inconsistent
        units if they like, so in general this routine cannot be guaranteed to infer the correct units. Human
        cross-checks are strongly advised.

        Note that this does not convert the array to the new units, it only sets the units attribute. To convert,
        use :func:`in_units`, :func:`convert_units` or :func:`in_original_units`.
        """

        if self.sim is not None:
            self.units = self.sim.infer_original_units(new_unit)
        else:
            raise RuntimeError("No link to SimSnap")

    def set_default_units(self, quiet=False):
        """Set the units for this array by guessing the ``sim``'s unit scheme and known dimensionality information.

        For example, if ``sim`` has units of ``Msol`` and ``kpc`` and this array is the ``rho`` array, the
        units of this array will be set to ``Msol kpc^-3``.

        The underlying code uses dimensional analysis; of course, simulation codes are free to use inconsistent
        units if they like, so in general this routine cannot be guaranteed to infer the correct units. Human
        cross-checks are strongly advised.

        Note that this does not convert the array to the new units, it only sets the units attribute. To convert,
        use :func:`in_units`, :func:`convert_units` or :func:`in_original_units`.
        """

        if self.sim is not None:
            try:
                self.units = self.sim._default_units_for(self.name)
            except (KeyError, units.UnitsException):
                if not quiet:
                    raise
        else:
            raise RuntimeError("No link to SimSnap")

    def in_original_units(self):
        """Return a copy of this array expressed in the file's internal unit scheme.

        For example, if ``sim`` has units of ``Msol`` and ``kpc`` and this array is the ``rho`` array, a copy of the
        array in units of ``Msol kpc^-3`` will be returned, even if the current units are something else like
        ``kg m^-3``.

        The underlying code uses dimensional analysis; of course, simulation codes are free to use inconsistent
        units if they like, so in general this routine cannot be guaranteed to infer the correct units. Human
        cross-checks are strongly advised.
        """

        return self.in_units(self.sim.infer_original_units(self.units))

    def in_units(self, new_unit, **context_overrides):
        """Return a copy of this array, expressed relative to an alternative unit ``new_unit``.

        Additional keyword arguments are interpreted as context overrides for the unit conversion. For example, if the
        array is a comoving distance and you want to convert it to a physical distance, you might call

        >>> x.in_units('kpc', a=0.1)

        to get the result assuming a scalefactor 0.1. If no context overrides are given, the context of the underlying
        ``sim`` is adopted.
        """

        context = self.conversion_context()
        context.update(context_overrides)

        if self.units is not None:
            r = self * self.units.ratio(new_unit,
                                        **context)
            r.units = new_unit
            return r
        else:
            raise ValueError("Units of array unknown")

    def convert_units(self, new_unit):
        """Convert units of this array in-place. If this is a sub-view, the entire base array will be converted."""

        if self.base is not None and hasattr(self.base, 'units'):
            self.base.convert_units(new_unit)
        else:
            ratio = self.units.ratio(new_unit,
                                     **(self.conversion_context()))
            logger.debug("Converting %s units from %s to %s; ratio = %.3e" %
                         (self.name, self.units, new_unit, ratio))
            self *= ratio
            self.units = new_unit

    def write(self, **kwargs):
        """
        Write this array to disk according to the standard method
        associated with its base file. This is equivalent to calling

        >>> sim.gas.write_array('array')

        in the case of writing out the array 'array' for the gas
        particle family.  See the description of
        :func:`pynbody.snapshot.SimSnap.write_array` for options.
        """

        if self.sim and self.name:
            self.sim.write_array(self.name, fam=self.family, **kwargs)
        else:
            raise RuntimeError("No link to SimSnap")


# Now add dirty bit setters to all the operations which are known
# to modify the numpy array

def _dirty_fn(w):
    def q(a, *y, **kw):
        if a.sim is not None and a.name is not None:
            a.sim._dirty(a.name)

        if kw != {}:
            return w(a, *y, **kw)
        else:
            return w(a, *y)

    q.__name__ = w.__name__
    return q

_dirty_fns = ['__setitem__', '__setslice__',
              '__irshift__',
              '__imod__',
              '__iand__',
              '__ifloordiv__',
              '__ilshift__',
              '__imul__',
              '__ior__',
              '__ixor__',
              '__isub__',
              '__invert__',
              '__iadd__',
              '__itruediv__',
              '__idiv__',
              '__ipow__']

for x in _dirty_fns:
    if hasattr(SimArray, x): # numpy 2 doesn't have __idiv__
        setattr(SimArray, x, _dirty_fn(getattr(SimArray, x)))


def _get_units_or_none(*a):
    r = []
    for x in a:
        if isinstance(x, units.UnitBase):
            r.append(x)
        elif hasattr(x, "units"):
            r.append(x.units)
        else:
            r.append(None)

    return r

#
# Now we have the rules for unit outputs after numpy built-in ufuncs
#
# Note if these raise UnitsException, a standard numpy array is returned
# from the ufunc to indicate the units can't be calculated. That means
# ufuncs can do 'non-physical' things, but then return ndarrays instead
# of SimArrays.


@SimArray.ufunc_rule(np.sqrt)
def _sqrt_units(a):
    if a.units is not None:
        return a.units ** (1, 2)
    else:
        return None


@SimArray.ufunc_rule(np.multiply)
def _mul_units(a, b, catch=None):
    a_units, b_units = _get_units_or_none(a, b)
    if a_units is not None and b_units is not None:
        return a_units * b_units
    elif a_units is not None:
        return a_units
    else:
        return b_units


@SimArray.ufunc_rule(np.divide, np.true_divide)
def _div_units(a, b, catch=True):
    a_units, b_units = _get_units_or_none(a, b)
    if a_units is not None and b_units is not None:
        return a_units / b_units
    elif a_units is not None:
        return a_units
    else:
        return 1 / b_units


@SimArray.ufunc_rule(np.add, np.subtract, np.negative, np.squeeze)
def _consistent_units(*arrays, catch=None):
    array_units = _get_units_or_none(*arrays)
    numpy_arrays = []
    for a in arrays:
        if isinstance(a, np.ndarray):
            numpy_arrays.append(a.view(np.ndarray))
        elif isinstance(a, units.UnitBase):
            numpy_arrays.append(1)
        else:
            numpy_arrays.append(a)

    if len(array_units)==0:
        return None
    if len(array_units)==1:
        return array_units[0]
    else:
        output_unit = None
        for i, (unit, numpy_array) in enumerate(zip(array_units, numpy_arrays)):
            if unit is None or isinstance(unit, units.NoUnit):
                continue
            elif output_unit is None:
                output_unit = unit
                continue
            else:
                if unit != output_unit:
                    conversion_ratio = unit.ratio(output_unit)
                    numpy_arrays[i] = numpy_array * conversion_ratio

        return output_unit, numpy_arrays


@SimArray.ufunc_rule(np.square)
def _square_units(a):
    if a.units is not None:
        return a.units ** 2
    else:
        return None

@SimArray.ufunc_rule(np.power)
def _pow_units(a, b):
    a_units,  = _get_units_or_none(a)

    numeric_b = b

    if isinstance(b, tuple):
        b = fractions.Fraction(b[0], b[1])
        numeric_b = float(b)

    if isinstance(b, float):
        b = fractions.Fraction(b).limit_denominator(1000)

    if isinstance(b, np.ndarray):
        # can never figure out units in this case
        return None

    if a_units is not None:
        if not (isinstance(numeric_b, int) or isinstance(numeric_b, float)):
            raise units.UnitsException(f"Don't know how to take the power of a unit with exponent of type {type(b)}")
        return a_units ** b, (a.view(np.ndarray), numeric_b)
    else:
        return None


@SimArray.ufunc_rule(np.arctan, np.arctan2, np.arcsin, np.arccos, np.arcsinh, np.arccosh, np.arctanh, np.sin, np.tan,
    np.cos, np.sinh, np.tanh, np.cosh)
def _trig_units(*a):
    return 1


@SimArray.ufunc_rule(np.greater, np.greater_equal, np.less, np.less_equal, np.equal, np.not_equal)
def _comparison_units(ar, other):
    if units.is_unit_like(other):
        if units.has_units(ar):
            # either other is a unit, or an array with a unit If
            # it's an array with a unit that matches our own, we
            # want to fall straight through to the comparison
            # operation. If it's an array with a unit that doesn't
            # match ours, OR it's a plain unit, we want to
            # convert first.
            if units.is_unit(other) or other.units != ar.units:
                other = other.in_units(ar.units)
        else:
            raise units.UnitsException("One side of a comparison has units and the other side does not")

    if isinstance(ar, SimArray):
        ar = ar.view(np.ndarray)
    if isinstance(other, SimArray):
        other = other.view(np.ndarray)

    return None, (ar, other)

@SimArray.ufunc_rule(np.linalg.norm)
def _norm_units(a, *args, **kwargs):
    return a.units


def _implement_array_functionality(class_):
    """Implement all the standard numpy array functionality on the given class.

    A function is automatically generated for each SimArray method, ensuring that class_ also implements them.
    The implementation is obtained simply by creating a SimArray when the method is called."""

    def _wrap_fn(w):
        @functools.wraps(w)
        def q(s, *y,  **kw):
            return w(SimArray(s), *y, **kw)

        #q.__name__ = w.__name__
        return q

    # functions we definitely want to wrap, even though there's an existing
    # implementation
    _override = "__eq__", "__ne__", "__gt__", "__ge__", "__lt__", "__le__"

    for x in set(np.ndarray.__dict__).union(SimArray.__dict__):
        _w = getattr(SimArray, x)
        if 'array' not in x and ((not hasattr(class_, x)) or x in _override) and hasattr(_w, '__call__') and x!="__buffer__":
            setattr(class_, x, _wrap_fn(_w))
    return class_


@_copy_docstring_from(SimArray)
@_copy_docstring_from(np.ndarray)
@_implement_array_functionality
class IndexedSimArray:
    """A view into a SimArray that allows for indexing and slicing.

    Unlike a numpy array constructed from indexing a parent, IndexedSimArrays do not hold a copy of the underlying data.

    For example:

    >>> a = pynbody.array.SimArray([1, 2, 3, 4])
    >>> b = a[[1, 3]]

    In this case, ``b`` copies the relevant part of ``a`` into a new array. Updating the elements of ``b`` leave
    ``a`` untouched.

    >>> b[:] = 0
    >>> a
    SimArray([1, 2, 3, 4])

    By contrast,

    >>> a = pynbody.array.SimArray([1, 2, 3, 4])
    >>> b = pynbody.array.IndexedSimArray(a, [1, 3])

    In this case, ``b`` provides a view into ``a``. Changes to ``b`` are reflected in ``a``. However, ``b`` is not a
    genuine numpy array.

    >>> b[:] = 0
    >>> a
    SimArray([1, 0, 3, 0])

    For most purposes, an IndexedSimArray should behave exactly like a SimArray/numpy array. However, advanced users may
    want to understand more about performance implications and ways to optimize code. This can be found in the
    :ref:`performance` section of the documentation.

    .. note::

      The methods of this class are the same as those of :class:`SimArray`, and mainly therefore correspond to the
      equivalents within ``numpy.ndarray``. See also the note on function documentation for :class:`SimArray`.

    """
    @property
    def derived(self):
        return self.base.derived

    @property
    def ancestor(self):
        return self.base.ancestor

    def __init__(self, array: SimArray, ptr: slice | np.ndarray):
        """Initialise an IndexedSimArray based on an underlying SimArray and a pointer into that array.

        The pointer can be a slice or an array of indexes.

        Parameters
        ----------

        array : SimArray
            The underlying :class:`SimArray`.

        ptr : slice or numpy.ndarray
            The slice or array of indexes into the underlying array.

        """
        self.base = array
        self._ptr = ptr

    def __array__(self, dtype=None, copy=None):
        return np.asanyarray(self.base[self._ptr], dtype=dtype)

    def _reexpress_index(self, index):
        if isinstance(index, tuple) or (isinstance(index, list) and len(index) > 0 and hasattr(index[0], '__len__')):
            return (self._ptr[index[0]],) + tuple(index[1:])
        else:
            return self._ptr[index]

    def __getitem__(self, item):
        return self.base[self._reexpress_index(item)]

    def __setitem__(self, item, to):
        self.base[self._reexpress_index(item)] = to

    def __getslice__(self, a, b):
        return self.__getitem__(slice(a, b))

    def __setslice__(self, a, b, to):
        self.__setitem__(slice(a, b), to)

    def __repr__(self):
        return self.__array__().__repr__()  # Could be optimized

    def __str__(self):
        return self.__array__().__str__()  # Could be optimized

    def __len__(self):
        return len(self._ptr)

    def __reduce__(self):
        return SimArray(self).__reduce__()

    @property
    def shape(self):
        x = [len(self._ptr)]
        x += self.base.shape[1:]
        return tuple(x)

    @property
    def ndim(self):
        return self.base.ndim

    @property
    def units(self):
        return self.base.units

    @units.setter
    def units(self, u):
        self.base.units = u

    @property
    def sim(self):
        if self.base.sim is not None:
            return self.base.sim[self._ptr]
        else:
            return None

    @sim.setter
    def sim(self, s):
        self.base.sim = s

    @property
    def dtype(self):
        return self.base.dtype

    def conversion_context(self):
        return self.base.conversion_context()

    def set_units_like(self, new_unit):
        self.base.set_units_like(new_unit)

    def in_units(self, new_unit, **context_overrides):
        return IndexedSimArray(self.base.in_units(new_unit, **context_overrides), self._ptr)

    def convert_units(self, new_unit):
        self.base.convert_units(new_unit)

    def write(self, **kwargs):
        self.base.write(**kwargs)




def array_factory(dims: int | tuple, dtype: np.dtype, zeros: bool, shared: bool) -> SimArray:
    """Create an array of dimensions *dims* with the given numpy *dtype*.

    Parameters
    ----------

    dims : int or tuple
        The dimensions of the array.
    dtype : numpy.dtype
        The data type of the array.
    zeros : bool
        If True, the array is guaranteed to be zeroed.
    shared : bool
        If True, the array uses shared memory and can be efficiently shared across processes.

    """
    if not hasattr(dims, '__len__'):
        dims = (dims,)

    if shared:
        from . import shared
        ret_ar = shared.make_shared_array(dims, dtype, zeros)

    else:
        if zeros:
            ret_ar = np.zeros(dims, dtype=dtype).view(SimArray)
        else:
            ret_ar = np.empty(dims, dtype=dtype).view(SimArray)
    return ret_ar
