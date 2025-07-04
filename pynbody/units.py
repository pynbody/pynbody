"""
A set of classes for tracking units.

Units are generated and used at various points through the pynbody
framework. Quite often the functions where users interact with units
simply accept strings.

You can also make units yourself in two ways. Either you can create a string, and
instantiate a Unit like this:

>>> units.Unit("Msol kpc**-3")
>>> units.Unit("2.1e12 m_p cm**-2/3")

Or you can do it within python, using the named Unit objects

>>> units.Msol * units.kpc**-3
>>> 2.1e12 * units.m_p * units.cm**(-2,3)

In the last example, either a tuple describing a fraction or a
Fraction instance (from the standard python module fractions) is
acceptable.


Getting conversion ratios
-------------------------

To convert one unit to another, use :meth:`~Unit.ratio`. For example:

>>> units.Msol.ratio(units.kg)
1.99e30
>>> (units.Msol / units.kpc**3).ratio(units.m_p/units.cm**3)
4.04e-8

If the units cannot be converted, a UnitsException is raised:

>>> units.Msol.ratio(units.kpc)
UnitsException

Specifying numerical values
---------------------------

Sometimes it's necessary to specify a numerical value in the course
of a conversion. For instance, consider a comoving distance; this
can be specified in pynbody units as follows:

>>> comoving_kpc = units.kpc * units.a

where units.a represents the scalefactor. We can attempt to convert
this to a physical distance as follows

>>> comoving_kpc.ratio(units.kpc)

but this fails, throwing a UnitsException. On the other hand we
can specify a value for the scalefactor when we request the conversion

>>> comoving_kpc.ratio(units.kpc, a=0.5)
0.5

and the conversion completes with the expected result. The units
module also defines units.h for the dimensionless hubble constant,
which can be used similarly. *By default, all conversions happening
within a specific simulation context should pass in values for
a and h as a matter of routine.*

Any IrreducibleUnit (see below) can have a value specified in this way,
but a and h are envisaged to be the most useful applications.

Defining new base units
-----------------------

The units module is fully extensible: you can define and name your own
units which then integrate with all the standard functions.

.. code-block:: python

   litre = units.NamedUnit("litre",0.001*units.m**3)
   gallon = units.NamedUnit("gallon",0.004546*units.m**3)
   gallon.ratio(litre) # 4.546
   (units.pc**3).ratio(litre) # 2.94e52


You can even define completely new dimensions.

.. code-block:: python

    V = units.IrreducibleUnit("V") # define a volt
    C = units.NamedUnit("C", units.J/V) # define a coulomb
    q = units.NamedUnit("q", 1.60217646e-19*C) # elementary charge
    F = units.NamedUnit("F", C/V) # Farad
    epsilon0 = 8.85418e-12 *F/units.m


>>> (q*V).ratio("eV")
1.000
>>> ((q**2)/(4*math.pi*epsilon0*units.m**2)).ratio("N")
2.31e-28


"""

import abc
import fractions
import functools
import keyword
import numbers
import re
from collections import defaultdict

import numpy as np

Fraction = fractions.Fraction

_registry = {}


class UnitsException(Exception):
    pass


class UnitBase(abc.ABC):

    """Abstract base class for units.

    To instantiate a unit, use instead class :class:`Unit`, which will return an object of an appropriate subclass.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    def __pow__(self, p):
        if isinstance(p, tuple):
            p = Fraction(p[0], p[1])
        if not (isinstance(p, Fraction) or isinstance(p, int)):
            if isinstance(p, float):
                raise ValueError("Units can only be raised to integer or fractional powers. Use python's built-in fractions module or a tuple: e.g. unit**(1,2) represents a square root.")
            else :
                raise ValueError("Units can only be raised to integer or fractional powers")
        return CompositeUnit(1, [self], [p]).simplify()

    def __truediv__(self, m):
        return self.__div__(m)

    def __rtruediv__(self, m):
        return self.__rdiv__(m)

    def __div__(self, m):
        if hasattr(m, "_no_unit"):
            return NoUnit()

        if isinstance(m, UnitBase):
            return CompositeUnit(1, [self, m], [1, -1]).simplify()
        else:
            return CompositeUnit(1.0 / m, [self], [1]).simplify()

    def __rdiv__(self, m):
        return CompositeUnit(m, [self], [-1]).simplify()

    def __mul__(self, m):
        if hasattr(m, "_no_unit"):
            return NoUnit()
        elif hasattr(m, "units"):
            return m * self
        elif isinstance(m, UnitBase):
            return CompositeUnit(1, [self, m], [1, 1]).simplify()
        else:
            return CompositeUnit(m, [self], [1]).simplify()

    def __rmul__(self, m):
        return CompositeUnit(m, [self], [1]).simplify()

    def __add__(self, m):
        scale = m.in_units(self) if hasattr(m, 'in_units') else m/float(self)
        if hasattr(scale, 'units'):
            scale.units = 1
        return self * (1.0 + scale)

    def __sub__(self, m):
        return self + (-m)

    def __repr__(self):
        return 'Unit("' + str(self) + '")'

    def __eq__(self, other):
        if not isinstance(other, UnitBase):
            other = Unit(other)
        _self_scale = getattr(self, '_scale', None)
        _other_scale = getattr(other, '_scale', None)
        if self.dimensionality_as_string() != other.dimensionality_as_string():
            return False
        elif _self_scale == 0.0 and _other_scale == 0.0:
            return True
        elif _self_scale == 0.0 or _other_scale == 0.0:
            return False
        else:
            try:
                return self.ratio(other) == 1.
            except UnitsException:
                return False


    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return self.ratio(other) < 1.

    def __gt__(self, other):
        return self.ratio(other) > 1.

    def __le__(self, other):
        return self.ratio(other) <= 1.

    def __ge__(self, other):
        return self.ratio(other) >= 1.

    def __neg__(self):
        return self * (-1)

    def __float__(self):
        return 1.

    def __hash__(self):
        return id(self)

    def copy(self):
        """Shortcut for ``copy.copy(unit)`` using python's copy module"""
        import copy
        return copy.copy(self)

    def _dimension_state(self):
        state = defaultdict(int)
        for base, power in zip(self._bases, self._powers):
            if isinstance(base, UnitBase):
                for k, v in base._dimension_state().items():
                    state[k] += v * power
            else:
                state[base] += power

        return state

    def dimensionality_as_string(self):
        """
        Returns the dimensionality of the Unit object.

        Example
        -------
        > pynbody.units.Unit("3e8 m s**-1 yr").dimensionality_as_string()
        'm^1'
        """
        state = self._dimension_state()
        return " ".join(
            f"{k}^{state[k]}" for k in sorted(state) if state[k] != 0
        )

    def simplify(self):
        """Return a simplified version of the unit."""
        return self

    @abc.abstractmethod
    def latex(self):
        r"""Returns a LaTeX representation of this unit.

        Prefactors are converted into exponent notation. Named units by default
        are represented by the string '\mathrm{unit_name}', although this can
        be overriden in the pynbody configuration files or by setting
        unit_name._latex."""

    def is_dimensionless(self):
        """Returns True if the unit is dimensionless, False otherwise."""
        return False

    def ratio(self, other, **substitutions):
        """Get the conversion ratio between this Unit and another specified unit.

        Parameters
        ----------

        other : Unit or str
            The other unit to convert to.

        substitutions : dict
            Substitutions to make for units in the conversion. For example, if the unit is a comoving distance, you can
            specify the scale factor a. More generally, you can specify a valuable for any IrreducibleUnit.

        Notes
        -----

        Usage examples:

        >>> Unit("1 Mpc a").ratio("kpc", a=0.25)
        250.0
        >>> Unit("1 Mpc").ratio("Msol")
        UnitsException: not convertible
        >>> Unit("1 Mpc").ratio("Msol", kg=25.0, m=50.0)
        3.1028701506345152e-08
        """

        if isinstance(other, str):
            other = Unit(other)

        if hasattr(other, "_no_unit"):
            raise UnitsException("Unknown units")

        try:
            return (self / other).dimensionless_constant(**substitutions)
        except UnitsException:
            raise UnitsException("Not convertible")

    def in_units(self, *a, **kw):
        """Alias for ratio"""

        return self.ratio(*a, **kw)

    @abc.abstractmethod
    def dimensionless_constant(self, **substitutions):
        """If this unit is dimensionless, return its scalar quantity.

        Direct use of this function is not recommended. It is generally
        better to use the ratio function instead.

        Provide keyword arguments to set values for named IrreducibleUnits --
        see the :meth:`ratio` function for more information."""
        pass

    @abc.abstractmethod
    def irrep(self):
        """Return a unit equivalent to this one in terms of the currently defined :class:`IrreducibleUnit`s

        Two units are equivalent if they have the same ``irrep``.
        """
        pass

    def _register_unit(self, st):
        if st in _registry:
            raise UnitsException("Unit with this name already exists")
        if "**" in st or "^" in st or " " in st:
            # will cause problems for simple string parser in Unit() factory
            raise UnitsException("Unit names cannot contain '**' or '^' or spaces")
        _registry[st] = self

    def __deepcopy__(self, memo):
        # This may look odd, but the units conversion will be very
        # broken after deep-copying if we don't guarantee that a given
        # physical unit corresponds to only one instance
        return self

    def dimensional_project(self, basis_units):
        """Work out how to express the dimensions of this unit relative to the
        specified list of basis units.

        This is used by the framework when making inferences about sensible units to
        use in various situations.

        For example, you can represent a length as an energy divided by a force:

           >>> Unit("23 kpc").dimensional_project(["J", "N"])
           array([1, -1], dtype=object)

        However it's not possible to represent a length by energy alone:

           >>> Unit("23 kpc").dimensional_project(["J"])
           UnitsException: Basis units do not span dimensions of specified unit

        This function also doesn't know what to do if the result is ambiguous:

           >>> Unit("23 kpc").dimensional_project(["J", "N", "kpc"])
           UnitsException: Basis units are not linearly independent

        """

        vec_irrep = [Unit(x).irrep() for x in basis_units]
        me_irrep = self.irrep()
        bases = set(me_irrep._bases)
        for vec in vec_irrep:
            bases.update(vec._bases)

        bases = list(bases)

        matrix = np.zeros((len(bases), len(vec_irrep)), dtype=Fraction)

        for base_i, base in enumerate(bases):
            for vec_i, vec in enumerate(vec_irrep):
                matrix[base_i, vec_i] = vec._power_of(base)

        # The matrix calculated above describes the transformation M
        # such that v = M.d where d is the sought-after powers of the
        # specified base vectors, and v is the powers in terms of the
        # base units in the list bases.
        #
        # To invert, since M is possibly rectangular, we use the
        # solution to the least-squares problem [minimize (v-M.d)^2]
        # which is d = (M^T M)^(-1) M^T v.
        #
        # If the solution to that does not solve v = M.d, there is no
        # admissable solution to v=M.d, i.e. the supplied base vectors do not
        # span the required space.
        #
        # If (M^T M) is singular, the vectors are not linearly independent, so
        # any
        # solution would not be unique.
        M_T_M = np.dot(matrix.transpose(), matrix)

        from . import util

        try:
            M_T_M_inv = util.rational_matrix_inv(M_T_M)
        except np.linalg.linalg.LinAlgError:
            raise UnitsException("Basis units are not linearly independent")

        my_powers = [me_irrep._power_of(base) for base in bases]

        candidate = np.dot(M_T_M_inv, np.dot(matrix.transpose(), my_powers))

        # Because our method involves a loss of information (multiplying
        # by M^T), we could get a spurious solution. Check this is not the
        # case...

        if any(np.dot(matrix, candidate) != my_powers):
            # Spurious solution, meaning the base vectors did not span the
            # units required in the first place.
            raise UnitsException(
                "Basis units do not span dimensions of specified unit")

        return candidate


class NoUnit(UnitBase):
    """Represents a state of units being unknown.

    Note that this is not the same as dimensionless units, which are known but have no physical dimensions.
    Dimensionless units are represented by ``Unit(1)``.
    """

    def __init__(self):
        self._no_unit = True

    def _dimension_state(self):
        return {}

    def ratio(self, other, **substitutions):
        if isinstance(other, NoUnit):
            return 1
        else:
            raise UnitsException("Unknown units")

    def dimensional_project(self, *args):
        raise UnitsException("Unknown units")

    def is_dimensionless(self):
        return True

    def irrep(self):
        return self

    def simplify(self):
        return self

    def __pow__(self, a):
        return self

    def __div__(self, a):
        return self

    def __rdiv__(self, a):
        return self

    def __mul__(self, a):
        return self

    def __rmul__(self, a):
        return self

    def __repr__(self):
        return "NoUnit()"

    def latex(self):
        return ""

    def irrep(self):
        return self

    def dimensionless_constant(self, **substitutions):
        raise UnitsException("Unknown units")

no_unit = NoUnit()


def _resurrect_named_unit(unit_name, unit_latex, represents):
    if unit_name in _registry:
        return _registry[unit_name]
    else:
        if represents is None:
            nu = IrreducibleUnit(unit_name)
        else:
            nu = NamedUnit(unit_name, represents)
            nu._latex = unit_latex
        return nu


class IrreducibleUnit(UnitBase):
    """Represents a unit which cannot be further simplified."""

    def __init__(self, st):
        self._st_rep = st
        self._register_unit(st)
        self._bases = [st]
        self._powers = [1]

    def __reduce__(self):
        return (_resurrect_named_unit, (self._st_rep, None, None))

    def __str__(self):
        return self._st_rep

    def latex(self):
        return r"\mathrm{" + self._st_rep + "}"

    def irrep(self):
        return CompositeUnit(1, [self], [1])

    def dimensionless_constant(self, **substitutions):
        return 1.0


class NamedUnit(UnitBase):
    """Represents a unit which has a name but can also be expressed as a composite of :class:`IrreducibleUnit`s."""
    def __init__(self, st, represents):
        self._st_rep = st
        if isinstance(represents, str):
            represents = Unit(represents)

        self._represents = represents
        self._register_unit(st)

    def __reduce__(self):
        return (_resurrect_named_unit, (self._st_rep, getattr(self, '_latex', None), self._represents))

    def __str__(self):
        return self._st_rep

    def _dimension_state(self):
        return self._represents._dimension_state()

    def latex(self):
        if hasattr(self, '_latex'):
            return self._latex
        return r"\mathrm{" + self._st_rep + "}"

    def irrep(self):
        return self._represents.irrep()

    def dimensionless_constant(self, **substitutions):
        return 1.0


class CompositeUnit(UnitBase):
    """Represents a unit which is a composite of other units."""

    def __init__(self, scale, bases, powers):
        """Initialize a composite unit.

        Direct use of this initializer is not recommended. Instead use :class:`Unit` to create a unit object."""

        if scale == 1.:
            scale = 1

        self._scale = scale
        self._bases = bases
        self._powers = powers

    def latex(self):
        if self._scale != 1:
            log10_scale = np.log10(abs(self._scale))

            # if we can compactly express the scale without exponents, do so
            force_no_exponent = str(self._scale).endswith(".0") and log10_scale < 4

            if (log10_scale < 2.5 and log10_scale > -1.0) or force_no_exponent:
                if log10_scale > 2.0:
                    s = "%.1f" % self._scale
                elif log10_scale > 1.0:
                    s = "%.2f" % self._scale
                elif log10_scale > 0.0:
                    s = "%.3f" % self._scale
                else:
                    s = "%.4f" % self._scale

                while "." in s and s.endswith("0") or s.endswith("."):
                    s = s[:-1]
            else:
                x = ("%.2e" % self._scale).split('e')
                s = x[0]
                ex = x[1].lstrip('0+')
                if len(ex) > 0 and ex[0] == '-':
                    ex = '-' + (ex[1:]).lstrip('0')
                if ex != '':
                    s += r"\times 10^{" + ex + "}"

        else:
            s = ""

        for b, p in zip(self._bases, self._powers):
            if s != "":
                s += r"\," + b.latex()
            else:
                s = b.latex()

            if p != 1:
                s += "^{"
                s += str(p)
                s += "}"
        return s

    def __str__(self):
        s = None
        if len(self._bases) == 0:
            return "%.2e" % self._scale

        if self._scale != 1:
            s = "%.2e" % self._scale

        for b, p in zip(self._bases, self._powers):
            if s is not None:
                s += " " + str(b)
            else:
                s = str(b)

            if p != 1:
                s += "**"
                if isinstance(p, Fraction):
                    s += str(p)
                else:
                    s += str(p)
        return s

    def __float__(self):
        return float(self._scale)

    def _expand(self, expand_to_irrep=False):
        """Internal routine to expand any pointers to composite units
        into direct pointers to the base units. If expand_to_irrep is
        True, everything is expressed in irreducible units.
        A _gather will normally be necessary to sanitize the unit
        after an _expand."""

        trash = []

        for i, (b, p) in enumerate(zip(self._bases, self._powers)):
            if isinstance(b, NamedUnit) and expand_to_irrep:
                b = b._represents.irrep()

            if isinstance(b, CompositeUnit):
                if expand_to_irrep:
                    b = b.irrep()

                trash.append(i)
                self._scale *= b._scale ** p
                for b_sub, p_sub in zip(b._bases, b._powers):
                    self._bases.append(b_sub)
                    self._powers.append(p_sub * p)

        trash.sort()
        for offset, i in enumerate(trash):
            del self._bases[i - offset]
            del self._powers[i - offset]

    def _gather(self):
        """Internal routine to gather together powers of the same base
        units, then order the base units by their power (descending)"""

        trash = []
        bases = list(set(self._bases))
        powers = [sum(p for bi, p in zip(self._bases, self._powers)
                       if bi is b)
                  for b in bases]

        bp = sorted((x for x in zip(powers, bases) if x[0] != 0),
                    reverse=True,
                    key=lambda x: x[0])

        if len(bp) != 0:
            self._powers, self._bases = list(map(list, list(zip(*bp))))
        else:
            self._powers, self._bases = [], []

    def __copy__(self):
        return CompositeUnit(self._scale, self._bases[:], self._powers[:])

    def simplify(self):
        self._expand()
        self._gather()
        return self

    def irrep(self):
        x = self.copy()
        x._expand(True)
        x._gather()
        return x

    def is_dimensionless(self):
        x = self.irrep()
        if len(x._powers) == 0:
            return True
        else:
            return False

    def dimensionless_constant(self, **substitutions):
        x = self.irrep()
        c = x._scale
        for xb, xp in zip(x._bases, x._powers):
            if str(xb) in substitutions:
                c *= substitutions[str(xb)] ** xp
            else:
                raise UnitsException("Not dimensionless")

        return c

    def _power_of(self, base):
        if base in self._bases:
            return self._powers[self._bases.index(base)]
        else:
            return 0


class Unit(UnitBase):
    """
    Class factory for units.
    """

    def __new__(cls, s):
        if isinstance(s, UnitBase):
            return s
        elif isinstance(s, numbers.Number):
            scale = float(s)
            units = []
            powers = []
        else:
            s = str(s)

            x = s.split()
            try:
                scale = float(x[0])
                del x[0]
            except (ValueError, IndexError):
                scale = 1.0

            units = []
            powers = []

            for com in x:
                if "**" in com or "^" in com:
                    s = com.split("**" if "**" in com else "^")
                    try:
                        u = _registry[s[0]]
                    except KeyError:
                        raise ValueError("Unknown unit " + s[0])
                    p = Fraction(s[1])
                    if p.denominator == 1:
                        p = p.numerator
                else:
                    u = _registry[com]
                    p = 1

                units.append(u)
                powers.append(p)

        return CompositeUnit(scale, units, powers)

    def __init__(self, s):
        """Given a string s, creates an appropriate Unit object.

        The string format is ``[<scale>] [<unit_name>][**<rational_power>] [[<unit_name>] ... ]``

        Examples: ``1.e30 kg``, ``kpc**2``, ``26.2 m s**-1``, ``15 m**1/2 s**-1/2``.

        Note that the actual class returned depends on the string provided. Generally
        you should not need to worry about this, as the class returned will be a
        subclass of :class:`UnitBase` and will behave as expected.
        """


def takes_arg_in_units(*args, **orig_kwargs):
    """

    Returns a decorator to create a function which auto-converts input
    to given units.

    **Usage:**

    .. code-block:: python

        @takes_arg_in_units((2, "Msol"), (1, "kpc"), ("blob", "erg"))
        def my_function(arg0, arg1, arg2, blob=22) :
           print("Arg 2 is",arg2,"Msol")
           print("Arg 1 is",arg1,"kpc")
           print("blob is",blob,"ergs")



    >>> My_function(22, "1.e30 kg", 23, blob="12 J")
    Input 3 is 0.5 Msol
    Input 2 is 23 kpc

    """

    context_arg = orig_kwargs.get('context_arg', None)

    kwargs = [x for x in args if hasattr(x[0], '__len__')]
    args = [x for x in args if not hasattr(x[0], '__len__')]

    def decorator_fn(x):
        @functools.wraps(x)
        def wrapper_fn(*fn_args, **fn_kwargs):
            context = {}
            if context_arg is not None:
                context = fn_args[context_arg].conversion_context()

            fn_args = list(fn_args)

            for arg_num, arg_units in args:

                if isinstance(fn_args[arg_num], str):
                    fn_args[arg_num] = Unit(fn_args[arg_num])

                if hasattr(fn_args[arg_num], "in_units"):
                    fn_args[arg_num] = fn_args[
                        arg_num].in_units(arg_units, **context)

            for arg_name, arg_units in kwargs:
                if isinstance(fn_kwargs[arg_name], str):
                    fn_kwargs[arg_name] = Unit(fn_kwargs[arg_name])

                if hasattr(fn_kwargs[arg_name], "in_units"):
                    fn_kwargs[arg_name] = fn_kwargs[
                        arg_name].in_units(arg_units, **context)

            return x(*fn_args, **fn_kwargs)

        return wrapper_fn

    return decorator_fn


from . import config_parser


def __is_clean_name(s):
    if re.search('[^0-9a-zA-Z_]', s):
        return False
    if re.search('^[^a-zA-Z_]+', s):
        return False
    if keyword.iskeyword(s):
        return False
    return True

for new_unit_name in map(str.strip, config_parser.get('irreducible-units', 'names').split(",")):
    new_unit = IrreducibleUnit(new_unit_name)
    if __is_clean_name(new_unit_name):
        globals()[new_unit_name] = new_unit

for new_unit_name, new_unit_definition in config_parser.items("named-units"):
    new_unit = NamedUnit(new_unit_name, new_unit_definition)
    if __is_clean_name(new_unit_name):
        globals()[new_unit_name] = new_unit

for unit_name, latex in config_parser.items("units-latex"):
    _registry[unit_name]._latex = latex


_default_units = {}

for a_, b_ in config_parser.items("default-array-dimensions"):
    _default_units[a_] = Unit(b_)


def has_unit(obj):
    """Returns True if the specified object has a meaningful units attribute"""
    if hasattr(obj, 'units') and isinstance(obj.units, UnitBase):
        return not hasattr(obj.units, '_no_unit')
    else:
        return False

has_units = has_unit

def get_item_with_unit(array, item):
    if has_unit(array) and len(array.shape)==1:
        return array[item]*array.units
    else:
        return array[item]


def is_unit(obj):
    """Returns True if the specified object represents a unit"""

    return isinstance(obj, UnitBase)


def is_unit_like(obj):
    """Returns True if the specified object is itself a unit or
    otherwise exposes unit information"""

    return is_unit(obj) or has_unit(obj)
