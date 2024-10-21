.. _derived:

Automatically Derived Arrays
============================

Pynbody includes a system which automatically derives one array from
others. There are two goals for this system:

(1) Help make user code more independent of the underlying file format,
    by providing missing arrays. For example, some simulation formats
    store temperature while others store internal energy. By always
    presenting a ``temp`` array to you, and handling the conversion
    automatically, the need to explicitly distinguish between different
    scenarios is removed.

(2) Keep arrays up-to-date when they depend on other arrays. For example,
    if you are interested in the radius of particles from the origin,
    a derived ``r`` array is offered which is automatically recalculated if
    the position of particles changes.


Example: the radius array
-------------------------

The quantities listed under :mod:`~pynbody.derived` are calculated for
all simulation types. If you want, for example, to calculate the
radius of particles away from the origin, you can just access the ``r`` array and
pynbody will calculate it:

.. ipython::

   In [3]: import pynbody

   In [4]: s = pynbody.load('testdata/gasoline_ahf/g15784.lr.01024.gz')

   In [5]: s['r']

Here, ``r`` has been calculated in the same units as the position array,
which in turn are in the units of the simulation.
You can call :meth:`~pynbody.snapshot.simsnap.SimSnap.physical_units` as
normal to convert to more recognizable units.
Note that you cannot directly alter a derived array:

.. ipython:: :okexcept:

   In [5]: s['r'][0] = 0

But the array updates itself when one of its dependencies changes:

.. ipython::

   In [5]: s['x']-=1

   In [5]: s['r']

The recalculation only takes place when required -- in the example
above the ``r`` array is stored (just like a normal array) until the ``pos`` array is updated,
at which point it is deleted. When you ask for the ``r`` array again,
it is recalculated.

This is why you're not allowed to change values in a derived array --
your changes could be overwritten, leading to confusing bugs. However,
if you want to make a derived array back into a normal array you can
do so.

.. ipython::

   In [5]: s['r'].derived = False

   In [5]: s['r'][0] = 0

   In [5]: s['r']


At this point you've taken full responsibility for the
array, so if you update its dependencies the framework *won't* help
you out any longer:

.. ipython::

   In [5]: s['x']+=2

   In [5]: s['r'] # NOT updated because we took control by setting derived = False


To get the framework back on your side, you can delete the modified
array:

.. ipython::

   In [5]: del s['r']

   In [5]: s['r']

Here we've deleted then re-derived the ``r`` array, so it's now accurate (and will start
auto-updating again).

Derived functions for specific formats
--------------------------------------

Some derived arrays are specific to certain simulation formats. For example, ramses simulations
need to derive masses for their gas cells and as such :func:`~pynbody.snapshot.ramses.mass` is registered
as a derived array specifically for the :class:`~pynbody.snapshot.ramses.RamsesSnap` class.


Defining your own deriving functions
------------------------------------

You can easily define your own derived arrays. The easiest way to do
this is using the decorator :func:`pynbody.snapshot.simsnap.SimSnap.derived_array`.
This is handily aliased to ``pynbody.derived_array``.

Here's an example of a derived array that calculates the specific
kinetic energy of a particle:

.. ipython::

   In [5]: @pynbody.derived_array
      ...: def specific_ke(sim):
      ...:     return 0.5 * (sim['vel']**2).sum(axis=1)

   In [7]: s['specific_ke']

When your function is called, the framework monitors any arrays it
retrieves from the simulation. It automatically marks the accessed
arrays as dependencies for your function. So, if the velocities now
get changed, your derived array will be recalculated:


.. ipython::

   In [7]: s['vel']*=2

   In [7]: s['specific_ke']

To create a derived array associated with a specific subclass, use the
:meth:`~pynbody.snapshot.simsnap.SimSnap.derived_array` method of that subclass,
e.g.

.. ipython::

   In [5]: @pynbody.snapshot.tipsy.TipsySnap.derived_array
      ...: def half_mass(sim):
      ...:     return 0.5 * sim['mass']

   In [7]: s['half_mass'] # this is a TipsySnap, so this will work

   In [8]: another_snap = pynbody.new(dm=100) # NOT a TipsySnap!

   In [9]: another_snap['half_mass']

The derived array will only be available for the class it was defined for, so the
final command raised an error.

.. versionchanged:: 2.0
    The method name has been changed from ``derived_quantity`` to ``derived_array``.
    The old name is still available but will be removed in a future version.


.. _stable_derived:

Stable derived arrays
----------------------

Occasionally, you may want to define a derived array that is not automatically recalulated
when its underlying dependencies change. An example from within the framework is the
smoothing length (``smooth``). This is expensive to calculate, and changes to the underlying position
coordinates -- while in theory capable of changing the smoothing length -- are far more commonly
associated with translations or rotations in the course of a normal analysis. In this case, it
would be wasteful to recalculate the smoothing length every time the position coordinates change.

To define a stable derived array, use :meth:`~pynbody.snapshot.simsnap.SimSnap.stable_derived_array`
in place of :meth:`~pynbody.snapshot.simsnap.SimSnap.derived_array`. The array will be derived for
the first time, but will not automatically update:

.. ipython::

   In [5]: @pynbody.snapshot.simsnap.SimSnap.stable_derived_array
      ...: def stable_x_copy(sim):
      ...:     return sim['x']

   In [7]: s['stable_x_copy']

   In [8]: s['x']+=1

   In [9]: s['x']

   In [9]: s['stable_x_copy']


.. seealso::
   More information about the derived array system can be found in the
   method documentation for :meth:`~pynbody.snapshot.simsnap.SimSnap.derived_array`
   and :meth:`~pynbody.snapshot.simsnap.SimSnap.stable_derived_array`. Information about
   built-in derived arrays can be found in the module :mod:`pynbody.derived`. The module
   :mod:`pynbody.analysis.luminosity` also defines an entire class of derived arrays for
   calculating magnitudes from stellar population tables.
