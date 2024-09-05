.. performance tutorial

.. _performance:

Performance optimisation in pynbody
===================================

``Pynbody`` is built on top of ``numpy``, which means that learning how to optimize ``numpy``
array manipulations is the most important route to writing efficient code; see, for example,
the `Scientific Python lectures <https://lectures.scientific-python.org/advanced/optimizing/index.html>`_
for an introduction.

However there are a couple of issues which are specific to ``pynbody``.

* Many of ``pynbody``'s most common operations are parallelized: make sure you have set up
  these routines to behave in a way that matches your needs by reading the page on :ref:`threads`.
* Sometimes, manipulating arrays from a :class:`~pynbody.snapshot.subsnap.SubSnap` can be slower than
  manipulating the equivalent arrays from the parent snapshot. This is because the arrays
  returned from a :class:`~pynbody.snapshot.subsnap.SubSnap` are not true ``numpy`` arrays but are
  instead :class:`~pynbody.array.IndexedSimArray` objects. This is explained in more detail below.

.. seealso::

   A :ref:`separate document <parallelism>` covers parallelism in ``pynbody``, which can also be
   important for performance-critical code.

Overheads of SubSnaps
------------------------

.. _template_performance_code:

A template for performance-critical code
********************************************

To cut a long story short, if your routine does a lot of array access on an object which might
be a :class:`~pynbody.snapshot.subsnap.SubSnap` of a certain flavour (explained further below),
you will find that wrapping your code as follows speeds it up.

.. code-block:: python

 def my_expensive_operation(sim_or_subsim) :
     with sim_or_subsim.immediate_mode :
         mass_array = sim_or_subsim['mass']
         other_array = sim_or_subsim['other']
         # ... get other arrays required...

         #
         # perform multiple operations on arrays
         #

         # At end, copy back results if the arrays have
         # changed
         sim_or_subsim['other'] = other_array

The remainder of this document unpacks what this does and why it should be necessary.

What is a SubSnap, really?
****************************

When you construct a sub-view of a simulation, the framework records which
particles of the underlying :class:`~pynbody.snapshot.simsnap.SimSnap` are included and which are not.
Thereafter, if you access an array from the sub-view, it is
constructed in one of two ways.

- If the set of particles in the sub-view is expressible as
  a slice, the type of the sub-view is :class:`~pynbody.snapshot.subsnap.SubSnap`,
  and any arrays accessed are still returned as ``numpy`` arrays, albeit with non-contiguous strides.
  This will be the
  case if you explicitly slice the simulation (e.g. ``f[2:100:3]``), or if you ask for a
  specific particle family (e.g. ``f.dm``).

- If the set of particles is not expressible in this way, instead of a
  :class:`~pynbody.snapshot.subsnap.SubSnap`
  you will get a :class:`~pynbody.snapshot.subsnap.IndexedSubSnap`. In this case,
  arrays returned from the new view are
  *emulating* ``numpy`` arrays and this can become expensive (see below). This will
  be the case if you ask for a list of particles (e.g. ``f[[2,10,15,22]]``) use a ``numpy``-like
  indexing trick (e.g. ``f[f['x']>10]`` or ``f[numpy.where(f['x']>10)]``) or use a
  :py:mod:`filter <pynbody.filt>`.

In the case of :class:`~pynbody.snapshot.subsnap.IndexedSubSnap`, performance
can be rather different from that of the parent snapshot.
To understand how and why, we
need to look at the difference between an indexed and a sliced ``numpy`` array.


The need for array emulation
****************************

When you get an array from a :class:`~pynbody.snapshot.subsnap.IndexedSubSnap`, it is of
type :class:`~pynbody.array.IndexedSimArray`.
This section explains why the reason and implications.

The ``pynbody`` framework is designed to allow users to interact with data without worrying
too much about whether it is an entire simulation or a small portion of a simulation.
Consistency then requires all sub-arrays to continue pointing to the original data.
But a simple experiment with numpy shows that it does not enable this behaviour in all
cases that we want to cover.

Here's what happens when you use a slice of an existing ``numpy`` array.

.. ipython::

 In [2]: import numpy as np

 In [3]: a = np.zeros(10)

 In [4]: b = a[3:5]

 In [5]: b[1] = 100

 In [6]: a
 Out[6]: array([   0.,    0.,    0.,    0.,  100.,    0.,    0.,    0.,    0.,    0.])

The ``a`` array has been updated as required, because the ``b`` and ``a`` objects
actually point back to the same part of the computer memory.

On the other hand, when you *index* a ``numpy`` array, the behaviour is different.

.. ipython::

 In [7]: c = a[[4,5,6]]

 In [8]: c[1] = 200

 In [9]: a
 Out[9]: array([   0.,    0.,    0.,    0.,  100.,    0.,    0.,    0.,    0.,    0.])

Here changing ``c`` has not updated ``a``. That's because the construction of ``c`` actually
*copied* the relevant data instead of just pointing back at it.  This is necessitated by
the underlying design of ``numpy`` arrays requiring the data to lie in a predictable
pattern in the memory.

The :class:`~pynbody.array.IndexedSimArray` class fixes this problem:

.. ipython ::

 In [10]: import pynbody

 In [12]: d = pynbody.array.IndexedSimArray(a, [4,5,6])

 In [13]: d[1] = 200

 In [14]: a
 Out[14]: array([   0.,    0.,    0.,    0.,  100.,  200.,    0.,    0.,    0.,    0.])

Note that ``a`` has been updated correctly. This is achieved by the ``IndexedSimArray``
*emulating*, rather than *wrapping*, a ``numpy`` array; internally
the syntax ``d[1]=200`` is then translated into ``a[[4,5,6][1]]=200``.

The cost of this is that each time you call a function that requires a ``numpy`` array
as an input, the ``IndexedSimArray`` has to generate a proxy for this purpose. This can become slow.
Have a look at the following timings:

.. ipython ::

 In [22]: %time for i in range(10000) : d+=1
 CPU times: user 0.25 s, sys: 0.03 s, total: 0.28 s
 Wall time: 0.26 s

 In [23]: %time for i in range(10000) : a+=1
 CPU times: user 0.04 s, sys: 0.00 s, total: 0.04 s
 Wall time: 0.04 s

Adding to the subarray is *slower* than adding to the entire array!
This is because of the overheads of continually constructing proxy
``numpy`` arrays to pass to the ``__add__`` method.


How to remove this bottleneck
*****************************

We should emphasize that the example above is quite contrived, since it forces
re-construction of the ``numpy`` proxy 10000 times. In user code,
the number of ``numpy`` proxies that have to be constructed will be vastly smaller,
so the fractional overheads are normally quite small.

Nonetheless, construction of these proxy arrays does sometimes become a problem for
performance-critical code. For that reason, it's possible to avoid constructing
:class:`~pynbody.array.IndexedSimArray` s altogether
and force only :class:`~pynbody.array.SimArray` to be returned. This is a thin wrapper
around a ``numpy`` array (see :ref:`overhead_simarray` below) and, as such, is enormously more efficient.
But it can be less convenient since you have to keep track of when to copy data back.

Pynbody refers to this approach as *immediate mode*; it can be activated using a context manager
(i.e. python's ``with`` keyword).
Let's create a test snapshot and a subview into that snapshot to try it out.

.. ipython ::

 In [24]: f = pynbody.new(dm=100)

 In [25]: sub_f = f[[20,21,22]]

Under normal conditions, the type of arrays returned from ``sub_f`` is
:class:`~pynbody.array.IndexedSimArray`.
Updating one of these arrays will transparently update the main snapshot.

.. ipython ::

 In [36]: sub_mass = sub_f['mass']

 In [30]: type(sub_mass)

 In [37]: sub_mass[:]=3

 In [35]: f['mass'][[20,21,22]]
 [ 3.  3.  3.]


Conversely, in ``immediate mode``, the type of arrays returned from ``sub_f`` is
:class:`~pynbody.array.SimArray`.
This is faster, but updating the returned ``numpy`` array has *no effect* on the
parent snapshot.

.. ipython ::

 In [32]: with f.immediate_mode :
    ....:     sub_mass = sub_f['mass']
    ....:

 In [30]: type(sub_mass)

 In [30]: sub_mass

 In [64]: sub_mass[:]=5

 In [30]: sub_mass # updated as expected

 In [39]: f['mass'][[20,21,22]] # NOT updated - should still be 3,3,3!
 Out[39]: SimArray([ 3.,  3.,  3.])


So it becomes your responsibility to copy the results back in this case, if required. A template for performance
critical code which might be operating on a ``SubSnap`` was given above, in
:ref:`template_performance_code`.

In summary, the template code:

 - stores a *copy* of the data for the subset of particles
 - works on the copy
 - (if necessary) updates the main snapshot data explicitly before returning


.. note::

 ``with f_sub.immediate_mode``
 is equivalent to ``with f.immediate_mode`` where ``f_sub`` is any
 sub-view of ``f``.

.. _overhead_simarray:

Overheads of SimArrays
----------------------

.. note::

 This information is provided for interest. We have never come across a realistic use case
 where the following is necessary.

In ``pynbody``, arrays are implemented by the class :class:`~pynbody.array.SimArray`. This is a thin wrapper
around a ``numpy`` array. There is a small extra cost associated with every operation to allow
units to be matched and updated. For long arrays such as those found in typical simulations, this is usually a tiny fraction of the
actual computation time. We have never found it to be a problem, but if you want to disable the
unit tracking you can always do so using ``numpy``'s view mechanism to get a raw ``numpy`` array.
Suppose you have a ``SimSnap`` ``f``; then ``pos = f['pos'].view(numpy.ndarray)`` (for example) will return the position
array without any of the ``SimArray`` wrapping. The new ``pos`` variable can be manipulated without
any unit handling code being called.
