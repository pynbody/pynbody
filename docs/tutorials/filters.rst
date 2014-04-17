.. filters tutorial

.. _filters_tutorial

Subviews and Filters
====================

Analyzing simulations often boils down to comparing properties of
groups of particles or sections of the simulated volume. Therefore a
generic task in analysis routines is to search for indices of
particles (or cells) corresponding to some interesting property, so
analysis code often looks something like this:

>>> index = numpy.where(sim["group"]==group_id)
>>> (sim["mass"][index]*sim["pos"][index]).mean(axis=0)

In this example, we obtained the center of mass of some group
specified by `group_id`. Such patterns are fine, but they are ugly to
look at and illegible when written out in long complicated chains.

For this reason, `pynbody` has the concept of sub-views of
simulations. Subviews behave exactly like a full simulation, but they
reference only a subset of the data of that full simulation.

This makes writing code which processes individual halos trivial; for
instance

>>> (sim["mass"]*sim["pos"]).mean(axis=0)

will find the centre of mass of whatever you throw at it. So if we had
some way of defining a *sub-view* of the whole simulation that refers
to just the halo that we want, called `halo`, then we could do
something like

>>> (halo["mass"]*halo["pos"]).mean(axis=0)

The indexing is taken care of under the hood and all the boilerplate
indexing code is gone. In this tutorial, we discuss some of the ways
that subviews and filters can be used in `pynbody`.

How do I create a simple subview?
---------------------------------

Quite simply, indexing or slicing operations commute with array-fetch
operations, i.e.

>>> subsim = sim[slice] # -> new SimSnap object 
>>> sim[slice]["x"] == sim["x"][slice] # -> True 
>>> subsim["x"] == sim["x"][slice] # -> True

`slice` here can literally be a python slice (e.g. `sim[22:73:2]`,
starting at the 22nd particle, finishing at the 72nd, taking every
second particle). It can also be a list of particle indices.

Or, more interestingly, it can be a `filter`. More on that below, but first...

Conceptually, everything is a pointer
-------------------------------------

Whenever you create a sub-view of your simulation, you should think of
it as a _pointer_ into, not a copy of, the original data.

Thus changing anything in the arrays of your sub-view makes the
corresponding change in the full data.

>>> subsim['x'][0] = 22

You can verify that the corresponding element of `sim['x']` has now changed

Filters
-------

:mod:`pynbody.filt` defines abstract filters which can be used in place of
index lists. For instance,

>>> radius = "1 kpc" # or might be a float if you already know your units
>>> centre = (0,0,0) # Take the origin for now
>>> sphere = sub[pynbody.filt.Sphere(radius, centre)]

`sphere` is now a standard subsim, as though you'd generated it with
your own index list.

>>> sphere["mass"].sum() # -> total mass in sphere
>>> sphere["pos"] # etc 


Under the hood, filters are very simple objects which have a
`__call__` method taking a simulation as the sole parameter, and
returning a boolean array representing whether particles are included
(`True`) or excluded (`False`) according to the given filter
object. The framework uses this to generate an index list and proceeds
as though you'd passed that index list in.

The benefits that this brings are:

  * Code clarity
  * Code factorization (guaranteed that the 'sphere' filter works,
    rather than having to recode a sphere index-list operation each
    time)
  * Future extensibility (e.g. we can just start having Sphere's
    accept quantities with units, rather than having to reimplement to
    include units in all code which pulls out spheres)
  * Parallelizability (the 'filter' objects can be trivially sent to
    another node, whereas an index list is irrelevant on another node)

Filters can be combined using the following operators, which can of course be chained together:

============  ========================
**Operator**  **Particle included if** 
============  ========================
`f1 & f2`     Filters `f1` and `f2` both include particle 
`f1 | f2`     Filter `f1` OR `f2` includes particle 
`~f`          Filter `f` does NOT include particle 
============  ========================

There are several filter classes currently implemented. Generic filter
classes work like band-pass filters and are conveniently named so. For example, 

>>> young = pynbody.filt.LowPass('age', '1 Gyr') 
>>> old = pynbody.filt.HighPass('age', '10 Gyr')
>>> intermediate = pynbody.filt.BandPass('age', '3 Gyr', '7 Gyr')

would create three filters that each select particles with different
value ranges for the `age` array.

Additionaly, several "geometric" filter classes are available. We
already saw :class:`~pynbody.filt.Sphere` used above, but
:class:`~pynbody.filt.Disc`, :class:`~pynbody.filt.Cuboid`, and two
convenience functions :func:`~pynbody.filt.SolarNeighborhood` and
:func:`~pynbody.filt.Annulus` are also available.


