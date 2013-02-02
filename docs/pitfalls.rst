.. pitfalls Common Pitfalls


Common Pitfalls
===============

I get errors like "Unknown units" or "Not convertible" from analysis or plotting routines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the great things about `pynbody` is that it takes care of units, but
if it can't figure out what units to use, everything is assumed to be
dimensionless. Some analysis functions then get grumpy, because
they're trying to be smart about using sensible units but the
information just isn't available. The simplest way to avoid this
situation is to make sure `pynbody` can work out the units for itself.

In particular, *for gasoline/PKD/tipsy users, make sure you have a
param file in the directory where you are analyzing a tipsy file, and
make sure that it defines dKpcUnit and dMsolUnit.*

Even if you are analyzing a DM only simulation, it's can be easier to play
along and assign units even though they weren't needed for the simulation.

I tried to make an image but got a generic-looking error spewed back at me
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most common causes of uninformative errors in the
:ref:`image-making process <pictures>` are that:

 1. there are no particles in the view you tried to generate; or
 2. all the particles are clustered in the central pixel. 

To tackle both of these issues in turn:

 1. The image is *always* centred on ``(0,0,0)``, so you need to offset
 the simulation before you start. The most common way to do this
 is with the function :func:`pynbody.analysis.halo.center`; see
 :ref:`snapshot_manipulation` for an introduction.

 2. The `width` keyword for the image function
 expects a floating point number in the current units of the
 snapshot. It also defaults to the number `10`, which may be
 very large compared to your snapshot, depending on the units you
 have adopted. That means either you should *specify* a width which
 is more appropriate (i.e. your call might look like
 ``pynbody.plot.sph.image(my_snapshot, width=0.01)``) or *convert
 to sensible units first*. The easiest way to do the latter is to call
 :func:`~pynbody.snapshot.SimSnap.physical_units` on your snapshot,
 e.g. ``s.physical_units()`` if your simulation is called ``s``. 


