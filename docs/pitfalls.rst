.. pitfalls Common Pitfalls


Common Pitfalls
===============

.. _paramfiles_are_good:

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

.. note:: Point 2. above should be fixed by commit a11df5af1f10. 
 


I'm trying to load a file which I'm sure is fine, but it says the format isn't recognized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you load a file with `pynbody` you don't have to specify what
format it is. This is quite handy when everything works. However, if
there is a minor problem with the file, `pynbody` may move onto trying
to interpret it as a different format and ultimately conclude it just
doesn't understand what's going on. At that point you'll get an unhelpful error
like this: ``File 'filename': format not understood or does not exist``.

To expose the underlying problem, you need to explicitly tell pynbody
what file format you think it is. For instance, if you have a tipsy
file, try replacing your ``pynbody.load(filename)`` with
``pynbody.tipsy.TipsySnap(filename)``. This will then show you the
actual error. Most likely it's to do with file permissions, or a
problem with a ``.param`` file. If at this point you can't see what's
going wrong, do `drop us a line
<https://groups.google.com/forum/?fromgroups#!forum/pynbody-users>`_.


I'm using a RAMSES snapshot, but when I try to open it I get bizarre errors about "No space left on device" or something similar
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This will happen if you are using the parallel loader of RAMSES
snapshots and your shared memory has filled up. Look in your /dev/shm
for files named "pynbody-*" -- there are probably many of them. If
your code exits gracefully, these files are deleted, but if your
session crashes or you kill your python shell for whatever reason,
these files will stick around. You should periodically delete them by
hand if they start to accumulate.
