.. snapshot_manipulation tutorial

.. _snapshot_manipulation:


Basic snapshot manipulation
===========================


Once you've :ref:`installed pynbody <pynbody-installation>`, you will
probably want to have a quick look at your simulation and maybe make a
pretty picture or two.

``Pynbody`` includes some essential tools that allow you to quickly
generate (and visualize) physically interesting quantities. Some of
the snapshot manipulation functions are included in the low-level
:class:`pynbody.snapshot.SimSnap` class, others can be found in two
analysis modules :mod:`pynbody.analysis.angmom` and
:mod:`pynbody.analysis.halo`. This brief walkthrough will show you
some of their capabilities that form the first steps of any simulation
analysis workflow and by the end you should be able to display a basic
image of your simulation.


Setting up the interactive environment
--------------------------------------

In this walkthrough (and in others found in this documentation) we
will use the `ipython <http://ipython.org>`_ interpreter which offers a
much richer interactive environment over the vanilla ``python``
interpreter. This is also the same as using a `Jupyter <https://jupyter.org>`_ notebook.
However, you can type exactly the same commands into
vanilla ``python``; only the formatting will look slightly
different.

.. note:: Before you start make sure ``pynbody`` is properly
 installed. See :ref:`pynbody-installation` for more information. You
 will also need the standard ``pynbody`` test files, so that you can
 load the exact same data as used to write the tutorial. You need to
 download these separately here:
 `testdata.tar.gz <http://star.ucl.ac.uk/~app/testdata.tar.gz>`_.
 You can then extract them in a directory of your
 choice with ``tar -zxvf testdata.tar.gz``


First steps
-----------

The first step of any analysis is to load the data. Afterwards, we
will want to center it on the halo of interest (in this case the main
halo) to analyze its contents.


.. ipython::

 In [1]: import pynbody

 In [1]: import pylab

 In [2]: s = pynbody.load('testdata/gasoline_ahf/g15784.lr.01024.gz')

This loads the snapshot ``s`` (make sure you use the correct path to
the ``testdata`` directory). Now we load the halos and center on the
main halo (see the :ref:`halo_tutorial` tutorial for more detailed
information on how to deal with halos):

.. ipython::

 In [3]: h = s.halos()

For later convenience, we can store the main halo in a separate
variable:

.. ipython::

 In [1]: main_halo = h[0]


And perhaps check quickly how many particles of each type are identified there:

.. ipython::

 In [1]: print('ngas = %e, ndark = %e, nstar = %e\n'%(len(main_halo.gas),len(main_halo.dark),len(main_halo.star)))


The halos of ``s`` are now loaded in ``h`` and ``h[0]`` yields the
:class:`~pynbody.snapshot.SubSnap` of ``s`` that corresponds to
halo 0.

.. note:: The halo numbers by default are those used by the halo finder, which (depending
          on your specific finder) may not start at zero, and may even be *random numbers*!
          You can see all the available halos using ``h.keys()``.

          Older versions of ``pynbody`` renumbered AHF halos to start at 1, regardless
          of the internal numbering used by AHF. This inconsistency has been fixed in
          version 2, but to get the same results as in the previous versions, you need to
          specifically request it. ``h = s.halos(halo_number='v1')`` provides
          this backwards-compatibility.

Quick-look at the data and units
--------------------------------

In pynbody, the 3D position array is always known as ``pos``, the velocity array as ``vel``,
and the mass array as ``mass``. The units of these arrays are accessible through the
``units`` attribute, and may be converted to something more useful using the ``in_units`` method.

.. ipython::

     In [1]: s['pos']

     In [2]: s['pos'].units

     In [3]: s['pos'].in_units('kpc')

     In [4]: s['pos'].in_units('Mpc')

     In [5]: s['pos'].in_units('Mpc a h**-1')

Note here that the ``a`` is the cosmological expansion factor, i.e. its appearance in a unit
indicates that the unit is comoving. The ``h`` is the Hubble parameter in units of 100 km/s/Mpc.

The mass array is also accessible in the same way:

.. ipython::

     In [6]: s['mass']

     In [7]: s['mass'].units

     In [8]: s['mass'].in_units('Msol')

For convenience, you can also convert the entire snapshot to physical units:

.. ipython::

     In [9]: s.physical_units()

     In [10]: s['pos']

For pynbody, the default units are kpc, km/s, and Msol, but you can also specify them directly:

.. ipython::

    In [11]: s.physical_units("Mpc", "km s^-1", "1e5 Msol")

    In [12]: s['pos']

    In [13]: s['vel']

    In [14]: s['mass']


For now, we will stick to the default units.

.. ipython::

    In [15]: s.physical_units()

For more information on the unit system, see the reference section on :ref:`units`.

.. _centering:

Centering on something interesting
----------------------------------

Several built-in functions (e.g. those that plot images and make
profiles) in pynbody like your data to be centered on a point of
interest.  The most straight-forward way to center your snapshot on a
halo is as follows:

.. ipython ::

 In [4]: pynbody.analysis.halo.center(main_halo)
 Out [4]: <pynbody.transformation.GenericTranslation at 0x10a61e790>

We passed ``h[1]`` to the function
:func:`~pynbody.analysis.halo.center` to center the *entire* snapshot
on the largest halo. The default centring uses the *shrinking sphere* method,
which normally gives a really stable and precise centre for galaxies and halos
(see the documentation for :func:`~pynbody.analysis.halo.center` for
more details).

Suppose we now want to center only the contents of halo 5, leaving the
rest of the simulation untouched. This is no problem. Let's check
where a particle in halo 5 is, then shift it and try again. You'll
notice halo 1 doesn't move at all.

.. ipython ::

 In [4]: h[1]['pos'][0]

 In [4]: h[5]['pos'][0]

 In [4]: h5 = h[5]

 In [4]: my_h5_transform = pynbody.analysis.halo.center(h5, move_all=False)

 In [4]: h[1]['pos'][0] # should be unchanged

 In [4]: h5['pos'][0] # should be changed

Note however that the data inside ``h5`` (or any halo) just *points*
to a subset of the data in the full simulation. So you now have an
inconsistent state where part of the simulation has been translated
and the rest of it is where it started out. For that reason, functions
that transform data return a ``Tranformation`` object that conveniently
allows you to undo the operation:

.. ipython ::

 In [5]: my_h5_transform.revert()

 In [5]: print(h5['pos'][0]) # back to where it started

 In [5]: print(h[1]['pos'][0]) # still hasn't changed, of course


In fact, there's a more pythonic and compact way to do this. Suppose
you want to process ``h[5]`` in some way, but be sure that the
centering is unaffected after you are done. This is the thing to do:

.. ipython ::

 In [6]: with pynbody.analysis.halo.center(h[5]):
    ...:     print("Position when inside with block: ", h[5]['pos'][0])
    ...: print("Position when outside with block: ", h[5]['pos'][0])


Inside the ``with`` code block, ``h[5]`` is centered. The moment the block
exits, the transformation is undone -- even if the block exits with an
exception.

Note that :func:`~pynbody.analysis.halo.center` also by default velocity-centers
the halo, based on the centre of mass velocity of the innermost particles.
You can turn this off by passing ``do_velocity=False``. For more
information, visit the documentation for :func:`~pynbody.analysis.halo.center`.

Making some images
------------------

Enough centering! We can take a look at what we have in the centre of
our box now, using :func:`pynbody.plot.sph.image`

.. ipython::

 @savefig snapshot_manipulation_fig1.png width=5in
 In [9]: pynbody.plot.sph.image(main_halo.g, width=100, cmap='Blues')

This has used one of pynbody's built-in plotting routines to make an
SPH-interpolated image. It automatically estimates smoothing lengths and
densities if needed, and stores them in the variables ``smooth`` and ``rho``
respectively. The return value from :func:`~pynbody.plot.sph.image` is a numpy
array of the pixel values, which you can then manipulate further if you wish.

Here's a slightly more complicated example showing the larger-scale
dark-matter distribution -- note that you can conveniently specify the
width as a string with a unit.

.. ipython::

 @savefig snapshot_manipulation_fig1_wide.png width=5in
 In [1]: pynbody.plot.image(s.d[pynbody.filt.Sphere('10 Mpc')],
    ...:                    width='10 Mpc', units = 'Msol kpc^-2',
    ...:                    cmap='Greys')

.. note:: see the :doc:`pictures` tutorial for more examples and help regarding images.
          Pynbody also has a companion package, `topsy <https://github.com/pynbody/topsy>`_,
          which enables real-time rendering of snapshots on a GPU. See its separate website
          for more information.

.. _aligning:

Aligning the Snapshot
---------------------

In the above example, the disk seems to be aligned more or less face-on,
but let's say we want it edge-on:

.. ipython::

 In [12]: pynbody.analysis.angmom.sideon(main_halo, cen=(0,0,0))

 @savefig snapshot_manipulation_fig2.png width=5in
 In [12]: pynbody.plot.image(main_halo.g, width=100, cmap='Blues');


Note that the function :func:`~pynbody.analysis.angmom.sideon` will
actually by default center the snapshot first, unless you feed it the
``cen`` keyword. We did that here since we already centered it
earlier. It then calculates the angular momentum vector in a sphere
around the center and rotates the snapshot such that the angular
momentum vector is parallel to the ``y``-axis. If, instead, you'd like
the disk face-on, you can call the equivalent
:func:`pynbody.analysis.angmom.faceon`. Alternatively, if you
want to just rotate the snapshot by arbitrary angles, the
:class:`~pynbody.snapshot.SimSnap` class includes functions
:func:`~pynbody.snapshot.SimSnap.rotate_x`,
:func:`~pynbody.snapshot.SimSnap.rotate_y`,
:func:`~pynbody.snapshot.SimSnap.rotate_z` that rotate the snapshot
about the respective axes.


We can use this to rotate the disk into a face-on orientation:

.. ipython::

 In [21]: s.rotate_x(90)

All of these transformations behave in the way that was specified for
centering. That is, you can revert them by using a ``with`` block or
by storing the transformation and applying the ``revert`` method
later.

.. note:: High-level snapshot manipulation functions defined in
  ``pynbody.analysis`` typically transform the *entire* simulation,
  even if you only pass in a :class:`~pynbody.snapshot.SubSnap`. This
  is because you normally want to *calculate* the transform
  from a subset of particles, but *apply* the transform to the full
  simulation (e.g. when centering on a particular halo). So, for
  instance, ``pynbody.analysis.angmom.sideon(main_halo)`` calculates the
  transforms for halo 1, but then applies them to the entire snapshot,
  unless you specifically ask otherwise.
  However, *core* routines (i.e. those that are not part of the
  ``pynbody.analysis`` module) typically operate on exactly what you
  ask them to, so ``s.g.rotate_x(90)`` rotates only the gas while
  ``s.rotate_x(90)`` rotates the entire simulation.



Where next?
-----------

* For more about *images*, see the :doc:`pictures` cookbook.
* For more about *profiles*, such as density profiles or rotation curves, see the :doc:`profile` walk-through.
* For more about the low-level data access facilities, see the :ref:`data-access`
  walk-through.
* For more about *halos*, see the :ref:`halos` cookbook.
* Or go back to the table of contents for all :ref:`tutorials`.
