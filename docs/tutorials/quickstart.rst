.. _quickstart:

Quick-start: first steps with pynbody
=====================================

``Pynbody`` includes some essential tools that allow you to retrieve, explore
and visualize physically interesting quantities. The most basic functionality
is found in a pair of classes:

* the :class:`~pynbody.snapshot.simsnap.SimSnap` class, which makes
  the data from a simulation snapshot available, as well as providing
  facilities to convert units and perform transformations;
* the :class:`~pynbody.halo.HaloCatalogue` class, which provides access to
  halo catalogues and the halos they contain.

But to make best use of ``pynbody`` it is also
necessary to use the :mod:`pynbody.analysis`  and :mod:`pynbody.plot` modules.
This brief walkthrough will show you some of the capabilities in these,
and subsequent walkthroughs expand in more detail.

In all walkthroughs in the ``pynbody`` documentation we
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

 Code snippets can be copied from this page and pasted into
 python, ipython or jupyter. Hover over the code and click the
 button that appears; the commands will be copied to your clipboard.


Loading snapshots and halos
---------------------------

The first step of any analysis is to load the data. Afterwards, we
will want to center it on the halo of interest (in this case the main
halo) to analyze its contents.


.. ipython::

 In [1]: import pynbody

 In [1]: import pylab

 In [2]: s = pynbody.load('testdata/gasoline_ahf/g15784.lr.01024.gz')

This loads the snapshot ``s`` (make sure you use the correct path to
the ``testdata`` directory). The object ``s`` is an instance of
:class:`~pynbody.snapshot.simsnap.SimSnap`, and handles loading data
on your behalf. Note that on the initial call to :class:`~pynbody.snapshot.load`,
only the header is loaded. Actual data will be loaded only when it is needed.

Now we load the halos:

.. ipython::

 In [3]: h = s.halos()

Note that ``h`` is now a :class:`~pynbody.halo.HaloCatalogue` object, containing the
halo catalogue which ``pynbody`` could locate for the snapshot ``s``. Generally, it is not
necessary to tell ``pynbody`` what format the halos are in or their location; it can infer
it in most situations.

For later convenience, we can store the first halo in a separate
variable. The particular snapshot we have loaded here is a zoom cosmological simulation,
and halo ``0`` contains the central galaxy.

.. ipython::

 In [1]: main_halo = h[0]

.. note:: The halo numbers by default are those used by the halo finder, which (depending
          on your specific finder) may not start at zero, and may even be *random numbers*!
          You can see all the available halos using ``h.keys()``.

          Older versions of ``pynbody`` renumbered AHF halos to start at 1, regardless
          of the internal numbering used by AHF. This inconsistency has been fixed in
          version 2, but to get the same results as in the previous versions, you need to
          specifically request it. ``h = s.halos(halo_number='v1')`` provides
          this backwards-compatibility.

We can check quickly how many particles of each type are identified there:

.. ipython::

 In [1]: print('ngas = %e, ndark = %e, nstar = %e\n'%(len(main_halo.gas),len(main_halo.dark),len(main_halo.star)))

``pynbody`` refers to different particle types as "families". Here, we have accessed the ``gas``, ``dark``
and ``star`` families of the halo. There are also convenient one-letter aliases for these
regularly-used families: ``.g``, ``.d`` and ``.s`` respectively.
And, as you might expect, the python ``len`` function returns the number of particles in each family.

We could similarly have applied similar code to the entire snapshot, or to any other halo:

.. ipython::

 In [1]: print('Whole snapshot ngas = %e, ndark = %e, nstar = %e\n'%(len(s.gas),len(s.dark),len(s.star)))

 In [1]: print('Halo 5 ngas = %e, ndark = %e, nstar = %e\n'%(len(h[5].gas),len(h[5].dark),len(h[5].star)))


.. seealso::
  * For a more in-depth look at loading snapshot data, see the :ref:`data-access` tutorial.

  * For more information on handling halos with ``pynbody``, start with the tutorial :ref:`halo_tutorial`.

Making some images
------------------

Let's skip straight to making some images. The following code will make a simple density
interpolation of the gas particles in the main halo.

.. ipython::

 In [8]: s.physical_units()

 In [9]: pynbody.analysis.center(main_halo)

 @savefig snapshot_manipulation_fig1.png width=5in
 In [10]: image_values = pynbody.plot.image(main_halo.gas, width=100, cmap='Blues')

This has used three of ``pynbody``'s routines:

1) :meth:`~pynbody.snapshot.SimSnap.physical_units` to convert the units of the snapshot to
   physical units (unless otherwise specified, this means kpc, Msol and km/s for distances
   masses and velocities respectively);
2) :meth:`pynbody.analysis.center` to center the halo on the central density peak of the halo;
3) :meth:`pynbody.plot.image` to make an SPH-interpolated image of ``main_halo`` gas particles.

The latter automatically estimates smoothing lengths and
densities if needed, even if these are not stored in the file explicitly.
The returned ``image_values`` from :func:`~pynbody.plot.sph.image` is a numpy
array of the pixel values, which you can then manipulate further if you wish.

Here's another example showing the larger-scale
dark-matter distribution -- note that you can conveniently specify the
width as a string with a unit. The ``units`` keyword is used to specify the units of the
output, and notice here that we have specified a mass per unit area, which
pynbody takes as an indication that we want a projected density map (rather than a slice
through z=0, which is what we obtained in the gas case above).

.. ipython::

 @savefig snapshot_manipulation_fig1_wide.png width=5in
 In [1]: pynbody.plot.image(s.d[pynbody.filt.Sphere('10 Mpc')],
    ...:                    width='10 Mpc', units = 'Msol kpc^-2',
    ...:                    cmap='Greys')

.. seealso::

          See the :doc:`pictures` tutorial for more examples and help regarding images.

          ``pynbody`` also has a companion package, `topsy <https://github.com/pynbody/topsy>`_,
          which enables real-time rendering of snapshots on a GPU. See its separate website
          for more information.

.. _aligning:

Aligning the Snapshot
---------------------

In the above example, the disk seems to be aligned more or less face-on. Pynbody images
are *always* in the x-y plane; if they are projected, then the z-axis is the line of sight.
To cut or project the simulation along another direction, we need to align it. For example,
to align the disk, we can use the :func:`~pynbody.analysis.sideon` function:

.. ipython:: python

 @suppress
 pylab.clf()

 pynbody.analysis.sideon(main_halo)

 @savefig snapshot_manipulation_fig2.png width=5in
 pynbody.plot.image(main_halo.g, width=100, cmap='Blues');

Note that the function :func:`~pynbody.analysis.sideon` also calls
:func:`~pynbody.analysis.center` to center the halo, so it doesn't matter if the
halo isn't centered when you start. It then calculates the
angular momentum vector in a sphere
around the center and rotates the snapshot such that the angular
momentum vector is parallel to the ``y``-axis. If, instead, you'd like
the disk face-on, you can call the equivalent
:func:`pynbody.analysis.faceon`.

.. note:: High-level snapshot manipulation functions defined in
  ``pynbody.analysis`` transform the *entire* simulation,
  even if you only pass in a subset of particles like a halo. That is why we could pass
  ``main_halo`` to :func:`~pynbody.analysis.center` but still plot
  *all* the dark matter particles in the simulation in the example in the previous
  section. The particles in ``main_halo`` were used to calculate the right
  center, but the transformation was applied to all particles. If this is not the
  behaviour you want, you can pass ``move_all = False`` to these routines, and only
  the particles you pass in will be transformed.

  By contrast, *core* routines (i.e. those that are not part of the
  ``pynbody.analysis`` module) always operate on exactly what you
  apply them to, so ``s.g.rotate_x(90)`` rotates only the gas while
  ``s.rotate_x(90)`` rotates the entire simulation.


.. seealso::

 See the next tutorial's section on :ref:`centering <centering>`
 and reference documentation for the :mod:`~pynbody.transformation` module for more
 information about how coordinate transformations are handled in pynbody, including
 how to revert back to the original orientation.


Quick-look at the data and units
--------------------------------

Most analyses require you to get closer to the raw data arrays, and ``pynbody`` makes these
readily accessible through a dictionary-like interface. The 3D position array is always known as ``pos``, the velocity array as ``vel``,
and the mass array as ``mass``. The units of these arrays are accessible through the
``units`` attribute, and may be converted to something more useful using the ``in_units`` method.

.. ipython::

     In [1]: s['pos']

     In [2]: s['pos'].units

Earlier on, we converted the snapshot to physical units. We can easily undo that and see the
data in its original units:

.. ipython::

     In [3]: s.original_units()

     In [4]: s['pos']


Equally, we can manually convert units to whatever we wish:

.. ipython::

     In [4]: s['pos'].in_units('Mpc')

     In [5]: s['pos'].in_units('Mpc a h**-1')

Note here that the ``a`` is the cosmological expansion factor, i.e. its appearance in a unit
indicates that the unit is comoving. The ``h`` is the Hubble parameter in units of 100 km/s/Mpc.
The :meth:`~pynbody.snapshot.SimSnap.in_units` method makes a copy of the array in the new units,
leaving the original array unchanged. There is also a :meth:`~pynbody.snapshot.SimSnap.convert_units`
method that changes the units of the array in-place.

Now let's convert the entire snapshot back to kpc, Msol and km/s, and check the units of the
``pos`` array again:

.. ipython::

     In [9]: s.physical_units()

     In [10]: s['pos']

Of course, ``vel`` and ``mass`` arrays can be handled in exactly the same way. Pynbody also
loads all the other arrays inside a snapshot, standardizing the names where possible. If no
standardized name is available, the array is loaded with the name it has in the snapshot file.

.. seealso::

    * For more information about loading snapshot data and units, see the :ref:`data-access` tutorial.
    * For in-depth information on the unit system, see the reference section on :ref:`units`.


Making a density profile
------------------------

Another component of ``pynbody``'s scientific analysis tools is the ability to make profiles of
any quantity. The :mod:`pynbody.analysis.profile` module is powerful and flexible, but here we
will simply make a simple density profile of the gas, dark matter, and stars in the main halo.

Remember that the halo is already centred on the origin. We can therefore make 3d density
profiles as follows:

.. ipython::

 In [1]: star_profile = pynbody.analysis.Profile(main_halo.s, min=0.2, max=50,
    ...:                                         type='log', nbins=50, ndim=3)

 In [2]: dm_profile = pynbody.analysis.Profile(main_halo.d, min=0.2, max=50,
    ...:                                       type='log', nbins=50, ndim=3)

 In [3]: gas_profile = pynbody.analysis.Profile(main_halo.g, min=0.2, max=50,
    ...:                                        type='log', nbins=50, ndim=3)

The ``min`` and ``max`` arguments specify the minimum and maximum radii of the profile, and the
``nbins`` argument specifies the number of bins. The ``type`` argument specifies the binning
scheme, which can be 'log', 'lin' or 'equaln'. Finally, the ``ndim`` argument specifies the
dimensionality. Note the use of the ``s``, ``d`` and ``g`` shortcuts for the star, dark matter
and gas families respectively.

Let's now plot the profiles:

.. ipython:: python

 @suppress
 pylab.clf()

 pylab.plot(star_profile['rbins'], star_profile['density'], 'r', label='Stars')
 pylab.plot(dm_profile['rbins'], dm_profile['density'], 'k', label='Dark Matter')
 pylab.plot(gas_profile['rbins'], gas_profile['density'], 'b', label='Gas')
 pylab.loglog()
 pylab.xlabel('r [kpc]')
 pylab.ylabel(r'$\rho$ [M$_\odot$/kpc$^3$]')

 @savefig snapshot_manipulation_denpro.png width=5in
 pylab.legend()


Where next?
-----------

This tutorial has shown you how to load a snapshot, access the data, make some simple images
and a density profile.

* In the :ref:`next tutorial <data-access>`, we will go into more depth on how to manipulate the
  data inside a snapshot.
* For more about *images*, see the :doc:`images` cookbook.
* For more about *profiles*, such as density profiles or rotation curves, see the :doc:`profile` walk-through.
* For more about the low-level data access facilities, see the :ref:`data-access`
  walk-through.
* For more about *halos*, see the :ref:`halos` cookbook.
* Or go back to the table of contents for all :ref:`tutorials`.
