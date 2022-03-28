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

 In [2]: s = pynbody.load('testdata/g15784.lr.01024.gz')

This loads the snapshot ``s`` (make sure you use the correct path to
the ``testdata`` directory). Now we load the halos and center on the
main halo (see the :ref:`halo_tutorial` tutorial for more detailed
information on how to deal with halos):

.. ipython::

 In [3]: h = s.halos()

For later convenience, we can store the main halo in a separate
variable:

.. ipython::

 In [1]: h1 = h[1]


And perhaps check quickly how many particles of each type are identified there:

.. ipython::

 In [1]: print('ngas = %e, ndark = %e, nstar = %e\n'%(len(h1.gas),len(h1.dark),len(h1.star)))


The halos of ``s`` are now loaded in ``h`` and ``h[1]`` yields the
:class:`~pynbody.snapshot.SubSnap` of ``s`` that corresponds to
halo 1.

Centering on something interesting
----------------------------------

Several built-in functions (e.g. those that plot images and make
profiles) in pynbody like your data to be centered on a point of
interest.  The most straight-forward way to center your snapshot on a
halo is as follows:

.. ipython ::

 In [4]: pynbody.analysis.halo.center(h1,mode='hyb')
 Out [4]: <pynbody.transformation.GenericTranslation at 0x10a61e790>

We passed ``h[1]`` to the function
:func:`~pynbody.analysis.halo.center` to center the *entire* snapshot
on the largest halo. We specify the mode of centering using the
keyword ``mode`` - here, we used ``hyb``, which stands for hybrid: the
snapshot is first centered on the particle with the lowest potential,
and this guess is then refined using the *shrinking sphere* method
(see the documentation for :func:`~pynbody.analysis.halo.center` for
more details).

Suppose we now want to center only the contents of halo 5, leaving the
rest of the simulation untouched. This is no problem. Let's check
where a particle in halo 5 is, then shift it and try again. You'll
notice halo 1 doesn't move at all.

.. ipython ::

 In [4]: print(h[1]['pos'][0])

 In [4]: print(h[5]['pos'][0])

 In [4]: h5 = h[5]

 In [4]: my_h5_transform = pynbody.analysis.halo.center(h5, mode='hyb', move_all=False)

 In [4]: print(h[1]['pos'][0]) # should be unchanged

 In [4]: print(h5['pos'][0]) # should be changed

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

 In [6]: with pynbody.analysis.halo.center(h[5], mode='hyb'): print(h[5]['pos'][0])

 In [7]: print(h[5]['pos'][0])


Inside the ``with`` code block, ``h[5]`` is centered. The moment the block
exits, the transformation is undone -- even if the block exits with an
exception.


Taking even more control
------------------------

If you want to make sure that the coordinates which pynbody finds for
the center are reasonable before recentering, supply
:func:`~pynbody.analysis.halo.center` with the ``retcen`` keyword and
change the positions manually. This is useful for comparing the
results of different centering schemes, when accurate center
determination is essential. So lets repeat some of the previous steps
to illustrate this:

.. ipython::

 In [2]: s = pynbody.load('testdata/g15784.lr.01024.gz'); h1 = s.halos()[1];

 In [4]: cen_hyb = pynbody.analysis.halo.center(h1,mode='hyb',retcen=True)

 In [5]: cen_pot = pynbody.analysis.halo.center(h1,mode='pot',retcen=True)

 In [6]: print(cen_hyb)

 In [7]: print(cen_pot)

 In [7]: s['pos'] -= cen_hyb

In this case, we decided that the ``hyb`` center was better, so we use
it for the last step.

.. note:: When calling :func:`~pynbody.analysis.halo.center` without
          the ``retcen`` keyword, the particle velocities are also
          centered according to the mean velocity around the
          center. If you perform the centering manually, this is not done.
          You have to determine the bulk velocity separately using
          :func:`~pynbody.analysis.halo.vel_center`.


Making some images
------------------

Enough centering! We can take a look at what we have at the center
now, but to make things easier to interpret we convert to physical
units first:

.. ipython::

 In [5]: s.physical_units()

 @savefig snapshot_manipulation_fig1.png width=5in
 In [9]: pynbody.plot.image(h1.g, width=100, cmap='Blues')

Here's a slightly more complicated example showing the larger-scale
dark-matter distribution -- note that you can conveniently specify the
width as a string with a unit.

.. ipython::

 @savefig snapshot_manipulation_fig1_wide.png width=5in
 In [1]: pynbody.plot.image(s.d[pynbody.filt.Sphere('10 Mpc')], width='10 Mpc', units = 'Msol kpc^-2', cmap='Greys');

.. note:: see the :doc:`pictures` tutorial for more examples and help regarding images.


Aligning the Snapshot
---------------------

In this example, the disk seems to be aligned more or less face-on,
but let's say we want it edge-on:

.. ipython::

 In [12]: pynbody.analysis.angmom.sideon(h1, cen=(0,0,0))

 @savefig snapshot_manipulation_fig2.png width=5in
 In [12]: pynbody.plot.image(h1.g, width=100, cmap='Blues');


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
  instance, ``pynbody.analysis.angmom.sideon(h1)`` calculates the
  transforms for halo 1, but then applies them to the entire snapshot,
  unless you specifically ask otherwise.
  However, *core* routines (i.e. those that are not part of the
  ``pynbody.analysis`` module) typically operate on exactly what you
  ask them to, so ``s.g.rotate_x(90)`` rotates only the gas while
  ``s.rotate_x(90)`` rotates the entire simulation.

In the face-on orientation, we may wish to make a profile of the stars:

.. ipython::

 In [23]: ps = pynbody.analysis.profile.Profile(h1.s, min = 0.01, max = 50, type = 'log')

 In [25]: pylab.clf()

 In [25]: pylab.plot(ps['rbins'], ps['density']);

 In [26]: pylab.semilogy();

 In [28]: pylab.xlabel('$R$ [kpc]');

 @savefig snapshot_manipulation_fig3.png width=5in
 In [29]: pylab.ylabel('$\Sigma$ [M$_\odot$/kpc$^2$]');

We can also generate other profiles, like the rotation curve:

.. ipython::

 In [1]: pylab.figure()

 In [1]: pd = pynbody.analysis.profile.Profile(h1.d,min=.01,max=50, type = 'log')

 In [2]: pg = pynbody.analysis.profile.Profile(h1.g,min=.01,max=50, type = 'log')

 In [3]: p = pynbody.analysis.profile.Profile(h1,min=.01,max=50, type = 'log')

 In [4]: for prof, name in zip([p,pd,ps,pg],['total','dm','stars','gas']) : pylab.plot(prof['rbins'],prof['v_circ'],label=name)

 In [5]: pylab.xlabel('$R$ [kpc]');

 In [6]: pylab.ylabel('$v_{circ}$ [km/s]');

 @savefig vcirc_profiles.png width=5in
 In [5]: pylab.legend()

See the :doc:`profile` tutorial or the
:class:`~pynbody.analysis.profile.Profile` documentation for more
information on available options and other profiles that you can
generate.

We've only touched on the basic information that ``pynbody`` is able to
provide about your simulation snapshot. To learn a bit more about how
to get closer to your data, have a look at the :ref:`data-access`
tutorial.
