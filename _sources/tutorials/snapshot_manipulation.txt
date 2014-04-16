.. snapshot_manipulation tutorial

.. _snapshot_manipulation: 


Basic snapshot manipulation
===========================


Once you've :ref:`installed pynbody <pynbody-installation>`, you will
probably want to have a quick look at your simulation and maybe make a
pretty picture or two.

`Pynbody` includes some essential tools that allow you to quickly
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
much richer interactive environment over the vanilla `python`
interpreter. However, you can type exactly the same commands into
vanilla `python`; only the formatting will look slightly
different. For instance, the `ipython` prompt looks like 

::

  In [1]:


while the `python` prompt looks like 

::

   >>>


We highly recommend `ipython` for interactive data analysis. You should also
install `matplotlib <http://matplotlib.org/>`_ to generate the plots at the
end of the walkthrough (see the :ref:`pynbody-installation` documentation for
more details).

Once `ipython` and `matplotlib` are installed, you can start the
`ipython` shell with the ``--pylab`` flag to automatically load the
interactive plotting environment:

:: 

  [user@domain ~]$ ipython --pylab

  Python 2.7.2 |EPD 7.1-2 (64-bit)| (default, Jul 27 2011, 14:50:45) 
  Type "copyright", "credits" or "license" for more information.

  IPython 0.13.1 -- An enhanced Interactive Python.
  ?         -> Introduction and overview of IPython's features.
  %quickref -> Quick reference.
  help      -> Python's own help system.
  object?   -> Details about 'object', use 'object??' for extra details.

  Welcome to pylab, a matplotlib-based Python environment [backend: MacOSX].
  For more information, type 'help(pylab)'.

  In [1]: 


Now we can get started with the analysis. 


.. note:: Before you start make sure `pynbody` is properly
 installed. See :ref:`pynbody-installation` for more information. You
 will also need the standard `pynbody` test files, so that you can
 load the exact same data as used to write the tutorial. You need to
 download these separately here:
 <https://code.google.com/p/pynbody/downloads/list>
 (`testdata.tar.gz`). You can then extract them in a directory of your
 choice with ``tar -zxvf testdata.tar.gz``


Centering the snapshot
----------------------

The first step of any analysis is to load the data. Afterwards, we
will want to center it on the halo of interest (in this case the main
halo) to analyze its contents.


.. ipython::

 In [1]: import pynbody

 In [2]: s = pynbody.load('testdata/g15784.lr.01024.gz')

This loads the snapshot ``s`` (make sure you use the correct path to
the `testdata` directory). Now we load the halos and center on the
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

 In [1]: print 'ngas = %e, ndark = %e, nstar = %e\n'%(len(h1.gas),len(h1.dark),len(h1.star)) 
 
 In [4]: pynbody.analysis.halo.center(h1,mode='hyb')

The halos of ``s`` are now loaded in ``h`` and ``h[1]`` yields the
:class:`~pynbody.snapshot.SubSnap` of `s` that corresponds to
halo 1. We pass ``h[1]`` to the function
:func:`~pynbody.analysis.halo.center` to center the *entire* snapshot
on the largest halo. We specify the mode of centering using the
keyword ``mode`` - here, we used ``hyb``, which stands for hybrid: the
snapshot is first centered on the particle with the lowest potential,
and this guess is then refined using the `shrinking sphere` method
(see the documentation for :func:`~pynbody.analysis.halo.center` for
more details).

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
  
 In [6]: print cen_hyb

 In [7]: print cen_pot

 In [7]: s['pos'] -= cen_hyb

In this case, we decided that the `hyb` center was better, so we use
it for the last step.

.. note:: When calling :func:`~pynbody.analysis.halo.center` without the ``retcen`` keyword, the particle velocities are also centered according to the mean velocity around the center. If you do the centering manually, this is not done and you have to determine the bulk velocity separately using :func:`~pynbody.analysis.halo.vel_center`.

  
We can take a look at what we have at the center now, but to make
things easier to interpret we convert to physical units first:

.. ipython::

 In [5]: s.physical_units()
 
 @savefig snapshot_manipulation_fig1.png width=5in
 In [9]: pynbody.plot.image(h1.g, width=100, cmap='Blues');

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
but lets say we want it edge-on:

.. ipython::

 In [12]: pynbody.analysis.angmom.sideon(h1, cen=(0,0,0))

 @savefig snapshot_manipulation_fig2.png width=5in
 In [13]: pynbody.plot.image(h1.g, width=100, cmap='Blues');


Note that the function :func:`~pynbody.analysis.angmom.sideon` will
actually by default center the snapshot first, unless you feed it the
``cen`` keyword. We did that here since we already centered it
earlier. It then calculates the angular momentum vector in a sphere
around the center and rotates the snapshot such that the angular
momentum vector is parallel to the `y`-axis. If, instead, you'd like
the disk face-on, you can call the equivalent
:func:`pynbody.analysis.angmom.faceon`. Alternatively, if you
want to just rotate the snapshot by arbitrary angles, the
:class:`~pynbody.snapshot.SimSnap` class includes functions
:func:`~pynbody.snapshot.SimSnap.rotate_x`,
:func:`~pynbody.snapshot.SimSnap.rotate_y`,
:func:`~pynbody.snapshot.SimSnap.rotate_z` that rotate the snapshot
about the respective axes. We can use this to rotate the disk into a
face-on orientation:

.. ipython::

 In [21]: s.rotate_x(90)

.. note:: High-level snapshot manipulation functions defined in
  ``pynbody.analysis`` typically transform the *entire* simulation,
  even if you only pass in a :class:`~pynbody.snapshot.SubSnap`. This 
  is because you normally want to *calculate* the transform
  from a subset of particles, but *apply* the transform to the full
  simulation (e.g. when centering on a particular halo). So, for
  instance, ``pynbody.analysis.angmom.sideon(h1)`` calculates the
  transforms for halo 1, but then applies them to the entire snapshot.
  However, *core* routines (i.e. those that are not part of the
  ``pynbody.analysis`` module) typically operate on exactly what you 
  ask them to, so ``s.g.rotate_x(90)`` rotates only the gas while
  ``s.rotate_x(90)`` rotates the entire simulation.

In the face-on orientation, we may wish to make a profile of the stars: 

.. ipython:: 

 In [23]: ps = pynbody.analysis.profile.Profile(h1.s, min = 0.01, max = 50, type = 'log')
 
 In [25]: import matplotlib.pylab as plt

 In [25]: plt.clf()

 In [25]: plt.plot(ps['rbins'], ps['density']);

 In [26]: plt.semilogy();

 In [28]: plt.xlabel('$R$ [kpc]');

 @savefig snapshot_manipulation_fig3.png width=5in
 In [29]: plt.ylabel('$\Sigma$ [M$_\odot$/kpc$^2$]');

We can also generate other profile, like the rotation curve: 

.. ipython::

 In [1]: plt.figure()

 In [1]: pd = pynbody.analysis.profile.Profile(h1.d,min=.01,max=50, type = 'log')

 In [2]: pg = pynbody.analysis.profile.Profile(h1.g,min=.01,max=50, type = 'log')

 In [3]: p = pynbody.analysis.profile.Profile(h1,min=.01,max=50, type = 'log')

 In [4]: for prof, name in zip([p,pd,ps,pg],['total','dm','stars','gas']) : plt.plot(prof['rbins'],prof['v_circ'],label=name)

 In [5]: plt.xlabel('$R$ [kpc]');

 In [6]: plt.ylabel('$v_{circ}$ [km/s]');

 @savefig vcirc_profiles.png width=5in
 In [5]: plt.legend()

See the :doc:`profile` tutorial or the
:class:`~pynbody.analysis.profile.Profile` documentation for more
information on available options and other profiles that you can
generate. 

We've only touched on the basic information that `pynbody` is able to
provide about your simulation snapshot. To learn a bit more about how
to get closer to your data, have a look at the :ref:`data-access`
tutorial.


