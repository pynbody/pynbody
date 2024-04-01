.. filters tutorial

.. _filters_tutorial

Subviews and Filters
====================

Central to pynbody is the ability to work with sub-views of the simulation, and treat them as if they were the full
simulation. There are two key concepts:

* **Sub-views** are slices of the simulation data that behave like the full simulation,
  but only reference a subset of the data. When you modify a sub-view, you modify the original data.
  Sub-views can be created using a slice, an list of particle indices, a particular particle family,
  or a *filter*.
* **Filters** are objects that can be used to select particles based on some criterion. They can be combined
  using logical operators to create complex selections. Filters can be applied to simulations to create a
  sub-view.


Creating a simple sub-view
--------------------------

We will use the testdata provided with pynbody to demonstrate how to create a sub-view of a simulation.
If you do not have the testdata, see :ref:`obtaining_testdata`.

First, let's load a simulation to play with:

.. ipython::

    In [1]: import pynbody

    In [2]: import numpy as np

    In [2]: f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")

    In [3]: f.physical_units()

Now we can create a sub-view of the simulation. These are represented by the class
:class:`~pynbody.snapshot.subsnap.SubSnap`, so we somtimes call them subsnaps. To create a subsnap, we can use a slice
or a list of particle indices. (We can also use a filter, which we will discuss later.)

.. ipython::

        In [4]: subsnap_slice = f[100:200:2]

        In [5]: subsnap_indexed = f[[0, 90, 200, 1000]]

These syntaxes are hopefully self-explanatory if you are familiar with Python and numpy indexing. ``subsnap_slice``
represents every second particle between the 100th and 200th particles, while ``subsnap_indexed`` represents the
particles at indices 0, 90, 200, and 1000.

You can verify that these sub-views behave like the full simulation:

.. ipython::

    In [6]: np.all(f['pos'][100] == subsnap_slice['pos'][0])
    Out[6]: True

    In [7]: np.all(f['pos'][[0, 90, 200, 1000]] == subsnap_indexed['pos'])
    Out[7]: True

This becomes useful if the sub-view is a physically-meaningful selection of particles. For example, pynbody loads
halos from various halo-finding algorithms as sub-views of the simulation.

.. ipython::

    In [8]: h = f.halos()

    In [9]: subsnap_halo0 = h[0]

    In [10]: (subsnap_halo0['mass'][:,np.newaxis]*subsnap_halo0['pos']).sum(axis=0)/subsnap_halo0['mass'].sum()

Actually mass-weighted mean values are such a common thing to want to do that pynbody provides a convenience function
to do them from any subsnap, called ``mean_by_mass``. So the above code can be replaced with:

.. ipython::

    In [10]: subsnap_halo0.mean_by_mass('pos')

In the above we have very quickly found the centre of mass of the first halo in the simulation. More information about
halo catalogues can be found in the :ref:`halo_tutorial`, but for now we just note that all halos returned by a
halo catalogue are a subsnap.

The relationship is two-way
---------------------------

A subsnap is not just a static view of the data; it is a pointer to the original data. This means that if you modify
the subsnap, you modify the original data. For example:

.. ipython::

    In [11]: subsnap_slice['pos'][0] = [1., 2., 3.]

    In [12]: f['pos'][100]
    Out[12]: SimArray([1., 2., 3.], 'kpc')

Similarly, if you modify the original data, the subsnap will reflect the change:

.. ipython::

    In [13]: f['pos'][100] = [4., 5., 6.]

    In [14]: subsnap_slice['pos'][0]
    Out[14]: SimArray([4., 5., 6.], 'kpc')



Filters
-------

:mod:`pynbody.filt` defines abstract filters which can be used in place of
index lists. For instance,

.. ipython::

   In [15]: cen = subsnap_halo0.mean_by_mass('pos')

   In [16]: sphere_filter = pynbody.filt.Sphere('200 kpc', cen)

   In [17]: sphere_view = f[sphere_filter]

   In [18]: f"DM, gas, star mass: {sphere_view.dm['mass'].sum():.1e}, \
      ....: {sphere_view.g['mass'].sum():.1e}, and {sphere_view.s['mass'].sum():.1e} Msol."

The above created a filter that selects all particles within 200 kpc of the centre of mass of the first halo, and then
used that filter to create a sub-view of the simulation. The mass of dark matter, gas, and stars within 200 kpc of the
halo's centre of mass was then calculated.

Filters can be combined using logical operators ``&``, ``|``, and ``~`` to create complex selections. For example,
to select all particles within 200 kpc of the centre but outside 25 kpc, you can use:

.. ipython::

    In [19]: sphere_filter_outer = pynbody.filt.Sphere('200 kpc', cen) \
       ....:                       & ~pynbody.filt.Sphere('25 kpc', cen)

    In [20]: sphere_outer_view = f[sphere_filter_outer]

    In [21]: f"DM, gas, star mass: \
       ....: {sphere_outer_view.dm['mass'].sum():.1e}, {sphere_outer_view.g['mass'].sum():.1e}, \
       ....: and {sphere_outer_view.s['mass'].sum():.1e} Msol."

Other than :class:`~pynbody.filt.Sphere`, there are several other filters available in pynbody including
:class:`~pynbody.filt.Disc`, :class:`~pynbody.filt.Cuboid`, and :class:`~pynbody.filt.Annulus`. Filters can also
be more abstract and select particles based on non-geometric criterion. For example, :class:`~pynbody.filt.LowPass`
selects particles with values below a certain threshold of a specified array, :class:`~pynbody.filt.HighPass` selects
particles with values above a certain threshold, and :class:`~pynbody.filt.BandPass` selects particles with values
within a certain range.

.. note::

    For a full list of available filters, see the :mod:`pynbody.filt` module documentation.

For example, to select all stars with age between 1 Gyr and 10 Gyr inside 25kpc:


.. ipython::

    In [22]: age_filter = pynbody.filt.BandPass('age', '1 Gyr', '10 Gyr')

    In [23]: age_sphere_filter = age_filter & pynbody.filt.Sphere('25 kpc', cen)

    In [24]: age_sphere_view = f.star[age_sphere_filter]

    In [25]: f"Stellar mass in range: {age_sphere_view['mass'].sum():.1e} Msol."


.. _filters_tutorial_performance_implications:

Performance implications
------------------------

Many filters are evaluated using OpenMP parallelism, so they can be very fast. Furthermore, if a KD tree has been
constructed for the simulation (via :meth:`~pynbody.snapshot.simsnap.SimSnap.build_tree`), then
:class:`~pynbody.filt.Sphere` and :class:`~pynbody.filt.Cuboid` filters automatically evaluate using that tree.
This can amount to a significant speedup for very large simulations, although the effect on smaller simulations is less
pronounced, especially if many CPU cores are available, making the brute force search pretty fast in the first place.
