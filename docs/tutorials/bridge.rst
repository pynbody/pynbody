.. bridge tutorial

.. _bridge_tutorial:

Tracing Particles between Snapshots
===================================

The :mod:`~pynbody.bridge` module has tools for connecting different
outputs which allows you to trace particles from one snapshot of the
simulation to another.

In pynbody, a :class:`~pynbody.bridge.Bridge` is an object that links two snapshots
together. Once connected, a bridge object called on a specific subset
of particles in one snapshot will trace these particles back (or
forward) to the second snapshot. Constructing bridges is also very straight-forward
in most cases.

Since the pynbody test data does not include a simulation with two different outputs,
we will borrow some test data from `tangos <https://pynbody.github.io/tangos/>`_ to illustrate the point.
Click `here <https://zenodo.org/records/10825178/files/tutorial_gadget4.tar.gz?download=1>`_ to download, or type

.. code:: bash

   $ wget https://zenodo.org/records/10825178/files/tutorial_gadget4.tar.gz?download=1 -O tutorial_gadget4.tar.gz
   $ tar -xvzf tutorial_gadget4.tar.gz


Basic usage
------------

Load the data at high and low redshift:

.. ipython::

   In [1]: import pynbody

   In [2]: f1 = pynbody.load("tutorial_gadget4/snapshot_033.hdf5")

   In [3]: f2 = pynbody.load("tutorial_gadget4/snapshot_035.hdf5")

Verify the redshifts:

.. ipython::

   In [9]: f"f1 redshift={f1.properties['z']:.2f}; f2 redshift={f2.properties['z']:.2f}"


Load the halo catalogue at low redshift:

.. ipython::

   In [10]: h2 = f2.halos()


Create the bridge object:

.. ipython::

   In [11]: b = f2.bridge(f1)


``b`` is now an :class:`~pynbody.bridge.Bridge` object that links
the two outputs ``f1`` and ``f2`` together. Note that you can either
bridge from ``f2`` to ``f1`` (as here) or from ``f1`` to ``f2``and it makes no difference at all to basic
functionality: the bridge can be traversed in either direction.

.. note::

    There are different subclasses of bridge that implement the mapping back and forth in different ways, and by default
    the:func:`~pynbody.snapshot.SimSnap.bridge` method will attempt to choose the best possible option, by inspecting the
    file formats. However, if you prefer, you can explicitly instantiate the bridge for yourself (see below).


Passing a ``SubSnap`` from one of the two linked snapshots to ``b`` will return a ``SubSnap`` with the same particles
in the *other* snapshot.  To take a really simple example, we might want to calculate the typical comoving distance
travelled by particles between the two snapshots. Without a bridge, this is hard; specifically, note that the following
gives the *wrong* answer:

.. ipython::

  In [30]: displacement = np.linalg.norm(f2['pos'] - f1['pos'], axis=1).in_units("Mpc") # <-- wrong thing to do

  In [31]: displacement.mean() # <-- will give wrong answer
  Out[31]: SimArray(2.6222425, dtype=float32, 'Mpc')

This seems like a very long way for a particle to have travelled on average between two quite closely spaced
snapshots — because it's wrong. Gadbget has re-ordered the particles between the two snapshots, the particle with index
0 in the first snapshot is not the same particle as the one with index 0 in the second snapshot. So the above answer
involves randomly shuffling particles. What we actually wanted to do was to trace the particles from one snapshot to
the other, and then calculate the distance travelled by each particle. This is what the bridge does:

.. ipython::

    In [33]: f2_particles_reordered = b(f1)

    In [35]: displacement = np.linalg.norm(f2_particles_reordered['pos'] - f1['pos'], axis=1).in_units("Mpc")

    In [36]: displacement.mean()
    Out[36]: SimArray(0.39596564, dtype=float32, 'Mpc')

This is the correct (and much more reasonable) answer.

Tracing subregions
------------------

Bridges are not just about correcting the order of particles for comparisons like this; we can also select subsets
of the full snapshot. If we want to see where all the particles that are in halo 9 in the low-redshift
snapshot (``f1``) came from at low redshift (``f2``), we can simply do:

.. ipython::

   In [13]: progenitor_particles = b(h2[9])

``progenitor_particles`` now contains the particles in snapshot 1 that will later collapse into halo 9 in snapshot 2.
To verify, we can explicitly check that pynbody has selected out the correct particles according to their unique
identifier (``iord``):

.. ipython::

   In [14]: h2[9]['iord']

   In [15]: progenitor_particles['iord']

   In [15]: all(h2[9]['iord'] == progenitor_particles['iord'])

But of course the actual particle properties are different in the two cases,
being taken from the two snapshots, e.g.

.. ipython::

   In [17]: progenitor_particles['x']

   In [16]: h2[9]['x']

We can now make a plot to see where the particles in halo 8 at low redshift were in the higher redshift snapshot:

.. ipython::

   In [17]: import matplotlib.pyplot as p

   In [18]: p.plot(h2[7]['x'], h2[7]['y'], 'b.', label=f"z={f2.properties['z']:.2f} halo 7")

   In [19]: p.plot(b(h2[7])['x'], b(h2[7])['y'], 'r.', label=f"Tracked to z={f1.properties['z']:.2f}")

   In [20]: p.ylim(27.25, 27.75); p.xlim(24.6, 25.2); p.gca().set_aspect('equal')

   In [21]: p.legend()

   @savefig tracing_particles.png width=6in
   In [22]: p.xlabel('x / code units'); p.ylabel('y / code units')

From this we can see that the particles in halo 7 at z=1.06 (blue) are more compact than the same particles
at z=1.35 (red), and that the comoving position of the halo centre has also drifted as expected from the earlier
calculations.

Identifying halos between different outputs
-------------------------------------------

.. versionchanged:: 2.0

    Interface for halo matching

    The methods :func:`~pynbody.bridge.AbstractBridge.match_halos`, :func:`~pynbody.bridge.AbstractBridge.fuzzy_match_halos`
    and an underlying method :func:`~pynbody.bridge.AbstractBridge.count_particles_in_common` were added in version 2.0,
    and should be preferred to older methods (:func:`~pynbody.bridge.AbstractBridge.match_catalog`,
    :func:`~pynbody.bridge.AbstractBridge.fuzzy_match_catalog`). The latter
    provide similar functionality but with an inconsistent interface; they are now deprecated and should not be used
    in new code.

You may wish to work out how a halo catalogue maps onto a halo catalogue for a different output. Just as with
particles, the ordering of halos can be expected to change between snapshots, so if we get a halo catalogue for
the earlier snapshot, we'll find halo 7 is not the same as halo 7 in the later snapshot:

.. ipython::

   In [23]: h1 = f1.halos()

   In [24]: h1[7]['pos'].mean(axis=0)

   In [25]: b(h2[7])['pos'].mean(axis=0)

A glance at the positions of these rough halo centres show they can't be the same set of particles.

To map correctly between halo catalogues at different redshifts, we can use
:func:`~pynbody.bridge.AbstractBridge.match_halos`:

.. ipython::

   In [26]: matching = b.match_halos(h2, h1)

   In [27]: matching[7]

Here, ``matching`` is a dictionary that maps from halo numbers in ``h2`` (the low redshift snapshot) to halo numbers in
``h1`` (the high redshift snapshot). The above code is telling
us that halo 7 in the low-redshift snapshot is the same as halo 8 in the high-redshift snapshot. Let's test that
graphically:

.. ipython::

  In [18]: p.plot(h1[8]['x'], h1[8]['y'], 'k.', label=f"z={f1.properties['z']:.2f} halo 8")

  @savefig tracing_particles_and_halo.png width=6in
  In [19]: p.legend()


As expected, the particles that make up halo 8 (black) in the high-redshift snapshot are almost coincident with those
that we tracked from halo 7 in the low-redshift snapshot (red). Some of the tracked halo 7 particles haven't yet
accreted, so it's smaller, but the centres are almost coincident.

We can also see if there were any mergers or transfer between different structures by calling
:func:`~pynbody.bridge.AbstractBridge.fuzzy_match_halos`:

.. ipython::

   In [28]: fuzzy_matching = b.fuzzy_match_halos(h2, h1)

   In [29]: fuzzy_matching[7]

This tells us that as well as halo 8, which contributed most of the particles, about 1.8% of the particles were
contributed by halo 770. Let's plot that too, for good measure:


.. ipython::

  In [18]: p.plot(h1[770]['x'], h1[770]['y'], 'y.', label=f"z={f1.properties['z']:.2f} halo 770")

  @savefig tracing_particles_and_halo_and_accretion.png width=6in
  In [19]: p.legend()

It shows up in yellow and, as expected, it looks like it's falling in.

.. note::

   Some halo finders generate a merger tree that can provide some of this information, in which case it is available
   through the properties of the halo catalogues themselves. (See :ref:`halo_tutorial` for more information on halo
   properties.) However, the bridge is a more general tool that can be used to trace any subset
   of particles between two snapshots, not just those that are part of a halo catalogue, and furthermore can be
   applied to snapshots which are not necessarily adjacent in time. It can also be used to match halos between
   different simulations, e.g. DMO and hydro runs.

   There can be an overwhelming amount of information returned by the bridge. To digest cosmological information,
   we recommend the use of pynbody's sister package, `tangos <https://pynbody.github.io/tangos/>`_.

Which class to use?
-------------------

There is a built-in-logic which selects the best possible subclass of
:class:`~pynbody.bridge.Bridge` when you call the method
:func:`~pynbody.snapshot.SimSnap.bridge`.
However, you can equally well choose the bridge and its options
for yourself, and sometimes :func:`~pynbody.snapshot.SimSnap.bridge` will tell you it can't decide what kind of bridge
to generate.

For files where the particle ordering is static, so that the particle
with index i in the first snapshot also has index i in the second
snapshot, use the :class:`~pynbody.bridge.Bridge` class, as follows: ::

   b = pynbody.bridge.Bridge(f1, f2)

For files which can spawn new particles, and therefore have a monotonically
increasing particle ordering array (e.g. "iord" in gasoline), use the
:class:`~pynbody.bridge.OrderBridge` class: ::

   b = pynbody.bridge.OrderBridge(f1, f2)

Snapshot formats where the particle ordering can change require a more
processor and memory intensive mapping algorithm to be used, which
you can enable by asking for it explicitly: ::

   b = pynbody.bridge.OrderBridge(f1, f2, monotonic=False)
