.. bridge tutorial

.. _bridge_tutorial:

Tracing Particles between Snapshots
===================================

The :mod:`~pynbody.bridge` module has tools for connecting different
outputs which allows you to trace particles from one snapshot of the
simulation to another. Normally, matching up particle IDs between
different snapshots can be a pain, but :mod:`~pynbody.bridge` does all
of this for you transparently.

In pynbody, a :class:`~pynbody.bridge.Bridge` is an object that links two snapshots
together. Once connected, a bridge object called on a specific subset
of particles in one snapshot will trace these particles back (or
forward) to the second snapshot. Constructing bridges is also very straight-forward
in most cases. The easiest way is to try it in action.

Since the pynbody test data does not include a simulation with two different outputs,
we will borrow the ``test_gadget`` data from tangos to illustrate the point. Click
`here <ftp://ftp.star.ucl.ac.uk/app/tangos/tutorial_gadget.tar.gz>`_ to download.

Basic usage
------------

Load the data at high and low redshift:

.. ipython::

   In [1]: import pynbody

   In [2]: f1 = pynbody.load("tutorial_gadget/snapshot_018")

   In [3]: f2 = pynbody.load("tutorial_gadget/snapshot_020")

Verify the redshifts:

.. ipython::

   In [9]: "f1 redshift=%.2f; f2 redshift=%.2f"%(f1.properties['z'], f2.properties['z'])


Load the halo catalogue at low redshift:

.. ipython::

   In [10]: h2 = f2.halos()


Create the bridge object:

.. ipython::

   In [11]: b = f2.bridge(f1)


``b`` is now an :class:`~pynbody.bridge.Bridge` object that links
the two outputs ``f1`` and ``f2`` together. Note that you can either
bridge from ``f2`` to ``f1`` (as here) or from ``f1`` to ``f2`` and it
makes no difference at all to basic functionality: the bridge can be traversed
in either direction.

There are different
subclasses of bridge that implement the mapping back and forth in different ways,
and by default the :func:`~pynbody.snapshot.SimSnap.bridge`
method will attempt to choose the best
possible option, by inspecting the file formats. However, if you
prefer, you can explicitly instantiate the bridge for yourself (see
below).

Passing a ``SubSnap`` from
one of the two linked snapshots to ``b`` will return a ``SubSnap``
with the same particles in the *other* snapshot. So, if we want to see
where all the particles that are in halo 9 in the low-redshift
snapshot (``f1``) came from at low redshift (``f2``), we can simply do:

.. ipython::

   In [13]: progenitor_particles = b(h2[9])

``progenitor_particles`` now contains the particles which were in halos_1[9]
in the high redshift output.  This will have been achieved by matching the
unique particle indexes, also known in pynbody as the ``iord`` array. To verify,

.. ipython::

   In [14]: h2[9]['iord']

   In [15]: progenitor_particles['iord']

   In [15]: all(h2[9]['iord'] == progenitor_particles['iord'])

But of course the actual particle properties are different in the two cases,
being taken from the two snapshots, e.g.

.. ipython::

   In [17]: progenitor_particles['x']

   In [16]: h2[9]['x']


Identifying halos between different outputs
-------------------------------------------

You may wish to work out how a halo catalogue maps onto a halo
catalogue for a different output. For this purpose a simple function,
:func:`~pynbody.bridge.Bridge.match_catalog, is provided. Extending the example above,
this would be called as follows:

.. ipython::

   In [17]: cat = b.match_catalog()

   In [18]: cat


The ith element of the returned array indicates that ``f1.halos()[cat[i]]`` is
the major progenitor for ``f2.halos()[i]``. Negative
values indicate a halo that either does not exist (e.g. halo 0 in
SubFind catalogues) or that cannot be matched in the counterpart
catalogue.

If you want multiple matches per halo, e.g. to identify mergers, you
can usee :func:`~pynbody.bridge.Bridge.fuzzy_match_catalog`; see the reference documentation.




Which class to use?
-------------------

There is a built-in-logic which selects the best possible subclass of
:class:`~pynbody.bridge.Bridge` when you call the method
:func:`~pynbody.snapshot.SimSnap.bridge`.
However, you can equally well choose the bridge and its options
for yourself.

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
