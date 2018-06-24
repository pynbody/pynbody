.. bridge tutorial

.. _bridge_tutorial:

Tracing Particles between Snapshots
===================================

The :mod:`~pynbody.bridge` module has tools for connecting different
outputs which allows you to trace particles from one snapshot of the
simulation to another. Normally, matching up particle IDs between
different snapshots can be a pain, but :mod:`~pynbody.bridge` does all
of this for you transparently. All that is needed is to initialize a
:class:`~pynbody.bridge.Bridge` object that links two snapshots
together. Once connected, a bridge object called on a specific subset
of particles in one snapshot will trace these particles back (or
forward) to the second snapshot.

Basic usage
------------

Load the data: 

>>> f1 = pynbody.load(high_redshift_file)
>>> f2 = pynbody.load(low_redshift_file)

Load the halo catalogue:

>>> h_high_z = f1.halos() 

Create the bridge object:

>>> b = pynbody.bridge.OrderBridge(f1, f2) # Or a different class, see "Which class to use" below

``b`` is now an :class:`~pynbody.bridge.OrderBridge` object that links
the two outputs ``f1`` and ``f2`` together. Passing a ``SubSnap`` from
one of the two linked snapshots to ``b`` will return a ``SubSnap``
with the same particles in the *other* snapshot. So, if we want to see
where all the particles that were in halo 1 in the high-redshift
snapshot (``f1``) end up at low redshift (``f2``), we can simply do:

>>> h1_at_low_z = b(h[1]) 

``h1_at_low_z now`` contains the particles which were in h[1] in the high redshift output


Identifying halos between different outputs
-------------------------------------------

You may wish to work out how a halo catalogue maps onto a halo
catalogue for a different output. For this purpose a simple function,
match_catalog, is provided. Extending the example above,
this would be called as follows:

>>> cat = b.match_catalog()

``cat`` is now a numpy index array such that ``f1.halos()[i]`` is
(probably!) the major progenitor for ``f2.halos()[cat[i]]``.


Which class to use?
-------------------

For files where the particle ordering is static, so that the particle
with index i in the first snapshot also has index i in the second
snapshot, use the :class:`~pynbody.bridge.Bridge` class.

For files which can spawn new particles, and therefore have a monotonically
increasing particle ordering array (e.g. "iord" in gasoline), use the
:class:`~pynbody.bridge.OrderBridge` class.

Snapshot formats where the particle ordering can change are not currently supported.
