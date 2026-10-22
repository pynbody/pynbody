"""Running an analysis in several processes, which share snapshots and halo catalogues with the main process.

The user-facing part is :class:`ProcessPool` (also available as ``pynbody.ProcessPool``); see
:ref:`portable_parallelism` for an introduction.

Behind it, the main process acts as a server for the worker processes. Each worker starts with a
:class:`~pynbody.parallel.served.ServedSimSnap` made from the portable state of each snapshot (see
:meth:`~pynbody.snapshot.simsnap.SimSnap.get_portable_state`), and asks the
:class:`~pynbody.parallel.server.SnapshotServer` for anything else it needs. Three parts are kept separate, so that
each could be replaced, e.g. to work between machines:

* :mod:`~pynbody.parallel.transport` -- how arrays travel (on one machine, through shared memory);
* :mod:`~pynbody.parallel.channel` -- how requests and replies travel;
* the launching and pausing of workers, in :mod:`~pynbody.parallel.pool`.

.. versionadded:: 2.9.0

"""

from .pool import ProcessPool

__all__ = ['ProcessPool']
