"""The main process's side of a :class:`~pynbody.parallel.ProcessPool`: providing what workers ask for.

When a worker's :class:`~pynbody.parallel.served.ServedSimSnap` needs an array, derived array or KD-tree that it
does not have, it sends a request to the :class:`SnapshotServer`. The server obtains it in the original snapshot
-- loading, deriving or building it, once -- and replies with a description from which the worker installs it.

The server deals only with snapshots. How requests arrive is up to a channel (see :mod:`pynbody.parallel.channel`),
how arrays travel is up to an :class:`~pynbody.parallel.transport.ArrayTransport`, and how workers are paused is
up to whatever launched them. To add a new kind of request, add a method named ``request_<name>``.

.. versionadded:: 2.9.0

"""

from __future__ import annotations

import contextlib
import threading

from .. import family
from ..snapshot.portable import describe_array

#: The reply to a request to derive an array, if the worker should derive it itself
DERIVE_LOCALLY = 'derive-locally'


class SnapshotServer:
    """Provides arrays, derived arrays and KD-trees of snapshots to workers that request them.

    Snapshots are addressed by a *view key*, ``(snapshot_id, selection)``, where *snapshot_id* is the index of the
    snapshot in the list the server was given. *selection* identifies a part of the snapshot; at present only
    ``None``, the whole snapshot, is supported.
    """

    def __init__(self, snapshots, transport, pause_workers=contextlib.nullcontext):
        """Create a server for the given snapshots.

        Parameters
        ----------

        snapshots : list[SimSnap]
            The snapshots to serve, which should have shared arrays enabled if the transport uses shared memory.

        transport : ArrayTransport
            Encodes the arrays sent back to the workers.

        pause_workers : callable
            Returns a context manager, within which the workers are paused. This is used while the server does
            work that can use every CPU (building a tree, or deriving an array for the whole snapshot).
        """
        self._snapshots = list(snapshots)
        self._transport = transport
        self._pause_workers = pause_workers
        # Requests are handled one at a time, so that two workers asking for the same array get it loaded once
        self._lock = threading.Lock()

    def handle(self, name, args):
        """Carry out the request *name*, by calling the method ``request_<name>``, and return its reply"""
        method = getattr(self, 'request_' + name, None)
        if method is None:
            raise ValueError(f"Unknown request {name!r}")
        with self._lock:
            return method(*args)

    def request_load_array(self, view_key, name, family_name):
        """Load an array from disk, if it is not already in memory.

        Returns a list of descriptions of the array (see :meth:`_describe`), or None if it cannot be loaded."""
        sim, target = self._view(view_key, family_name)
        with sim.lazy_derive_off:
            try:
                target[name]
            except KeyError:
                return None
        return self._describe(sim, name, family_name)

    def request_derive_array(self, view_key, name, family_name):
        """Load or derive an array, if it is not already in memory.

        Returns a list of descriptions of the array (see :meth:`_describe`), or :data:`DERIVE_LOCALLY` if it
        cannot be obtained here, e.g. because it depends on an array that only the worker has."""
        sim, target = self._view(view_key, family_name)
        if not self._has_array(target, name):
            with self._pause_workers():
                try:
                    target[name]
                except KeyError:
                    return DERIVE_LOCALLY
        return self._describe(sim, name, family_name)

    def request_kdtree(self, view_key):
        """Build the snapshot's KD-tree, if it has not been built, and return its encoded portable state"""
        sim, _ = self._view(view_key, None)
        if not hasattr(sim, 'kdtree'):
            with self._pause_workers():
                sim.build_tree()
        return self._transport.encode_state(sim.kdtree.get_portable_state(include_particle_arrays=False))

    def _view(self, view_key, family_name):
        snapshot_id, selection = view_key
        if selection is not None:
            raise NotImplementedError("Only whole snapshots can be served at present")
        sim = self._snapshots[snapshot_id]
        target = sim if family_name is None else sim[family.get_family(family_name)]
        return sim, target

    @staticmethod
    def _has_array(target, name):
        with target.lazy_off:
            try:
                target[name]
            except KeyError:
                return False
        return True

    def _describe(self, sim, name, family_name):
        """Describe what the snapshot now holds by the given name, for the worker to install.

        Each description is a dictionary like those in a portable state (see
        :func:`~pynbody.snapshot.portable.describe_array`), with the array encoded for transport, the name of the
        array (which for a 1D slice like ``vx`` is that of the whole array) and the family it belongs to, or None
        for the whole snapshot. A request for one family may be answered with an array for the whole snapshot, and
        a request for the whole snapshot with arrays for each family that has one."""
        name = sim._array_name_1D_to_ND(name) or name
        if name in sim._arrays:
            found = [(None, sim._arrays[name], sim.is_derived_array(name))]
        else:
            by_family = sim._family_arrays.get(name, {})
            families = by_family.keys() if family_name is None else \
                [f for f in by_family if f.name == family_name]
            found = [(f, by_family[f], name in sim._family_derived_array_names[f]) for f in families]

        descriptions = []
        for fam, array, derived in found:
            description = describe_array(array, derived)
            description['data'] = self._transport.encode(description['data'])
            description['name'] = name
            description['family'] = None if fam is None else fam.name
            descriptions.append(description)
        return descriptions
