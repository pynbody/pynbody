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
        before = self._array_keys(sim)
        with sim.lazy_derive_off:
            try:
                target[name]
            except KeyError:
                return None
        return self._describe(sim, name, family_name, before)

    def request_derive_array(self, view_key, name, family_name):
        """Load or derive an array, if it is not already in memory.

        Returns a list of descriptions of the array (see :meth:`_describe`), or :data:`DERIVE_LOCALLY` if it
        cannot be obtained here, e.g. because it depends on an array that only the worker has."""
        sim, target = self._view(view_key, family_name)
        # The snapshot's lock is taken before the workers are paused, so that they are not left paused while it
        # waits for another thread (e.g. the main thread, deriving something itself)
        with sim._get_array_lock:
            before = self._array_keys(sim)
            if not self._has_array(target, name):
                with self._pause_workers():
                    try:
                        target[name]
                    except KeyError:
                        return DERIVE_LOCALLY
        return self._describe(sim, name, family_name, before)

    def request_kdtree(self, view_key):
        """Build the snapshot's KD-tree, if it has not been built, and return its encoded portable state"""
        sim, _ = self._view(view_key, None)
        with sim._get_array_lock:
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

    @staticmethod
    def _array_keys(sim):
        """The (name, family) of every array in the snapshot, with family None for arrays of the whole snapshot.

        The 1D slices of 3D arrays (like ``x`` of ``pos``) are not included, since they come with their arrays."""
        keys = {(name, None) for name in sim._arrays}
        keys |= {(name, fam) for name, by_family in sim._family_arrays.items() for fam in by_family}
        return {(name, fam) for name, fam in keys
                if not SnapshotServer._is_slice(sim, name, fam)}

    @staticmethod
    def _is_slice(sim, name, fam):
        nd_name = sim._array_name_1D_to_ND(name)
        if nd_name is None:
            return False
        return nd_name in sim._arrays if fam is None else fam in sim._family_arrays.get(nd_name, {})

    def _describe(self, sim, name, family_name, before):
        """Describe the requested array, and any others that have appeared alongside it, for the worker to install.

        Arrays often appear together: some formats load several at once, and deriving one array may create others
        (e.g. SPH densities create smoothing lengths). The worker is given all those that have appeared since
        *before* (see :meth:`_array_keys`), saving it from asking for each, as well as the one requested.

        Each description is a dictionary like those in a portable state (see
        :func:`~pynbody.snapshot.portable.describe_array`), with the array encoded for transport, the name of the
        array (which for a 1D slice like ``vx`` is that of the whole array) and the family it belongs to, or None
        for the whole snapshot. A request for one family may be answered with an array for the whole snapshot, and
        a request for the whole snapshot with arrays for each family that has one."""
        name = sim._array_name_1D_to_ND(name) or name
        if name in sim._arrays:
            requested = [(name, None)]
        else:
            families = sim._family_arrays.get(name, {}).keys()
            requested = [(name, fam) for fam in families if family_name is None or fam.name == family_name]

        descriptions = []
        for array_name, fam in requested + sorted(self._array_keys(sim) - before - set(requested), key=str):
            if fam is None:
                array, derived = sim._arrays[array_name], sim.is_derived_array(array_name)
            else:
                array = sim._family_arrays[array_name][fam]
                derived = array_name in sim._family_derived_array_names[fam]
            description = describe_array(array, derived)
            description['data'] = self._transport.encode(description['data'])
            description['name'] = array_name
            description['family'] = None if fam is None else fam.name
            if array_name == 'smooth':
                # Whether these are the smoothing lengths the tree itself uses (see pynbody.sph)
                description['kdtree_derived_smoothing'] = getattr(sim, '_kdtree_derived_smoothing', False)
            descriptions.append(description)
        return descriptions
