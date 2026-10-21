"""Packing the objects given to a :class:`~pynbody.parallel.ProcessPool`, so that each worker can recreate them.

.. versionadded:: 2.9.0

"""

from __future__ import annotations

import warnings

import numpy as np

from .. import family
from ..halo import HaloCatalogue
from ..kdtree import KDTree
from ..snapshot import SimSnap
from ..snapshot.subsnap import FamilySubSnap


def pack_objects(objects, transport):
    """Express snapshots, halo catalogues, trees, arrays and other objects so that they can be sent to workers.

    Returns ``(packed, snapshots)``: *packed* is a picklable description of the objects, to be passed to
    :func:`unpack_objects` in a worker; *snapshots* lists the snapshots it involves, including those that halo
    catalogues belong to. The arrays involved are encoded by *transport*."""
    snapshots = []

    def snapshot_index(sim):
        if sim is not sim.ancestor:
            raise TypeError(f"{sim!r} is part of a snapshot. Only whole snapshots can be shared with a "
                            f"ProcessPool; share the whole snapshot, and select the part you need in your function.")
        for i, s in enumerate(snapshots):
            if s is sim:
                return i
        snapshots.append(sim)
        return len(snapshots) - 1

    def catalogue_base(h):
        """Return the index of the snapshot a catalogue belongs to, and the name of its family if it is of one"""
        base = h.base
        if isinstance(base, FamilySubSnap):
            return snapshot_index(base.ancestor), base._unifamily.name
        elif base is not base.ancestor:
            raise TypeError(f"{h!r} belongs to {base!r}, which is part of a snapshot. A halo catalogue can only be "
                            f"shared with a ProcessPool if it belongs to a whole snapshot, or to a family.")
        return snapshot_index(base), None

    # Snapshots are gathered first, so that a tree or catalogue belonging to one of them can refer to it,
    # whatever the order in which they were given
    for obj in objects:
        if isinstance(obj, SimSnap):
            snapshot_index(obj)
        elif isinstance(obj, HaloCatalogue):
            catalogue_base(obj)

    descriptions = []
    for obj in objects:
        if isinstance(obj, SimSnap):
            descriptions.append(('snapshot', snapshot_index(obj)))
        elif isinstance(obj, HaloCatalogue):
            descriptions.append(('halos', *catalogue_base(obj), transport.encode_state(obj.get_portable_state())))
        elif isinstance(obj, KDTree):
            owner = [i for i, s in enumerate(snapshots) if getattr(s, 'kdtree', None) is obj]
            if owner:
                descriptions.append(('tree_of_snapshot', owner[0]))
            else:
                descriptions.append(('tree', transport.encode_state(obj.get_portable_state())))
        elif isinstance(obj, np.ndarray):
            descriptions.append(('array', transport.encode(obj)))
        else:
            descriptions.append(('object', obj))

    snapshot_states = []
    for sim in snapshots:
        bytes_before = transport.bytes_copied
        snapshot_states.append(transport.encode_state(sim.get_portable_state()))
        copied = transport.bytes_copied - bytes_before
        if copied > 0:
            warnings.warn(f"Arrays of {sim!r} have been copied into shared memory to share them between "
                          f"processes, using an extra {copied / 2**20:.1f} MB until the pool is shut down. To avoid "
                          f"this, call enable_shared_arrays() on the snapshot before loading arrays or building "
                          f"its tree.", RuntimeWarning, stacklevel=3)

    return (snapshot_states, descriptions), snapshots


def unpack_objects(packed, transport, make_snapshot):
    """Recreate the objects packed by :func:`pack_objects`.

    Snapshots are recreated by calling ``make_snapshot(state, view_key)``, with their decoded portable state and
    the key by which the server knows them (see :class:`~pynbody.parallel.server.SnapshotServer`).

    Returns ``(objects, keep_alive)``. Halo catalogues refer to their snapshots only weakly, so *keep_alive*, a
    list of the snapshots (and families of them) involved, must be kept for as long as the objects are used."""
    snapshot_states, descriptions = packed
    snapshots = [make_snapshot(transport.decode_state(state), (i, None)) for i, state in enumerate(snapshot_states)]

    objects = []
    keep_alive = list(snapshots)
    for kind, *details in descriptions:
        if kind == 'snapshot':
            objects.append(snapshots[details[0]])
        elif kind == 'halos':
            snapshot_number, family_name, state = details
            base = snapshots[snapshot_number]
            if family_name is not None:
                base = base[family.get_family(family_name)]
                keep_alive.append(base)
            objects.append(HaloCatalogue.from_portable_state(transport.decode_state(state), base))
        elif kind == 'tree_of_snapshot':
            objects.append(snapshots[details[0]].kdtree)
        elif kind == 'tree':
            objects.append(KDTree.from_portable_state(transport.decode_state(details[0])))
        elif kind == 'array':
            objects.append(transport.decode(details[0]))
        else:
            objects.append(details[0])

    return tuple(objects), keep_alive
