"""

bridge
======

The bridge module has tools for connecting different outputs. For instance,
it's possible to take a subview (e.g. a halo) from one snapshot and 'push'
it into the other. This is especially useful if the two snapshots are
different time outputs of the same simulation.

Once connected, bridge called on a specific subset of particles in
output1 will trace these particles back (or forward) to the output2,
enabling observing a change in their properties, such as position,
temperature, etc.

For a tutorial on how to use the bridge module to trace the particles
in your simulation, see the `bridge tutorial
<http://pynbody.github.io/pynbody/tutorials/bridge.html>`_.

"""


import weakref
import numpy as np
import math
from . import _bridge


class Bridge(object):

    """Generic Bridge class"""

    def __init__(self, start, end):
        self._start = weakref.ref(start)
        self._end = weakref.ref(end)
        assert len(start) == len(end)

    def is_same(i1, i2):
        """Returns true if the particle i1 in the start point is the same
        as the particle i2 in the end point."""
        return i1 == i2

    def __call__(self, s):
        """Given a subview of either the start or end point of the bridge,
        generate the corresponding subview of the connected snapshot"""
        start, end = self._get_ends()

        if s.is_descendant(start):
            return end[s.get_index_list(start)]
        elif s.is_descendant(end):
            return start[s.get_index_list(end)]
        else:
            raise RuntimeError, "Not a subview of either end of the bridge"

    def _get_ends(self):
        start = self._start()
        end = self._end()
        if start is None or end is None:
            raise RuntimeError, "Stale reference to start or endpoint"
        return start, end

    def match_catalog(self, min_index=1, max_index=30, threshold=0.5):
        """Given a Halos object groups_1, a Halos object groups_2 and a
        Bridge object connecting the two parent simulations, this identifies
        the most likely ID's in groups_2 of the objects specified in groups_1.

        Parameters min_index and max_index are the minimum and maximum halo
        numbers to be matched (in both ends of the bridge). If max_index is
        too large, the catalogue matching can take prohibitively long (it
        scales as max_index^2).

        This routine currently uses particle number as a proxy for mass, so that the
        main simulation data does not need to be loaded.

        If b links snapshot f1 (high redshift) to f2 (low redshift) and we set

          cat = b.match_catalog()

        then cat is now a numpy index array such that f1.halos()[i] is the
        major progenitor for f2.halos()[cat[i]], assuming cat[i] is positive.

        cat[0:min_index+1] is set to -2. Halos which cannot be matched because
        they have too few particles in common give the result -1. This is determined
        by the given threshold fraction of particles in common (by default, 50%).

        """
        start, end = self._get_ends()
        groups_1 = start.halos()
        groups_2 = end.halos()

        restriction_end = self(self(end)).get_index_list(end.ancestor)
        restriction_start = self(self(start)).get_index_list(start.ancestor)

        assert len(restriction_end) == len(
            restriction_start), "Internal consistency failure in match_catalog2"
        g1 = groups_1.get_group_array()[restriction_start]
        g2 = groups_2.get_group_array()[restriction_end]

        mass = _bridge.match(g1, g2, min_index, max_index)

        identification = np.argmax(mass, axis=1) + min_index
        frac_shared = np.array(mass[np.arange(
            len(identification)), identification - min_index], dtype=float) / mass.sum(axis=1)

        identification[
            (frac_shared != frac_shared) | (frac_shared < threshold)] = -1

        return np.concatenate(([-2] * min_index, identification))


class OrderBridge(Bridge):

    """An OrderBridge uses integer arrays in two simulations
    (start,end) where particles i_start and i_end are
    defined to be the same if and only if
    start[order_array][i_start] == start[order_array][i_end].

    If monotonic is True, order_array must be monotonically increasing
    in both ends of the bridge (and this is not checked for you). If
    monotonic is False, the bridging is slower but this is the
    failsafe option.
    """

    def __init__(self, start, end, order_array="iord", monotonic=True):
        self._start = weakref.ref(start)
        self._end = weakref.ref(end)
        self._order_array = order_array
        self.monotonic = monotonic

    def is_same(self, i1, i2):

        start, end = self._get_ends()
        return start[order_array][i1] == end[order_array][i2]

    def __call__(self, s):

        start, end = self._get_ends()

        if s.is_descendant(start):
            from_ = start
            to_ = end

        elif s.is_descendant(end):
            from_ = end
            to_ = start

        else:
            raise RuntimeError, "Not a subview of either end of the bridge"

        iord_to = np.asarray(to_[self._order_array]).view(np.ndarray)
        iord_from = np.asarray(s[self._order_array]).view(np.ndarray)

        if not self.monotonic:
            iord_map_to = np.argsort(iord_to)
            iord_map_from = np.argsort(iord_from)
            iord_to = iord_to[iord_map_to]
            iord_from = iord_from[iord_map_from]

        output_index = _bridge.bridge(iord_to, iord_from)

        if not self.monotonic:
            output_index = iord_map_to[output_index[np.argsort(iord_map_from)]]

        return to_[output_index]


def bridge_factory(a, b):
    """Create a bridge connecting the two specified snapshots. For
    more information see :ref:`bridge-tutorial`."""

    from ..snapshot import tipsy, gadget, ramses, nchilada, gadgethdf
    a_top = a.ancestor
    b_top = b.ancestor

    if type(a_top) is not type(b_top):
        raise RuntimeError, "Don't know how to automatically bridge between two simulations of different formats. You will need to create your bridge manually by instantiating either the Bridge or OrderBridge class appropriately."

    if (isinstance(a_top, tipsy.TipsySnap) or isinstance(a_top, nchilada.NChiladaSnap)):
        if "iord" in a_top.loadable_keys():
            return OrderBridge(a_top, b_top, monotonic=True)
        else:
            return Bridge(a_top, b_top)
    elif isinstance(a_top, gadget.GadgetSnap) or isinstance(a_top, gadgethdf.GadgetHDFSnap):
        return OrderBridge(a_top, b_top, monotonic=False)
    elif isinstance(a_top, ramses.RamsesSnap):
        if len(a.gas) > 0 or len(b.gas) > 0:
            raise RuntimeError, "Cannot bridge AMR gas cells"
        return OrderBridge(a_top, b_top, monotonic=False)
    else:
        raise RuntimeError, "Don't know how to automatically bridge between these simulations. You will need to create your bridge manually by instantiating either the Bridge or OrderBridge class appropriately."
