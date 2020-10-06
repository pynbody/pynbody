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
            raise RuntimeError("Not a subview of either end of the bridge")

    def _get_ends(self):
        start = self._start()
        end = self._end()
        if start is None or end is None:
            raise RuntimeError("Stale reference to start or endpoint")
        return start, end

    def match_catalog(self, min_index=1, max_index=30, threshold=0.5,
                      groups_1 = None, groups_2 = None,
                      use_family = None):
        """Given a Halos object groups_1, a Halos object groups_2 and a
        Bridge object connecting the two parent simulations, this identifies
        the most likely ID's in groups_2 of the objects specified in groups_1.

        If groups_1 and groups_2 are not specified, they are automatically obtained
        using the SimSnap.halos method.

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

        If use_family is specified, only particles from that family are cross-matched.
        This can be useful e.g. if matching between two different simulations where
        the relationship between DM particles is known, but perhaps the relationship
        between star particles is random.

        """
        fuzzy_matches = self.fuzzy_match_catalog(min_index, max_index, threshold, groups_1, groups_2, use_family)

        identification = np.zeros(max_index+1,dtype=int)

        for i,row in enumerate(fuzzy_matches):
            if len(row)>0:
                identification[i] = row[0][0]
            elif i<min_index:
                identification[i] = -2
            else:
                identification[i] = -1

        return identification

    def fuzzy_match_catalog(self, min_index=1, max_index=30, threshold=0.01,
                            groups_1 = None, groups_2 = None, use_family=None, only_family=None):
        """fuzzy_match_catalog returns, for each halo in groups_1, a list of possible
        identifications in groups_2, along with the fraction of particles in common
        between the two.

        Normally, match_catalog is simpler to use, but this routine offers greater
        flexibility for advanced users. The first entry for each halo corresponds
        to the output from match_catalog.

        If no identification is found, the entry is the empty list [].
        """

        transfer_matrix = self.catalog_transfer_matrix(min_index,max_index,groups_1,groups_2,use_family,only_family)

        output = [[]]*min_index
        for row in transfer_matrix:
            this_row_matches = []
            if row.sum()>0:
                frac_particles_transferred = np.array(row,dtype=float)/row.sum()
                above_threshold = np.where(frac_particles_transferred>threshold)[0]
                above_threshold = above_threshold[np.argsort(frac_particles_transferred[above_threshold])[::-1]]
                for column in above_threshold:
                    this_row_matches.append((column+min_index, frac_particles_transferred[column]))

            output.append(this_row_matches)

        return output

    def catalog_transfer_matrix(self, min_index=1, max_index=30, groups_1=None, groups_2=None,use_family=None,only_family=None):
        """Return a max_index x max_index matrix with the number of particles transferred from
        the row group in groups_1 to the column group in groups_2.

        Normally, match_catalog (or fuzzy_match_catalog) are easier to use, but this routine
        provides the maximal information."""

        start, end = self._get_ends()
        if groups_1 is None:
            groups_1 = start.halos()
        else:
            assert groups_1.base.ancestor is start.ancestor

        if groups_2 is None:
            groups_2 = end.halos()
        else:
            assert groups_2.base.ancestor is end.ancestor

        if use_family:
            end = end[use_family]
            start = start[use_family]

        restricted_start_particles = self(self(start)) # map back and forth to get only particles that are held in common
        restricted_end_particles = self(restricted_start_particles) # map back to start in case a reordering is required

        restriction_end_indices = restricted_end_particles.get_index_list(end.ancestor)
        restriction_start_indices = restricted_start_particles.get_index_list(start.ancestor)

        assert len(restriction_end_indices) == len(
            restriction_start_indices), "Internal consistency failure in catalog_transfer_matrix: particles supposedly common to both simulations have two different lengths"


        if only_family is None:


            g1 = groups_1.get_group_array()[restriction_start_indices]
            g2 = groups_2.get_group_array()[restriction_end_indices]

        else:

            g1 = groups_1.get_group_array(family=only_family)[restriction_start_indices]
            g2 = groups_2.get_group_array(family=only_family)[restriction_end_indices]

        if max_index is None:
            max_index = max(len(groups_1), len(groups_2))
        if min_index is None:
            min_index = min(g1.min(),g2.min())

        transfer_matrix = _bridge.match(g1, g2, min_index, max_index)

        return transfer_matrix


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

    def __init__(self, start, end, order_array="iord", monotonic=True, allow_family_change=False):
        self._start = weakref.ref(start)
        self._end = weakref.ref(end)
        self._order_array = order_array
        self.monotonic = monotonic
        self.allow_family_change = allow_family_change

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
            raise RuntimeError("Not a subview of either end of the bridge")

        iord_to = np.asarray(to_[self._order_array]).view(np.ndarray)
        iord_from = np.asarray(s[self._order_array]).view(np.ndarray)

        if not self.monotonic:
            iord_map_to = np.argsort(iord_to)
            iord_map_from = np.argsort(iord_from)
            iord_to = iord_to[iord_map_to]
            iord_from = iord_from[iord_map_from]

        output_index, found_match = _bridge.bridge(iord_to, iord_from)

        if not self.monotonic:
            output_index = iord_map_to[output_index[np.argsort(iord_map_from)][found_match]]
        else:
            output_index = output_index[found_match]

        if self.allow_family_change:
            new_family_index = to_._family_index()[output_index]

            # stable sort by family:
            output_index = output_index[np.lexsort((new_family_index,))]

        return to_[output_index]


def bridge_factory(a, b):
    """Create a bridge connecting the two specified snapshots. For
    more information see :ref:`bridge-tutorial`."""

    from ..snapshot import tipsy, gadget, ramses, nchilada, gadgethdf
    a_top = a.ancestor
    b_top = b.ancestor

    if type(a_top) is not type(b_top):
        raise RuntimeError("Don't know how to automatically bridge between two simulations of different formats. You will need to create your bridge manually by instantiating either the Bridge or OrderBridge class appropriately.")

    if (isinstance(a_top, tipsy.TipsySnap) or isinstance(a_top, nchilada.NchiladaSnap)):
        if "iord" in a_top.loadable_keys():
            return OrderBridge(a_top, b_top, monotonic=True)
        else:
            return Bridge(a_top, b_top)
    elif isinstance(a_top, gadget.GadgetSnap) or isinstance(a_top, gadgethdf.GadgetHDFSnap):
        return OrderBridge(a_top, b_top, monotonic=False, allow_family_change=True)
    elif isinstance(a_top, ramses.RamsesSnap):
        if len(a.gas) > 0 or len(b.gas) > 0:
            raise RuntimeError("Cannot bridge AMR gas cells")
        return OrderBridge(a_top, b_top, monotonic=False)
    else:
        raise RuntimeError("Don't know how to automatically bridge between these simulations. You will need to create your bridge manually by instantiating either the Bridge or OrderBridge class appropriately.")
