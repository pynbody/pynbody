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
in your simulation, see the :doc:`tutorial </tutorials/bridge>`.

"""

from __future__ import annotations

import abc
import typing
import weakref

import numpy as np

from . import _bridge

if typing.TYPE_CHECKING:
    from .. import snapshot

class AbstractBridge(abc.ABC):
    """The abstract base class for bridges between two snapshots.

    For more information see the module documentation for :mod:`pynbody.bridge`, or the
    :doc:`bridge tutorial </tutorials/bridge>`.
    """

    def __init__(self, start: snapshot.SimSnap, end: snapshot.SimSnap):
        self._start = weakref.ref(start)
        self._end = weakref.ref(end)

    @abc.abstractmethod
    def __call__(self, s: snapshot.SubSnap) -> snapshot.SubSnap:
        """Map from a ``SubSnap`` at one end of the bridge, to the corresponding ``SubSnap`` at the other end"""


    def _get_ends(self):
        start = self._start()
        end = self._end()
        if start is None or end is None:
            raise RuntimeError("Stale reference to start or endpoint")
        return start, end

    def match_catalog(self, min_index=1, max_index=30, threshold=0.5, groups_1 = None, groups_2 = None,
                      use_family = None):
        """Given HaloCatalogues, identify the likely halo number in the second catalogue for each halo in the first.

        For example, if a bridge ``b`` links snapshot ``f1`` (high redshift) to ``f2`` (low redshift) and we set

        >>> cat = b.match_catalog()

        then ``cat`` is now a numpy index array such that f1.halos()[i] is the major progenitor for
        ``f2.halos()[cat[i]]``, assuming ``cat[i]`` is positive.

        ``cat[0:min_index+1]`` is set to ``-2``. Halos which cannot be matched because
        they have too few particles in common give the result ``-1``. This is determined
        by the given threshold fraction of particles in common (by default, 50%).

        Parameters
        ----------

        min_index : int
            The minimum halo number to be matched. Default is 1.
        max_index: int
            The maximum halo number to be matched. Default is 30.
        threshold : float
            The minimum fraction of particles in common for a match to be considered. Default is 0.5.
        groups_1 : pynbody.halo.HaloCatalogue
            The HaloCatalogue for the first snapshot. Default is None, in which case it is obtained from the start
            point of the bridge via ``SimSnap.halos()``.
        groups_2 : pynbody.halo.HaloCatalogue
            The HaloCatalogue for the second snapshot. Default is None, in which case it is obtained from the end
            point of the bridge via ``SimSnap.halos()``.
        use_family : str
            Only match particles of this family. Default is None, in which case all particles are matched.
            Setting this to a family name can be useful if matching between two different simulations where the
            relationship between DM particles is known, but perhaps the relationship between gas particles is not
            (e.g. a Ramses simulation where actually the gas 'particles' are cells).

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
        """Given HaloCatalogues, return multiple possible matches within the second catalogue for each halo in the first.

        For details about the parameters to this function, see the documentation for :func:`match_catalog`.

        Returns
        -------
        list
            A list of lists, where the i-th element of the list is a list of tuples. Each tuple contains the
            index of a halo in the second catalogue and the fraction of particles in the first halo that are
            in the second halo. The list is sorted in descending order of the fraction of particles in common.
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
        """Return a matrix with the number of particles transferred from ``groups_1`` to groups_2.

        For more information on the arguments to this function, see the documentation for :func:`match_catalog`.

        Normally, :func:`match_catalog` (or :func:`fuzzy_match_catalog`) are easier to use, but this routine
        provides the maximal information.

        Returns
        -------
        numpy.ndarray
            A matrix with the number of particles transferred from each halo in the first catalogue to each halo in the
            second.
        """

        if only_family is not None:
            if use_family!=only_family and use_family is not None:
                raise ValueError("use_family and only_family must be the same if both specified")
            use_family = only_family

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
            # Need to account for the fact that get_group_array(only_family) will return an array of length Nfamily
            # while restriction_start_indices are global indeces to the ancestor snapshot,
            # which can start anywhere depending on the family ordering ==> Need to offset them back to start at 0.
            restriction_start_indices -= start.ancestor._get_family_slice(only_family).start
            restriction_end_indices -= end.ancestor._get_family_slice(only_family).start

            g1 = groups_1.get_group_array(family=only_family)[restriction_start_indices]
            g2 = groups_2.get_group_array(family=only_family)[restriction_end_indices]

        if max_index is None:
            max_index = max(len(groups_1), len(groups_2))
        if min_index is None:
            min_index = min(g1.min(),g2.min())

        transfer_matrix = _bridge.match(g1, g2, min_index, max_index)

        return transfer_matrix

class OneToOneBridge(AbstractBridge):
    """Connects two snapshots with identical particle numbers and file layout.

    Particle i in the start point is the same as particle i in the end point. As such, bridging is essential trivial.
    """

    def __init__(self, start: snapshot.SimSnap, end: snapshot.SimSnap):
        if len(start) != len(end):
            raise ValueError("OneToOneBridge requires snapshots of the same length")
        super().__init__(start, end)

    def __call__(self, s):
        start, end = self._get_ends()

        if s.is_descendant(start):
            return end[s.get_index_list(start)]
        elif s.is_descendant(end):
            return start[s.get_index_list(end)]
        else:
            raise RuntimeError("Not a subview of either end of the bridge")

class OrderBridge(AbstractBridge):
    """Connects to snapshots that both have ``iord`` arrays (or similar) to identify particles.

    Particles ``i_start`` and ``i_end`` are defined to be the same if and only if
    ``start[order_array][i_start] == start[order_array][i_end]``.
    """

    def __init__(self, start, end, order_array="iord", monotonic=True, allow_family_change=False):
        """Initialise the ``OrderBridge``

        Parameters
        ----------

        start : snapshot.SimSnap
            The start point of the bridge

        end : snapshot.SimSnap
            The end point of the bridge

        order_array : str
            The name of the array that is used to identify particles. Default is ``iord``.

        monotonic : bool
            If ``True``, the order_array must be monotonically increasing in both ends of the bridge.
            Note that this is not checked for you. If ``False``, the bridging is slower but this is the failsafe option.

        allow_family_change : bool
            If ``True``, the bridge will allow particles to change family going from one end to the other of the
            bridge. Otherwise, it is assumed that the family of a particle is conserved.

        """
        super().__init__(start, end)
        self._order_array = order_array
        self.monotonic = monotonic
        self.allow_family_change = allow_family_change

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


def bridge_factory(a: snapshot.SimSnap, b: snapshot.SimSnap) -> AbstractBridge:
    """Create a bridge connecting the two specified snapshots.

    This function will determine the best type of :ref:`Bridge` to construct between the two snapshots, and return it.
    It is called by :func:`pynbody.snapshot.simsnap.SimSnap.bridge`.

    For more information see :doc:`the bridge tutorial </tutorials/bridge>`."""

    not_sure_error = "Don't know how to automatically bridge between two simulations of different formats. "\
                     "You will need to create your bridge manually by instantiating either the OneToOneBridge or "\
                     "OrderBridge class appropriately."

    from ..snapshot import gadget, gadgethdf, nchilada, ramses, tipsy
    a_top = a.ancestor
    b_top = b.ancestor

    if type(a_top) is not type(b_top):
        raise RuntimeError(not_sure_error)

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
        raise RuntimeError(not_sure_error)
