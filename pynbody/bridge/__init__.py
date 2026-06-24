"""
Tools for connecting different simulation snaphots

The bridge module allows you to take a subview (e.g. a halo) from one snapshot and 'push' it into the other. This is
especially useful if the two snapshots are different time outputs of the same simulation, or closely-related
simulations (e.g. one DMO and one hydro).

Once connected, bridge called on a specific subset of particles in output1 will trace these particles back (or
forward) to the output2, enabling observing a change in their properties, such as position, temperature, etc.

For an introduction on how to use bridges, see the :doc:`tutorial </tutorials/bridge>`.

"""

from __future__ import annotations

import abc
import typing
import warnings
import weakref

import numpy as np

from .. import family, util
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

    def match_halos(self, halos_1, halos_2, /, threshold=0.5, use_family=None, fill_value=-1, use_halo_indexes=False):
        """Given halo catalogues, identify the likely halo number in the second catalogue for each halo in the first.

        For example, if a bridge ``b`` links snapshot ``f1`` (high redshift) to ``f2`` (low redshift) and we perform:

        >>> h1 = f1.halos()
        >>> h2 = f2.halos()
        >>> cat = b.match_halos(h1, h2)

        then ``cat`` is now a dictionary such that f1.halos()[i] is the major progenitor for``f2.halos()[cat[i]]``,
        assuming ``cat[i]`` is positive.

        Halos which cannot be matched because they have too few particles in common give the result ``fill_value``,
        default -1. This is determined by the given threshold fraction of particles in common (specified by
        ``threshold``, default 50%)

        Parameters
        ----------

        halos_1 : pynbody.halo.HaloCatalogue
            The HaloCatalogue for the first snapshot.

        halos_2 : pynbody.halo.HaloCatalogue
            The HaloCatalogue for the second snapshot.

        threshold : float
            The minimum fraction of particles in common for a match to be considered. Default is 0.5.

        use_family : str
            Only match particles of this family. Default is None, in which case all particles are matched.
            Setting this to a family name can be useful if matching between two different simulations where the
            relationship between DM particles is known, but perhaps the relationship between gas particles is not
            (e.g. a Ramses simulation where actually the gas 'particles' are cells)

        use_halo_indexes : bool
            If True, instead of returning a dictionary mapping to halo numbers, return a numpy array that matches
            halo IDs (which are zero-based indexes into the full list of halos; for more information see
            the nomenclature guide in the documentation for :class:`~pynbody.halo.HaloCatalogue`. The default is
            False, so that halo numbers are used throughout.

        fill_value : int
            The value to use for halos that cannot be matched. Default is -1.

        """

        self._check_compatible_halo_catalogues(halos_1, halos_2)

        particles_in_common_matrix = self.count_particles_in_common(halos_1, halos_2, use_family=use_family)

        highest_commonality_index = particles_in_common_matrix.argmax(axis=1)
        highest_commonality = particles_in_common_matrix[np.arange(particles_in_common_matrix.shape[0]),
        highest_commonality_index]
        # return nan for zero division
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_commonality = highest_commonality / particles_in_common_matrix.sum(axis=1)
        frac_commonality[~np.isfinite(frac_commonality)] = 0

        invalid_matches = frac_commonality < threshold

        if use_halo_indexes:
            result = highest_commonality_index
            result[invalid_matches] = fill_value
            return result
        else:
            mapper_1 = halos_1.number_mapper
            mapper_2 = halos_2.number_mapper

            sources = mapper_1.index_to_number(np.arange(len(halos_1)))
            destinations = mapper_2.index_to_number(highest_commonality_index)
            destinations[invalid_matches] = fill_value

            return dict(zip(sources, destinations))

    def _check_compatible_halo_catalogues(self, halos_1, halos_2):
        if halos_1.base.ancestor is not self._start().ancestor or halos_2.base.ancestor is not self._end().ancestor:
            raise ValueError("The HaloCatalogues must be based on the correct snapshots")

    def fuzzy_match_halos(self, halos_1, halos_2, /, threshold=0.01, use_family=None,
                          use_halo_indexes=False) -> dict[int, list[tuple[int, float]]]:
        """Given halo catalogues, return possible matches within the second catalogue for each halo in the first.

        For details about the parameters to this function, see the documentation for :func:`match_catalog`.

        Parameters
        ----------

        halos_1 : pynbody.halo.HaloCatalogue
            The HaloCatalogue for the source snapshot

        halos_2 : pynbody.halo.HaloCatalogue
            The HaloCatalogue for the destination snapshot

        threshold : float
            The minimum fraction of particles in common for a match to be considered. Default is 0.01.

        use_family : str
            Only match particles of this family. Default is None, in which case all particles are matched.

        use_halo_indexes : bool
            If True, instead of returning halo numbers, return halo indexes. Default is False.

        Returns
        -------
        dict[int, list[tuple[int, float]]]
            Maps from halo number in the first catalogue to a list of tuples. Each tuple contains the halo number in the
            second catalogue and the fraction of particles in the first halo that are in the second halo. If
            ``use_halo_indexes`` is True, the halo numbers are replaced by halo indexes.
        """

        self._check_compatible_halo_catalogues(halos_1, halos_2)

        def map_index_to_output(index, for_halos):
            if use_halo_indexes:
                return index
            else:
                return for_halos.number_mapper.index_to_number(index)

        particles_in_common_matrix = self.count_particles_in_common(halos_1, halos_2, use_family=use_family)

        output = {}

        for source_index, row in enumerate(particles_in_common_matrix):
            if source_index >= len(halos_1):
                # count_particles_in_common returns a square matrix, so we might run over the end of the first catalogue
                break

            this_row_matches = []
            row_sum = row.sum()
            if row_sum > 0:
                frac_particles_transferred = row / row_sum
                above_threshold = np.where(frac_particles_transferred > threshold)[0]
                above_threshold = above_threshold[np.argsort(frac_particles_transferred[above_threshold])[::-1]]
                for column in above_threshold:
                    this_row_matches.append((map_index_to_output(column, halos_2), frac_particles_transferred[column]))

            output[map_index_to_output(source_index, halos_1)] = this_row_matches

        return output

    def count_particles_in_common(self, halos_1, halos_2, /, max_num_halos=None, use_family=None) -> np.ndarray:
        """Return a matrix with the number of particles transferred from ``groups_1`` to groups_2.

        Normally, :func:`match_catalog` (or :func:`fuzzy_match_catalog`) are easier to use, but this routine
        provides the maximal information.

        .. warning::
            This routine returns results in terms of halo indexes, rather than halo numbers. This is because halo
            indexes are guaranteed to be continuous starting at zero, while halo numbers may have gaps. If you need
            to convert from halo indexes to halo numbers, you can use :meth:`~pynbody.halo.HaloCatalogue.number_mapper`
            attribute of :class:`~pynbody.halo.HaloCatalogue`. Alternatively, use the :func:`match_catalog` method
            which returns results in terms of halo numbers.

        Parameters
        ----------
        halos_1 : pynbody.halo.HaloCatalogue
            The HaloCatalogue for the first snapshot (the one at the start of the bridge).
        halos_2 : pynbody.halo.HaloCatalogue
            The HaloCatalogue for the second snapshot (the one at the end of the bridge).
        max_num_halos : int, optional
            The maximum number of halos
        use_family : str
            Only match particles of this family. Default is None, in which case all particles are matched.

        Returns
        -------
        numpy.ndarray
            A matrix with the number of particles transferred from each halo in the first catalogue to each halo in the
            second. The size of the matrix is determined by the maximum number of halos in either catalogue, or
            by the value of ``max_num_halos`` if specified.

        """

        self._check_compatible_halo_catalogues(halos_1, halos_2)

        start, end = self._get_ends()

        if use_family:
            end = end[use_family]
            start = start[use_family]

        restricted_start_particles = self(
            self(start))  # map back and forth to get only particles that are held in common
        restricted_end_particles = self(
            restricted_start_particles)  # map back to start in case a reordering is required

        restriction_end_indices = restricted_end_particles.get_index_list(end.ancestor)
        restriction_start_indices = restricted_start_particles.get_index_list(start.ancestor)

        assert len(restriction_end_indices) == len(restriction_start_indices), \
            ("Internal consistency failure in catalog_transfer_matrix: particles supposedly "
             "common to both simulations have two different lengths")

        # Need to account for the fact that get_group_array(only_family) will return an array of length Nfamily
        # while restriction_start_indices are global indeces to the ancestor snapshot,
        # which can start anywhere depending on the family ordering ==> Need to offset them back to start at 0.
        restriction_start_indices -= start.ancestor._get_family_slice(use_family).start
        restriction_end_indices -= end.ancestor._get_family_slice(use_family).start

        g1 = halos_1.get_group_array(family=use_family, use_index=True)[restriction_start_indices]
        g2 = halos_2.get_group_array(family=use_family, use_index=True)[restriction_end_indices]

        if max_num_halos is None:
            max_num_halos = max(len(halos_1), len(halos_2))

        transfer_matrix = _bridge.match(g1, g2, 0, max_num_halos)

        return transfer_matrix

    @util.deprecated("match_catalog is deprecated; use match_halos instead")
    def match_catalog(self, min_index=1, max_index=30, threshold=0.5, groups_1=None, groups_2=None,
                      use_family=None):
        """Deprecated alternative to :func:`match_halos`. Use that method instead."""
        fuzzy_matches = self.fuzzy_match_catalog(min_index, max_index, threshold, groups_1, groups_2, use_family)

        identification = np.zeros(max_index + 1, dtype=int)

        for i, row in enumerate(fuzzy_matches):
            if len(row) > 0:
                identification[i] = row[0][0]
            elif i < min_index:
                identification[i] = -2
            else:
                identification[i] = -1

        return identification

    @util.deprecated("fuzzy_match_catalog is deprecated; use fuzzy_match_halos instead")
    def fuzzy_match_catalog(self, min_index=1, max_index=30, threshold=0.01,
                            groups_1=None, groups_2=None, use_family=None, only_family=None):
        """Deprecated alternative to :func:`fuzzy_match_halos`. Use that method instead."""

        transfer_matrix = self.catalog_transfer_matrix(min_index, max_index, groups_1, groups_2, use_family,
                                                       only_family)

        output = [[]] * min_index
        for row in transfer_matrix:
            this_row_matches = []
            if row.sum() > 0:
                frac_particles_transferred = np.array(row, dtype=float) / row.sum()
                above_threshold = np.where(frac_particles_transferred > threshold)[0]
                above_threshold = above_threshold[np.argsort(frac_particles_transferred[above_threshold])[::-1]]
                for column in above_threshold:
                    this_row_matches.append((column + min_index, frac_particles_transferred[column]))

            output.append(this_row_matches)

        return output

    @util.deprecated("catalog_transfer_matrix is deprecated; use count_particles_in_common instead")
    def catalog_transfer_matrix(self, min_index=1, max_index=30, groups_1=None, groups_2=None,
                                use_family=None, only_family=None):
        """Deprecated interface to :func:`count_particles_in_common`. Use that method instead."""

        # the 'index' naming of the parameter is misleading -- in pynbody v1, the distinction between number and
        # index had not been drawn. In fact, by pynbody v2 nomenclature, this function accepts halo numbers, so
        # we rename internally to prevent confusion in the compatibility code below.

        min_number = min_index
        max_number = max_index

        del min_index, max_index

        if only_family is not None:
            if use_family != only_family and use_family is not None:
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

        max_number_1 = min(max_number, max(groups_1.keys()))
        max_number_2 = min(max_number, max(groups_2.keys()))

        indices_to_use_1 = groups_1.number_mapper.number_to_index(np.arange(min_number, max_number_1 + 1))
        indices_to_use_2 = groups_2.number_mapper.number_to_index(np.arange(min_number, max_number_2 + 1))

        max_index = max(indices_to_use_1.max(), indices_to_use_2.max())

        transfer_matrix = self.count_particles_in_common(groups_1, groups_2,
                                                         max_num_halos=max_index, use_family=use_family)

        transfer_matrix_restricted = transfer_matrix[indices_to_use_1, :][:, indices_to_use_2]

        return transfer_matrix_restricted


class OneToOneBridge(AbstractBridge):
    """Connects two snapshots with identical particle numbers and file layout.

    Particle ``i`` in the start point is the same as particle ``i`` in the end point.
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
    """Connects to snapshots that both have arrays of identity integers (``iord`` or similar) to identify particles.

    Particles ``i_start`` and ``i_end`` are defined to be the same if and only if
    ``start[order_array][i_start] == start[order_array][i_end]``.
    """

    def __init__(self, start, end, order_array="iord", monotonic=True, allow_family_change=False,
                 only_families=None):
        """Initialise the ``OrderBridge``

        .. versionchanged:: 2.3.0

           The ``only_families`` parameter was added to allow bridging only between specific families. This is especially
           helpful where the order array is not defined for all families, such as in some RAMSES simulations.


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

        only_families : list of str or family.Family, optional
            If specified, only particles in these families will be considered for the bridge. This is useful if
            the order array is not defined for all families.

        """
        super().__init__(start, end)
        self._order_array = order_array
        self.monotonic = monotonic
        self.allow_family_change = allow_family_change
        if only_families is not None:
            self._only_families = [family.get_family(f) for f in only_families]
        else:
            self._only_families = None

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

        if self._only_families is None:
            iord_to = self._get_iord_array(to_)
            iord_from = self._get_iord_array(s)
            output_index = self._get_particle_indices_from_source_and_target_iords(iord_from, iord_to)
        else:
            output_index = []
            for f in s.families():
                if f in self._only_families:
                    iord_from = self._get_iord_array(s[f])
                    iord_to = self._get_iord_array(to_[f])
                    family_offset = to_._get_family_slice(f).start
                    output_index_this_fam = self._get_particle_indices_from_source_and_target_iords(iord_from, iord_to)
                    output_index.append(output_index_this_fam + family_offset)
            if len(output_index) == 0:
                output_index = np.array([], dtype=np.int64)
            else:
                output_index = np.concatenate(output_index)

        if self.allow_family_change:
            new_family_index = to_._family_index()[output_index]

            # stable sort by family:
            output_index = output_index[np.lexsort((new_family_index,))]

        return to_[output_index]

    def _get_iord_array(self, snap):
        return np.asarray(snap[self._order_array]).view(np.ndarray)

    def _get_particle_indices_from_source_and_target_iords(self, selected_iords_in_source_simulation,
                                                           all_iords_in_target_simulation):
        if not self.monotonic:
            iord_map_to = np.argsort(all_iords_in_target_simulation)
            iord_map_from = np.argsort(selected_iords_in_source_simulation)
            all_iords_in_target_simulation = all_iords_in_target_simulation[iord_map_to]
            selected_iords_in_source_simulation = selected_iords_in_source_simulation[iord_map_from]
        output_index, found_match = _bridge.bridge(all_iords_in_target_simulation, selected_iords_in_source_simulation)
        if not self.monotonic:
            output_index = iord_map_to[output_index[np.argsort(iord_map_from)][found_match]]
        else:
            output_index = output_index[found_match]
        return output_index


class RamsesBugOrderBridge(OrderBridge):
    def __init__(self, start, end, order_array="iord", monotonic=False, allow_family_change=False, only_families=None):
        """A special case of OrderBridge for tracking between Ramses snapshots affected by int32 iord truncation bug.

        In this bug, the iord array is truncated to `int32` when transmitted from one CPU to another. We first truncate
        all iords to `int32`, then use heuristics to try to disambiguate collisions that occur due to this bit loss.
        The heuristics are that star particles must always map onto star particles, and DM particles must map onto
        DM particles of the same level.
        """

        if only_families is not None:
            families = [family.get_family(f) for f in only_families]
        else:
            families = start.families()

        for fam_identifier in families:
            fam = family.get_family(fam_identifier)
            use_level_hashing = fam == family.dm
            start[fam]['pynbody_iord_recreation'] = self._make_new_iord_array(start[fam], order_array,
                                                                              use_level_hashing)
            end[fam]['pynbody_iord_recreation'] = self._make_new_iord_array(end[fam], order_array, use_level_hashing)

        if monotonic is not False:
            warnings.warn("RamsesBugOrderBridge does not support monotonic iord arrays; setting monotonic to False")

        super().__init__(start, end, 'pynbody_iord_recreation', False, allow_family_change, only_families)

    @classmethod
    def _make_new_iord_array(cls, snapshot, order_array_name, use_level_hashing=False):
        new_order_array = snapshot[order_array_name].astype(np.int32).astype(np.int64)

        if use_level_hashing:
            level_guess = np.log2(snapshot['mass']).astype(np.int64)
            level_guess -= level_guess.max()

            # we put each level onto its own high-order bits, in the hope this resolves most collisions.
            new_order_array += (1 + level_guess) * 2 ** 32

        # Check that the newly created iords are all unique
        if len(np.unique(new_order_array)) != len(new_order_array):
            warnings.warn(
                "Failed to resolve all conflicts when recreating the iord array in RamsesBugOrderBridge. "
                "This is likely due to having non-unique iords in one of the two ends of the bridge"
                "which cannot be mapped out to unique iords on the other end.")

        return new_order_array


def bridge_factory(a: snapshot.SimSnap, b: snapshot.SimSnap) -> AbstractBridge:
    """Create a bridge connecting the two specified snapshots.

    This function will determine the best type of :ref:`Bridge` to construct between the two snapshots, and return it.
    It is called by :func:`pynbody.snapshot.simsnap.SimSnap.bridge`.

    For more information see :doc:`the bridge tutorial </tutorials/bridge>`.
    """

    not_sure_error = "Don't know how to automatically bridge between two simulations of different formats. " \
                     "You will need to create your bridge manually by instantiating either the OneToOneBridge or " \
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
            return OneToOneBridge(a_top, b_top)
    elif isinstance(a_top, gadget.GadgetSnap) or isinstance(a_top, gadgethdf.GadgetHDFSnap):
        return OrderBridge(a_top, b_top, monotonic=False, allow_family_change=True)
    elif isinstance(a_top, ramses.RamsesSnap):
        if len(a.gas) > 0 or len(b.gas) > 0:
            raise RuntimeError("Cannot bridge AMR gas cells")
        if a_top.has_potential_negative_iords_bug or b_top.has_potential_negative_iords_bug:
            warnings.warn("Due to the unexpected presence of negative iord values, one of your snapshots has been identified "
                          "as potentially affected by a bug in RAMSES which leads to bit-loss in the iords. Pynbody will attempt "
                          "to correct for this bug. However if you intentionally have negative iords in your snapshot, this is not "
                          "what you want. In such cases, please reload your file with `pynbody.load(..., "
                          "negative_iords_on_purpose=True)`. For more background see: "
                          "https://github.com/pynbody/pynbody/pull/914, https://github.com/pynbody/pynbody/pull/961")
            return RamsesBugOrderBridge(a_top, b_top, monotonic=False, only_families=["dm", "star"])
        return OrderBridge(a_top, b_top, monotonic=False, only_families=["dm", "star"])
    else:
        raise RuntimeError(not_sure_error)
