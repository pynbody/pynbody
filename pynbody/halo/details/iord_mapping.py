from __future__ import annotations

import abc
import warnings

import numpy as np

from pynbody.util import binary_search, is_sorted

NO_OFFSET = -1
"""Sentinel returned by :meth:`IordToOffset.map_ignoring_order` in place of iords that are not in the snapshot.

Only returned if ``allow_missing=True``; otherwise an :class:`IllegalIordError` is raised."""


class IllegalIordError(ValueError):
    """Raised when a mapping is requested for iord values that do not appear in the snapshot.

    This most commonly arises when a halo catalogue refers to particles that have not been loaded, e.g.
    because only a subset of the file (a region, or a subset of families) was loaded."""


def _describe_missing(iord_values: np.ndarray, missing: np.ndarray, max_examples: int = 5) -> str:
    """Generate an error message describing the iord values flagged by the boolean array missing"""
    missing_values = np.atleast_1d(np.asarray(iord_values))[np.atleast_1d(missing)]
    num_missing = len(missing_values)
    examples = ", ".join(str(int(v)) for v in missing_values[:max_examples])
    if num_missing > max_examples:
        examples += ", ..."
    return (f"{num_missing} particle ID(s) do not correspond to any particle in the snapshot (e.g. {examples}). "
            f"This normally means the snapshot has been only partially loaded, while the halo catalogue "
            f"refers to particles that are not present.")


def _check_none_missing(iord_values, result) -> None:
    """Raise an IllegalIordError if any entry of result is NO_OFFSET.

    Since all genuine offsets are non-negative, the check in the usual case (nothing missing) is a single
    reduction over result with no temporary allocations; only if something is missing do we identify what."""
    if np.ndim(result) == 0:
        if result == NO_OFFSET:
            raise IllegalIordError(_describe_missing(iord_values, np.bool_(True)))
    elif len(result) > 0 and result.min() < 0:
        raise IllegalIordError(_describe_missing(iord_values, result == NO_OFFSET))


class IordToOffset(abc.ABC):
    @abc.abstractmethod
    def map_ignoring_order(self, i: np.ndarray | int, allow_missing: bool = False) -> np.ndarray | int:
        """Given an array of iord values, return the corresponding fpos values.

        .. warning::

            The returned values are not guaranteed to be in the same order as the input iord array.

        Parameters
        ----------
        i :
            The iord value(s) to map. An array maps to an array; a scalar maps to a scalar.
        allow_missing :
            If False (default), an :class:`IllegalIordError` is raised if any of the requested iords are
            not present in the snapshot. If True, :data:`NO_OFFSET` (-1) is returned in their place.
        """
        pass


class IordToOffsetDense(IordToOffset):
    def __init__(self, iord_array, max_iord=None):
        if max_iord is None:
            max_iord = int(iord_array.max())
        self._iord_to_offset = np.empty(max_iord + 1, dtype=np.int64)
        self._iord_to_offset.fill(NO_OFFSET)
        self._iord_to_offset[iord_array] = np.arange(len(iord_array), dtype=np.int64)

    def _map_out_of_range(self, i):
        """Map iords, at least some of which lie outside the range covered by the lookup table"""
        in_range = (i >= 0) & (i < len(self._iord_to_offset))
        result = np.empty(len(i), dtype=self._iord_to_offset.dtype)
        result.fill(NO_OFFSET)
        result[in_range] = self._iord_to_offset[i[in_range]]
        return result

    def map_ignoring_order(self, i, allow_missing=False):
        if np.ndim(i) == 0:
            if 0 <= i < len(self._iord_to_offset):
                result = self._iord_to_offset[i]
            else:
                result = np.int64(NO_OFFSET)
        else:
            i = np.asarray(i)
            # Reductions over the query cost far less than the gather itself, and let us keep the common case
            # (everything in range) as a single fancy-index with no temporaries. They are also essential for
            # correctness: numpy would silently wrap negative iords around the end of the lookup table.
            if len(i) == 0:
                result = np.empty(0, dtype=self._iord_to_offset.dtype)
            elif (i.max() >= len(self._iord_to_offset)
                  or (i.dtype.kind != 'u' and i.min() < 0)):
                result = self._map_out_of_range(i)
            else:
                result = self._iord_to_offset[i]

        if not allow_missing:
            _check_none_missing(i, result)

        return result


class IordToOffsetSparse(IordToOffset):
    """Class for efficiently mapping from iords to offsets in the iord array, even if iord values are large.

    WARNING: if a query is made with iords that are not themselves in ascending order, a sort takes place
    ahead of the query and therefore the set returned is correct but the ordering is not preserved."""
    def __init__(self, iord_array):
        self._iord = iord_array
        self._iord_argsort = np.argsort(iord_array)

    def map_ignoring_order(self, iord_values: np.ndarray | int, allow_missing: bool = False) -> np.ndarray | int:
        if not hasattr(iord_values, "__len__"):
            iord_values = np.array([iord_values])
            singleton = True
        else:
            iord_values = np.asarray(iord_values)
            singleton = False

            if is_sorted(iord_values) != 1:
                iord_values = np.sort(iord_values)

        result = binary_search(np.asarray(iord_values), self._iord, self._iord_argsort)

        # binary_search flags values it could not find with the length of the array being searched, which is
        # also the largest value it can return; so a single reduction detects them without allocating a mask
        if len(result) > 0 and result.max() == len(self._iord):
            missing = result == len(self._iord)
            if not allow_missing:
                raise IllegalIordError(_describe_missing(iord_values, missing))
            result[missing] = NO_OFFSET

        if singleton:
            return result[0]
        else:
            return result


class IordOffsetModifier(IordToOffset):
    """A wrapper around an IordToOffset which adds a constant offset to the result of the underlying mapping.

    Useful if the iord values e.g. are only available for a single family; then the fpos_offset will correspond
    to the first index of that family in the pynbody snapshot.
    """

    def __init__(self, iord_to_offset: IordToOffset, fpos_offset: int):
        self._underlying = iord_to_offset
        self._fpos_offset = fpos_offset

    def map_ignoring_order(self, i: np.ndarray | int, allow_missing: bool = False) -> np.ndarray | int:
        result = self._underlying.map_ignoring_order(i, allow_missing=allow_missing)

        if np.ndim(result) == 0:
            if allow_missing and result == NO_OFFSET:
                return result
            return result + self._fpos_offset

        if allow_missing:
            # the sentinel must not be shifted along with the genuine offsets
            missing = result == NO_OFFSET
            result += self._fpos_offset
            result[missing] = NO_OFFSET
        else:
            result += self._fpos_offset

        return result


def make_iord_to_offset_mapper(iord: np.ndarray) -> IordToOffset:
    """Given an array of unique integers, iord, make an object which maps from an iord value to offset in the array.

    i.e. given an iord array and a subset of values my_iord_values,

     make_iord_to_offset_mapper(iord).map_ignoring_order(my_iord_values)

    returns the indexes of my_iord_values in the iord array.

    If any of my_iord_values are not in the iord array, an :class:`IllegalIordError` is raised, unless
    ``allow_missing=True`` is passed, in which case :data:`NO_OFFSET` is returned in their place.
    """

    min_iord = int(iord.min())
    max_iord = int(iord.max())

    if (min_iord >= 0) and (max_iord < 2 * len(iord)):
        # maximum iord is not very big, just do a direct in-memory mapping for speed
        return IordToOffsetDense(iord, max_iord)
    else:
        # maximum iord is large, so we'll use util.binary_search to save memory at the cost of speed
        return IordToOffsetSparse(iord)
