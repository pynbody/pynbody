from __future__ import annotations

import abc

import numpy as np

from pynbody.util import binary_search, is_sorted


class IordToOffset(abc.ABC):
    @abc.abstractmethod
    def map_ignoring_order(self, i: np.ndarray | int) -> np.ndarray | int:
        """Given an array of iord values, return the corresponding fpos values.

        Warning: The returned values are not guaranteed to be in the same order as the input iord array."""
        pass


class IordToOffsetDense(IordToOffset):
    def __init__(self, iord_array, max_iord=None):
        if max_iord is None:
            max_iord = int(iord_array.max())
        self._iord_to_offset = np.empty(max_iord + 1, dtype=np.int64)
        self._iord_to_offset.fill(-1)
        self._iord_to_offset[iord_array] = np.arange(len(iord_array), dtype=np.int64)

    def map_ignoring_order(self, i):
        return self._iord_to_offset[i]


class IordToOffsetSparse(IordToOffset):
    """Class for efficiently mapping from iords to offsets in the iord array, even if iord values are large.

    WARNING: if a query is made with iords that are not themselves in ascending order, a sort takes place
    ahead of the query and therefore the set returned is correct but the ordering is not preserved."""
    def __init__(self, iord_array):
        self._iord = iord_array
        self._iord_argsort = np.argsort(iord_array)

    def map_ignoring_order(self, iord_values: np.ndarray | int) -> np.ndarray | int:
        if not hasattr(iord_values, "__len__"):
            iord_values = np.array([iord_values])
            singleton = True
        else:
            iord_values = np.asarray(iord_values)
            singleton = False

            if is_sorted(iord_values) != 1:
                iord_values = np.sort(iord_values)

        result = binary_search(np.asarray(iord_values), self._iord, self._iord_argsort)

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

    def map_ignoring_order(self, i: np.ndarray | int) -> np.ndarray | int:
        result = self._underlying.map_ignoring_order(i)
        result += self._fpos_offset
        return result

def make_iord_to_offset_mapper(iord: np.ndarray) -> IordToOffset:
    """Given an array of unique integers, iord, make an object which maps from an iord value to offset in the array.

    i.e. given an iord array and a subset of values my_iord_values,

     make_iord_to_offset_mapper(iord).map_ignoring_order(my_iord_values)

    returns the indexes of my_iord_values in the iord array.
    """

    max_iord = int(iord.max())
    assert iord.min() >= 0, "Can't handle negative iord values"

    if max_iord < 2 * len(iord):
        # maximum iord is not very big, just do a direct in-memory mapping for speed
        return IordToOffsetDense(iord, max_iord)
    else:
        # maximum iord is large, so we'll use util.binary_search to save memory at the cost of speed
        return IordToOffsetSparse(iord)
