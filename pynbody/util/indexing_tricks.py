"""Tricks for manipulating indexes and slices into arrays for internal use by pynbody."""

from __future__ import annotations

import numpy as np


def _gcf(a, b):
    """Return the greatest common factor of a and b"""
    while b > 0:
        a, b = b, a % b
    return a


def _lcm(a, b):
    """Return the least common multiple of a and b"""
    return (a * b) // _gcf(a, b)


def intersect_slices(s1, s2, array_length=None):
    """Given two python slices s1 and s2, return a slice that picks out all members of s1 and s2.

    That is, if d is an array, then ``d[intersect_slices(s1,s2,len(d))]`` returns all elements of ``d`` that are in
    both ``d[s1]`` and ``d[s2]``.

    Note that it may not be possible to do this without information on the length of the array referred to, hence
    all slices with end-relative indexes are first converted into begin-relative indexes. The slice returned may be
    specific to the length specified.

    If the slices are mutually exclusive, a zero-length slice is returned.

    Parameters
    ----------

    s1 : slice
        The first slice

    s2 : slice
        The second slice

    array_length : int, optional
        The length of the array to which the slices refer. If not specified, the slices must have positive start and
        stop.

    Returns
    -------

    slice
        A slice that picks out all elements of s1 and s2, as described above.

    """

    assert array_length is not None or \
        (s1.start >= 0 and s2.start >= 0 and s1.stop >= 0 and s2.start >= 0)

    s1_start = s1.start
    s2_start = s2.start
    s1_stop = s1.stop
    s2_stop = s2.stop
    s1_step = s1.step
    s2_step = s2.step

    if s1_step is None:
        s1_step = 1
    if s2_step is None:
        s2_step = 1

    assert s1_step > 0 and s2_step > 0

    if s1_start < 0:
        s1_start = array_length + s1_start
    if s1_start < 0:
        return slice(0, 0)

    if s2_start < 0:
        s2_start = array_length + s2_start
    if s2_start < 0:
        return slice(0, 0)

    if s1_stop < 0:
        s1_stop = array_length + s1_stop
    if s1_stop < 0:
        return slice(0, 0)

    if s2_stop < 0:
        s2_stop = array_length + s2_stop
    if s2_stop < 0:
        return slice(0, 0)

    step = _lcm(s1_step, s2_step)

    start = max(s1_start, s2_start)
    stop = min(s1_stop, s2_stop)

    if stop <= start:
        return slice(0, 0)

    s1_offset = start - s1_start
    s2_offset = start - s2_start
    s1_offset_x = int(s1_offset)
    s2_offset_x = int(s2_offset)

    if s1_step == s2_step and s1_offset % s1_step != s2_offset % s1_step:
        # slices are mutually exclusive
        return slice(0, 0)

    # There is surely a more efficient way to do the following, but
    # it eludes me for the moment
    while s1_offset % s1_step != 0 or s2_offset % s2_step != 0:
        start += 1
        s1_offset += 1
        s2_offset += 1
        if s1_offset % s1_step == s1_offset_x % s1_step and s2_offset % s2_step == s2_offset_x % s2_step:
            # slices are mutually exclusive
            return slice(0, 0)

    if step == 1:
        step = None

    return slice(start, stop, step)


def relative_slice(s_relative_to, s):
    """Return a slice s_prime with the property that ar[s_relative_to][s_prime] == ar[s].

    Clearly this will not be possible for arbitrarily chosen s_relative_to and s, but it should be possible
    for any ``s = intersect_slices(s_relative_to, s_any)``, which is the use case envisioned here.

    If impossible, a ValueError is raised.

    This code does not work with end-relative (i.e. negative) start or stop positions.
    """

    assert (s_relative_to.start >= 0 and s.start >= 0 and s.stop >= 0)

    if s.start == s.stop:
        return slice(0, 0, None)

    s_relative_to_step = s_relative_to.step if s_relative_to.step is not None else 1
    s_step = s.step if s.step is not None else 1

    if (s.start - s_relative_to.start) % s_relative_to_step != 0:
        raise ValueError("Incompatible slices")
    if s_step % s_relative_to_step != 0:
        raise ValueError("Incompatible slices")

    start = (s.start - s_relative_to.start) // s_relative_to_step
    step = s_step // s_relative_to_step
    stop = start + \
        (s_relative_to_step - 1 + s.stop - s.start) // s_relative_to_step

    if step == 1:
        step = None

    return slice(start, stop, step)


def chained_slice(s1, s2):
    """Return a slice s3 with the property that ar[s1][s2] == ar[s3]"""

    assert (s1.start >= 0 and s2.start >= 0 and s1.stop >= 0 and s2.stop >= 0)
    s1_start = s1.start or 0
    s2_start = s2.start or 0
    s1_step = s1.step or 1
    s2_step = s2.step or 1

    start = s1_start + s2_start * s1_step
    step = s1_step * s2_step
    if s1.stop is None and s2.stop is None:
        stop = None
    elif s1.stop is None:
        stop = start + step * (s2.stop - s2_start) // s2_step
    elif s2.stop is None:
        stop = s1.stop
    else:
        stop_s2 = start + step * (s2.stop - s2_start) // s2_step
        stop_s1 = s1.stop
        stop = stop_s2 if stop_s2 < stop_s1 else stop_s1
    return slice(start, stop, step)


def index_before_slice(s, index):
    """Return an index array new_index such that ``ar[s][new_index] == ar[index]``.

    Arguments
    ---------
    s : slice
        The slice to apply to the array

    index : array-like
        The index array to apply to the sliced array

    Returns
    -------
    new_index : array-like
        The index array that will pick out the same elements of the sliced array as index does of the original array
    """

    start = s.start or 0
    step = s.step or 1

    assert start >= 0
    assert step >= 0
    assert s.stop is None or s.stop >= 0

    new_index = start + index * step
    if s.stop is not None:
        new_index = new_index[np.where(new_index < s.stop)]

    return new_index


def concatenate_indexing(i1, i2):
    """Given either a numpy array or slice for both i1 and i2, return an object such that ar[i3] == ar[i1][i2].

    As a convenience, if i2 is None, i1 is returned.

    Parameters
    ----------
    i1 : array-like or slice
        The first indexing or slicing operation to apply

    i2 : array-like or slice
        The second indexing or slicing operation to apply

    Returns
    -------
    i3 : array-like or slice
        The combined indexing or slicing operation
    """
    if isinstance(i1, tuple) and len(i1) == 1:
        i1 = i1[0]
    if isinstance(i2, tuple) and len(i2) == 1:
        i2 = i2[0]

    if i2 is None:
        return i1
    if isinstance(i1, slice) and isinstance(i2, slice):
        return chained_slice(i1, i2)
    elif isinstance(i1, slice) and isinstance(i2, (np.ndarray, list)):
        return index_before_slice(i1, i2)
    elif isinstance(i1, (np.ndarray, list)) and isinstance(i2, (slice, np.ndarray)):
        return np.asarray(i1, dtype=np.int64)[i2]
    else:
        raise TypeError("Don't know how to chain these index types")


def indexing_length(sl_or_ar):
    """Given either an array or slice, return ``len(ar[sl_or_ar])``

    This assumes that the slice does not overrun the array, e.g. if an array ``ar`` is shorter than the stop index of
    the slice ``sl`` then this routine will return an inaccurate result for ``len(ar[sl])``.

    Parameters
    ----------
    sl_or_ar : slice or array-like
        The slice or array-like object to consider

    Returns
    -------
    int
        The length of the array after slicing
    """

    if isinstance(sl_or_ar, slice):
        step = sl_or_ar.step or 1
        diff = (sl_or_ar.stop - sl_or_ar.start)
        return diff // step + (diff % step > 0)
    else:
        return len(sl_or_ar)
