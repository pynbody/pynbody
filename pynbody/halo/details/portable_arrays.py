"""Helpers for putting numpy arrays into a halo catalogue's portable state.

.. versionadded:: 2.7.1

"""

from __future__ import annotations

import numpy as np

from ...array import SimArray


def as_portable_array(value) -> np.ndarray:
    """Return *value* as a numpy array suitable for inclusion in a portable state.

    Unlike ``np.asarray``, this preserves the identity of array subclasses that pynbody understands.
    That matters for arrays which live in shared memory: viewing one as a base-class ``ndarray``
    discards the record of where the memory came from, so it can no longer be turned into a
    :class:`~pynbody.array.shared.SharedArrayReference` and would have to be copied to reach another
    process. See :meth:`pynbody.halo.HaloCatalogue.get_portable_state`.

    Subclasses which are not pynbody's own are converted to a plain ``ndarray``, since a recipient of
    the state is entitled to assume that the arrays in it behave in the ordinary way; a
    :class:`~pynbody.array.SimArray` with units is fine, because units are recorded separately by
    whatever is doing the encoding (and are ignored by everything else).
    """
    value = np.asanyarray(value)
    if isinstance(value, SimArray) or type(value) is np.ndarray:
        return value
    return np.asarray(value)
