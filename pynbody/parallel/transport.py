"""How arrays travel from the main process to the workers of a :class:`~pynbody.parallel.ProcessPool`.

An :class:`ArrayTransport` turns each array in a portable state, or in a reply from the
:class:`~pynbody.parallel.server.SnapshotServer`, into a picklable *token* (:meth:`ArrayTransport.encode`), and
turns the token back into an array at the other end (:meth:`ArrayTransport.decode`). Within one machine,
:class:`SharedMemoryTransport` makes the token a reference to shared memory, so that no data is copied. Other
transports (e.g. between machines) can copy the data instead, as :class:`CopyTransport` does; either way, the
arrays a worker receives are read-only, so that code behaves the same whichever transport is in use.

.. versionadded:: 2.9.0

"""

from __future__ import annotations

import numpy as np

from ..array import SimArray, shared
from ..halo.portable import map_arrays

#: Arrays smaller than this number of bytes are sent to each worker process directly, rather than through shared
#: memory, unless they are already in shared memory. Every shared array is a separate block of shared memory,
#: and there is little to gain from sharing small ones.
MIN_SHARED_BYTES = 1 << 16


class ArrayTransport:
    """Base class for the ways of getting arrays from the main process to the workers.

    A transport object is created in the main process, which calls :meth:`encode`; each worker is given the object
    returned by :meth:`for_workers`, which calls :meth:`decode`."""

    #: The types of token that :meth:`decode` accepts
    token_types: tuple = (np.ndarray,)

    #: The number of bytes of array data that the transport has had to duplicate in the main process
    bytes_copied = 0

    def encode(self, array: np.ndarray):
        """Return a picklable token from which :meth:`decode` can recreate *array* in another process"""
        raise NotImplementedError

    def decode(self, token) -> np.ndarray:
        """Return a read-only array from a token made by :meth:`encode`"""
        raise NotImplementedError

    def release(self):
        """Free any resources held for the arrays that have been encoded"""
        pass

    def for_workers(self) -> ArrayTransport:
        """Return a picklable transport for the workers to decode with, holding none of the main process's state"""
        return type(self)()

    def encode_state(self, state):
        """Return a copy of a portable state (or any part of one) with every array encoded"""
        return map_arrays(state, self.encode)

    def decode_state(self, state):
        """Return a copy of an encoded portable state, with every token decoded"""
        return map_arrays(state, self.decode, types=self.token_types)


class CopyTransport(ArrayTransport):
    """Copies arrays into the token. This is suitable for any channel that can carry pickled data."""

    def encode(self, array):
        return np.array(array)

    def decode(self, token):
        token.flags['WRITEABLE'] = False
        return token


class SharedMemoryTransport(ArrayTransport):
    """Sends references to shared memory, copying arrays there if they are not there already.

    Small arrays are sent as they are (see :data:`MIN_SHARED_BYTES`). Copies made in shared memory are held until
    :meth:`release` is called."""

    token_types = (shared.SharedArrayReference, np.ndarray)

    def __init__(self):
        self.copies = []
        self.bytes_copied = 0
        # Maps the memory an array occupies to (array, token). Keeping the array serves two purposes: its memory
        # cannot be freed and reused by a different array, which would then be given the same token; and a worker
        # started late, which decodes tokens made when the pool was created, still finds the memory there.
        self._tokens = {}

    def encode(self, array):
        key = (array.__array_interface__['data'][0], array.shape, array.strides, array.dtype.str)
        if key in self._tokens:
            return self._tokens[key][1]

        token = None
        if isinstance(array, SimArray):
            try:
                token = shared.to_shared_reference(array)
            except TypeError:
                pass  # not in shared memory

        if token is None:
            if array.nbytes < MIN_SHARED_BYTES or array.dtype.hasobject:
                token = np.asarray(array)
            else:
                copy = shared.make_shared_array(array.shape, array.dtype)
                copy[...] = array
                self.copies.append(copy)
                self.bytes_copied += array.nbytes
                token = shared.to_shared_reference(copy)

        self._tokens[key] = (array, token)
        return token

    def decode(self, token):
        if isinstance(token, shared.SharedArrayReference):
            token = shared.from_shared_reference(token)
        token.flags['WRITEABLE'] = False
        return token

    def release(self):
        self._tokens = {}
        self.copies = []
