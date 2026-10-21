"""The workers' side of a :class:`~pynbody.parallel.ProcessPool`: a read-only snapshot that asks for what it lacks.

.. versionadded:: 2.9.0

"""

from __future__ import annotations

from .. import family
from ..snapshot.portable import PortableSimSnap
from ..snapshot.simsnap import SimSnap
from .server import DERIVE_LOCALLY


class ServedSimSnap(PortableSimSnap):
    """A read-only copy of a snapshot, which obtains arrays and KD-trees it lacks from the main process.

    It starts with everything in the portable state of the original snapshot. When an array that it does not
    hold is needed, it asks the :class:`~pynbody.parallel.server.SnapshotServer`, which loads or derives it in the
    original snapshot and sends it back -- so that every worker shares the one copy. Similarly,
    :meth:`build_tree` fetches the original snapshot's tree, building it there if need be.

    Like any :class:`~pynbody.snapshot.portable.PortableSimSnap`, the snapshot cannot be modified, but arrays can
    be created in it; these belong to the worker alone. Arrays that the server cannot provide, e.g. because they
    depend on such an array, are derived by the worker itself.

    .. versionadded:: 2.9.0

    """

    def __init__(self, state, channel, view_key, transport):
        """Create a snapshot from a portable state, which obtains what it lacks through *channel*.

        Parameters
        ----------

        state : dict
            The portable state of the original snapshot, already decoded by *transport*.

        channel :
            An object with a ``request(name, *args)`` method, such as a
            :class:`~pynbody.parallel.channel.LocalChannel`, that passes requests to the server.

        view_key : tuple
            What the server calls this snapshot (see :class:`~pynbody.parallel.server.SnapshotServer`).

        transport : ArrayTransport
            Decodes the arrays that the server sends.
        """
        super().__init__(state)
        self._channel = channel
        self._view_key = view_key
        self._transport = transport

    def _load_array(self, array_name, fam=None):
        descriptions = self._channel.request('load_array', self._view_key, array_name,
                                             None if fam is None else fam.name)
        if descriptions is None:
            raise OSError(f"{array_name!r} cannot be loaded")
        self._install_from_server(descriptions)

    def _derive_array(self, name, fam=None):
        descriptions = self._channel.request('derive_array', self._view_key, name,
                                             None if fam is None else fam.name)
        if descriptions == DERIVE_LOCALLY:
            SimSnap._derive_array(self, name, fam)
        else:
            self._install_from_server(descriptions)

    def build_tree(self, num_threads=None, shared_mem=None):
        """Use the KD-tree of the original snapshot, asking for it to be built if it has not been.

        The arguments are those of :meth:`SimSnap.build_tree <pynbody.snapshot.simsnap.SimSnap.build_tree>`;
        *num_threads* applies to operations on the tree in this process, and *shared_mem* is ignored."""
        if not hasattr(self, 'kdtree'):
            tree_state = self._transport.decode_state(self._channel.request('kdtree', self._view_key))
            self._attach_tree(tree_state, num_threads=num_threads)

    def loadable_keys(self, fam=None):
        """The arrays that the original snapshot can load, which can therefore be obtained here too."""
        return list(self._loadable_keys_in_original.get(fam, []))

    def _install_from_server(self, descriptions):
        for description in descriptions:
            name = description['name']
            fam = None if description['family'] is None else family.get_family(description['family'])
            if fam is None:
                if name in self._arrays:
                    continue
                # The server may have turned family arrays into one for the whole snapshot; if so, so must this
                self._discard_family_arrays(name)
            elif name in self._arrays or fam in self._family_arrays.get(name, {}):
                continue
            description['data'] = self._transport.decode(description['data'])
            self._install_array(name, fam, description)

    def _discard_family_arrays(self, name):
        for name_or_slice in [name] + list(self._array_name_ND_to_1D(name)):
            self._family_arrays.pop(name_or_slice, None)
            for derived_names in self._family_derived_array_names.values():
                if name_or_slice in derived_names:
                    derived_names.remove(name_or_slice)
