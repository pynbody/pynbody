"""Implements classes to automatically copy data from another snapshot when needed.

Most users will not need to use this module directly. It is used by `tangos <https://pynbody.github.io/tangos>`_
to provide a transparent view of a snapshot that is actually stored in a different location.
"""

import copy

from .simsnap import SimSnap


class UnderlyingClassMixin:
    """Mixin for a SimSnap that allows it to derive quantities associated with another class."""
    def __init__(self, underlying_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._underlying_class = underlying_class

    def find_deriving_function(self, name):
        cl = self._underlying_class
        if cl in self._derived_array_registry \
                and name in self._derived_array_registry[cl]:
            return self._derived_array_registry[cl][name]
        else:
            return super().find_deriving_function(name)

class CopyOnAccessSimSnap(UnderlyingClassMixin, SimSnap):
    """SimSnap that copies data from another SimSnap when that data is needed.

    To the user, data which is already loaded in the underlying snapshot presents merely as 'loadable'
    (i.e. in :meth:`loadable_keys`)."""

    def __init__(self, base: SimSnap, underlying_class=None):
        self._copy_from = base
        if underlying_class is None:
            ancestor = base.ancestor
            if hasattr(ancestor, "_underlying_class"):
                underlying_class = ancestor._underlying_class
            else:
                underlying_class = type(ancestor)

        super().__init__(underlying_class)

        self._unifamily = base._unifamily
        self._file_units_system = base._file_units_system
        self._num_particles = len(base)
        self._family_slice = {f: base._get_family_slice(f) for f in base.families()}
        self._filename = base._filename+":copied_on_access"
        self._dont_try_accessing = []
        self.properties = copy.deepcopy(base.properties)


    def _load_array(self, array_name, fam=None):
        if (array_name, fam) in self._dont_try_accessing:
            raise OSError("Previously tried to get this array without success; not trying again")
        try:
            with self._copy_from.lazy_derive_off, self.lazy_derive_off, self.auto_propagate_off:
                if fam is None:
                    self[array_name] = self._copy_from[array_name]
                else:
                    self[fam][array_name] = self._copy_from[fam][array_name]
        except KeyError as e:
            self._dont_try_accessing.append((array_name, fam))
            raise OSError("Not found in underlying snapshot") from e

    def loadable_keys(self, fam=None):
        """Return the names of the arrays that can be copied from the underlying snapshot.

        Only ND arrays are named, not the 1D slices that accompany them: ``vel`` is reported but
        ``vz`` is not, matching what a snapshot loaded from disk offers. The slices remain
        accessible, since a request for one is mapped onto its parent when it is loaded.

        .. versionchanged:: 2.6.0

          The 1D slices of ND arrays are no longer reported. Previously, an in-memory ``vel`` in
          the underlying snapshot caused ``vx``, ``vy`` and ``vz`` to be listed as loadable as
          well, which no file-backed snapshot does. Code inspecting ``loadable_keys`` to establish
          what a snapshot can provide consequently got a different answer for a copy-on-access
          view than for the snapshot it was copying from.
        """
        if fam is None:
            loaded_keys_in_parent = self._copy_from.keys()
        else:
            loaded_keys_in_parent = self._copy_from.family_keys(fam)
        keys = set(self._copy_from.loadable_keys(fam)).union(loaded_keys_in_parent)

        # A 1D name is only dropped when its parent is genuinely present, so that a name which
        # merely looks like a slice is left alone.
        return [k for k in keys if self._array_name_1D_to_ND(k) not in keys]
