import numpy as np

from pynbody import filt, util
from pynbody.snapshot import SimSnap


class ExposedBaseSnapshotMixin:
    # The following will be objects common to a SimSnap and all its SubSnaps
    _inherited = ["_immediate_cache_lock",
                  "lazy_off", "lazy_derive_off", "lazy_load_off", "auto_propagate_off",
                  "properties", "_derived_array_names", "_family_derived_array_names",
                  "_dependency_tracker", "immediate_mode", "delay_promotion"]

    def __init__(self, base, *args, **kwargs):
        self.base = base
        super().__init__(base, *args, **kwargs)

    def _inherit(self):
        self._file_units_system = self.base._file_units_system
        self._unifamily = self.base._unifamily

        for x in self._inherited:
            setattr(self, x, getattr(self.base, x))

class SubSnapBase(SimSnap):
    def __init__(self, base):
        self._subsnap_base = base

    def _get_array(self, name, index=None, always_writable=False):
        if self.immediate_mode:
            return self._get_from_immediate_cache(name,
                                                  lambda: self._subsnap_base._get_array(
                                                      name, None, always_writable)[self._slice])

        else:
            ret = self._subsnap_base._get_array(name, util.concatenate_indexing(
                self._slice, index), always_writable)
            ret.family = self._unifamily
            return ret

    def _set_array(self, name, value, index=None):
        self._subsnap_base._set_array(
            name, value, util.concatenate_indexing(self._slice, index))

    def _get_family_array(self, name, fam, index=None, always_writable=False):
        base_family_slice = self._subsnap_base._get_family_slice(fam)
        sl = util.relative_slice(base_family_slice,
                                 util.intersect_slices(self._slice, base_family_slice, len(self._subsnap_base)))
        sl = util.concatenate_indexing(sl, index)
        if self.immediate_mode:
            return self._get_from_immediate_cache((name, fam),
                                                  lambda: self._subsnap_base._get_family_array(
                                                      name, fam, None, always_writable)[sl])
        else:
            return self._subsnap_base._get_family_array(name, fam, sl, always_writable)

    def _set_family_array(self, name, family, value, index=None):
        fslice = self._get_family_slice(family)
        self._subsnap_base._set_family_array(
            name, family, value, util.concatenate_indexing(fslice, index))

    def _promote_family_array(self, *args, **kwargs):
        self._subsnap_base._promote_family_array(*args, **kwargs)

    def __delitem__(self, name):
        # is this the right behaviour?
        raise RuntimeError("Arrays can only be deleted from the base snapshot")

    def _del_family_array(self, name, family):
        # is this the right behaviour?
        raise RuntimeError("Arrays can only be deleted from the base snapshot")

    @property
    def _filename(self):
        return str(self._subsnap_base._filename) + ":" + self._descriptor

    def keys(self):
        return list(self._subsnap_base.keys())

    def loadable_keys(self, fam=None):
        if self._unifamily:
            return self._subsnap_base.loadable_keys(self._unifamily)
        else:
            return self._subsnap_base.loadable_keys(fam)

    def derivable_keys(self):
        return self._subsnap_base.derivable_keys()

    def infer_original_units(self, *args):
        """Return the units on disk for a quantity with the specified dimensions"""
        return self._subsnap_base.infer_original_units(*args)

    def _get_family_slice(self, fam):
        sl = util.relative_slice(self._slice,
                                 util.intersect_slices(self._slice, self._subsnap_base._get_family_slice(fam), len(self._subsnap_base)))
        return sl

    def _load_array(self, array_name, fam=None, **kwargs):
        self._subsnap_base._load_array(array_name, fam)

    def write_array(self, array_name, fam=None, **kwargs):
        fam = fam or self._unifamily
        if not fam or self._get_family_slice(fam) != slice(0, len(self)):
            raise OSError(
                "Array writing is available for entire simulation arrays or family-level arrays, but not for arbitrary subarrays")

        self._subsnap_base.write_array(array_name, fam=fam, **kwargs)

    def _derive_array(self, array_name, fam=None):
        self._subsnap_base._derive_array(array_name, fam)

    def family_keys(self, fam=None):
        return self._subsnap_base.family_keys(fam)

    def _create_array(self, *args, **kwargs):
        self._subsnap_base._create_array(*args, **kwargs)

    def _create_family_array(self, *args, **kwargs):
        self._subsnap_base._create_family_array(*args, **kwargs)

    def physical_units(self, *args, **kwargs):
        self._subsnap_base.physical_units(*args, **kwargs)

    def is_derived_array(self, v, fam=None):
        fam = fam or self._unifamily
        return self._subsnap_base.is_derived_array(v, fam)

    def unlink_array(self, name):
        self._subsnap_base.unlink_array(name)

    def get_index_list(self, relative_to, of_particles=None):
        if of_particles is None:
            of_particles = np.arange(len(self))

        if relative_to is self:
            return of_particles

        return self._subsnap_base.get_index_list(relative_to, util.concatenate_indexing(self._slice, of_particles))

class SubSnap(ExposedBaseSnapshotMixin, SubSnapBase):
    """Represent a sub-view of a SimSnap, initialized by specifying a
    slice.  Arrays accessed through __getitem__ are automatically
    sub-viewed using the given slice."""

    def __init__(self, base, _slice):
        super().__init__(base)
        self._inherit()

        if isinstance(_slice, slice):
            # Various slice logic later (in particular taking
            # subsnaps-of-subsnaps) requires having positive
            # (i.e. start-relative) slices, so if we have been passed a
            # negative (end-relative) index, fix that now.
            if _slice.start is None:
                _slice = slice(0, _slice.stop, _slice.step)
            if _slice.start < 0:
                _slice = slice(len(
                    base) + _slice.start, _slice.stop, _slice.step)
            if _slice.stop is None or _slice.stop > len(base):
                _slice = slice(_slice.start, len(base), _slice.step)
            if _slice.stop < 0:
                _slice = slice(_slice.start, len(
                    base) + _slice.stop, _slice.step)

            self._slice = _slice

            descriptor = "[" + str(_slice.start) + ":" + str(_slice.stop)
            if _slice.step is not None:
                descriptor += ":" + str(_slice.step)
            descriptor += "]"

        else:
            raise TypeError("Unknown SubSnap slice type")

        self._num_particles = util.indexing_length(_slice)

        self._descriptor = descriptor






class IndexingViewMixin:
    def __init__(self, *args, **kwargs):
        index_array = kwargs.pop('index_array', None)
        iord_array = kwargs.pop('iord_array', None)
        allow_family_sort = kwargs.pop('allow_family_sort', False)

        super().__init__(*args, **kwargs)
        self._descriptor = "indexed"

        self._unifamily = self._subsnap_base._unifamily
        self._file_units_system = self._subsnap_base._file_units_system

        if index_array is None and iord_array is None:
            raise ValueError(
                "Cannot define a subsnap without an index_array or iord_array.")
        if index_array is not None and iord_array is not None:
            raise ValueError(
                "Cannot define a subsnap without both and index_array and iord_array.")
        if iord_array is not None:
            index_array = self._iord_to_index(iord_array)

        if isinstance(index_array, filt.Filter):
            self._descriptor = index_array._descriptor
            index_array = index_array.where(self._subsnap_base)[0]

        elif isinstance(index_array, tuple):
            if isinstance(index_array[0], np.ndarray):
                index_array = index_array[0]
            else:
                index_array = np.array(index_array)
        else:
            index_array = np.asarray(index_array)

        findex = self._subsnap_base._family_index()[index_array]

        if allow_family_sort:
            sort_ar = np.argsort(findex)
            index_array = index_array[sort_ar]
            findex = findex[sort_ar]
        else:
            # Check the family index array is monotonically increasing
            # If not, the family slices cannot be implemented
            if not all(np.diff(findex) >= 0):
                raise ValueError(
                    "Families must retain the same ordering in the SubSnap")

        self._slice = index_array
        self._family_slice = {}
        self._family_indices = {}
        self._num_particles = len(index_array)

        # Find the locations of the family slices
        for i, fam in enumerate(self._subsnap_base.ancestor.families()):
            ids = np.where(findex == i)[0]
            if len(ids) > 0:
                new_slice = slice(ids.min(), ids.max() + 1)
                self._family_slice[fam] = new_slice
                self._family_indices[fam] = np.asarray(index_array[
                                                       new_slice]) - self._subsnap_base._get_family_slice(fam).start
    def _iord_to_index(self, iord):
        # Maps iord to indices. Note that this requires to perform an argsort (O(N log N) operations)
        # and a binary search (O(M log N) operations) with M = len(iord) and N = len(self._subsnap_base).

        if not util.is_sorted(iord) == 1:
            raise Exception('Expected iord to be sorted in increasing order.')

        # Find index of particles using a search sort
        iord_base = self._subsnap_base['iord']
        iord_base_argsort = self._subsnap_base['iord_argsort']
        index_array = util.binary_search(iord, iord_base, sorter=iord_base_argsort)

        # Check that the iord match
        if np.any(index_array == len(iord_base)):
            raise Exception('Some of the requested ids cannot be found in the dataset.')

        return index_array


class IndexedSubSnap(IndexingViewMixin, ExposedBaseSnapshotMixin, SubSnapBase):
    """Represents a subset of the simulation particles according
        to an index array.

        Parameters
        ----------
        base : SimSnap object
            The base snapshot
        index_array : integer array or None
            The indices of the elements that define the sub snapshot. Set to None to use iord-based instead.
        iord_array : integer array or None
            The iord of the elements that define the sub snapshot. Set to None to use index-based instead.
            This may be computationally expensive. See note below.

        Notes
        -----
        `index_array` and `iord_array` arguments are mutually exclusive.
        In the case of `iord_array`, an sorting operation is required that may take
        a significant time and require O(N) memory.
        """

    def __init__(self, base, index_array=None, iord_array=None, *args, **kwargs):
        super().__init__(base, index_array=index_array, iord_array=iord_array, *args, **kwargs)
        self._inherit()

    def _get_family_slice(self, fam):
        # A bit messy: jump out the SubSnap inheritance chain
        # and call SimSnap method directly...
        return SimSnap._get_family_slice(self, fam)

    def _get_family_array(self, name, fam, index=None, always_writable=False):
        sl = self._family_indices.get(fam,slice(0,0))
        sl = util.concatenate_indexing(sl, index)

        return self._subsnap_base._get_family_array(name, fam, sl, always_writable)

    def _set_family_array(self, name, family, value, index=None):
        self._subsnap_base._set_family_array(name, family, value,
                                    util.concatenate_indexing(self._family_indices[family], index))

    def _create_array(self, *args, **kwargs):
        self._subsnap_base._create_array(*args, **kwargs)


class FamilySubSnap(SubSnap):

    """Represents a one-family portion of a parent snap object"""

    def __init__(self, base, fam):
        super().__init__(base, base._get_family_slice(fam))

        self._unifamily = fam
        self._descriptor = ":" + fam.name


    def __delitem__(self, name):
        if name in list(self._subsnap_base.keys()):
            raise ValueError(
                "Cannot delete global simulation property from sub-view")
        elif name in self._subsnap_base.family_keys(self._unifamily):
            self._subsnap_base._del_family_array(name, self._unifamily)

    def keys(self):
        global_keys = list(self._subsnap_base.keys())
        family_keys = self._subsnap_base.family_keys(self._unifamily)
        return list(set(global_keys).union(family_keys))

    def family_keys(self, fam=None):
        # We now define there to be no family-specific subproperties,
        # because all properties can be accessed through standard
        # __setitem__, __getitem__ methods
        return []

    def _get_family_slice(self, fam):
        if fam is self._unifamily:
            return slice(0, len(self))
        else:
            return slice(0, 0)

    def _get_array(self, name, index=None, always_writable=False):
        try:
            return SubSnap._get_array(self, name, index, always_writable)
        except KeyError:
            return self._subsnap_base._get_family_array(name, self._unifamily, index, always_writable)

    def _create_array(self, array_name, ndim=1, dtype=None, zeros=True, derived=False, shared=None):
        # Array creation now maps into family-array creation in the parent
        self._subsnap_base._create_family_array(
            array_name, self._unifamily, ndim, dtype, derived, shared)

    def _set_array(self, name, value, index=None):
        if name in list(self._subsnap_base.keys()):
            self._subsnap_base._set_array(
                name, value, util.concatenate_indexing(self._slice, index))
        else:
            self._subsnap_base._set_family_array(name, self._unifamily, value, index)

    def _create_family_array(self, array_name, family, ndim, dtype, derived, shared):
        self._subsnap_base._create_family_array(
            array_name, family, ndim, dtype, derived, shared)

    def _promote_family_array(self, *args, **kwargs):
        pass

    def _load_array(self, array_name, fam=None, **kwargs):
        if fam is self._unifamily or fam is None:
            self._subsnap_base._load_array(array_name, self._unifamily)

    def _derive_array(self, array_name, fam=None):
        if fam is self._unifamily or fam is None:
            self._subsnap_base._derive_array(array_name, self._unifamily)
