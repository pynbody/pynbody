# Contains classes to wrap h5py.Datasets to provide views which return only
# a subset of elements. These elements are identified by a list of slices to
# apply in the first dimension.
#

from collections.abc import Mapping

import numpy as np

try:
    import hdfstream
except ImportError:
    hdfstream = None


def _split_path(path):
    """
    Split a HDF5 path into a leading group name and remainder of the path
    """
    fields = [field for field in path.split('/') if field]
    if len(fields) == 0:
        raise ValueError("Path has no non-zero length components")
    elif len(fields) > 1:
        prefix = fields[0]
        remainder = "/".join(fields[1:])
    else:
        prefix = None
        remainder = fields[0]
    return prefix, remainder


def _join_slices(slices):
    """
    Join any consecutive slices in a list of slices.

    Must have step size of 1 and increasing start values.
    Slices must not overlap.
    """
    slices_out = []
    for s in slices:
        if s.start < 0 or s.stop < 0:
            raise ValueError("Slice start and stop must be positive")
        if s.stop < s.start:
            raise ValueError("Slices must have non-zero size")
        if s.step != 1 and s.step is not None:
            raise ValueError("Slice step must be one")
        if len(slices_out) > 0 and s.start < slices_out[-1].stop:
            raise ValueError("Slices must not overlap")
        if len(slices_out) > 0 and slices_out[-1].stop == s.start:
            slices_out[-1] = slice(slices_out[-1].start, s.stop, 1)
        else:
            slices_out.append(s)
    return slices_out


class SlicedDatasetView:
    """
    This wraps a h5py.Dataset or similar object.
    """
    def __init__(self, obj, slices=None):

        # Store a reference to the wrapped dataset
        self._obj = obj

        # Store selected slices, if any
        if slices is not None:
            if len(slices) == 0:
                slices = [slice(0,0),] # empty list indicates zero selected elements
            else:
                slices = _join_slices(slices)
        self._slices = slices

        # Compute total number of elements after slicing
        self.size = len(self)
        for s in obj.shape[1:]:
            self.size *= s

        # Store some other dataset properties
        self.shape = (len(self),) + self._obj.shape[1:]
        self.dtype = self._obj.dtype
        self.attrs = self._obj.attrs

    def _load(self):
        """
        Returns the specified slices of the underlying dataset
        """
        # If we're accessing a file using hdfstream we can fetch all slices
        # with one request
        if hdfstream is not None:
            if isinstance(self._obj, hdfstream.RemoteDataset):
                return self._obj.request_slices(self._slices)

        # Otherwise, fetch all slices
        data = [self._obj[s,...] for s in self._slices]

        # Concatenate if necessary (avoids a copy if we have only one slice)
        if len(data) > 1:
            return np.concatenate(data, axis=0)
        elif len(data) == 1:
            return data[0]
        else:
            raise RuntimeError("Requesting zero slices is not supported!")

    def __getitem__(self, key):
        if self._slices:
            return self._load()[key]
        else:
            return self._obj[key]

    def __iter__(self):
        if self._slices:
            return self._load().__iter__()
        else:
            return self._obj.__iter__()

    def __len__(self):
        if self._slices:
            return sum([s.stop-s.start for s in self._slices])
        else:
            return self._obj.__len__()

    def read_direct(self, array, source_sel=None, dest_sel=None):

        # If this is an un-sliced HDF5 dataset, can just call its read_direct()
        if self._slices is None:
            self._obj.read_direct(array, source_sel, dest_sel)
            return

        # May be able to download slices directly to destination if using hdfstream
        if hdfstream is not None:
            if isinstance(self._obj, hdfstream.RemoteDataset) and source_sel is None and dest_sel is None:
                self._obj.request_slices(self._slices, dest=target)
                return

        # Otherwise we need to read all slices and copy to the destination
        array[dest_sel] = self._load()[source_sel]


class GroupView(Mapping):
    """
    Class to wrap a h5py.Group or similar. This needs to ensure that if
    we open a sub-group or dataset we return a wrapped group or dataset
    rather than the underlying h5py object.

    The slices parameter should be a dict of {group_name : list_of_slices}
    pairs. If a group's name is in the dict then we only read the
    specified slices from any datasets in that group.
    """
    def __init__(self, obj, slices=None, parent=None):
        self._obj = obj
        self._slices = slices
        self.parent = self if parent is None else parent
        self.attrs = self._obj.attrs
        self.name = self._obj.name

    def _is_dataset_like(self, obj):
        return hasattr(obj, "shape")

    def _is_group_like(self, obj):
        return hasattr(obj, "visititems")

    def __getitem__(self, key):

        # Check if key refers to an object in a sub-group
        prefix, remainder = _split_path(key)
        if prefix is not None:
            return self[prefix][remainder]

        # Otherwise, key refers to something in this group
        obj = self._obj[key]
        if self._is_dataset_like(obj):
            # If it's a dataset, check if we need to slice it
            name = self._obj.name.lstrip("/")
            if self._slices is not None and name in self._slices:
                slices = self._slices[name]
            else:
                slices = None
            result = SlicedDatasetView(obj, slices)
        elif self._is_group_like(obj):
            # If it's a group, pass on the dict of slices
            result = GroupView(obj, self._slices, parent=self)
        else:
            # And if it's anything else, return the underlying object
            result = obj
        return result

    def __iter__(self):
        yield from self._obj.keys()

    def __len__(self):
        return self._obj.__len__()

    def visit(self, func):
        self._obj.visit(func)

    def visititems(self, func):
        """
        visititems() needs to pass wrapped objects to func()
        """
        def wrapper_func(name):
            return func(name, self[name])
        self._obj.visit(wrapper_func)
