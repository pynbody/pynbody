#!/bin/env python
#
# Contains classes to wrap h5py.Datasets to provide views which return only
# a subset of elements. These elements are identified by a list of slices to
# apply in the first dimension.
#

import numpy as np
from collections.abc import Mapping


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


class BaseView(Mapping):
    """
    Base class for the sliced Group, Dataset and File objects.
    Implements the (immutable) Mapping interface.

    This wraps the input object and forwards any attribute
    or item access to the wrapped object.
    """
    def __init__(self, obj):
        self.obj = obj

    def __getattr__(self, name):
        return getattr(self.obj, name)

    def __getitem__(self, key):
        return self.obj[key]

    def __iter__(self):
        return self.obj.__iter__()

    def __len__(self):
        return self.obj.__len__()


class SlicedDatasetView(BaseView):
    """
    This wraps a h5py.Dataset or similar object.
    """
    def __init__(self, obj, slices=None):
        if slices is not None:
            if len(slices) == 0:
                slices = [slice(0,0),] # empty list indicates zero selected elements
            else:
                slices = _join_slices(slices)
        self._slices = slices
        super(SlicedDatasetView, self).__init__(obj)
        # Compute total number of elements after slicing
        self.size = len(self)
        for s in obj.shape[1:]:
            self.size *= s

    def _load(self):
        """
        Returns the specified slices of the underlying dataset
        """
        return np.concatenate([self.obj[s,...] for s in self._slices], axis=0)

    def __getitem__(self, key):
        if self._slices:
            return self._load()[key]
        else:
            return self.obj[key]

    def __iter__(self):
        if self._slices:
            return self._load().__iter__()
        else:
            return self.obj.__iter__()

    def __len__(self):
        if self._slices:
            return sum([s.stop-s.start for s in self._slices])
        else:
            return self.obj.__len__()

    def __getattr__(self, name):
        if self._slices and name == "shape":
            shape = list(self.obj.shape)
            shape[0] = len(self)
            return tuple(shape)
        else:
            return getattr(self.obj, name)

    def read_direct(self, target):
        target[...] = self[...]


class GroupView(BaseView):
    """
    Class to wrap a h5py.Group or similar. This needs to ensure that if
    we open a sub-group or dataset we return a wrapped group or dataset
    rather than the underlying h5py object.

    The slices parameter should be a dict of {group_name : list_of_slices}
    pairs. If a group's name is in the dict then we only read the
    specified slices from any datasets in that group.

    TODO: might need to wrap visititems() and anything else that returns
    a group or dataset?
    """
    def __init__(self, obj, slices=None):
        self._slices = slices
        super(GroupView, self).__init__(obj)

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
        obj = self.obj[key]
        if self._is_dataset_like(obj):
            # If it's a dataset, check if we need to slice it
            name = self.obj.name.lstrip("/")
            if self._slices is not None and name in self._slices:
                slices = self._slices[name]
            else:
                slices = None
            result = SlicedDatasetView(obj, slices)
        elif self._is_group_like(obj):
            # If it's a group, pass on the dict of slices
            result = GroupView(obj, self._slices)
        else:
            # And if it's anything else, return the underlying object
            result = obj
        return result
