from __future__ import annotations

import os
import tempfile
from typing import Iterable

import h5py


class HdfVdsMaker:
    def __init__(self, hdf_files: list[h5py.File | str]):
        self._files = []
        for f in hdf_files:
            if isinstance(f, h5py.File):
                self._files.append(f)
            else:
                self._files.append(h5py.File(f, 'r'))

    def concatenation_keys(self) -> Iterable[str]:
        """Returns all keys to concatenate as VDS"""
        # TODO - make this general
        return['Subhalos', 'SubhaloParticles', 'NestedSubhalos']


    def copy_keys(self) -> Iterable[str]:
        """Returns all keys to copy from the first file into the VDS.

        Examples of copy keys are headers or one-off arrays"""
        # TODO - make this general
        return ['SnapshotId']

    def make_hdf_vfile(self, filepath: str) -> h5py.File:
        with h5py.File(name=filepath, mode='w') as hdf_vfile:
            for k in self.concatenation_keys():
                self.write_single_vds(k, hdf_vfile)

            for k in self.copy_keys():
                self.write_single_vds(k, hdf_vfile, first_only=True)

        return hdf_vfile

    def write_single_vds(self, key: str, target_hdf_file: h5py.File, first_only: bool=False):
        shape = None
        sources = []
        slices = []
        dtype = None
        offset = 0

        files = [self._files[0]] if first_only else self._files

        for f in files:
            source_dataset: h5py.Dataset = f[key]
            if shape is None:
                dtype = source_dataset.dtype
                shape = source_dataset.shape
            else:
                if dtype != source_dataset.dtype:
                    raise ValueError(f"The dtypes of array {key} are inconsistent between files")
                if source_dataset.shape[1:] != shape[1:]:
                    raise ValueError(f"The shapes of array {key} are inconsistent between files")
                shape = (source_dataset.shape[0] + shape[0],)+shape[1:]
            sources.append(h5py.VirtualSource(source_dataset))
            slices.append(slice(offset, offset+len(source_dataset)))
            offset+=len(source_dataset)

        layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
        for slice_, vsource in zip(slices, sources):
            layout[slice_] = vsource

        target_hdf_file.create_virtual_dataset(key, layout)


    def get_temporary_hdf_vfile(self) -> h5py.File:
        """Create the HDF file with virtual datasets in a temporary directory, such that it is deleted on closure"""

        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "nofile.hdf5")
            # ideally one would simply use  backing_store=False, to File but then there doesn't seem to be a way
            # to actually use the file (the VDS views just returns zeros).
            # Instead we write then re-read it, which presumably carries minimal overhead but is a bit ugly.

            self.make_hdf_vfile(filepath)

            hdf_vfile = h5py.File(name=filepath, mode='r')
        # on exiting the with block, the temporary directory and file are unlinked, but the file won't actually be
        # erased until it's closed.
        return hdf_vfile
