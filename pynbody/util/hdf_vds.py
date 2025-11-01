"""Toolset for creating a virtual dataset from multiple HDF5 files."""

from __future__ import annotations

import atexit
import os
import tempfile
import weakref
from typing import Iterable

import h5py


class TempHDF5File(h5py.File):
    """HDF5 file that auto-deletes on close or program exit."""
    
    def __init__(self, path, *args, **kwargs):
        # Create temporary file
        self._temp_path = path
        
        # Initialize h5py.File with temp path
        super().__init__(self._temp_path, *args, **kwargs)
        
        # Register cleanup handlers
        self._cleanup_registered = True
        atexit.register(self._cleanup)
        self._weakref = weakref.ref(self, lambda ref: self._cleanup())
    
    def _cleanup(self):
        """Remove temporary file if it exists."""
        if hasattr(self, '_cleanup_registered') and self._cleanup_registered:
            self._cleanup_registered = False
            if os.path.exists(self._temp_path):
                try:
                    os.unlink(self._temp_path)
                except OSError:
                    pass  # File might already be deleted
    
    def close(self):
        """Close file and delete temp file."""
        super().close()
        self._cleanup()
    
    @property
    def temp_path(self):
        """Get the temporary file path."""
        return self._temp_path


class HdfVdsMaker:
    """Tool for creating a virtual dataset from multiple HDF5 files."""
    def __init__(self, hdf_files: list[h5py.File | str]):
        self._files = []
        for f in hdf_files:
            if isinstance(f, h5py.File):
                self._files.append(f)
            else:
                self._files.append(h5py.File(f, 'r', locking=False))

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
        """Create an HDF file with virtual datasets combining the datasets in the input files."""
        with h5py.File(name=filepath, mode='w') as hdf_vfile:
            for k in self.concatenation_keys():
                self.write_single_vds(k, hdf_vfile)

            for k in self.copy_keys():
                self.write_single_vds(k, hdf_vfile, first_only=True)

        return hdf_vfile

    def write_single_vds(self, key: str, target_hdf_file: h5py.File, first_only: bool=False):
        """Write a single virtual dataset to the target HDF file."""
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

        # An ideal solution is to make a file then unlink it while keep it open, but Windows doesn't like that.
        # Instead, we create a temporary file then use a wrapper around HDF5 that deletes it on closure.
        self._temp_fd, self._temp_path = tempfile.mkstemp(suffix='.h5')
        os.close(self._temp_fd)

        hdf_vfile = h5py.File(name=self._temp_path, mode='w')
        for k in self.concatenation_keys():
            self.write_single_vds(k, hdf_vfile)

        for k in self.copy_keys():
            self.write_single_vds(k, hdf_vfile, first_only=True)

        hdf_vfile.close()

        # Windows seems to have a problem with reading when these files are still open, so close them
        for f in self._files:
            f.close()

        return TempHDF5File(self._temp_path, mode='r')
