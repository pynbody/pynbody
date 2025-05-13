import pathlib
import numpy as np

from .swift import SwiftSnap
from .gadgethdf import _GadgetHdfMultiFileManager
from ..util import slice_dataset

try:
    import hdfstream
except ImportError:
    hdfstream = None


class _RemoteSwiftMultiFileManager(_GadgetHdfMultiFileManager):

    def __init__(self, filename: pathlib.Path, take_cells, take_region, mode='r'):

        filename = str(filename)
        # TODO: mechanism to specify URL, user, password
        server = "https://dataweb.cosma.dur.ac.uk:8443/hdfstream"
        self._rootdir = hdfstream.open(server, "/")

        # Determine number of files and open the first file
        if self._rootdir.is_hdf5(filename):
            # Single file snapshot
            self._open_files = {0 : self._rootdir[filename]}
            self._filenames = [filename]
            self._fileindex = [0,]
            self._numfiles = 1
        else:
            # Multi file snapshot
            self._open_files = {0 : self._rootdir[filename+".0.hdf5"]}
            self._numfiles = self._open_files[0]["Header"].attrs["NumFilesPerSnapshot"][0]
            self._filenames = [f"{filename}.{i}.hdf5" for i in range(self._numfiles)]
            self._fileindex = list(range(self._numfiles))

        # Determine which cells we need
        if take_cells is not None and take_region is not None:
            raise ValueError("Either take_cells or take_region must be specified, not both")
        self._take_cells = take_cells
        if take_region is not None:
            self._take_cells = self._identify_cells_to_take(take_region)

        if self._take_cells is not None:
            # If we're reading only specific cells we may not need all of the files.
            # Determine which files the required cells are in.
            file0 = self[0]
            self._files_needed = set()
            for groupname in self._all_group_names():
                cell_file_nr = file0[f"Cells/Files/{groupname}"][...]
                self._files_needed |= set(cell_file_nr[np.asarray(self._take_cells, dtype=int)])
            # Should not have opened all of the files yet
            assert len(self._open_files) == 1
            # Prune the list of required files and store index of files we're keeping
            filenames = []
            fileindex = []
            for name, index in zip(self._filenames, self._fileindex):
                if index in self._files_needed:
                    filenames.append(name)
                    fileindex.append(index)
            self._filenames = filenames
            self.fileindex = fileindex
            self._numfiles = len(self._filenames)
            # Close file 0 if we don't need it
            if 0 not in self._files_needed:
                del self._open_files[0]

    def _identify_cells_to_take(self, take):
        centres = self[0]['Cells/Centres'][:]
        return np.where(take.cubic_cell_intersection(centres))[0]

    def get_unit_attrs(self):
        return self[0].parent['InternalCodeUnits'].attrs

    def get_header_attrs(self):
        return self[0].parent['Parameters'].attrs

    def iter_particle_groups_with_name(self, hdf_family_name):
        if hdf_family_name in self._open_files[0]:
            if self._size_from_hdf5_key in self._open_files[0][hdf_family_name]:
                yield self._open_files[0][hdf_family_name]

    def _all_group_names(self):
        return self[0]['Cells/Counts'].keys()

    def _ensure_file_open(self, i):
        if i not in self._open_files:
            self._open_files[i] = self._rootdir[self._filenames[i]]
            if self._subgroup_name is not None:
                self._open_files[i] = self._open_files[i][self._subgroup_name]

    def __iter__(self) :
        for i in range(self._numfiles) :
            self._ensure_file_open(i)
            yield self._open_files[i]

    def __getitem__(self, i) :
        self._ensure_file_open(i)
        return self._open_files[i]

    def iter_particle_groups_with_name(self, hdf_family_name):
        for hdf in self:
            if hdf_family_name in hdf:
                if self._size_from_hdf5_key in hdf[hdf_family_name]:
                    yield hdf[hdf_family_name]


class RemoteSwiftSnap(SwiftSnap):
    _multifile_manager_class = _RemoteSwiftMultiFileManager
