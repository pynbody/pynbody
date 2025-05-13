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
        self._slices = None

        # Determine which cells we need
        if take_cells is not None and take_region is not None:
            raise ValueError("Either take_cells or take_region must be specified, not both")
        self._take_cells = take_cells
        if take_region is not None:
            self._take_cells = self._identify_cells_to_take(take_region)

        if self._take_cells is not None:

            # Avoid unwanted conversion of counts, offsets to scalar if _take_cells is a list
            # with only one element
            self._take_cells = np.asarray(self._take_cells, dtype=int)

            # Read the cell information for each particle type and make nested
            # dicts of the form slices_in_file[file_nr][particle_type] = list_of_slices.
            slices_in_file = {}
            all_files = set()
            file0 = self[0]
            for ptype in self._all_group_names():
                # Read cells for this particle type
                counts = file0["Cells"]["Counts"][ptype][...][self._take_cells]
                offsets = file0["Cells"]["OffsetsInFile"][ptype][...][self._take_cells]
                files = file0["Cells"]["Files"][ptype][...][self._take_cells]
                # Find slices for each particle type in each file
                for count, offset, file in zip(counts, offsets, files):
                    if count > 0:
                        if file not in slices_in_file:
                            slices_in_file[file] = {}
                        if ptype not in slices_in_file[file]:
                            slices_in_file[file][ptype] = []
                        slices_in_file[file][ptype].append(slice(offset, offset+count))
                        all_files.add(file)
                # Sort the list of slices by starting offset
                slices_in_file[file][ptype].sort(key=lambda x: x.start)
            self._slices_in_file = slices_in_file

            # If we're reading only specific cells we may not need all of the files.
            # Prune the list of required files and store index of files we're keeping
            filenames = []
            fileindex = []
            for name, index in zip(self._filenames, self._fileindex):
                if index in all_files:
                    filenames.append(name)
                    fileindex.append(index)
            self._filenames = filenames
            self.fileindex = fileindex
            self._numfiles = len(self._filenames)
            # Close file 0 if we don't need it
            if 0 not in all_files:
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
