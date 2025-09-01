import pathlib

import h5py
import numpy as np

from ..util import dataset_view
from .gadgethdf import _GadgetHdfMultiFileManager
from .swift import SwiftSnap

try:
    import hdfstream
except ImportError:
    hdfstream = None


class _BaseSwiftMultiFileManager(_GadgetHdfMultiFileManager):

    def __init__(self, filename: pathlib.Path, take_cells, take_region, mode='r'):

        filename = str(filename)
        self._rootdir = None

        # Determine number of files and open the first file
        if self._is_hdf5(filename):
            # We've been given the name of a single snapshot file. This might
            # be a "virtual" snapshot which includes all sub-files, a complete
            # single file snapshot, or one of the sub-files in a multi file
            # snapshot. In the latter case we don't allow extracting regions
            # because they would be incomplete.
            file0 = self._open_hdf5_file(filename)
            if file0["Header"].attrs["NumFilesPerSnapshot"][0] > 1:
                if take_cells or take_region:
                    raise ValueError("Unable to extract regions from a SWIFT snapshot sub-file")
            # Store the name and index of the requested file
            self._filenames = [filename,]
            self._fileindex = [int(file0["Header"].attrs["ThisFile"][0]),]
            self._numfiles = 1
        else:
            # We're reading all files from a multi file snapshot
            file0 = self._open_hdf5_file(filename+".0.hdf5")
            self._numfiles = file0["Header"].attrs["NumFilesPerSnapshot"][0]
            self._filenames = [f"{filename}.{i}.hdf5" for i in range(self._numfiles)]
            self._fileindex = list(range(self._numfiles))
        self._open_files = {}

        # Determine which cells we need
        if take_cells is not None and take_region is not None:
            raise ValueError("Either take_cells or take_region must be specified, not both")
        self._take_cells = take_cells
        if take_region is not None:
            self._take_cells = self._identify_cells_to_take(file0, take_region)

        if self._take_cells is not None:

            # Avoid unwanted conversion of counts, offsets to scalar if _take_cells is a list
            # with only one element. Also ensure cell indexes are in ascending order.
            self._take_cells = np.sort(np.asarray(self._take_cells, dtype=int))

            # Read the cell information for each particle type and make nested
            # dicts of the form slices_in_file[file_nr][particle_type] = list_of_slices.
            slices_in_file = {}
            all_files = set()
            all_type_names = list(file0["Cells/Counts"])
            for ptype in all_type_names:
                # Read cells for this particle type
                counts = file0["Cells"]["Counts"][ptype][...][self._take_cells]
                offsets = file0["Cells"]["OffsetsInFile"][ptype][...][self._take_cells]
                files = file0["Cells"]["Files"][ptype][...][self._take_cells]
                # Find slices for each particle type in each file
                for count, offset, file in zip(counts, offsets, files):
                    if count > 0:
                        if file not in slices_in_file:
                            # This is the first slice in this file. Set an empty list of
                            # slices for all particle types.
                            slices_in_file[file] = {name : [] for name in all_type_names}
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
            if len(filenames) > 0:
                self._filenames = filenames
                self._fileindex = fileindex
                self._numfiles = len(self._filenames)
            else:
                # This can happen if using take_region or take_cells and reading a single sub-file
                raise ValueError("Snapshot file does not contain any part of the requested region")

    def _identify_cells_to_take(self, file0, take):
        centres = file0['Cells/Centres'][...]
        return np.where(take.cubic_cell_intersection(centres))[0]

    def get_unit_attrs(self):
        return self[0].parent['InternalCodeUnits'].attrs

    def get_header_attrs(self):
        return self[0].parent['Parameters'].attrs

    def _all_group_names(self):
        return self[0]['Cells/Counts'].keys()

    def _ensure_file_open(self, i):
        if i not in self._open_files:
            f = self._open_hdf5_file(self._filenames[i])
            if self._take_cells is not None:
                index = self._fileindex[i]
                f = dataset_view.GroupView(f["/"], slices=self._slices_in_file[index])
            self._open_files[i] = f

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

#
# Class for reading remote snapshots using the hdfstream service
#
# Remote snapshots take an extra parameter with the server URL, and we need to
# arrange for this to reach the multi file manager class.
#
class _RemoteSwiftMultiFileManager(_BaseSwiftMultiFileManager):

    def __init__(self, server, user, password, *args, **kwargs):
        self._server = server
        self._user = user
        self._password = password
        super().__init__(*args, **kwargs)

    def _connect(self):
        if self._rootdir is None:
            # Set lazy loading parameters such that we're likely to fetch the
            # cell metadata with the initial request.
            self._rootdir = hdfstream.open(self._server, "/", user=self._user, password=self._password,
                                           max_depth=3, data_size_limit=1048576)

    def _open_hdf5_file(self, filename):
        self._connect()
        return self._rootdir[filename]

    def _is_hdf5(self, filename):
        self._connect()
        return self._rootdir.is_hdf5(filename)
#
# To open a RemoteSwiftSnap we need the server URL and we might have a
# username and password if authentication is required.
#
class RemoteSwiftSnap(SwiftSnap):
    _max_buf = 2**63-1 # do not split requests into multiple chunks
    _multifile_manager_class = _RemoteSwiftMultiFileManager
    def __init__(self, *args, **kwargs):
        self._server = kwargs.pop("server", None)
        self._user = kwargs.pop("user", None)
        self._password = kwargs.pop("password", None)
        super().__init__(*args, **kwargs)

    def _init_hdf_filemanager(self, filename):
        self._hdf_files = self._multifile_manager_class(self._server, self._user, self._password,
                                                        filename, self._take_swift_cells, self._take_region)
#
# Class for reading local snapshots using h5py (e.g. to test _BaseSwiftMultiFileManager)
#
class _LocalSwiftMultiFileManager(_BaseSwiftMultiFileManager):

    def _open_hdf5_file(self, filename):
        return h5py.File(filename, "r")

    def _is_hdf5(self, filename):
        return h5py.is_hdf5(filename)

class LocalSwiftSnap(SwiftSnap):
    _multifile_manager_class = _LocalSwiftMultiFileManager
