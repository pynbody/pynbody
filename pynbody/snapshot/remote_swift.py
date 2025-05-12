import pathlib

from .swift import SwiftSnap
from .gadgethdf import _GadgetHdfMultiFileManager

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

        if self._rootdir.is_hdf5(filename):
            self._filenames = [filename]
            self._numfiles = 1
        else:
            # Don't attempt to support multi file yet
            raise IOError(f"Unable to open file: {filename}")

        self._open_files = {0 : self._rootdir[self._filenames[0]]}

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

    def __iter__(self) :
        yield self._open_files[0]

class RemoteSwiftSnap(SwiftSnap):
    _multifile_manager_class = _RemoteSwiftMultiFileManager
