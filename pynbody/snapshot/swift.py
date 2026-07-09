import pathlib

import h5py
import numpy as np

from .. import halo, units, util
from ..util import dataset_view
from .gadgethdf import GadgetHDFSnap, _GadgetHdfMultiFileManager

try:
    import hdfstream
except ImportError:
    hdfstream = None


class _BaseSwiftMultiFileManager(_GadgetHdfMultiFileManager):

    def __init__(self, filename: pathlib.Path, take_cells, take_region, mode='r'):

        filename = str(filename)

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

    def is_virtual(self):
        try:
            return self[0].parent['Header'].attrs['Virtual'][0] == 1
        except KeyError:
            return False

#
# Class for reading remote snapshots using the hdfstream service
#
class _RemoteSwiftMultiFileManager(_BaseSwiftMultiFileManager):

    def __init__(self, remote_dir, *args, **kwargs):
        self._rootdir = remote_dir
        assert self._rootdir is not None
        super().__init__(*args, **kwargs)

    def _open_hdf5_file(self, filename):
        return self._rootdir[filename]

    def _is_hdf5(self, filename):
        return self._rootdir.is_hdf5(filename)

#
# Class for reading local snapshots using h5py
#
class _LocalSwiftMultiFileManager(_BaseSwiftMultiFileManager):

    def _open_hdf5_file(self, filename):
        return h5py.File(filename, "r")

    def _is_hdf5(self, filename):
        return h5py.is_hdf5(filename)


class ExtractScalarWrapper:
    def __init__(self, underlying):
        self.underlying = underlying

    def __getitem__(self, name):
        val = self.underlying[name]
        try:
            return val[0]
        except (TypeError, IndexError):
            return val


class BaseSwiftSnap(GadgetHDFSnap):
    _readable_hdf5_test_key = "Policy"

    _velocity_unit_key = None
    _length_unit_key = 'Unit length in cgs (U_L)'
    _mass_unit_key = 'Unit mass in cgs (U_M)'
    _time_unit_key = 'Unit time in cgs (U_t)'

    _namemapper_config_section = 'swift-name-mapping'

    def __init__(self, filename, take_swift_cells=None, take_region=None):
        self._take_swift_cells = take_swift_cells
        self._take_region = take_region
        super().__init__(filename)


    def _init_hdf_filemanager(self, filename):
        self._hdf_files = self._multifile_manager_class(filename, self._take_swift_cells, self._take_region)

    def _is_cosmological(self):
        cosmo = ExtractScalarWrapper(self._hdf_files[0]['Cosmology'].attrs)
        return cosmo['Cosmological run'] == 1
    def _init_properties(self):
        params = ExtractScalarWrapper(self._hdf_files[0]['Parameters'].attrs)
        header = ExtractScalarWrapper(self._hdf_files[0]['Header'].attrs)
        cosmo = ExtractScalarWrapper(self._hdf_files[0]['Cosmology'].attrs)

        cosmological = self._is_cosmological()

        assert header['Dimension'] == 3, "Sorry, pynbody is only set up to deal with 3-dimensional swift simulations"

        if cosmological:
            self.properties['z'] = 1./(1.+cosmo['Scale-factor'])
            self.properties['a'] = cosmo['Scale-factor']
            self.properties['h'] = cosmo['h']
            # TODO: check these params are OK even at higher redshift (sample file is z=0)
            self.properties['omegaM0'] = cosmo['Omega_m']
            self.properties['omegaL0'] = cosmo['Omega_lambda']
            self.properties['omegaB0'] = cosmo['Omega_b']
            self.properties['omegaC0'] = cosmo['Omega_cdm']
            self.properties['omegaNu0'] = cosmo['Omega_nu_0']

            self.properties['boxsize'] = header['BoxSize']*self.infer_original_units('m')
            # Swift writes out 3D box sizes. Check it's actually a cube and if not emit a warning
            boxsize_3d = header.underlying['BoxSize']
            assert np.allclose(boxsize_3d[0], boxsize_3d)

        self.properties['time'] = header['Time']*self._hdf_unitvar['U_t']
        # Above should NOT be infer_original_units('s'), which assumes a three-way consistency between
        # position, velocity and time units that swift does not respect for cosmo sims.


    def _get_units_from_hdf_attr(self, hdfattrs):
        this_unit = units.Unit("1")
        for k in hdfattrs.keys():
            if k.endswith('exponent'):
                unitname = k.split(" ")[0]
                exponent = util.fractions.Fraction.from_float(float(ExtractScalarWrapper(hdfattrs)[k])).limit_denominator()

                if exponent != 0:
                    this_unit *= self._hdf_unitvar.get(unitname, units.no_unit)**exponent

        return this_unit

    def _init_unit_information(self):
        atr = ExtractScalarWrapper(self._hdf_files.get_unit_attrs())
        dist_unit = atr['Unit length in cgs (U_L)'] * units.cm
        mass_unit = atr['Unit mass in cgs (U_M)'] * units.g
        time_unit = atr['Unit time in cgs (U_t)'] * units.s
        temp_unit = atr['Unit temperature in cgs (U_T)'] * units.K
        vel_unit = dist_unit / time_unit

        unitvar = {'U_V': vel_unit,
                   'U_L': dist_unit,
                   'U_M': mass_unit,
                   'U_t': time_unit,
                   'U_T': temp_unit,
                   'a-scale': 1.0, # non-cosmo sims bizarrely still have non-zero scalefactor exponents
                   'h-scale': 1.0,}

        if self._is_cosmological():
            unitvar.update({
                'a-scale': units.a,
                'h-scale': units.h
            })


        self._hdf_unitvar = unitvar

        if self._is_cosmological():
            # when using hdf-provided exponents, the scalefactor will automatically be included. However, if using
            # infer_original_units, our best guess is that distances are comoving. So include the scalefactor
            # in self._file_units_system but not in self._hdf_unitvar.
            dist_unit = dist_unit * units.a

        self._file_units_system = [vel_unit, dist_unit, mass_unit, temp_unit]

    def halos(self, **kwargs):
        h = super().halos(**kwargs)

        if isinstance(h, halo.number_array.HaloNumberCatalogue):
            ignore = int(self._hdf_files.get_parameter_attrs()['FOF:group_id_default'])
            return halo.number_array.HaloNumberCatalogue(self, ignore=ignore, **kwargs)

        return h

#
# To open a RemoteSwiftSnap we need a hdfstream.RemoteDirectory object which contains the file(s)
#
class RemoteSwiftSnap(BaseSwiftSnap):
    _max_buf = 2**63-1 # do not split requests into multiple chunks
    _multifile_manager_class = _RemoteSwiftMultiFileManager
    def __init__(self, *args, **kwargs):
        self._remote_dir = kwargs.pop("remote_dir")
        super().__init__(*args, **kwargs)

    @classmethod
    def _can_load(cls, f):
        return False

    @classmethod
    def _can_load_remote(cls, f, *args, **kwargs):
        remote_dir = kwargs.get("remote_dir")
        return (hdfstream is not None) and (remote_dir is not None) and (f in remote_dir) and (remote_dir[f].is_hdf5())

    def _init_hdf_filemanager(self, filename):
        self._hdf_files = self._multifile_manager_class(self._remote_dir, filename, self._take_swift_cells, self._take_region)


class SwiftSnap(BaseSwiftSnap):
    _multifile_manager_class = _LocalSwiftMultiFileManager
