import os
import pathlib
import tempfile

import h5py
import numpy as np

from .. import halo, units, util
from .gadgethdf import GadgetHDFSnap, _GadgetHdfMultiFileManager


class SwiftMultiFileManager(_GadgetHdfMultiFileManager):

    def __init__(self, filename: pathlib.Path, take_cells, take_region, mode='r'):
        self._take_cells = take_cells

        if take_cells is not None and take_region is not None:
            raise ValueError("Either take_cells or take_region must be specified, not both")

        if take_cells is not None or take_region is not None:
            # we need to avoid loading a VDS file, as it won't have the info needed to make our own VDS
            # pointers to the right cells
            try:
                with h5py.File(filename, mode='r') as f:
                    if f['Header'].attrs['Virtual'][0] == 1 and filename.suffix == '.hdf5':
                        # if we simply strip off .hdf5, GadgetHDFSnap will find the underlying files (hopefully..)
                        filename = filename.with_suffix('')
                    # it's not a virtual file anyway, so we're ok
            except FileNotFoundError:
                pass # perfect, this is either going to be pieced together by pynbody, or it'll fail anyway
        super().__init__(filename, mode)

        if take_region is not None:
            # convert take to take_cells
            take_cells = self._identify_cells_to_take(take_region)

        # hopefully the shenanigans above mean we don't end up with a VDS, but just in case we do, check again:
        if self.is_virtual() and take_cells is not None:
            raise ValueError("Can't take a subset of cells from a VDS-based HDF5 file pointing to multiple underlying files")

        self._make_hdf_vfile(take_cells)

    def _identify_cells_to_take(self, take):
        centres = self[0]['Cells/Centres'][:]
        return np.where(take.cubic_cell_intersection(centres))[0]

    def get_unit_attrs(self):
        return self[0].parent['InternalCodeUnits'].attrs

    def get_header_attrs(self):
        return self[0].parent['Parameters'].attrs

    def is_virtual(self):
        return self[0].parent['Header'].attrs['Virtual'][0] == 1

    def iter_particle_groups_with_name(self, hdf_family_name):
        if hdf_family_name in self._hdf_vfile:
            if self._size_from_hdf5_key in self._hdf_vfile[hdf_family_name]:
                yield self._hdf_vfile[hdf_family_name]

    def _all_group_names(self):
        return self[0]['Cells/Counts'].keys()

    def _make_hdf_vfile(self, take_cells):
        temp_fd, temp_path = tempfile.mkstemp(suffix='.h5')
        os.close(temp_fd)

        from ..util.hdf_vds import TempHDF5File

        with h5py.File(name=temp_path, mode='w') as hdf_vfile:
            # ideally one would simply use  backing_store=False, to File but then there doesn't seem to be a way
            # to actually use the file (the VDS views just returns zeros).
            # Instead we write then re-read it, which presumably carries minimal overhead but is a bit ugly.

            for group_name in self._all_group_names():

                if take_cells is None:
                    source_groups, source_slices = self._generate_groups_and_slices_for_full_file(group_name)
                else:
                    source_groups, source_slices = self._generate_groups_and_slices_from_cells(group_name,
                                                                                                take_cells)

                target_group = hdf_vfile.create_group(group_name)
                self._make_hdf_group_with_slicing(source_groups, source_slices, target_group)

        self._hdf_vfile = TempHDF5File(temp_path, mode='r')


    def _generate_groups_and_slices_for_full_file(self, group_name):
        source_groups = [hdf_file[group_name] for hdf_file in self]
        source_lens = [len(f[group_name][self._size_from_hdf5_key]) for f in self]
        source_slices = [[slice(0, l)] for l in source_lens]
        return source_groups, source_slices

    def _generate_groups_and_slices_from_cells(self, group_name, take_cells):
        # we get the cell info from the first file, and assume it's consistent between files
        offsets = self[0]['Cells']['OffsetsInFile'][group_name]
        lens = self[0]['Cells']['Counts'][group_name]
        file_responsible = self[0]['Cells']['Files'][group_name]

        # First, make a map from the file ID to the slices to be taken from that file
        file_id_to_slices_map = {}
        for cell in take_cells:
            if file_responsible[cell] not in file_id_to_slices_map:
                file_id_to_slices_map[file_responsible[cell]] = []
            sl = slice(offsets[cell], offsets[cell] + lens[cell])
            file_id_to_slices_map[file_responsible[cell]].append(sl)

        # Now, reformat everything into the format expected by _make_hdf_group_with_slicing
        source_slices = []
        source_groups = []

        for file_id in sorted(list(file_id_to_slices_map)):
            source_groups.append(self[file_id][group_name])
            source_slices.append(sorted(file_id_to_slices_map[file_id]))

        return source_groups, source_slices

    def _make_hdf_group_with_slicing(self, source_groups, source_slices, target_group):
        total_len = 0
        for take_slices in source_slices:
            for s in take_slices:
                assert s.step is None
                total_len += s.stop - s.start

        for array_name in source_groups[0]:
            offset = 0
            target_dims = (total_len,) + source_groups[0][array_name].shape[1:]
            layout = h5py.VirtualLayout(shape=target_dims, dtype=source_groups[0][array_name].dtype)

            for source_group, take_slices in zip(source_groups, source_slices):
                source = h5py.VirtualSource(source_group[array_name])
                for s in take_slices:
                    layout[slice(offset, offset+s.stop-s.start)] = source[s]
                    offset += s.stop-s.start

            target_group.create_virtual_dataset(array_name, layout)


class ExtractScalarWrapper:
    def __init__(self, underlying):
        self.underlying = underlying

    def __getitem__(self, name):
        val = self.underlying[name]
        try:
            return val[0]
        except (TypeError, IndexError):
            return val

class SwiftSnap(GadgetHDFSnap):
    _multifile_manager_class = SwiftMultiFileManager
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

        self.properties['time'] = header['Time']*self.infer_original_units("s")


    def _get_units_from_hdf_attr(self, hdfattrs):
        cosmological = self._is_cosmological()
        this_unit = units.Unit("1")

        for k in hdfattrs.keys():
            if k.endswith('exponent'):
                if (k.startswith('a-scale') or k.startswith('h-scale')) and not cosmological:
                    continue
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
                   'a-scale': units.a,
                   'h-scale': units.h}


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
