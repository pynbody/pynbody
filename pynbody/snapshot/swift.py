import os
import pathlib
import shutil
import tempfile
import weakref

import h5py
import numpy as np

from .. import halo, units, util
from .gadgethdf import GadgetHDFSnap, _GadgetHdfMultiFileManager


class SwiftMultiFileManager(_GadgetHdfMultiFileManager):

    def get_unit_attrs(self):
        return self[0].parent['InternalCodeUnits'].attrs

    def get_header_attrs(self):
        return self[0].parent['Parameters'].attrs

    def is_virtual(self):
        try:
            return self[0].parent['Header'].attrs['Virtual'][0] == 1
        except KeyError:
            return False

    def _all_group_names(self):
        return self[0]['Cells/Counts'].keys()


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


    def _get_take_parameter(self, **kwargs):
        if len({"take", "take_swift_region", "take_swift_cells"} & kwargs.keys()) > 1:
            raise ValueError("Can only specify one of take, take_cells or take_region")
        if "take" in kwargs:
            return kwargs.pop("take")
        elif "take_swift_cells" in kwargs:
            return self._take_from_cells(kwargs.pop("take_swift_cells"))
        elif "take_region" in kwargs:
            return self._take_from_region(kwargs.pop("take_region"))
        else:
            return None

    def _take_from_cells(self, take_cells):
        if take_cells is None:
            return None

        # Check if we're reading one file from a multi file snapshot
        snap = self._hdf_files[0]
        header = ExtractScalarWrapper(snap["Header"].attrs)
        nr_files = header["NumFilesPerSnapshot"]
        if len(self._hdf_files) < nr_files:
            raise NotImplementedError("Can't use take_cells or take_region on a single sub-file")

        # Validate the input cell index array
        take_cells = np.unique(np.asarray(take_cells, dtype=int))
        nr_cells = snap["Cells/Centres"].shape[0]
        if np.any(take_cells < 0) or np.any(take_cells >= nr_cells):
            raise ValueError("SWIFT cell index is out of range!")

        # Compute particle indices corresponding to the selected cells
        take = []
        ptype_offset = 0
        for family in self._families_ordered():
            for name in self._family_to_group_map[family]:
                cell_file_index = snap["Cells/Files"][name][...]
                cell_file_offset = snap["Cells/OffsetsInFile"][name][...]
                particles_per_cell = snap["Cells/Counts"][name][...]
                particles_per_file = np.bincount(cell_file_index, weights=particles_per_cell, minlength=nr_files)
                offset_to_file = np.cumsum(particles_per_file, dtype=np.int64) - particles_per_file
                offset_to_cell = ptype_offset + offset_to_file[cell_file_index] + cell_file_offset
                cell_offsets_to_take = offset_to_cell[take_cells]
                cell_counts_to_take = particles_per_cell[take_cells]
                order = np.argsort(cell_offsets_to_take)
                for cell_count, cell_offset in zip(cell_counts_to_take[order], cell_offsets_to_take[order]):
                    take.append(np.arange(cell_offset, cell_offset+cell_count, dtype=np.int64))
                ptype_offset += np.sum(particles_per_cell, dtype=np.int64)
        return np.concatenate(take)

    def _take_from_region(self, take_region):
        if take_region is None:
            return None
        centres = self._hdf_files[0]['Cells/Centres'][:]
        take_cells = np.where(take_region.cubic_cell_intersection(centres))[0]
        return self._take_from_cells(take_cells)

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
