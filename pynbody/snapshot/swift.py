import re
import warnings

import h5py
import numpy as np

from .. import halo, units, util
from .gadgethdf import GadgetHDFSnap, _GadgetHdfMultiFileManager


class SwiftMultiFileManager(_GadgetHdfMultiFileManager):

    def __init__(self, filename, mode='r', take_swift_cells=None, take_region=None, **kwargs):

        super().__init__(filename, mode, **kwargs)

        if take_swift_cells is not None and take_region is not None:
            raise ValueError("Either take_swift_cells or take_region must be specified, not both")

        # We can only cut out regions or cells if we're reading all files in the set
        file0 = self[0] # the file might already have been opened
        if self._get_num_files(file0) != len(self._filenames) and (take_swift_cells is not None or take_region is not None):
            raise ValueError("Cannot select part of a sub-file from a multi file snapshot")

        # Determine which cells we need. Avoid reading cells if we don't
        # need them in case the snapshot doesn't have valid metadata (e.g.
        # the planetary example in the test data).
        if take_region is not None or take_swift_cells is not None:
            self._read_cell_metadata(file0)
            if take_region is not None:
                take_swift_cells = self._identify_cells_to_take(take_region)
            if take_swift_cells is not None:
                take_swift_cells = np.unique(np.asarray(take_swift_cells, dtype=int))
        self._take_swift_cells = take_swift_cells

        # Determine which files we need to read
        self._file_mask = self._select_files(self._filenames, take_swift_cells)

        # Discard the files we're not reading
        self._filenames = [filename for (filename, keep) in zip(self._filenames, self._file_mask) if keep]
        self._numfiles = len(self._filenames)
        self._open_files = {0: file0} if self._file_mask[0] else {}

    def _select_files(self, filenames, take_swift_cells):
        """Choose which files to open and return a boolean mask"""
        # If we're not reading a subset, open all of the files
        if take_swift_cells is None:
            return np.ones(len(filenames), dtype=bool)

        # Find the set of files which contain selected cells
        required_files = set()
        for name in self._cells:
            counts = self._cells[name]["counts"][take_swift_cells].astype(np.int64)
            files = self._cells[name]["files"][take_swift_cells].astype(np.int64)
            required_files = required_files.union(set(files[counts>0]))
            # Gadget and older Swift versions completely omit particle type
            # groups which would contain zero particles. We don't want to be
            # missing a particle type just because the selected region happens
            # not to contain that type, so ensure we open a file containing at
            # least one particle of the current type. This also prevents us
            # from opening zero files when no particles are in the region.
            if sum(counts) == 0:
                all_counts = self._cells[name]["counts"]
                all_files = self._cells[name]["files"]
                files_with_type = np.unique(all_files[all_counts>0])
                if len(files_with_type) > 0:
                    required_files.add(files_with_type[0])

        # Return the file selection mask
        mask = np.zeros(len(filenames), dtype=bool)
        mask[list(required_files)] = True
        return mask

    def get_take_parameter(self, families_ordered, family_to_group_map):
        """Return the array of particle indexes to read
        """
        # Handle the case where we're reading everything
        if self._take_swift_cells is None:
            return None

        # Compute particle "take" indices corresponding to the selected cells
        take = []
        ptype_offset = 0
        for family in families_ordered:
            for name in family_to_group_map[family]:
                # Find the cell metadata
                cell_file_index = self._cells[name]["files"]
                cell_file_offset = self._cells[name]["offsets"]
                particles_per_cell = self._cells[name]["counts"].copy()
                # Get the original number of files in the full snapshot
                nr_files_total = np.amax(cell_file_index) + 1
                # Treat cells in files we don't open as empty, because we're
                # computing an index into the subset of files we opened.
                particles_per_cell[self._file_mask[cell_file_index]==False] = 0
                # Compute the offset to the first particle in each file in the subset.
                # Note that the file index refers to the full set of snapshot files here.
                particles_per_file = np.zeros(nr_files_total, dtype=np.int64)
                np.add.at(particles_per_file, cell_file_index, particles_per_cell)
                offset_to_file = np.cumsum(particles_per_file, dtype=np.int64) - particles_per_file
                # Compute the offset to the first particle in each cell
                offset_to_cell = ptype_offset + offset_to_file[cell_file_index] + cell_file_offset
                # Extract offsets and lengths of the selected subsets of cells
                cell_offsets_to_take = offset_to_cell[self._take_swift_cells]
                cell_counts_to_take = particles_per_cell[self._take_swift_cells]
                # Construct a sorted array of particle indexes to take
                order = np.argsort(cell_offsets_to_take)
                for cell_count, cell_offset in zip(cell_counts_to_take[order], cell_offsets_to_take[order]):
                    take.append(np.arange(cell_offset, cell_offset+cell_count, dtype=np.int64))
                ptype_offset += np.sum(particles_per_cell, dtype=np.int64)
        return np.concatenate(take) if len(take) > 0 else np.zeros(0, dtype=np.int64)

    def _read_cell_metadata(self, h1):
        num_files = self._get_num_files(h1)
        numpart_total = (h1["Header"].attrs["NumPart_Total"].astype(np.int64) +
                         (h1["Header"].attrs["NumPart_Total_HighWord"].astype(np.int64) << 32))
        self._cells = {}
        for name in h1["Cells/Counts"]:
            # Get the total number of particles of this type. Read the header
            # because the PartType group might not be present if this sub-file
            # happens to contain zero particles.
            m = re.match(r"PartType([0-9]+)", name)
            if m is None:
                raise ValueError(f"Unexpected particle group name: {name}")
            else:
                nptot = numpart_total[int(m.group(1))]
            # Read the cell information
            self._cells[name] = {}
            self._cells[name]["counts"] = h1["Cells/Counts"][name][...]
            self._cells[name]["files"] = h1["Cells/Files"][name][...]
            self._cells[name]["offsets"] = h1["Cells/OffsetsInFile"][name][...]
            # Do some sanity checks
            if np.all(self._cells[name]["counts"] == 0):
                raise ValueError("No spatial index found in this snapshot; unable to load a region.")
            if np.any(self._cells[name]["counts"] < 0):
                raise ValueError("A SWIFT cell contains a negative number of particles!")
            if np.any((self._cells[name]["files"] < 0) | (self._cells[name]["files"] >= num_files)):
                raise ValueError("A SWIFT cell's file index is out of range!")
            if np.any(self._cells[name]["offsets"] < 0):
                raise ValueError("A SWIFT cell offset is negative!")
            if np.any((self._cells[name]["offsets"] + self._cells[name]["counts"]) > nptot):
                raise ValueError("A SWIFT cell's offset+count is out of range")
            if np.sum(self._cells[name]["counts"], dtype=np.int64) != nptot:
                raise ValueError("The total number of particles in all cells does not match the snapshot header")
        self._cell_centres = h1["Cells/Centres"][...]

    def _identify_cells_to_take(self, take):
        return np.where(take.cubic_cell_intersection(self._cell_centres))[0]

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

    def __init__(self, filename, take_swift_cells=None, take_region=None, **kwargs):
        """Initialise a SWIFT snapshot.

        Extra parameters can be passed to the multi file manager by
        storing them here and overriding the _init_hdf_filemanager
        method.
        """
        self._take_swift_cells = take_swift_cells
        self._take_region = take_region
        super().__init__(filename, **kwargs)

    def _init_hdf_filemanager(self, filename, **kwargs):
        self._hdf_files = self._multifile_manager_class(filename, take_swift_cells=self._take_swift_cells,
                                                        take_region=self._take_region, **kwargs)

    def _get_take_parameter(self, **kwargs):
        return self._hdf_files.get_take_parameter(list(self._families_ordered()), self._family_to_group_map)

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

    @staticmethod
    def _get_hdf_allarray_keys(group):
        """
        Return all HDF array keys underneath group (includes nested groups)

        Swift snapshots do not have nested groups in the PartTypeX groups and
        checking the type of every key can be slow when there are many keys.
        Here we check that we have the expected number of keys (given by the
        NumberOfFields attribute) and assume that they're all datasets.
        """
        if "NumberOfFields" in group.attrs:
            if group.attrs["NumberOfFields"][0] == len(group):
                return list(group.keys())
            else:
                # NumberOfFields can be wrong if we wrote a new field to the snapshot, for example.
                warnings.warn(f"The NumberOfFields attribute of group {group.name} does not match the number of keys in the group",
                              RuntimeWarning)
                return GadgetHDFSnap._get_hdf_allarray_keys(group)
        else:
            raise ValueError(f"The expected NumberOfFields attribute of group {group.name} is not present")

    def write_array(self, *args, **kwargs):
        """
        We can't yet write arrays if we selected a region or cells
        """
        if self._take_region is not None or self._take_swift_cells is not None:
            raise NotImplementedError("Unable to write array to a selected region or cells")
        super().write_array(*args, **kwargs)

    def _update_group_attributes_on_write(self, hdf):
        """
        Update group's NumberOfFields attribute on writing an array
        """
        if "NumberOfFields" in hdf.attrs:
            hdf.attrs.modify("NumberOfFields", (len(hdf),))
