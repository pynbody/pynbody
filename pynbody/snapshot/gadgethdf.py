"""
Implements reading HDF5 Gadget files, in various variants.

The gadget array names are mapped into pynbody array names according
to the mappings given by the config.ini section ``[gadgethdf-name-mapping]``.

The gadget particle groups are mapped into pynbody families according
to the mappings specified by the config.ini section ``[gadgethdf-type-mapping]``.
This can be many-to-one (many gadget particle types mapping into one
pynbody family), but only datasets which are common to all gadget types
will be available from pynbody.

Spanned files are supported. To load a range of files ``snap.0.hdf5``, ``snap.1.hdf5``, ... ``snap.n.hdf5``,
pass the filename ``snap``. If you pass e.g. ``snap.2.hdf5``, only file 2 will be loaded.
"""

import configparser
import functools
import itertools
import logging
import os
import warnings

import numpy as np

from .. import chunk, config_parser, family, units, util
from . import SimSnap, namemapper

logger = logging.getLogger('pynbody.snapshot.gadgethdf')

try:
    import h5py
except ImportError:
    h5py = None

_default_type_map = {}
for x in family.family_names():
    try:
        _default_type_map[family.get_family(x)] = \
                 [q.strip() for q in config_parser.get('gadgethdf-type-mapping', x).split(",")]
    except configparser.NoOptionError:
        pass

_all_hdf_particle_groups = []
for hdf_groups in _default_type_map.values():
    for hdf_group in hdf_groups:
        _all_hdf_particle_groups.append(hdf_group)


class _DummyHDFData:

    """A stupid class to allow emulation of mass arrays for particles
    whose mass is in the header"""

    def __init__(self, value, length, dtype):
        self.value = value
        self.length = length
        self.size = length
        self.shape = (length, )
        self.dtype = np.dtype(dtype)

    def __len__(self):
        return self.length

    def read_direct(self, target, source_sel=None):
        """Emulate h5py read_direct.

        The source_sel is ignored as filled with a single value.
        """
        target[:] = self.value


class _GadgetHdfMultiFileManager:
    _nfiles_groupname = "Header"
    _nfiles_attrname = "NumFilesPerSnapshot"
    _size_from_hdf5_key = "ParticleIDs"
    _subgroup_name = None

    def __init__(self, filename, mode='r') :
        filename = str(filename)
        self._mode = mode
        if h5py.is_hdf5(filename):
            self._filenames = [filename]
            self._numfiles = 1
        else:
            h1 = h5py.File(self._make_filename_for_cpu(filename, 0), mode)
            self._numfiles = self._get_num_files(h1)
            if hasattr(self._numfiles, "__len__"):
                assert len(self._numfiles) == 1
                self._numfiles = self._numfiles[0]
            self._filenames = [self._make_filename_for_cpu(filename, i) for i in range(self._numfiles)]

        self._open_files = {}

    def _get_num_files(self, first_file):
        return first_file[self._nfiles_groupname].attrs[self._nfiles_attrname]

    def _make_filename_for_cpu(self, filename, n):
        return filename + f".{n}.hdf5"

    def __len__(self):
        return self._numfiles

    def __iter__(self) :
        for i in range(self._numfiles) :
            if i not in self._open_files:
                self._open_files[i] = h5py.File(self._filenames[i], self._mode)
                if self._subgroup_name is not None:
                    self._open_files[i] = self._open_files[i][self._subgroup_name]

            yield self._open_files[i]

    def __getitem__(self, i) :
        try :
            return self._open_files[i]
        except KeyError :
            self._open_files[i] = next(itertools.islice(self,i,i+1))
            return self._open_files[i]

    def iter_particle_groups_with_name(self, hdf_family_name):
        for hdf in self:
            if hdf_family_name in hdf:
                if self._size_from_hdf5_key in hdf[hdf_family_name]:
                    yield hdf[hdf_family_name]

    def get_header_attrs(self):
        return self[0].parent['Header'].attrs

    def get_parameter_attrs(self):
        try:
            attrs = self[0].parent['Parameters'].attrs
        except KeyError:
            attrs = []

        if len(attrs) == 0:
            return self.get_header_attrs()
        else:
            return attrs


    def get_unit_attrs(self):
        return self[0].parent['Units'].attrs

    def get_file0_root(self):
        return self[0].parent

    def iterroot(self):
        for item in self:
            yield item.parent

    def reopen_in_mode(self, mode):
        if mode!=self._mode:
            self._open_files = {}
            self._mode = mode

class _GizmoHdfMultiFileManager(_GadgetHdfMultiFileManager):

    def get_unit_attrs(self):
        return self[0].parent['Header'].attrs


class _SubfindHdfMultiFileManager(_GadgetHdfMultiFileManager):
    _nfiles_groupname = "FOF"
    _nfiles_attrname = "NTask"
    _subgroup_name = "FOF"

class _HDFFileIterator:
    def __init__(self, hdf_file_iterator):
        """
        Initialize the HDF file iterator. specifically used for LoadControl.iterate_with_interrupts
        """
        self._hdf_file_iterator = hdf_file_iterator
        self.current_hdf_file = None
        self.file_index = -1
        self.particle_offset = 0 
        self.select_file(0)

    def select_file(self, offset):
        try:
            self.current_hdf_file = next(self._hdf_file_iterator)
            self.file_index += 1 # next file
            self.particle_offset = 0 # Reset offset for the new file
        except StopIteration:
            self.current_hdf_file = None
            self.file_index = -1
            self.particle_offset = 0
class _HDFArrayFiller:
    """A helper class to fill a pynbody array from an HDF5 dataset."""

    def __init__(self, sim_array_to_fill = None, hdf_dataset = None):
        
        # default element size for simulation arrays
        self.sim_element_size = 1 if sim_array_to_fill is None else self._get_element_size(sim_array_to_fill)
        self.file_element_size = 1 if hdf_dataset is None else self._get_element_size(hdf_dataset)
        self._update_scaling_factor()

    def _update_sim_element_size(self, sim_array_to_fill):
        """Update the element size for the simulation array."""
        self.sim_element_size = self._get_element_size(sim_array_to_fill)
        self._update_scaling_factor()
    
    def _update_file_element_size(self, hdf_dataset):
        """Update the element size for the HDF5 dataset."""
        self.file_element_size = self._get_element_size(hdf_dataset)
        self._update_scaling_factor()

    def _update_scaling_factor(self):
        """Update the scaling factor based on the current element sizes."""
        self.scaling_factor = self.sim_element_size / self.file_element_size
        self.need_rescale = (self.sim_element_size != self.file_element_size)

    def fill_array_from_hdf_dataset(self, sim_array_to_fill, hdf_dataset, source_sel: slice | np.ndarray | None, offset: int = 0):
        """Fill a simulation array from an HDF5 dataset, handling various indexing and data shapes."""
        if isinstance(hdf_dataset, _DummyHDFData):
            hdf_dataset.read_direct(sim_array_to_fill)
            return

        source_sel = self._preprocess_source_selection(source_sel, offset)

        if isinstance(source_sel, np.ndarray):
            self._fill_from_fancy_index(sim_array_to_fill, hdf_dataset, source_sel)
        elif isinstance(source_sel, slice):
            self._fill_from_slice(sim_array_to_fill, hdf_dataset, source_sel)
        elif source_sel is None:
            self._fill_entire_dataset(sim_array_to_fill, hdf_dataset)
        else:
            raise TypeError(f"Unsupported source_sel type: {type(source_sel)}. "
                            "Expected numpy.ndarray, slice, or None.")

    def _get_element_size(self, array):
        """Get the size of a single element in an array."""
        if hasattr(array, 'ndim') and array.ndim > 1:
            return int(np.prod(array.shape[1:]))
        elif hasattr(array, 'shape') and len(array.shape) > 1:
            return int(np.prod(array.shape[1:]))
        else:
            return 1

    def _preprocess_source_selection(self, source_sel, offset):
        """Apply offset to the source selection and optimize if possible."""
        if isinstance(source_sel, slice):
            return slice(source_sel.start + offset, source_sel.stop + offset)
        elif isinstance(source_sel, np.ndarray):
            source_sel = source_sel + offset
            # convert to slice for efficiency if the indices are contiguous
            if len(source_sel) > 1 and source_sel[-1] - source_sel[0] == len(source_sel) - 1:
                return slice(source_sel[0], source_sel[-1] + 1)
        return source_sel

    def _fill_from_fancy_index(self, sim_array_to_fill, hdf_dataset, source_sel):
        """Fill array from a non-contiguous (fancy) index."""
        id_min, id_max = source_sel[0], source_sel[-1]
        num_read = id_max - id_min + 1
        indices_in_read_chunk = source_sel - id_min

        contiguous_hdf_slice = self._get_contiguous_hdf_slice(id_min, id_max)

        data_chunk_from_hdf = hdf_dataset[contiguous_hdf_slice]
        data_chunk_from_hdf = data_chunk_from_hdf.reshape(num_read, *sim_array_to_fill.shape[1:])

        final_data_to_fill = data_chunk_from_hdf[indices_in_read_chunk]

        if sim_array_to_fill.shape == final_data_to_fill.shape:
            sim_array_to_fill[:] = final_data_to_fill
        else:
            sim_array_to_fill.reshape(final_data_to_fill.shape)[:] = final_data_to_fill

    def _fill_from_slice(self, sim_array_to_fill, hdf_dataset, source_sel):
        """Fill array from a contiguous slice."""
        
        if self.need_rescale:
            source_sel = self._get_contiguous_hdf_slice(source_sel.start, source_sel.stop - 1)

        num_elements = source_sel.stop - source_sel.start
        if len(hdf_dataset.shape) > 1:
            expected_chunk_shape = (num_elements,) + hdf_dataset.shape[1:]
        else:
            expected_chunk_shape = (num_elements,)

        assert sim_array_to_fill.size == np.prod(expected_chunk_shape)

        sim_array_reshaped = sim_array_to_fill.reshape(expected_chunk_shape)
        hdf_dataset.read_direct(sim_array_reshaped, source_sel=source_sel)

    def _fill_entire_dataset(self, sim_array_to_fill, hdf_dataset):
        """Fill array with the entire content of an HDF5 dataset."""
        assert sim_array_to_fill.size == np.prod(hdf_dataset.shape)
        sim_array_reshaped = sim_array_to_fill.reshape(hdf_dataset.shape)
        hdf_dataset.read_direct(sim_array_reshaped, source_sel=None)

    def _get_contiguous_hdf_slice(self, id_min, id_max):
        """Calculates the slice to select from an HDF5 file to get a contiguous block of data
        covering the particle range [id_min, id_max], accounting for differing data layouts
        between the file and memory.

        For example, a 3D position array in memory might be stored as a flat 1D array in the file.
        This function computes the correct start and end indices for the slice in the flat array.
        """
        if self.need_rescale:
            return np.s_[int(id_min * self.scaling_factor): int((id_max + 1) * self.scaling_factor)]
        else:
            return np.s_[id_min: id_max + 1]

class HDFArrayLoader:
    """A helper class to handle the loading of particle data arrays from Gadget HDF5 files.

    This class abstracts the logic for reading data from potentially multiple HDF5 files,
    mapping the on-disk particle types to pynbody's family structure, and handling
    partial loading of snapshots (i.e., loading only a subset of particles). It also
    handles chunked reading of the data to keep memory usage under control (see
    pynbody.chunk.LoadControl).
    """
    def __init__(self, hdf_files: _GadgetHdfMultiFileManager, all_families: list[family.Family], family_to_group_map: dict[family.Family, list[str]], max_buf: int, take: np.ndarray | None = None):
        """Initializes the HDFArrayLoader.

        Parameters
        ----------
        hdf_files : _GadgetHdfMultiFileManager
            The manager for the set of HDF5 files belonging to the snapshot.
        all_families : list of pynbody.family.Family
            A list of all pynbody families present in the simulation, correctly ordered.
        family_to_group_map : dict[pynbody.family.Family, list[str]]
            A dictionary mapping each pynbody family to a list of Gadget HDF particle
            group names (e.g., 'PartType0', 'PartType1').
        take : np.ndarray or None, optional
            If not None, this is an array of particle indices to load from the snapshot.
            This enables partial loading. If None, all particles are loaded. Default is None.
        """

        self._hdf_files = hdf_files
        self._all_families = all_families
        self._family_to_group_map = family_to_group_map
        
        self.partial_load = take is not None
        self.__init_file_map()
        self.__init_load_map(max_buf, take=take)

    def __init_file_map(self):
        """ Initialize the file map for particle types and families """

        family_slice_start = 0

        self._file_ptype_slice = {} # will map from gadget particle type to location in pynbody logical file map
        self._file_family_slice = {}
        self._file_interrupt_points = {} # Records cumulative particle counts at each file boundary for each type
        for fam in self._all_families:
            family_length = 0


            # A simpler and more readable version of the code below would be:
            #
            # for hdf_group in self._all_hdf_groups_in_family(fam):
            #     family_length += hdf_group[self._size_from_hdf5_key].size
            #
            # However, occasionally we need to know where in the pynbody file map the gadget particle types lie.
            # (Specifically this is used when loading subfind data.) So we need to expand that out a bit and also
            # keep track of the slice for each gadget particle type.

            ptype_slice_start = family_slice_start

            for particle_type in self._family_to_group_map[fam]:

                ptype_slice_len = 0
                self._file_interrupt_points[particle_type] = []
                for hdf_group in self._hdf_files.iter_particle_groups_with_name(particle_type):
                    ptype_slice_len += hdf_group[self._hdf_files._size_from_hdf5_key].size
                    self._file_interrupt_points[particle_type].append(ptype_slice_len)
                self._file_ptype_slice[particle_type] = slice(ptype_slice_start, ptype_slice_start + ptype_slice_len)
                family_length += ptype_slice_len
                ptype_slice_start += ptype_slice_len

            self._file_family_slice[fam] = slice(family_slice_start, family_slice_start + family_length)
            family_slice_start += family_length

        self._num_file_particle = family_slice_start
        
    def __init_load_map(self, max_buf, take = None):
        """ Set up family slice and particle count for loading """

        self._load_control = chunk.LoadControl(self._file_ptype_slice, max_buf, take) # use HDF groups type instead of family type here
        self._family_slice_to_load = {}
        self._num_particles_to_load = self._load_control.mem_num_particles

        family_slice_start = 0
        for fam in self._all_families:
            family_length = 0

            ptype_slice_start = family_slice_start

            for particle_type in self._family_to_group_map[fam]:

                ptype_slice_len = self._load_control.mem_family_slice[particle_type].stop - self._load_control.mem_family_slice[particle_type].start

                family_length += ptype_slice_len
                ptype_slice_start += ptype_slice_len

            self._family_slice_to_load[fam] = slice(family_slice_start, family_slice_start + family_length)
            family_slice_start += family_length
        

    def load_arrays(self, all_fams_to_load: list[family.Family], sim: SimSnap, array_name: str, translated_names: list[str]):
        """Load an array from the HDF files into the simulation snapshot.
        
        Parameters
        ----------
        all_fams_to_load : list of pynbody.family.Family
            A list of pynbody families for which to load the array.
        sim : SimSnap
            The parent SimSnap object.
        array_name : str
            The pynbody standard name for the array to load.
        translated_names : list of str
            A list of possible names for the array in the Gadget HDF file.

        """

        for loading_fam in all_fams_to_load:
            
            sim_fam_array, array_filler = self._get_array_filler(array_name, loading_fam, sim, translated_names)

            i0 = 0 # current write position in sim_fam_array

            # A 'gadget group name' is e.g. 'PartType0', 'PartType1' etc.
            for hdf_group_name in self._family_to_group_map[loading_fam]:
                if self._file_ptype_slice[hdf_group_name].stop <= self._file_ptype_slice[hdf_group_name].start:
                    continue
                # Create iterator for this group type across all files
                file_iterator = _HDFFileIterator(iter(self._hdf_files.iter_particle_groups_with_name(hdf_group_name)))

                last_file_index = -1
                for readlen, buf_index, mem_index in self._load_control.iterate_with_interrupts(
                        hdf_group_name, 
                        hdf_group_name, 
                        self._file_interrupt_points[hdf_group_name],  # file offset
                        file_iterator.select_file): 
                    if mem_index is None or file_iterator.current_hdf_file is None:
                        # Skip-read: advance on-disk cursor even when we don't copy into memory,
                        # otherwise the next actual read will start from the wrong disk position
                        # at chunk/file boundaries (e.g. slice at start == _max_buf). Refs #955
                        file_iterator.particle_offset += readlen
                        continue
                    i1 = i0 + mem_index.stop - mem_index.start

                    # Check if we need to load a new dataset
                    if last_file_index != file_iterator.file_index:
                        dataset = self._get_dataset_from_translated_names(sim, file_iterator.current_hdf_file, translated_names)
                        last_file_index = file_iterator.file_index

                    if dataset is not None:
                        target_array = sim_fam_array[i0:i1]
                        array_filler.fill_array_from_hdf_dataset(target_array, dataset, source_sel=buf_index,
                                                                       offset=file_iterator.particle_offset)
                    file_iterator.particle_offset += readlen
                    i0 = i1

    def _get_array_filler(self, array_name: str, loading_fam: family.Family, sim: SimSnap, translated_names: list[str]):
        """
        Set up and return the simulation family array and its corresponding HDF array filler.
        """
        
        sim_fam_array = sim[loading_fam][array_name]
        
        # Find the first dataset for this family to determine file element size
        first_dataset = None
        for hdf_group_name in self._family_to_group_map[loading_fam]:
            if self._file_ptype_slice[hdf_group_name].stop <= self._file_ptype_slice[hdf_group_name].start:
                continue
            for hdf_group in self._hdf_files.iter_particle_groups_with_name(hdf_group_name):
                first_dataset = self._get_dataset_from_translated_names(sim, hdf_group, translated_names)
                if first_dataset is not None:
                    break
            if first_dataset is not None:
                break

        if first_dataset is None:
            raise KeyError(f"No dataset found for {array_name} in family {loading_fam}")
        
        array_filler = _HDFArrayFiller(sim_fam_array, first_dataset)
        return sim_fam_array, array_filler

    def _get_dataset_from_translated_names(self, sim_obj, hdf_particle_group, translated_names):
        """Retrieve an HDF5 dataset from translated_names."""
        for translated_name in translated_names:
            try:
                dataset = sim_obj._get_hdf_dataset(hdf_particle_group, translated_name)
                return dataset
            except KeyError:
                continue
            

class GadgetHDFSnap(SimSnap):
    """
    Class that reads HDF Gadget snapshots.
    """

    _multifile_manager_class = _GadgetHdfMultiFileManager
    _readable_hdf5_test_key = "PartType?"
    _readable_hdf5_test_attr = None # if None, no attribute is checked
    _size_from_hdf5_key = "ParticleIDs"
    _namemapper_config_section = "gadgethdf-name-mapping"
    _softening_class_key = "SofteningClassOfPartType"
    _softening_comoving_key = "SofteningComovingClass"
    _softening_max_phys_key = "SofteningMaxPhysClass"

    _mass_pynbody_name = "mass"
    _eps_pynbody_name = "eps"
    _max_buf = 1024 * 512 # max_chunk for chunk.LoadControl

    _velocity_unit_key = 'UnitVelocity_in_cm_per_s'
    _length_unit_key = 'UnitLength_in_cm'
    _mass_unit_key = 'UnitMass_in_g'
    _time_unit_key = 'UnitTime_in_s'

    _units_need_hubble_factors = True

    def __init__(self, filename, **kwargs):
        """Initialise a Gadget HDF snapshot.

        Spanned files are supported. To load a range of files ``snap.0.hdf5``, ``snap.1.hdf5``, ... ``snap.n.hdf5``,
        pass the filename ``snap``. If you pass e.g. ``snap.2.hdf5``, only file 2 will be loaded.
        """

        super().__init__()

        self._filename = filename

        self._init_hdf_filemanager(filename)

        self._translate_array_name = namemapper.AdaptiveNameMapper(self._namemapper_config_section,
                                                                   return_all_format_names=True) # required for swift
        self._init_unit_information()
        self.__init_family_map()

        take = kwargs.pop("take", None)
        self.partial_load = take is not None
        self.__init_file_map(take)
        self._remove_empty_particle_groups()
        self.__init_loadable_keys()
        self.__infer_mass_dtype()
        self._init_properties()
        self._decorate()

    def _have_softening_for_particle_type(self, particle_type):
        attrs = self._get_hdf_parameter_attrs()
        class_name = self._softening_class_key + str(particle_type)
        return class_name in attrs

    def _get_softening_for_particle_type(self, particle_type):
        attrs = self._get_hdf_parameter_attrs()

        class_name = self._softening_class_key + str(particle_type)
        if class_name not in attrs:
            return None

        class_number = attrs[class_name]

        comoving_softening = attrs[self._softening_comoving_key + str(class_number)]
        max_physical_softening = attrs[self._softening_max_phys_key + str(class_number)]

        if comoving_softening > max_physical_softening / self.properties['a']:
            comoving_softening = max_physical_softening / self.properties['a']

        return comoving_softening

    def _have_softening_for_particle_group(self, particle_group):
        return self._have_softening_for_particle_type(int(particle_group.name[-1]))

    def _get_softening_for_particle_group(self, particle_group):
        return self._get_softening_for_particle_type(int(particle_group.name[-1]))


    def _get_hdf_header_attrs(self):
        return self._hdf_files.get_header_attrs()

    def _get_hdf_parameter_attrs(self):
        return self._hdf_files.get_parameter_attrs()

    def _get_hdf_unit_attrs(self):
        return self._hdf_files.get_unit_attrs()

    def _init_hdf_filemanager(self, filename):
        self._hdf_files = self._multifile_manager_class(filename)

    def __init_loadable_keys(self):

        self._loadable_family_keys = {}
        all_fams = self.families()
        if len(all_fams)==0:
            return

        for fam in all_fams:
            self._loadable_family_keys[fam] = {self._mass_pynbody_name}
            can_get_eps = True
            for hdf_group in self._all_hdf_groups_in_family(fam):
                for this_key in self._get_hdf_allarray_keys(hdf_group):
                    ar_name = self._translate_array_name(this_key, reverse=True)
                    self._loadable_family_keys[fam].add(ar_name)

                can_get_eps &= self._have_softening_for_particle_group(hdf_group)

            if can_get_eps:
                self._loadable_family_keys[fam].add(self._eps_pynbody_name)
            self._loadable_family_keys[fam] = list(self._loadable_family_keys[fam])

        self._loadable_keys = set(self._loadable_family_keys[all_fams[0]])
        for fam_keys in self._loadable_family_keys.values():
            self._loadable_keys.intersection_update(fam_keys)

        self._loadable_keys = list(self._loadable_keys)

    def _all_hdf_groups(self):
        for hdf_family_name in _all_hdf_particle_groups:
            yield from self._hdf_files.iter_particle_groups_with_name(hdf_family_name)

    def _all_hdf_groups_in_family(self, fam):
        for hdf_family_name in self._family_to_group_map[fam]:
            yield from self._hdf_files.iter_particle_groups_with_name(hdf_family_name)


    def __init_file_map(self, take):
        self._array_loader = HDFArrayLoader(self._hdf_files, self._families_ordered(), self._family_to_group_map, self._max_buf, take)
        self._gadget_ptype_slice = self._array_loader._file_ptype_slice
        self._family_slice = self._array_loader._family_slice_to_load
        self._num_particles = self._array_loader._num_particles_to_load

    def _remove_empty_particle_groups(self):
        """Remove particle groups that contain no particles from the internal family mapping.

        This is important for some formats like Arepo where tracer particles might be defined
        but not present, which can cause issues with `HaloCatalogue`.
        """
        for family_name in self._family_to_group_map:
            self._family_to_group_map[family_name] = [group_name for group_name in self._family_to_group_map[family_name]
                                           if self._gadget_ptype_slice[group_name].stop > self._gadget_ptype_slice[group_name].start]

    def __infer_mass_dtype(self):
        """Some files have a mixture of header-based masses and, for other partile types, explicit mass
        arrays. This routine decides in advance the correct dtype to assign to the mass array, whichever
        particle type it is loaded for."""
        mass_dtype = np.float64
        for hdf in self._all_hdf_groups():
            if "Coordinates" in hdf:
                mass_dtype = hdf['Coordinates'].dtype
        self._mass_dtype = mass_dtype

    def _families_ordered(self):
        # order by the PartTypeN
        all_families = list(self._family_to_group_map.keys())
        all_families_sorted = sorted(all_families, key=lambda v: self._family_to_group_map[v][0])
        return all_families_sorted

    def __init_family_map(self):
        type_map = {}
        for fam, g_types in _default_type_map.items():
            my_types = []
            for x in g_types:
                # Get all keys from all hdf files
                for hdf in self._hdf_files:
                    if x in list(hdf.keys()):
                        my_types.append(x)
                        break
            if len(my_types):
                type_map[fam] = my_types
        self._family_to_group_map = type_map

    def _family_has_loadable_array(self, fam, name):
        """Returns True if the array can be loaded for the specified family.
        If fam is None, returns True if the array can be loaded for all families."""
        return name in self.loadable_keys(fam)


    def _get_all_particle_arrays(self, gtype):
        """Return all array names for a given gadget particle type"""

        # this is a hack to flatten a list of lists
        l = [item for sublist in [self._get_hdf_allarray_keys(x[gtype]) for x in self._hdf_files] for item in sublist]

        # now just return the unique items by converting to a set
        return list(set(l))

    def loadable_keys(self, fam=None):
        if fam is None:
            return self._loadable_keys
        else:
            return self._loadable_family_keys[fam]


    @staticmethod
    def _write(self, filename=None):
        raise RuntimeError("Not implemented")

    def write_array(self, array_name, fam=None, overwrite=False):
        translated_name = self._translate_array_name(array_name)[0]

        self._hdf_files.reopen_in_mode('r+')

        try:
            if fam is None:
                target = self
                all_fams_to_write = self.families()
            else:
                target = self[fam]
                all_fams_to_write = [fam]
    
            for writing_fam in all_fams_to_write:
                i0 = 0
                target_array = self[writing_fam][array_name]
                for hdf in self._all_hdf_groups_in_family(writing_fam):
                    npart = hdf['ParticleIDs'].size
                    i1 = i0 + npart
                    target_array_this = target_array[i0:i1]
    
                    dataset = self._get_or_create_hdf_dataset(hdf, translated_name,
                                                              target_array_this.shape,
                                                              target_array_this.dtype)
    
                    dataset.write_direct(target_array_this.reshape(dataset.shape))
    
                    i0 = i1
        finally:
            self._hdf_files.reopen_in_mode('r')

    @staticmethod
    def _get_hdf_allarray_keys(group):
        """Return all HDF array keys underneath group (includes nested groups)"""
        keys = []

        def _append_if_array(to_list, name, obj):
            if not hasattr(obj, 'keys'):
                to_list.append(name)

        group.visititems(functools.partial(_append_if_array, keys))
        return keys

    def _get_or_create_hdf_dataset(self, particle_group, hdf_name, shape, dtype):
        if self._translate_array_name(hdf_name,reverse=True) == self._mass_pynbody_name:
            raise OSError("Unable to write the mass block due to Gadget header format")

        ret = particle_group
        for tpart in hdf_name.split("/")[:-1]:
            ret =ret[tpart]

        dataset_name = hdf_name.split("/")[-1]
        return ret.require_dataset(dataset_name, shape, dtype, exact=True)


    def _get_hdf_dataset(self, particle_group, hdf_name):
        """Return the HDF dataset resolving /'s into nested groups, and returning
        an apparent Mass array even if the mass is actually stored in the header"""
        if self._translate_array_name(hdf_name,reverse=True) == self._mass_pynbody_name:
            try:
                pgid = int(particle_group.name[-1])
                mtab = particle_group.parent['Header'].attrs['MassTable'][pgid]
                if mtab > 0:
                    return _DummyHDFData(mtab, particle_group[self._size_from_hdf5_key].size,
                                         self._mass_dtype)
            except (IndexError, KeyError):
                pass
        elif self._translate_array_name(hdf_name,reverse=True) == self._eps_pynbody_name:
            eps = self._get_softening_for_particle_group(particle_group)
            if eps is not None:
                return _DummyHDFData(eps, particle_group[self._size_from_hdf5_key].size, self._mass_dtype)


        ret = particle_group
        for tpart in hdf_name.split("/"):
            ret = ret[tpart]
        return ret

    @classmethod
    def _get_cosmo_factors(cls, hdf, arr_names) :
        """Return the cosmological factors for a given array"""
        matching_hdf_keys = lambda arr_name: [s for s in GadgetHDFSnap._get_hdf_allarray_keys(hdf)
                                             if ((s.endswith("/"+arr_name)) & ('PartType' in s))]
        if isinstance(arr_names, str):
            arr_name = arr_names
            match = matching_hdf_keys(arr_name)
        else:
            for arr_name in arr_names:
                match = matching_hdf_keys(arr_name)
                if len(match) > 0:
                    break

        if (arr_name == 'Mass' or arr_name == 'Masses') and len(match) == 0:
            # mass stored in header. We're out in the cold on our own.
            warnings.warn("Masses are either stored in the header or have another dataset name; assuming the cosmological factor %s" % units.h**-1)
            if cls._units_need_hubble_factors:
                return units.Unit('1.0'), units.h**-1
            else:
                return units.Unit('1.0'),
        if len(match) > 0 :
            attrs  = hdf[match[0]].attrs

            if 'aexp-scale-exponent' in attrs:
                aexp = attrs['aexp-scale-exponent']
            elif 'a_scaling' in attrs:
                # gadget4
                aexp = attrs['a_scaling']
            else:
                raise KeyError("Unable to find aexp-scale-exponent or a_scaling attribute for array %s" % arr_name)

            if 'h-scale-exponent' in attrs:
                hexp = attrs['h-scale-exponent']
            elif 'h_scaling' in attrs:
                # gadget4
                hexp = attrs['h_scaling']
            else:
                raise KeyError("Unable to find h-scale-exponent or h_scaling attribute for array %s" % arr_name)

            return units.a**util.fractions.Fraction.from_float(float(aexp)).limit_denominator(), units.h**util.fractions.Fraction.from_float(float(hexp)).limit_denominator()
        else :
            return units.Unit('1.0'), units.Unit('1.0')

    def _get_units_from_hdf_attr(self, hdfattrs) :
        """Return the units based on HDF attributes VarDescription"""
        if 'VarDescription' not in hdfattrs:
            warnings.warn("Unable to infer units from HDF attributes")
            return units.NoUnit()

        VarDescription = str(hdfattrs['VarDescription'])
        CGSConversionFactor = float(hdfattrs['CGSConversionFactor'])
        aexp = hdfattrs['aexp-scale-exponent']
        hexp = hdfattrs['h-scale-exponent']
        arr_units = self._get_units_from_description(VarDescription, CGSConversionFactor)
        if not np.allclose(aexp, 0.0):
            arr_units *= (units.a) ** util.fractions.Fraction.from_float(float(aexp)).limit_denominator()
        if not np.allclose(hexp, 0.0):
            arr_units *= (units.h) ** util.fractions.Fraction.from_float(float(hexp)).limit_denominator()
        return arr_units


    def _get_units_from_description(self, description, expectedCgsConversionFactor=None):
        arr_units = units.Unit('1.0')
        conversion = 1.0
        for unitname in list(self._hdf_unitvar.keys()):
            power = 1.
            if unitname in description:
                sstart = description.find(unitname)
                if sstart > 0:
                    if description[sstart - 1] == "/":
                        power *= -1.
                if len(description) > sstart + len(unitname):
                    # Just check we're not at the end of the line
                    if description[sstart + len(unitname)] == '^':
                        ## Has an index, check if this is negative
                        if description[sstart + len(unitname) + 1] == "-":
                            power *= -1.
                            power *= float(
                                description[sstart + len(unitname) + 2:-1].split()[0])  ## Search for the power
                        else:
                            power *= float(
                                description[sstart + len(unitname) + 1:-1].split()[0])  ## Search for the power
                if not np.allclose(power, 0.0):
                    arr_units *= self._hdf_unitvar[unitname] ** util.fractions.Fraction.from_float(
                        float(power)).limit_denominator()
                if not np.allclose(power, 0.0):
                    conversion *= self._hdf_unitvar[unitname].in_units(
                        self._hdf_cgsvar[unitname]) ** util.fractions.Fraction.from_float(
                        float(power)).limit_denominator()

        if expectedCgsConversionFactor is not None:
            if not np.allclose(conversion, expectedCgsConversionFactor, rtol=1e-3):
                raise units.UnitsException(
                    "Error with unit read out from HDF. Inferred CGS conversion factor is {!r} but HDF requires {!r}".format(
                    conversion, expectedCgsConversionFactor))

        return arr_units

    def _load_array(self, array_name, fam=None):
        if not self._family_has_loadable_array(fam, array_name):
            raise OSError("No such array on disk")
        else:

            translated_names = self._translate_array_name(array_name)
            dtype, dy, units = self.__get_dtype_dims_and_units(fam, translated_names)

            if array_name == self._mass_pynbody_name:
                dtype = self._mass_dtype
                # always load mass with this dtype, even if not the one in the file. This
                # is to cope with cases where it's partly in the header and partly not.
                # It also forces masses to the same dtype as the positions, which
                # is important for the KDtree code.

            if fam is None:
                target = self
                all_fams_to_load = self.families()
            else:
                target = self[fam]
                all_fams_to_load = [fam]

            target._create_array(array_name, dy, dtype=dtype)

            if units is not None:
                target[array_name].units = units
            else:
                target[array_name].set_default_units()

            self._array_loader.load_arrays(all_fams_to_load, self, array_name, translated_names)

    def __get_dtype_dims_and_units(self, fam, translated_names):
        if fam is None:
            fam = self.families()[0]

        inferred_units = units.NoUnit()
        representative_dset = None
        representative_hdf = None
        # not all arrays are present in all hdfs so need to loop
        # until we find one
        for hdf0 in self._hdf_files:
            for translated_name in translated_names:
                try:
                    representative_dset = self._get_hdf_dataset(hdf0[
                                                      self._family_to_group_map[fam][0]], translated_name)
                    break
                except KeyError:
                    continue

            if representative_dset is None:
                continue

            representative_hdf = hdf0
            if hasattr(representative_dset, "attrs"):
                inferred_units = self._get_units_from_hdf_attr(representative_dset.attrs)

            if len(representative_dset)!=0:
                # suitable for figuring out everything we need to know about this array
                break

        if representative_dset is None:
            raise KeyError("Array is not present in HDF file")


        assert len(representative_dset.shape) <= 2

        if len(representative_dset.shape) > 1:
            dy = representative_dset.shape[1]
        else:
            dy = 1

        # Some versions of gadget fold the 3D arrays into 1D.
        # So check if the dimensions make sense -- if not, assume we're looking at an array that
        # is 3D and cross your fingers
        npart = len(representative_hdf[self._family_to_group_map[fam][0]]['ParticleIDs'])

        if len(representative_dset) != npart:
            dy = len(representative_dset) // npart

        dtype = representative_dset.dtype
        if translated_name=="Mass":
            dtype = self._mass_dtype
        return dtype, dy, inferred_units

    def _init_unit_information(self):
        try:
            atr = self._hdf_files.get_unit_attrs()
        except KeyError:
            # Gadget 4 stores unit information in Parameters attr <sigh>
            atr = self._hdf_files.get_parameter_attrs()

        if self._velocity_unit_key not in atr.keys():
            warnings.warn("No unit information found in GadgetHDF file. Using gadget default units.", RuntimeWarning)
            vel_unit = config_parser.get('gadget-units', 'vel')
            dist_unit = config_parser.get('gadget-units', 'pos')
            mass_unit = config_parser.get('gadget-units', 'mass')
            self._file_units_system = [units.Unit(x) for x in [
                vel_unit, dist_unit, mass_unit, "K"]]
            return

        # Define the SubFind units, we will parse the attribute VarDescriptions for these
        if self._velocity_unit_key is not None:
            vel_unit = atr[self._velocity_unit_key]
        else:
            vel_unit = None

        dist_unit = atr[self._length_unit_key]
        mass_unit = atr[self._mass_unit_key]
        try:
            time_unit = atr[self._time_unit_key] * units.s
        except KeyError:
            # Gadget 4 seems not to store time units explicitly <sigh>
            time_unit = dist_unit/vel_unit

        if vel_unit is None:
            # Swift files don't store the velocity explicitly
            vel_unit = dist_unit / time_unit

        temp_unit = 1.0

        # Create a dictionary for the units, this will come in handy later
        unitvar = {'U_V': vel_unit * units.cm/units.s, 'U_L': dist_unit * units.cm,
                   'U_M': mass_unit * units.g,
                   'U_T': time_unit,
                   '[K]': temp_unit * units.K,
                   'SEC_PER_YEAR': units.yr,
                   'SOLAR_MASS': units.Msol,
                   'solar masses / yr': units.Msol/units.yr,
                   'BH smoothing': dist_unit}
        # Some arrays like StarFormationRate don't follow the pattern of U_ units
        cgsvar = {'U_M': 'g', 'SOLAR_MASS': 'g', 'U_T': 's',
                  'SEC_PER_YEAR': 's', 'U_V': 'cm s**-1', 'U_L': 'cm', '[K]': 'K',
                  'solar masses / yr': 'g s**-1', 'BH smoothing': 'cm'}

        self._hdf_cgsvar = cgsvar
        self._hdf_unitvar = unitvar

        cosmo = 'HubbleParam' in list(self._get_hdf_parameter_attrs().keys())
        if cosmo:
            try:
                for fac in self._get_cosmo_factors(self._hdf_files[0], 'Coordinates'): dist_unit *= fac
            except KeyError:
                if self._units_need_hubble_factors:
                    dist_unit *= units.a * units.h**-1
                else:
                    dist_unit *= units.a
                warnings.warn("Unable to find cosmological factors in HDF file; assuming position is %s" % dist_unit)
            try:
                for fac in self._get_cosmo_factors(self._hdf_files[0], 'Velocities'): vel_unit *= fac
            except KeyError:
                vel_unit *= units.a**(1,2)
                warnings.warn("Unable to find cosmological factors in HDF file; assuming velocity is %s" % vel_unit)
            try:
                for fac in self._get_cosmo_factors(self._hdf_files[0], ('Mass', 'Masses')): mass_unit *= fac
            except KeyError:
                if self._units_need_hubble_factors:
                    mass_unit *= units.h**-1
                warnings.warn("Unable to find cosmological factors in HDF file; assuming mass is %s" % mass_unit)

        self._file_units_system = [units.Unit(x) for x in [
            vel_unit*units.cm/units.s, dist_unit*units.cm, mass_unit*units.g, "K"]]

    @classmethod
    def _test_for_hdf5_key(cls, f):
        with h5py.File(f, "r") as h5test:
            test_key = cls._readable_hdf5_test_key
            found = False
            if test_key[-1]=="?":
                # try all particle numbers in turn
                for p in range(6):
                    test_key = test_key[:-1]+str(p)
                    if test_key in h5test:
                        found = True

            else:
                found = test_key in h5test

            if not found:
                return False

            if cls._readable_hdf5_test_attr is not None:
                location, attrname = cls._readable_hdf5_test_attr
                if location in h5test:
                    found = attrname in h5test[location].attrs
                else:
                    found = False
        return found

    @classmethod
    def _guess_file_ending(cls, f):
        return f.with_suffix(".0.hdf5")

    @classmethod
    def _can_load(cls, f):
        if hasattr(h5py, "is_hdf5"):
            if h5py.is_hdf5(f):
                return cls._test_for_hdf5_key(f)
            elif h5py.is_hdf5(cls._guess_file_ending(f)):
                return cls._test_for_hdf5_key(cls._guess_file_ending(f))
            else:
                return False
        else:
            if "hdf5" in f:
                warnings.warn(
                    "It looks like you're trying to load HDF5 files, but python's HDF support (h5py module) is missing.", RuntimeWarning)
            return False

    def _init_properties(self):
        atr = self._get_hdf_header_attrs()

        # expansion factor could be saved as redshift
        if 'ExpansionFactor' in atr:
            self.properties['a'] = atr['ExpansionFactor']
        elif 'Redshift' in atr:
            self.properties['a'] = 1. / (1 + atr['Redshift'])

        # Gadget 4 stores parameters in a separate dictionary <sigh>. For older formats, this will point back to the same
        # as the header attributes.
        atr = self._get_hdf_parameter_attrs()

        # not all omegas need to be specified in the attributes
        if 'OmegaBaryon' in atr:
            self.properties['omegaB0'] = atr['OmegaBaryon']
        if 'Omega0' in atr:
            self.properties['omegaM0'] = atr['Omega0']
        if 'OmegaLambda' in atr:
            self.properties['omegaL0'] = atr['OmegaLambda']
        if 'BoxSize' in atr:
            self.properties['boxsize'] = atr['BoxSize'] * self.infer_original_units('cm')
        if 'HubbleParam' in atr:
            self.properties['h'] = atr['HubbleParam']

        if 'a' in self.properties:
            self.properties['z'] = (1. / self.properties['a']) - 1


        # time unit might not be set in the attributes
        if "Time_GYR" in atr:
            self.properties['time'] = units.Gyr * atr['Time_GYR']
        else:
            from .. import analysis
            self.properties['time'] = analysis.cosmology.age(self) * units.Gyr

        for s,value in self._get_hdf_header_attrs().items():
            if s not in ['ExpansionFactor', 'Time_GYR', 'Time', 'Omega0', 'OmegaBaryon', 'OmegaLambda', 'BoxSize', 'HubbleParam']:
                self.properties[s] = value

class GizmoHDFSnap(GadgetHDFSnap):
    """
    Adapts the Gadget HDF reader to read recent versions of Gizmo snapshots.
    """
    # Use this test key, some Gizmo outputs (e.g. FIRE) lack Gizmo version info in header
    _readable_hdf5_test_key = "PartType0/ParticleIDGenerationNumber" 
    _multifile_manager_class = _GizmoHdfMultiFileManager
    _velocity_unit_key = 'UnitVelocity_In_CGS'
    _length_unit_key = 'UnitLength_In_CGS'
    _mass_unit_key = 'UnitMass_In_CGS'
    _param_file_velocity_unit_key = 'UnitVelocity_in_cm_per_s'
    _param_file_length_unit_key = 'UnitLength_in_cm'
    _param_file_mass_unit_key = 'UnitMass_in_g'
    _units_need_hubble_factors = False
    _namemapper_config_section = "gizmohdf-name-mapping"
    
    def __init__(self, filename, **kwargs):
    
        self._param_filename = kwargs.pop("param_filename", None)
        
        super().__init__(filename, **kwargs)

    def _get_units_from_hdf_attr(self, hdfattrs):
        # Gizmo doesn't seem to store any info about units in the attributes. Let pynbody use the default
        # dimensions combined with the file units system (if that's even available... otherwise just pure
        # guesswork!)
        return units.NoUnit()

    def _init_properties(self):
        atr = self._get_hdf_header_attrs()
        name_map = {'Omega_Baryon': 'omegaB0',
                    'Omega_Matter': 'omegaM0',
                    'Omega_Lambda': 'omegaL0'}
        for gizmo_name, pynbody_name in name_map.items():
            if gizmo_name in atr:
                self.properties[pynbody_name] = atr[gizmo_name]

        super()._init_properties()
        
        
    def _search_param_file(self):
        
        possible_paths = []
        
        sim_dir = os.path.dirname(self.filename)
        
        relative_paths = [
            "gizmo_parameters.txt-usedvalues",
            "../gizmo_parameters.txt-usedvalues",
            "../../gizmo_parameters.txt-usedvalues",
            "gizmo_parameters.txt",
            "../gizmo_parameters.txt",
            "../../gizmo_parameters.txt",
        ]
        
        for rel_path in relative_paths:
            abs_path = os.path.join(sim_dir, rel_path)
            abs_path = os.path.normpath(abs_path)
            possible_paths.append(abs_path)
        
        existing_files = []
        for file_path in possible_paths:
            if os.path.exists(file_path) and os.path.isfile(file_path):
                existing_files.append(file_path)
        
        if not existing_files:
            return None
        elif len(existing_files) == 1:
            return existing_files[0]
        else:
            warnings.warn(
                f"Multiple param files found. Using: {existing_files[0]}\n"
                f"Other found files: {existing_files[1:]}")
            return existing_files[0]
            
    def _get_gizmo_param_values(self, param_names):
        
        results = {name: None for name in param_names}
        found_count = 0
        
        with open(self._param_filename) as f:
            for line in f:
                if found_count >= len(param_names):
                    break
                
                # Remove inline comments
                if '#' in line:
                    line = line.split('#')[0]
                    
                if '%' in line:
                    line = line.split('%')[0]
                
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(None, 1)  # Split on whitespace, max 1 split
                
                if len(parts) >= 2:
                    current_param = parts[0]
                    value = parts[1].strip()
                    
                    if current_param in results and results[current_param] is None:
                        results[current_param] = value
                        found_count += 1
                    
        return results
        
    def _init_unit_information(self):
        try:
            atr = self._hdf_files.get_unit_attrs()
        except KeyError:
            atr = {"":""}
            
        if (self._velocity_unit_key not in atr.keys()):

            if self._param_filename is None:
                self._param_filename = self._search_param_file()

            if self._param_filename is not None:
                if not os.path.exists(self._param_filename):
                    raise FileNotFoundError(f"Parameter file not found: {self._param_filename}")

                keys = [self._param_file_velocity_unit_key, self._param_file_length_unit_key, self._param_file_mass_unit_key]
                atr = self._get_gizmo_param_values(keys)
                atr[self._velocity_unit_key] = units.Unit(atr[self._param_file_velocity_unit_key])
                atr[self._length_unit_key] = units.Unit(atr[self._param_file_length_unit_key])
                atr[self._mass_unit_key] = units.Unit(atr[self._param_file_mass_unit_key])
                self._units_need_hubble_factors = True
        
        if (self._velocity_unit_key not in atr.keys()):
            warnings.warn("No unit information found either in the HDF file itself, or as a Gizmo parameter file. Units will revert to defaults. To use a gizmo paramfile, provide its full path as 'param_filename'", RuntimeWarning)
            self._units_need_hubble_factors = False
            vel_unit = config_parser.get('gizmohdf-units', 'vel')
            dist_unit = config_parser.get('gizmohdf-units', 'pos')
            mass_unit = config_parser.get('gizmohdf-units', 'mass')
            self._file_units_system = [units.Unit(x) for x in [
                vel_unit, dist_unit, mass_unit, "K"]]
            return

        # Define the SubFind units, we will parse the attribute VarDescriptions for these
        if self._velocity_unit_key is not None:
            vel_unit = atr[self._velocity_unit_key]
        else:
            vel_unit = None

        dist_unit = atr[self._length_unit_key]
        mass_unit = atr[self._mass_unit_key]
        
        cosmo = 'HubbleParam' in list(self._get_hdf_parameter_attrs().keys())
        if cosmo:
            try:
                for fac in self._get_cosmo_factors(self._hdf_files[0], 'Coordinates'): dist_unit *= fac
            except KeyError:
                if self._units_need_hubble_factors:
                    dist_unit *= units.a * units.h**-1
                else:
                    dist_unit *= units.a
                warnings.warn("Unable to find cosmological factors in HDF file; assuming position is %s" % dist_unit)
            try:
                for fac in self._get_cosmo_factors(self._hdf_files[0], 'Velocities'): vel_unit *= fac
            except KeyError:
                vel_unit *= units.a**(1,2)
                warnings.warn("Unable to find cosmological factors in HDF file; assuming velocity is %s" % vel_unit)
            try:
                for fac in self._get_cosmo_factors(self._hdf_files[0], ('Mass', 'Masses')): mass_unit *= fac
            except KeyError:
                if self._units_need_hubble_factors:
                    mass_unit *= units.h**-1
                warnings.warn("Unable to find cosmological factors in HDF file; assuming mass is %s" % mass_unit)

        self._file_units_system = [units.Unit(x) for x in [
            vel_unit*units.cm/units.s, dist_unit*units.cm, mass_unit*units.g, "K"]]
            
                
class ArepoHDFSnap(GadgetHDFSnap):
    """
    Reads Arepo HDF snapshots.
    """
    _readable_hdf5_test_attr = "Config", "VORONOI"
    _softening_class_key = "SofteningTypeOfPartType"
    _softening_comoving_key = "SofteningComovingType"
    _softening_max_phys_key = "SofteningMaxPhysType"

    def _get_units_from_hdf_attr(self, hdfattrs):
        if 'length_scaling' not in hdfattrs:
            warnings.warn("Unable to infer units from HDF attributes")
            return units.NoUnit()
        l, m, v, a, h = (float(hdfattrs[x]) for x in ['length_scaling', 'mass_scaling', 'velocity_scaling', 'a_scaling', 'h_scaling'])
        base_units = [units.cm, units.g, units.cm/units.s, units.a, units.h]
        if float(hdfattrs['to_cgs'])==0.0:
            # 0.0 is used in dimensionless cases
            arr_units = units.Unit(1.0)
        else:
            arr_units = units.Unit(float(hdfattrs['to_cgs']))
        for exponent, base_unit in zip([l, m, v, a, h], base_units):
            if not np.allclose(exponent, 0.0):
                arr_units *= base_unit ** util.fractions.Fraction.from_float(float(exponent)).limit_denominator()
        return arr_units


class SubFindHDFSnap(GadgetHDFSnap) :
    """
    Reads the variant of Gadget HDF snapshots that include SubFind output inside the snapshot itself.
    """
    _multifile_manager_class = _SubfindHdfMultiFileManager
    _readable_hdf5_test_key = "FOF"


class EagleLikeHDFSnap(GadgetHDFSnap):
    """Reads Eagle-like HDF snapshots (download at http://data.cosma.dur.ac.uk:8080/eagle-snapshots/)"""
    _readable_hdf5_test_key = "PartType1/SubGroupNumber"

    def halos(self, subs=None):
        """Load the Eagle FOF halos, or if subs is specified the Subhalos of the given FOF halo number.

        *subs* should be an integer specifying the parent FoF number"""
        from .. import halo
        if subs:
            if not np.issubdtype(type(subs), np.integer):
                raise ValueError("The subs argument must specify the group number")
            parent_group = self[self['GroupNumber']==subs]
            if len(parent_group)==0:
                raise ValueError("No group found with id %d"%subs)

            cat = halo.number_array.HaloNumberCatalogue(parent_group,
                                     array="SubGroupNumber", ignore=np.max(self['SubGroupNumber']))
            cat._keep_subsnap_alive = parent_group # by default, HaloCatalogue only keeps a weakref (should this be changed?)
            return cat
        else:
            return halo.number_array.HaloNumberCatalogue(self, array="GroupNumber", ignore=np.max(self['GroupNumber']))

        
@GizmoHDFSnap.derived_array
def He(self) :
    He = self['metals_list'][:,1]
    return He
    
@GizmoHDFSnap.derived_array
def H(self) :
    H = 1 - self['metals_list'][:,0] - self['He']
    return H
   
@GizmoHDFSnap.derived_array
def C(self) :
    C = self['metals_list'][:,2]
    return C
    
@GizmoHDFSnap.derived_array
def N(self) :
    N = self['metals_list'][:,3]
    return N
    
@GizmoHDFSnap.derived_array
def O(self) :
    O = self['metals_list'][:,4]
    return O
    
@GizmoHDFSnap.derived_array
def Ne(self) :
    Ne = self['metals_list'][:,5]
    return Ne
     
@GizmoHDFSnap.derived_array
def Mg(self) :
    Mg = self['metals_list'][:,6]
    return Mg
    
@GizmoHDFSnap.derived_array
def Si(self) :
    Si = self['metals_list'][:,7]
    return Si
    
@GizmoHDFSnap.derived_array
def S(self) :
    S = self['metals_list'][:,8]
    return S
    
@GizmoHDFSnap.derived_array
def Ca(self) :
    Ca = self['metals_list'][:,9]
    return Ca
    
@GizmoHDFSnap.derived_array
def Fe(self) :
    Fe = self['metals_list'][:,10]
    return Fe
    
@GizmoHDFSnap.derived_array
def rprocess(self) :
    # Only stored in some FIRE simulations
    if self['metals_list'].shape[1] > 10:
        r_process_models = self['metals_list'][:,11:]
        return r_process_models
    
@GizmoHDFSnap.derived_array
def metals(self) :
    metals = self['metals_list'][:,0]
    # There's some small discrepancy with np.sum(self['metals_list'][:,2:11], axis = 1),
    # possibly due to reduced precission in float32.
    # FIRE-2 public release info is incorrect, as self['metals_list'][:,0] 
    # is clearly not equal to the H mass fraction
    return metals
    
## Gadget has internal energy variable
@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def u(self) :
    """Gas internal energy derived from snapshot variable or temperature"""
    try:
        u = self['InternalEnergy']
    except KeyError:
        gamma = 5./3
        u = self['temp']*units.k/(self['mu']*units.m_p*(gamma-1))

    return u

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def p(sim) :
    """Calculate the pressure for gas particles, including polytropic equation of state gas"""

    critpres = 2300. * units.K * units.m_p / units.cm**3 ## m_p K cm^-3
    critdens = 0.1 * units.m_p / units.cm**3 ## m_p cm^-3
    gammaeff = 4./3.

    oneos = sim.g['OnEquationOfState'] == 1.

    p = sim.g['rho'].in_units('m_p cm**-3') * sim.g['temp'].in_units('K')
    p[oneos] = critpres * (sim.g['rho'][oneos].in_units('m_p cm**-3')/critdens)**gammaeff

    return p

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def HII(sim) :
    """Number of HII ions per proton mass"""

    return sim.g["hydrogen"] - sim.g["HI"]

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def HeIII(sim) :
    """Number of HeIII ions per proton mass"""

    return sim.g["hetot"] - sim.g["HeII"] - sim.g["HeI"]

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def ne(sim) :
    """Number of electrons per proton mass, ignoring the contribution from He!"""
    ne = sim.g["HII"]  #+ sim["HeII"] + 2*sim["HeIII"]
    ne.units = units.m_p**-1

    return ne

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def rho_ne(sim) :
    """Electron number density per SPH particle, currently ignoring the contribution from He!"""

    return sim.g["ne"].in_units("m_p**-1") * sim.g["rho"].in_units("m_p cm**-3")

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def dm(sim) :
    """Dispersion measure per SPH particle currently ignoring n_e contribution from He """

    return sim.g["rho_ne"]

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def cosmodm(sim) :
    """Cosmological Dispersion measure per SPH particle includes (1+z) factor, currently ignoring n_e contribution from He """

    return sim.g["rho_ne"] * (1. + sim.g["redshift"])
@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def redshift(sim) :
    """Redshift from LoS Velocity 'losvel' """

    return np.exp( sim['losvel'].in_units('c') ) - 1.

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def doppler_redshift(sim) :
    """Doppler Redshift from LoS Velocity 'losvel' using SR """

    return np.sqrt( (1. + sim['losvel'].in_units('c')) / (1. - sim['losvel'].in_units('c'))  ) - 1.

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def em(sim) :
    """Emission Measure (n_e^2) per particle to be integrated along LoS"""

    return sim.g["rho_ne"]*sim.g["rho_ne"]

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def halpha(sim) :
    """H alpha intensity (based on Emission Measure n_e^2) per particle to be integrated along LoS"""

    ## Rate at which recombining electrons and protons produce Halpha photons.
    ## Case B recombination assumed from Draine (2011)
    #alpha = 2.54e-13 * (sim.g['temp'].in_units('K') / 1e4)**(-0.8163-0.0208*np.log(sim.g['temp'].in_units('K') / 1e4))
    #alpha.units = units.cm**(3) * units.s**(-1)

    ## H alpha intensity = coeff * EM
    ## where coeff is h (c / Lambda_Halpha) / 4Pi) and EM is int rho_e * rho_p * alpha
    ## alpha = 7.864e-14 T_1e4K from http://astro.berkeley.edu/~ay216/08/NOTES/Lecture08-08.pdf
    coeff = (6.6260755e-27) * (299792458. / 656.281e-9) / (4.*np.pi) ## units are erg sr^-1
    alpha = coeff * 7.864e-14 * (1e4 / sim.g['temp'].in_units('K'))

    alpha.units = units.erg * units.cm**(3) * units.s**(-1) * units.sr**(-1) ## It's intensity in erg cm^3 s^-1 sr^-1

    return alpha * sim["em"] # Flux erg cm^-3 s^-1 sr^-1

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def c_n_sq(sim) :
    """Turbulent amplitude C_N^2 for use in SM calculations (e.g. Eqn 20 of Macquart & Koay 2013 ApJ 776 2) """

    ## Spectrum of turbulence below the SPH resolution, assume Kolmogorov
    beta = 11./3.
    L_min = 0.1*units.Mpc
    c_n_sq = ((beta - 3.)/((2.)*(2.*np.pi)**(4.-beta)))*L_min**(3.-beta)*sim["em"]
    c_n_sq.units = units.m**(-20,3)

    return c_n_sq

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def hetot(self) :
    return self["He"]

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def hydrogen(self) :
    return self["H"]

## Need to use the ionisation fraction calculation here which gives ionisation fraction
## based on the gas temperature, density and redshift for a CLOUDY table
@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def HI(sim) :
    """Fraction of Neutral Hydrogen HI use limited CLOUDY table"""

    import pynbody.analysis.hifrac

    return pynbody.analysis.hifrac.calculate(sim.g,ion='hi')

## Need to use the ionisation fraction calculation here which gives ionisation fraction
## based on the gas temperature, density and redshift for a CLOUDY table, then applying
## selfshielding for the dense, star forming gas on the equation of state
@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def HIeos(sim) :
    """Fraction of Neutral Hydrogen HI use limited CLOUDY table, assuming dense EoS gas is selfshielded"""

    import pynbody.analysis.hifrac

    return pynbody.analysis.hifrac.calculate(sim.g,ion='hi', selfshield='eos')

## Need to use the ionisation fraction calculation here which gives ionisation fraction
## based on the gas temperature, density and redshift for a CLOUDY table, then applying
## selfshielding for the dense, star forming gas on the equation of state AND a further
## pressure based limit for
@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def HID12(sim) :
    """Fraction of Neutral Hydrogen HI use limited CLOUDY table, using the Duffy +12a prescription for selfshielding"""

    import pynbody.analysis.hifrac

    return pynbody.analysis.hifrac.calculate(sim.g,ion='hi', selfshield='duffy12')

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def HeI(sim) :
    """Fraction of Helium HeI"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='hei')

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def HeII(sim) :
    """Fraction of Helium HeII"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='heii')

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def OI(sim) :
    """Fraction of Oxygen OI"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='oi')

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def OII(sim) :
    """Fraction of Oxygen OII"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='oii')

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def OVI(sim) :
    """Fraction of Oxygen OVI"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='ovi')

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def CIV(sim) :
    """Fraction of Carbon CIV"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='civ')

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def NV(sim) :
    """Fraction of Nitrogen NV"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='nv')

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def SIV(sim) :
    """Fraction of Silicon SiIV"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='siiv')

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def MGII(sim) :
    """Fraction of Magnesium MgII"""

    import pynbody.analysis.ionfrac

    return pynbody.analysis.ionfrac.calculate(sim.g,ion='mgii')

# The Solar Abundances used in Gadget-3 OWLS / Eagle / Smaug sims
XSOLH=0.70649785
XSOLHe=0.28055534
XSOLC=2.0665436E-3
XSOLN=8.3562563E-4
XSOLO=5.4926244E-3
XSOLNe=1.4144605E-3
XSOLMg=5.907064E-4
XSOLSi=6.825874E-4
XSOLS=4.0898522E-4
XSOLCa=6.4355E-5
XSOLFe=1.1032152E-3

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def feh(self) :
    minfe = np.amin(self['Fe'][np.where(self['Fe'] > 0)])
    self['Fe'][np.where(self['Fe'] == 0)]=minfe
    return np.log10(self['Fe']/self['H']) - np.log10(XSOLFe/XSOLH)

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def sixh(self) :
    minsi = np.amin(self['Si'][np.where(self['Si'] > 0)])
    self['Si'][np.where(self['Si'] == 0)]=minsi
    return np.log10(self['Si']/self['Si']) - np.log10(XSOLSi/XSOLH)

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def sxh(self) :
    minsx = np.amin(self['S'][np.where(self['S'] > 0)])
    self['S'][np.where(self['S'] == 0)]=minsx
    return np.log10(self['S']/self['S']) - np.log10(XSOLS/XSOLH)

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def mgxh(self) :
    minmg = np.amin(self['Mg'][np.where(self['Mg'] > 0)])
    self['Mg'][np.where(self['Mg'] == 0)]=minmg
    return np.log10(self['Mg']/self['Mg']) - np.log10(XSOLMg/XSOLH)

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def oxh(self) :
    minox = np.amin(self['O'][np.where(self['O'] > 0)])
    self['O'][np.where(self['O'] == 0)]=minox
    return np.log10(self['O']/self['H']) - np.log10(XSOLO/XSOLH)

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def nexh(self) :
    minne = np.amin(self['Ne'][np.where(self['Ne'] > 0)])
    self['Ne'][np.where(self['Ne'] == 0)]=minne
    return np.log10(self['Ne']/self['Ne']) - np.log10(XSOLNe/XSOLH)

@SubFindHDFSnap.derived_array
def hexh(self) :
    minhe = np.amin(self['He'][np.where(self['He'] > 0)])
    self['He'][np.where(self['He'] == 0)]=minhe
    return np.log10(self['He']/self['He']) - np.log10(XSOLHe/XSOLH)

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def cxh(self) :
    mincx = np.amin(self['C'][np.where(self['C'] > 0)])
    self['C'][np.where(self['C'] == 0)]=mincx
    return np.log10(self['C']/self['H']) - np.log10(XSOLC/XSOLH)

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def caxh(self) :
    mincax = np.amin(self['Ca'][np.where(self['Ca'] > 0)])
    self['Ca'][np.where(self['Ca'] == 0)]=mincax
    return np.log10(self['Ca']/self['H']) - np.log10(XSOLCa/XSOLH)

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def nxh(self) :
    minnx = np.amin(self['N'][np.where(self['N'] > 0)])
    self['N'][np.where(self['N'] == 0)]=minnx
    return np.log10(self['N']/self['H']) - np.log10(XSOLH/XSOLH)

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def ofe(self) :
    minox = np.amin(self['O'][np.where(self['O'] > 0)])
    self['O'][np.where(self['O'] == 0)]=minox
    minfe = np.amin(self['Fe'][np.where(self['Fe'] > 0)])
    self['Fe'][np.where(self['Fe'] == 0)]=minfe
    return np.log10(self['O']/self['Fe']) - np.log10(XSOLO/XSOLFe)

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def mgfe(sim) :
    minmg = np.amin(sim['Mg'][np.where(sim['Mg'] > 0)])
    sim['Mg'][np.where(sim['Mg'] == 0)]=minmg
    minfe = np.amin(sim['Fe'][np.where(sim['Fe'] > 0)])
    sim['Fe'][np.where(sim['Fe'] == 0)]=minfe
    return np.log10(sim['Mg']/sim['Fe']) - np.log10(XSOLMg/XSOLFe)

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def nefe(sim) :
    minne = np.amin(sim['Ne'][np.where(sim['Ne'] > 0)])
    sim['Ne'][np.where(sim['Ne'] == 0)]=minne
    minfe = np.amin(sim['Fe'][np.where(sim['Fe'] > 0)])
    sim['Fe'][np.where(sim['Fe'] == 0)]=minfe
    return np.log10(sim['Ne']/sim['Fe']) - np.log10(XSOLNe/XSOLFe)

@GadgetHDFSnap.derived_array
@SubFindHDFSnap.derived_array
def sife(sim) :
    minsi = np.amin(sim['Si'][np.where(sim['Si'] > 0)])
    sim['Si'][np.where(sim['Si'] == 0)]=minsi
    minfe = np.amin(sim['Fe'][np.where(sim['Fe'] > 0)])
    sim['Fe'][np.where(sim['Fe'] == 0)]=minfe
    return np.log10(sim['Si']/sim['Fe']) - np.log10(XSOLSi/XSOLFe)
