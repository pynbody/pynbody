"""Support for the SubFind halo finder, in the HDF5 format used by Gadget3, 4 and Arepo."""

from __future__ import annotations

import os.path
import warnings

import h5py
import numpy as np

from .. import array, config_parser, snapshot, units
from ..snapshot import gadgethdf
from . import Halo, HaloCatalogue
from .details import number_mapping, particle_indices
from .subhalo_catalogue import SubhaloCatalogue


class SubFindHDFHaloCatalogue(HaloCatalogue) :
    """Handles catalogues produced by the SubFind halo finder, in the HDF5 format used by Gadget3, 4 and Arepo.

    Since the internal format differs quite substantially between these versions, child classes are provided for
    Gadget 4, Arepo and TNG. The base class is able to handle the common elements of the format, and is also used
    for Gadget 3 SubFind outputs.

    See :class:`Gadget4SubfindHDFCatalogue`, :class:`ArepoSubfindHDFCatalogue` and :class:`TNGSubfindHDFCatalogue`.

    .. warning::

        At present, the Gadget 4, Arepo and TNG subclasses of this class are not tested against multi-file
        outputs. If you encounter issues with these, please report them to the pynbody developers.

    """

    # Names of various groups and attributes in the hdf file (which vary in different versions of SubFind)

    _fof_name = 'FOF'
    _header_name = 'FOF'
    _subfind_name = 'SUBFIND'
    _subfind_grnr_name = 'GrNr'
    _subfind_first_gr_name = 'SUBFIND/FirstSubOfHalo'

    _numgrps_name = 'Total_Number_of_groups'
    _numsubs_name = 'Total_Number_of_subgroups'

    _grp_offset_name = 'Offset'
    _grp_len_name = 'Length'

    _sub_offset_name = 'SUB_Offset'
    _sub_len_name = 'SUB_Length'

    def __init__(self, sim, filename=None, subs=None, subhalos=False, _inherit_data_from=None) :
        """Initialise a SubFindHDF catalogue.

        By default, the FoF groups are imported, and subhalos are available via the 'subhalos' attribute of each
        halo object, e.g.

        >>> snap = pynbody.load('path/to/snapshot.hdf5')
        >>> halos = snap.halos()
        >>> halos[1].subhalos[2] # returns the third subhalo of FoF group 1

        However by setting ``subhalos=True``, the FoF groups are ignored and the catalogue is of all subhalos.

        .. note::

            Note that this constructor is common between :class:`SubFindHDFHaloCatalogue` and its subclasses
            :class:`Gadget4SubfindHDFCatalogue`, :class:`ArepoSubfindHDFCatalogue` and
            :class:`TNGSubfindHDFCatalogue`.

            For Gadget 3 outputs, the SubFind data is stored internally to the snapshot itself. Therefore
            passing a *filename* to the constructor for :class:`SubFindHDFHaloCatalogue` will result in an exception.
            For Gadget 4, Arepo and TNG SubFind outputs, a filename can be passed that points to the
            ``fof_subhalo_tab_XXX.hdf5`` file (see below).


        Parameters
        ----------

        sim : ~pynbody.snapshot.simsnap.SimSnap
            The simulation snapshot to which this catalogue applies.

        filename : str, optional
            The filename of the HDF5 file containing the SubFind catalogue. This is only used for Gadget 4, Arepo and
            TNG subclasses and **must** be None when used with the Gadget 3 base class. See the note above.

        subhalos : bool, optional
            If False (default), catalogue represents the FoF groups and subhalos are available through the
            :meth:`~pynbody.halo.Halo.subhalos` attribute of each group (see note above). If True, the catalogue
            represents the subhalos directly and FoF groups are not available.

        subs : bool, optional
            Deprecated alias for ``subhalos``.

        _inherit_data_from : SubfindCatalogue, optional
            For internal use only; allows subhalo catalogue to share data with its parent FOF catalogue
        """

        if subs is not None:
            warnings.warn("The 'subs' argument to SubFindHDFHaloCatalogue is deprecated. Use 'subhalos' instead.",
                          DeprecationWarning)
            subhalos = subs

        self._sub_mode = subhalos
        self._hdf_files = self._get_catalogue_multifile(sim, user_provided_filename=filename)

        self.__count_halos_and_subhalos()

        super().__init__(sim, number_mapping.SimpleHaloNumberMapper(0, self.__get_length()))

        if _inherit_data_from:
            self.__inherit_data(_inherit_data_from)
        else:
            self.__init_halo_offset_data()
            self.__init_subhalo_relationships()
            self.__init_subhalo_offset_data()
            self.__init_halo_properties()
            self.__reshape_multidimensional_properties()
            self.__reassign_properties_from_sub_to_fof()

        if not subhalos:
            self._subhalo_catalogue = type(self)(sim, subhalos=True, filename=filename, _inherit_data_from=self)

    def __inherit_data(self, parent):
        attrs_to_share = ["_fof_properties", "_sub_properties", "_fof_group_offsets", "_fof_group_lengths",
                         "_subfind_halo_offsets", "_subfind_halo_lengths",
                          "fof_ignore", "sub_ignore","_subfind_halo_parent_groups", "_ngroups", "_nsubhalos"]
        for attr in attrs_to_share:
            setattr(self, attr, getattr(parent, attr))


    def _get_catalogue_multifile(self, sim, user_provided_filename):
        """Some variants of Subfind put all the particle data in the catalogue files, in which case the catalogue
        HDF files are already present in the base simulation. In other cases, notably Gadget4/Arepo, this must be
        overridden to locate the actual HDF5 files for the catalogue"""
        if user_provided_filename is not None:
            raise ValueError("Filename must be None for loading a Gadget3-style SubFindHDFCatalogue")

        if not isinstance(sim, gadgethdf.SubFindHDFSnap):
            raise ValueError("SubFindHDFHaloCatalogue can only work with a SubFindHDFSnap simulation")

        return sim._hdf_files

    def __init_ignorable_keys(self):
        self.fof_ignore = list(map(str.strip,config_parser.get("SubfindHDF","FoF-ignore").split(",")))
        self.sub_ignore = list(map(str.strip,config_parser.get("SubfindHDF","Sub-ignore").split(",")))

        for t in list(self.base._family_to_group_map.values()):
            # Don't add SubFind particles ever as this list is actually spherical overdensity
            self.sub_ignore.append(t[0])
            self.fof_ignore.append(t[0])

    def __init_halo_properties(self):
        self.__init_ignorable_keys()
        self._fof_properties = self.__get_property_dictionary_from_hdf(self._fof_name)
        self._sub_properties = self.__get_property_dictionary_from_hdf(self._subfind_name)


    def __get_property_dictionary_from_hdf(self, hdf_key):

        props = self._get_properties_from_multifile(self._hdf_files, hdf_key)

        for property_key in list(props.keys()):
            arr_units = self._get_units(hdf_key, property_key)
            if property_key in props:
                props[property_key] = props[property_key].view(array.SimArray)
                props[property_key].units = arr_units
                props[property_key].sim = self.base

        return props

    def _get_properties_from_multifile(self, multifile, hdf_key):
        hdf0 = multifile.get_file0_root()
        props = {}
        for h in multifile.iterroot():
            for property_key in hdf0[hdf_key].keys():
                if property_key in self.fof_ignore or property_key not in h[hdf_key]:
                    continue
                if property_key in props:
                    props[property_key] = np.append(props[property_key], h[hdf_key][property_key][()])
                else:
                    props[property_key] = np.asarray(h[hdf_key][property_key])
        return props

    def _get_units(self, hdf_key, property_key):
        hdf0 = self._hdf_files.get_file0_root()
        return self.base._get_units_from_hdf_attr(hdf0[hdf_key][property_key].attrs)

    def __reshape_multidimensional_properties(self):
        self.__reshape_multidimensional_properties_one_dictionary(self._sub_properties, self._nsubhalos)
        self.__reshape_multidimensional_properties_one_dictionary(self._fof_properties, self._ngroups)

    def __reshape_multidimensional_properties_one_dictionary(self, properties_dict, expected_array_length):
        for key in list(properties_dict.keys()):
            # Test if there are no remainders, i.e. array is multiple of halo length
            # then solve for the case where this is 1, 2 or 3 dimension
            if len(properties_dict[key]) % expected_array_length == 0:
                ndim = len(properties_dict[key]) // expected_array_length
                if ndim > 1:
                    properties_dict[key] = properties_dict[key].reshape(expected_array_length, ndim)

    def __reassign_properties_from_sub_to_fof(self):
        reassign = []
        for k,v in self._sub_properties.items():
            if v.shape[0]==self._ngroups:
                reassign.append(k)

        for reassign_i in reassign:
            self._fof_properties[reassign_i] = self._sub_properties[reassign_i]
            del self._sub_properties[reassign_i]

    def __init_subhalo_relationships(self):
        nsub = 0
        nfof = 0
        self._subfind_halo_parent_groups = np.empty(self._nsubhalos, dtype=int)
        self._fof_group_first_subhalo = np.empty(self._ngroups, dtype=int)
        for h in self._hdf_files.iterroot():
            parent_groups = h[self._subfind_name].get(self._subfind_grnr_name, np.array([]))
            # .astype(int)[:] stopgap until h5py support numpy 2.0
            self._subfind_halo_parent_groups[nsub:nsub + len(parent_groups)] = parent_groups.astype(int)[:]
            nsub += len(parent_groups)

            first_groups = h.get(self._subfind_first_gr_name, np.array([]))
            # .astype(int)[:] stopgap until h5py support numpy 2.0
            self._fof_group_first_subhalo[nfof:nfof + len(first_groups)] = first_groups.astype(int)[:]
            nfof += len(first_groups)

    def __get_length(self):
        if self._sub_mode:
            return self._nsubhalos
        else:
            return self._ngroups

    def __count_halos_and_subhalos(self):
        hdf0 = self._hdf_files.get_file0_root()
        self._ngroups = int(hdf0[self._header_name].attrs[self._numgrps_name])
        self._nsubhalos = int(hdf0[self._header_name].attrs[self._numsubs_name])
    def __init_halo_offset_data(self):


        self._fof_group_offsets = {}
        self._fof_group_lengths = {}



        # Process FOF groups first
        for fam in self.base._families_ordered():
            ptypes = self.base._family_to_group_map[fam]
            ptype_group_offset = 0
            for ptype in ptypes:
                self._fof_group_offsets[ptype] = np.empty(self._ngroups, dtype='int64')
                self._fof_group_lengths[ptype] = np.empty(self._ngroups, dtype='int64')

                curr_groups = 0
                current_offset = 0

                for h in self._hdf_files:
                    length = self._get_halodata_array_with_default(h, self._grp_len_name, self._fof_name, ptype, np.array([]))

                    if len(length) == 0:
                        # Nothing to process in this file
                        continue

                    offset = self._get_halodata_array_with_default(h, self._grp_offset_name, self._fof_name, ptype,
                                                                   None)

                    if offset is None:
                        # Arepo doesn't store offsets, so we need to calculate them. Note these are relative to
                        # the specific PartType we are looking at, but across all files (ordered sequentially).
                        lengths_cumulative = np.cumsum(length)
                        offset = np.concatenate(([current_offset], current_offset+lengths_cumulative[:-1]))
                        current_offset += lengths_cumulative[-1]

                    self._fof_group_offsets[ptype][curr_groups:curr_groups + len(offset)] = offset.astype('int64')[:]
                    self._fof_group_lengths[ptype][curr_groups:curr_groups + len(offset)] = length.astype('int64')[:]
                    curr_groups += len(offset)

                if curr_groups != self._ngroups:
                    warnings.warn(
                        f"Incorrect number of groups recovered from HDF files. Expected {self._ngroups}, found {curr_groups}")
                    self._ngroups = curr_groups
                    self._fof_group_offsets[ptype] = self._fof_group_offsets[ptype][:curr_groups]
                    self._fof_group_lengths[ptype] = self._fof_group_lengths[ptype][:curr_groups]


    def __init_subhalo_offset_data(self):
        self._subfind_halo_offsets = {}
        self._subfind_halo_lengths = {}



        first_subs_in_groups = self._fof_group_first_subhalo.copy()
        first_subs_in_groups = first_subs_in_groups[(first_subs_in_groups > -1) & (
                    first_subs_in_groups < self._nsubhalos)]  # Just need actual first subhalo numbers, don't need to worry about haloes with no subhaloes. Accounts for formats where *no* first subhalo is recorded as i) '-1' and ii) 'total number of subhaloes'
        last_subs_in_groups = np.concatenate((first_subs_in_groups[1:], [self._nsubhalos])) - 1
        for fam in self.base._families_ordered():
            ptypes = self.base._family_to_group_map[fam]
            for ptype in ptypes:
                self._subfind_halo_offsets[ptype] = np.empty(self._nsubhalos, dtype='int64')
                self._subfind_halo_lengths[ptype] = np.empty(self._nsubhalos, dtype='int64')

                curr_subhalos = 0

                # Only get lengths
                for h in self._hdf_files:
                    length = self._get_halodata_array_with_default(h, self._sub_len_name, self._subfind_name, ptype, np.array([]))
                    self._subfind_halo_lengths[ptype][curr_subhalos:curr_subhalos + len(length)] = length.astype('int64')[:]
                    curr_subhalos += len(length)
                # Add offsets in blocks for all subhalos of a single halo
                for first_sub, last_sub in zip(first_subs_in_groups, last_subs_in_groups):
                    length = self._subfind_halo_lengths[ptype][first_sub:last_sub + 1]
                    offset = np.concatenate(([0], np.cumsum(length)[:-1]))
                    parent_fof_offset = self._fof_group_offsets[ptype][self._subfind_halo_parent_groups[first_sub]]
                    self._subfind_halo_offsets[ptype][first_sub:last_sub + 1] = offset + parent_fof_offset
                if curr_subhalos != self._nsubhalos:
                    warnings.warn(
                        f"Incorrect number of subhalos recovered from HDF files. Expected {self._nsubhalos}, found {curr_subhalos}")
                    self._nsubhalos = curr_subhalos
                    self._subfind_halo_offsets[ptype] = self._subfind_halo_offsets[ptype][:curr_groups]
                    self._subfind_halo_lengths[ptype] = self._subfind_halo_lengths[ptype][:curr_groups]

    def _get_halodata_array(self, hdf_file, array_name, halo_or_group, particle_type):
        # In gadget3 implementation, halo_or_group is not needed. In Gadget4 implementation (below), it is.
        return hdf_file[particle_type][array_name]

    def _get_halodata_array_with_default(self, hdf_file, array_name, halo_or_group, particle_type, default):
        try:
            return self._get_halodata_array(hdf_file, array_name, halo_or_group, particle_type)
        except KeyError:
            return default

    def get_properties_one_halo(self, i):
        def extract(arr, i):
            if np.issubdtype(arr.dtype, np.integer):
                return arr[i]
            else:
                return units.get_item_with_unit(arr, i)

        properties = {}
        if self._sub_mode:
            for key in self._sub_properties:
                properties[key] = extract(self._sub_properties[key], i)
        else:
            for key in self._fof_properties:
                properties[key] = extract(self._fof_properties[key], i)
            properties['children'], = np.where(self._subfind_halo_parent_groups == i)
        return properties

    def get_properties_all_halos(self, with_units=True) -> dict:
        if self._sub_mode:
            result = {'parent': self._subfind_halo_parent_groups}
            result.update(self._sub_properties)
        else:
            children = [[] for _ in range(self._ngroups)]
            for i, parent in enumerate(self._subfind_halo_parent_groups):
                children[parent].append(i)
            result = {'children': children}
            result.update(self._fof_properties)

        if with_units:
            return result
        else:
            result_nounits = {}
            for k, v in result.items():
                if hasattr(v, 'view'):
                    result_nounits[k] = v.view(np.ndarray)
                else:
                    result_nounits[k] = v
            return result_nounits

    def _get_particle_indices_one_halo(self, number):
        if self.base is None :
            raise RuntimeError("Parent SimSnap has been deleted")

        if number > len(self)-1 :
            description = "Subhalo" if self._sub_mode else "Group"
            raise ValueError(f"{description} {number} does not exist")

        type_map = self.base._family_to_group_map

        if self._sub_mode:
            lengths = self._subfind_halo_lengths
            offsets = self._subfind_halo_offsets
        else:
            lengths = self._fof_group_lengths
            offsets = self._fof_group_offsets


        # create the particle lists
        tot_len = 0
        for g_ptypes in list(type_map.values()) :
            for g_ptype in g_ptypes:
                tot_len += lengths[g_ptype][number]

        plist = np.empty(tot_len,dtype='int64')

        self._write_pynbody_index_list_into_array(plist, offsets, lengths, number, type_map)

        return plist

    def _write_pynbody_index_list_into_array(self, particle_indices, halo_or_group_offsets,
                                                      halo_or_group_lengths, halo_or_group_index, type_map):
        npart = 0
        for ptype in self.base._families_ordered():
            # family slice in the SubFindHDFSnap


            for g_ptype in type_map[ptype]:
                sl = self.base._gadget_ptype_slice[g_ptype]

                # add the particle indices to the particle list
                offset = halo_or_group_offsets[g_ptype][halo_or_group_index]
                length = halo_or_group_lengths[g_ptype][halo_or_group_index]

                ind = np.arange(sl.start + offset, sl.start + offset + length)
                particle_indices[npart:npart + length] = ind
                npart += length
        return npart

    def _get_all_particle_indices(self):
        if self.base is None:
            raise RuntimeError("Parent SimSnap has been deleted")

        type_map = self.base._family_to_group_map

        if self._sub_mode:
            lengths = self._subfind_halo_lengths
            offsets = self._subfind_halo_offsets
        else:
            lengths = self._fof_group_lengths
            offsets = self._fof_group_offsets

        # create the particle lists
        tot_len = 0
        for g_ptypes in list(type_map.values()):
            for g_ptype in g_ptypes:
                tot_len += lengths[g_ptype].sum()

        plist = np.empty(tot_len, dtype='int64')
        boundaries = np.empty((len(self), 2), dtype='int64')

        location = 0
        for i in range(len(self)):
            npart = self._write_pynbody_index_list_into_array(plist[location:], offsets, lengths, i, type_map)
            boundaries[i] = (location, location := location+npart)

        assert location == tot_len

        return particle_indices.HaloParticleIndices(plist, boundaries)

    def _group_and_halo_from_halo_index(self, i):
        """Return the group, halo pair of indices from the overall halo index i

        Subfind stores a long list of halos, gathered together into groups. This maps from the index in the long
        list to the particular group and subhalo indices."""

        if i>=self._nsubhalos:
            raise ValueError("Subhalo out of range")

        group = np.argmax(self._fof_group_first_subhalo>i)-1
        halo = i - self._fof_group_first_subhalo[group]
        return group, halo

    def _get_subhalo_catalogue(self, parent_halo_number):
        if not self._sub_mode:
            props = self.get_properties_one_halo(parent_halo_number)
            return SubhaloCatalogue(self._subhalo_catalogue, props['children'])
        else:
            return SubhaloCatalogue(self._subhalo_catalogue, [])

    @classmethod
    def _can_load(cls, sim, **kwargs):
        if isinstance(sim, gadgethdf.SubFindHDFSnap):
            return True
        else:
            return False

class Gadget4SubfindHDFCatalogue(SubFindHDFHaloCatalogue):
    """Handles catalogues produced by the SubFind halo finder, in the HDF5 format used by Gadget 4

    .. warning::

        At present, this is not tested against multi-file outputs. If you encounter issues with these, please report
        them to the pynbody developers.

    """

    _fof_name = 'Group'
    _subfind_name = 'Subhalo'
    _header_name = 'Header'
    _subfind_grnr_name = 'SubhaloGroupNr'
    _subfind_first_gr_name = 'Group/GroupFirstSub'

    _numgrps_name = 'Ngroups_Total'
    _numsubs_name = 'Nsubhalos_Total'

    _grp_offset_name = 'GroupOffsetType'
    _grp_len_name = 'GroupLenType'

    _sub_offset_name = 'SubhaloOffsetType'
    _sub_len_name = 'SubhaloLenType'

    def __init__(self, sim, filename=None, **kwargs):
        super().__init__(sim, filename, **kwargs)
        i = 0
        for prog_or_desc in "prog", "desc":
            try:
                files = self._get_progenitor_or_descendant_multifile(sim, prog_or_desc, filename)
            except FileNotFoundError:
                continue
            props = self._get_properties_from_multifile(files, 'Subhalo')
            self._sub_properties.update(props)

    def _get_halodata_array(self, hdf_file, array_name, halo_or_group, particle_type):
        return hdf_file[halo_or_group][array_name][:,int(particle_type[-1])]

    def _get_units(self, hdf_key, property_key):
        # Gadget4 doesn't seem to store unit information, so have a good guess
        if property_key == 'SubhaloVmax':
            dimensions = 'm s^-1'
        elif 'Mass' in property_key or '_M_' in property_key:
            dimensions = 'kg'
        elif 'Pos' in property_key or '_R_' in property_key or 'CM' in property_key or 'VmaxRad' in property_key:
            dimensions = 'm'
        elif 'Vel' in property_key:
            dimensions = 'm s^-1'
        else:
            return None

        return self.base.infer_original_units(dimensions)



    @classmethod
    def _get_catalogue_multifile(cls, sim, user_provided_filename):
        class Gadget4SubfindHdfMultiFileManager(gadgethdf._SubfindHdfMultiFileManager):
            _nfiles_groupname = cls._fof_name
            _nfiles_attrname = "NTask"
            _subgroup_name = None

        return Gadget4SubfindHdfMultiFileManager(cls._catalogue_filename(sim, user_provided_filename))

    @classmethod
    def _get_progenitor_or_descendant_multifile(cls, sim, prog_or_desc, user_provided_filename):
        class Gadget4SubfindHdfProgenitorsMultiFileManager(gadgethdf._SubfindHdfMultiFileManager):
            _nfiles_groupname = "Header"
            _nfiles_attrname = "NumFiles"
            _subgroup_name = None

        return Gadget4SubfindHdfProgenitorsMultiFileManager(cls._catalogue_filename(sim, user_provided_filename,
                                                                                    "subhalo_"+prog_or_desc+"_"))

    @classmethod
    def _catalogue_filename(cls, sim, user_provided_filename=None, namestem ="fof_subhalo_tab_"):
        if user_provided_filename is not None:
            user_provided_filename = str(user_provided_filename)
            if "fof_subhalo_tab_" not in user_provided_filename:
                raise ValueError("Filename must contain 'fof_subhalo_tab_'")
            return user_provided_filename.replace("fof_subhalo_tab_", namestem)

        snapnum = os.path.basename(sim.filename).split("_")[-1]
        parent_dir = os.path.dirname(os.path.abspath(sim.filename))
        return os.path.join(parent_dir, namestem + snapnum)


    @classmethod
    def _can_load(cls, sim, filename=None, **kwargs):
        try:
            file = cls._catalogue_filename(sim, user_provided_filename=filename)
        except ValueError:
            return False
        if not h5py.is_hdf5(file):
            file = file + ".0.hdf5"
            if not h5py.is_hdf5(file):
                return False
        with h5py.File(file, 'r') as f:
            if cls._header_name not in f:
                return False
            if cls._numsubs_name not in f[cls._header_name].attrs:
                return False
        return True


class ArepoSubfindHDFCatalogue(Gadget4SubfindHDFCatalogue):
    """Handles catalogues produced by the SubFind halo finder, in the HDF5 format used by Arepo

    .. warning::

        At present, this is not tested against multi-file outputs. If you encounter issues with these, please report
        them to the pynbody developers.

    """
    _numsubs_name = 'Nsubgroups_Total'

    _subfind_grnr_name = 'SubhaloGrNr'


class TNGSubfindHDFCatalogue(ArepoSubfindHDFCatalogue):
    """Handles catalogues produced by the SubFind halo finder, in the HDF5 format used by Arepo (TNG variant)

    .. warning::

        At present, this is not tested against multi-file outputs. If you encounter issues with these, please report
        them to the pynbody developers.

    """
    @classmethod
    def _catalogue_filename(cls, sim, user_provided_filename, namestem ="fof_subhalo_tab_"):
        if user_provided_filename is not None:
            return super()._catalogue_filename(sim, user_provided_filename, namestem)
        snapnum = os.path.basename(sim.filename).split("_")[-1]
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(sim.filename)))
        f = os.path.join(parent_dir, "groups_"+snapnum, namestem + snapnum)
        return f

    @classmethod
    def _get_catalogue_multifile(cls, sim, user_provided_filename):
        class TNGSubfindHdfMultiFileManager(gadgethdf._SubfindHdfMultiFileManager):
            _nfiles_groupname = "Header"
            _nfiles_attrname = "NumFiles"
            _subgroup_name = None

        return TNGSubfindHdfMultiFileManager(cls._catalogue_filename(sim, user_provided_filename))
