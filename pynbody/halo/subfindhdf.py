import os.path
import warnings
import weakref

import h5py
import numpy as np

from .. import array, config_parser, snapshot, units
from ..snapshot import gadgethdf
from . import Halo, HaloCatalogue


class SubFindHDFSubhaloCatalogue(HaloCatalogue) :
    """
    Gadget's SubFind HDF Subhalo catalogue.

    Initialized with the parent FOF group catalogue and created
    automatically when an fof group is created
    """

    def __init__(self, group_id, group_catalogue) :
        super().__init__(group_catalogue.base)

        self._group_id = group_id
        self._group_catalogue = group_catalogue
        self.__calc_len()

    def __calc_len(self):
        next_group_id = self._group_id + 1
        next_first_subhalo = self._group_catalogue.nsubhalos
        while next_group_id < len(self._group_catalogue._fof_group_first_subhalo)-1:
            if self._group_catalogue._fof_group_first_subhalo[self._group_id + 1]!=-1:
                next_first_subhalo = self._group_catalogue._fof_group_first_subhalo[next_group_id]
                break
            next_group_id+=1


        self._len = (next_first_subhalo - self._group_catalogue._fof_group_first_subhalo[self._group_id])

    def __len__(self):
        return self._len

    def _get_halo(self, i):
        if self.base is None :
            raise RuntimeError("Parent SimSnap has been deleted")

        if i > len(self)-1 :
            raise ValueError("FOF group %d does not have subhalo %d"%(self._group_id, i))

        # need this to index the global offset and length arrays
        absolute_id = self._group_catalogue._fof_group_first_subhalo[self._group_id] + i

        # now form the particle IDs needed for this subhalo
        type_map = self.base._family_to_group_map

        halo_lengths = self._group_catalogue._subfind_halo_lengths
        halo_offsets = self._group_catalogue._subfind_halo_offsets

        # create the particle lists
        tot_len = 0
        for g_ptypes in list(type_map.values()) :
            for g_ptype in g_ptypes:
                tot_len += halo_lengths[g_ptype][absolute_id]

        plist = np.zeros(tot_len,dtype='int64')

        npart = 0
        for ptype in self.base._families_ordered():
            # family slice in the SubFindHDFSnap
            sl = self.base._family_slice[ptype]

            for g_ptype in type_map[ptype]:
                # add the particle indices to the particle list
                offset = halo_offsets[g_ptype][absolute_id]
                length = halo_lengths[g_ptype][absolute_id]
                ind = np.arange(sl.start + offset, sl.start + offset + length)
                plist[npart:npart+length] = ind
                npart += length

        return SubFindHDFSubHalo(i, self._group_id, self._group_catalogue, self, self.base, plist)


    @property
    def base(self) :
        return self._base()


class SubFindHDFHaloCatalogue(HaloCatalogue) :
    """
    Gadget's SubFind Halo catalogue -- used in concert with :class:`~SubFindHDFSnap`
    """

    # Names of various groups and attributes in the hdf file (which seemingly may vary in different versions of SubFind?)

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

    def __init__(self, sim, subs=False, grp_array=False) :
        """Initialise the Subfind catalogue

        *sim*: The SimSnap
        *subs*: If True, enumerate the subhalos instead of the groups. Otherwise, the jth subhalo of FOF group i
        is still available, as halo_cat[i].sub[j].
        """

        super().__init__(sim)

        self._sub_mode = subs
        self._hdf_files = self._get_catalogue_multifile(sim)

        self.__init_halo_offset_data()
        self.__init_subhalo_relationships()
        self.__init_halo_properties()
        self.__reshape_multidimensional_properties()
        self.__reassign_properties_from_sub_to_fof()

    def _get_catalogue_multifile(self, sim):
        """Some variants of Subfind put all the particle data in the catalogue files, in which case the catalogue
        HDF files are already present in the base simulation. In other cases, notably Gadget4/Arepo, this must be
        overridden to locate the actual HDF5 files for the catalogue"""
        if not isinstance(sim, snapshot.gadgethdf.SubFindHDFSnap):
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
                if property_key in self.fof_ignore:
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
        self.__reshape_multidimensional_properties_one_dictionary(self._sub_properties, self.nsubhalos)
        self.__reshape_multidimensional_properties_one_dictionary(self._fof_properties, self.ngroups)

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
            if v.shape[0]==self.ngroups:
                reassign.append(k)

        for reassign_i in reassign:
            self._fof_properties[reassign_i] = self._sub_properties[reassign_i]
            del self._sub_properties[reassign_i]

    def __init_subhalo_relationships(self):
        nsub = 0
        nfof = 0
        self._subfind_halo_parent_groups = np.empty(self.nsubhalos, dtype=int)
        self._fof_group_first_subhalo = np.empty(self.ngroups, dtype=int)
        for h in self._hdf_files.iterroot():
            parent_groups = h[self._subfind_name][self._subfind_grnr_name]
            self._subfind_halo_parent_groups[nsub:nsub + len(parent_groups)] = parent_groups
            nsub += len(parent_groups)

            first_groups = h[self._subfind_first_gr_name]
            self._fof_group_first_subhalo[nfof:nfof + len(first_groups)] = first_groups
            nfof += len(first_groups)

    def __init_halo_offset_data(self):
        hdf0 = self._hdf_files.get_file0_root()

        self._fof_group_offsets = {}
        self._fof_group_lengths = {}
        self._subfind_halo_offsets = {}
        self._subfind_halo_lengths = {}

        self.ngroups = int(hdf0[self._header_name].attrs[self._numgrps_name])
        self.nsubhalos = int(hdf0[self._header_name].attrs[self._numsubs_name])

        for fam in self.base._families_ordered():
            ptypes = self.base._family_to_group_map[fam]
            for ptype in ptypes:
                self._fof_group_offsets[ptype] = np.empty(self.ngroups, dtype='int64')
                self._fof_group_lengths[ptype] = np.empty(self.ngroups, dtype='int64')
                self._subfind_halo_offsets[ptype] = np.empty(self.nsubhalos, dtype='int64')
                self._subfind_halo_lengths[ptype] = np.empty(self.nsubhalos, dtype='int64')

                curr_groups = 0
                curr_subhalos = 0

                for h in self._hdf_files:
                    # fof groups
                    offset = self._get_halodata_array(h, self._grp_offset_name, self._fof_name, ptype)
                    length = self._get_halodata_array(h, self._grp_len_name, self._fof_name, ptype)
                    self._fof_group_offsets[ptype][curr_groups:curr_groups + len(offset)] = offset
                    self._fof_group_lengths[ptype][curr_groups:curr_groups + len(offset)] = length
                    curr_groups += len(offset)

                    # subfind subhalos
                    offset = self._get_halodata_array(h, self._sub_offset_name, self._subfind_name, ptype)
                    length = self._get_halodata_array(h, self._sub_len_name, self._subfind_name, ptype)
                    self._subfind_halo_offsets[ptype][curr_subhalos:curr_subhalos + len(offset)] = offset
                    self._subfind_halo_lengths[ptype][curr_subhalos:curr_subhalos + len(offset)] = length
                    curr_subhalos += len(offset)

    def _get_halodata_array(self, hdf_file, array_name, halo_or_group, particle_type):
        # In gadget3 implementation, halo_or_group is not needed. In Gadget4 implementation (below), it is.
        return hdf_file[particle_type][array_name]

    def get_halo_properties(self, i, with_unit=True, subs=None):
        """Get just the properties for halo/group i

        Subs controls whether to get halo (True) or group (False) properties
        If subs is None, return the halo/group according to whether subs=True/False when constructing the catalogue.
        """

        if subs is None:
            subs = self._sub_mode

        if with_unit:
            extract = units.get_item_with_unit
        else:
            extract = lambda array, element: array[element]
        properties = {}
        if subs:
            for key in self._sub_properties:
                properties[key] = extract(self._sub_properties[key], i)
        else:
            for key in self._fof_properties:
                properties[key] = extract(self._fof_properties[key], i)
            properties['children'], = np.where(self._subfind_halo_parent_groups==i)
        return properties

    def get_group_array(self):
        if self._sub_mode:
            lengths = self._subfind_halo_lengths
            offsets = self._subfind_halo_offsets
        else:
            lengths = self._fof_group_lengths
            offsets = self._fof_group_offsets

        type_map = self.base._family_to_group_map

        grp = np.empty(len(self.base), dtype=np.int32)
        grp.fill(-1)

        for ptype in self.base._families_ordered():
            sl = self.base._family_slice[ptype]
            for g_ptype in type_map[ptype]:
                for i in range(len(self)):
                    offset = offsets[g_ptype][i]
                    length = lengths[g_ptype][i]
                    grp[sl.start + offset:sl.start + offset + length] = i
        return grp


    def _get_halo(self, i) :
        if self.base is None :
            raise RuntimeError("Parent SimSnap has been deleted")

        if i > len(self)-1 :
            description = "Subhalo" if self._sub_mode else "Group"
            raise ValueError(f"{description} {i} does not exist")

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
                tot_len += lengths[g_ptype][i]

        plist = np.zeros(tot_len,dtype='int64')

        npart = 0
        for ptype in self.base._families_ordered():
            # family slice in the SubFindHDFSnap
            sl = self.base._family_slice[ptype]

            for g_ptype in type_map[ptype]:
                # add the particle indices to the particle list
                offset = offsets[g_ptype][i]
                length = lengths[g_ptype][i]
                ind = np.arange(sl.start + offset, sl.start + offset + length)
                plist[npart:npart+length] = ind
                npart += length

        if self._sub_mode:
            # SubFindHDFSubHalo wants to know our parent group
            group, halo = self._group_and_halo_from_halo_index(i)
            return SubFindHDFSubHalo(halo, group, self, self, self.base, plist)
        else:
            return SubFindFOFGroup(i, self, self.base, plist)


    def _group_and_halo_from_halo_index(self, i):
        """Return the group, halo pair of indices from the overall halo index i

        Subfind stores a long list of halos, gathered together into groups. This maps from the index in the long
        list to the particular group and subhalo indices."""

        if i>=self.nsubhalos:
            raise ValueError("Subhalo out of range")

        group = np.argmax(self._fof_group_first_subhalo>i)-1
        halo = i - self._fof_group_first_subhalo[group]
        return group, halo


    def __len__(self) :
        if self._sub_mode:
            return self._hdf_files.get_file0_root()[self._header_name].attrs[self._numsubs_name]
        else:
            return self._hdf_files.get_file0_root()[self._header_name].attrs[self._numgrps_name]

    @property
    def base(self):
        return self._base()


class SubFindFOFGroup(Halo) :
    """
    SubFind FOF group class
    """

    def __init__(self, group_id, *args) :
        """Construct a special halo representing subfind's FOF group"""
        super().__init__(group_id, *args)

        self._subhalo_catalogue = SubFindHDFSubhaloCatalogue(group_id, self._halo_catalogue)

        self._descriptor = "fof_group_"+str(group_id)

        self.properties.update(self._halo_catalogue.get_halo_properties(group_id, subs=False))


    def __getattr__(self, name):
        if name == 'sub':
            return self._subhalo_catalogue
        else :
            return super().__getattr__(name)

class SubFindHDFSubHalo(Halo) :
    """
    SubFind subhalo class
    """

    def __init__(self, halo_id, group_id, subfind_data_object, *args) :
        """Construct a special halo representing subfind's subhalo

        *halo_id*: The halo_id, where 0 is the first halo within the specified group
        *group_id*: The group across the entire snapshot
        *subfind_data_object*: The object which actually holds the HDF data

        Other arguments get passed to the standard halo constructor
        """
        super().__init__(halo_id, *args)
        self._group_id = group_id
        self._descriptor = "fof_group_%d_subhalo_%d"%(group_id,halo_id)

        # need this to index the global offset and length arrays
        absolute_id = subfind_data_object._fof_group_first_subhalo[self._group_id] + halo_id

        # load properties
        self.properties.update(subfind_data_object.get_halo_properties(absolute_id, subs=True))


class Gadget4SubfindHDFCatalogue(SubFindHDFHaloCatalogue):

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

    def __init__(self, sim, subs=False, grp_array=False):
        super().__init__(sim, subs, grp_array)
        i = 0
        for prog_or_desc in "prog", "desc":
            try:
                files = self._get_progenitor_or_descendant_multifile(sim, prog_or_desc)
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
    def _get_catalogue_multifile(cls, sim):
        class Gadget4SubfindHdfMultiFileManager(gadgethdf.SubfindHdfMultiFileManager):
            _nfiles_groupname = cls._fof_name
            _nfiles_attrname = "NTask"
            _subgroup_name = None

        return Gadget4SubfindHdfMultiFileManager(cls._catalogue_filename(sim))

    @classmethod
    def _get_progenitor_or_descendant_multifile(cls, sim, prog_or_desc):
        class Gadget4SubfindHdfProgenitorsMultiFileManager(gadgethdf.SubfindHdfMultiFileManager):
            _nfiles_groupname = "Header"
            _nfiles_attrname = "NumFiles"
            _subgroup_name = None

        return Gadget4SubfindHdfProgenitorsMultiFileManager(cls._catalogue_filename(sim, "subhalo_"+prog_or_desc+"_"))

    @staticmethod
    def _catalogue_filename(sim, namestem ="fof_subhalo_tab_"):
        snapnum = os.path.basename(sim.filename).split("_")[-1]
        parent_dir = os.path.dirname(os.path.abspath(sim.filename))
        return os.path.join(parent_dir, namestem + snapnum)

    @classmethod
    def _can_load(cls, sim, **kwargs):
        file = Gadget4SubfindHDFCatalogue._catalogue_filename(sim)
        if os.path.exists(file) and (file.endswith(".hdf5") or os.listdir(file)[0].endswith(".hdf5")):
            # very hard to figure out whether it's the right sort of hdf5 file without just going ahead and loading it
            try:
                cls(sim, **kwargs)
                return True
            except:
                return False


class ArepoSubfindHDFCatalogue(Gadget4SubfindHDFCatalogue):
    _numsubs_name = 'Nsubgroups_Total'

    _subfind_grnr_name = 'SubhaloGrNr'


    def _get_halodata_array(self, hdf_file, array_name, halo_or_group, particle_type):
        if array_name.endswith("OffsetType"):
            len_array_name = array_name.replace("OffsetType", "LenType")
            # this doesn't exist in arepo
            lens = super()._get_halodata_array(hdf_file, len_array_name, halo_or_group, particle_type)
            return np.concatenate(([0],np.cumsum(lens)[:-1]))
        else:
            return super()._get_halodata_array(hdf_file, array_name, halo_or_group, particle_type)
