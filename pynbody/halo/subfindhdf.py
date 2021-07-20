import numpy as np
import os.path
import weakref
import warnings
import h5py

from . import HaloCatalogue, Halo
from .. import snapshot, config_parser, array, units

class SubFindHDFSubhaloCatalogue(HaloCatalogue) :
    """
    Gadget's SubFind HDF Subhalo catalogue.

    Initialized with the parent FOF group catalogue and created
    automatically when an fof group is created
    """

    def __init__(self, group_id, group_catalogue) :
        super(SubFindHDFSubhaloCatalogue,self).__init__(group_catalogue.base)

        self._group_id = group_id
        self._group_catalogue = group_catalogue



    def __len__(self):
        if self._group_id == (len(self._group_catalogue._fof_group_first_subhalo)-1) :
            return self._group_catalogue.nsubhalos - self._group_catalogue._fof_group_first_subhalo[self._group_id]
        else:
            return (self._group_catalogue._fof_group_first_subhalo[self._group_id + 1] -
                    self._group_catalogue._fof_group_first_subhalo[self._group_id])

    def _get_halo(self, i):
        if self.base is None :
            raise RuntimeError("Parent SimSnap has been deleted")

        if i > len(self)-1 :
            raise RuntimeError("FOF group %d does not have subhalo %d"%(self._group_id, i))

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

        return SubFindHDFSubHalo(i, self._group_id, self, self.base, plist)


    @property
    def base(self) :
        return self._base()


class SubFindHDFSubHalo(Halo) :
    """
    SubFind subhalo class
    """

    def __init__(self,halo_id, group_id, *args) :
        super(SubFindHDFSubHalo,self).__init__(halo_id, *args)

        self._group_id = group_id
        self._descriptor = "fof_group_%d_subhalo_%d"%(group_id,halo_id)

        # need this to index the global offset and length arrays
        absolute_id = self._halo_catalogue._group_catalogue._fof_group_first_subhalo[self._group_id] + halo_id

        # load properties
        sub_props = self._halo_catalogue._group_catalogue._sub_properties
        for key in sub_props :
            self.properties[key] = array.SimArray(sub_props[key][absolute_id], sub_props[key].units)
            self.properties[key].sim = self.base


class SubFindHDFHaloCatalogue(HaloCatalogue) :
    """
    Gadget's SubFind Halo catalogue -- used in concert with :class:`~SubFindHDFSnap`
    """

    # Names of various groups and attributes in the hdf file (which seemingly may vary in different versions of SubFind?)

    _fof_name = 'FOF'
    _subfind_name = 'SUBFIND'
    _subfind_grnr_name = 'GrNr'
    _subfind_first_gr_name = 'FirstSubOfHalo'

    _numgrps_name = 'Total_Number_of_groups'
    _numsubs_name = 'Total_Number_of_subgroups'

    _grp_offset_name = 'Offset'
    _grp_len_name = 'Length'

    _sub_offset_name = 'SUB_Offset'
    _sub_len_name = 'SUB_Length'

    def __init__(self, sim) :
        super(SubFindHDFHaloCatalogue,self).__init__(sim)

        if not isinstance(sim, snapshot.gadgethdf.SubFindHDFSnap):
            raise ValueError("SubFindHDFHaloCatalogue can only work with a SubFindHDFSnap simulation")

        self.__init_halo_offset_data()
        self.__init_subhalo_relationships()
        self.__init_halo_properties()
        self.__reshape_multidimensional_properties()
        self.__reassign_properties_from_sub_to_fof()

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
        sim = self.base
        hdf0 = sim._hdf_files.get_file0_root()

        props = {}
        for property_key in list(hdf0[hdf_key].keys()):
            if property_key not in self.fof_ignore:
                props[property_key] = np.array([])

        for h in sim._hdf_files.iterroot():
            for property_key in list(props.keys()):
                props[property_key] = np.append(props[property_key], h[hdf_key][property_key][()])

        for property_key in list(props.keys()):
            arr_units = sim._get_units_from_hdf_attr(hdf0[hdf_key][property_key].attrs)
            if property_key in props:
                props[property_key] = props[property_key].view(array.SimArray)
                props[property_key].units = arr_units
                props[property_key].sim = sim

        return props


    def __reshape_multidimensional_properties(self):
        sub_properties = self._sub_properties
        fof_properties = self._fof_properties

        for key in list(sub_properties.keys()):
            # Test if there are no remainders, i.e. array is multiple of halo length
            # then solve for the case where this is 1, 2 or 3 dimension
            if len(sub_properties[key]) % self.nsubhalos == 0:
                ndim = len(sub_properties[key]) // self.nsubhalos
                if ndim > 1:
                    sub_properties[key] = sub_properties[key].reshape(self.nsubhalos, ndim)

            try:
                # The case fof FOF
                if len(fof_properties[key]) % self.ngroups == 0:
                    ndim = len(fof_properties[key]) // self.ngroups
                    if ndim > 1:
                        fof_properties[key] = fof_properties[key].reshape(self.ngroups, ndim)
            except KeyError:
                pass

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
        for h in self.base._hdf_files.iterroot():
            parent_groups = h[self._subfind_name][self._subfind_grnr_name]
            self._subfind_halo_parent_groups[nsub:nsub + len(parent_groups)] = parent_groups
            nsub += len(parent_groups)

            first_groups = h[self._subfind_name][self._subfind_first_gr_name]
            self._fof_group_first_subhalo[nfof:nfof + len(first_groups)] = first_groups
            nfof += len(first_groups)

    def __init_halo_offset_data(self):
        hdf0 = self.base._hdf_files.get_file0_root()

        self._fof_group_offsets = {}
        self._fof_group_lengths = {}
        self._subfind_halo_offsets = {}
        self._subfind_halo_lengths = {}

        self.ngroups = hdf0[self._fof_name].attrs[self._numgrps_name]
        self.nsubhalos = hdf0[self._fof_name].attrs[self._numsubs_name]

        for fam in self.base._families_ordered():
            ptypes = self.base._family_to_group_map[fam]
            for ptype in ptypes:
                self._fof_group_offsets[ptype] = np.empty(self.ngroups, dtype='int64')
                self._fof_group_lengths[ptype] = np.empty(self.ngroups, dtype='int64')
                self._subfind_halo_offsets[ptype] = np.empty(self.ngroups, dtype='int64')
                self._subfind_halo_lengths[ptype] = np.empty(self.ngroups, dtype='int64')

                curr_groups = 0
                curr_subhalos = 0

                for h in self.base._hdf_files:
                    # fof groups
                    offset = h[ptype][self._grp_offset_name]
                    length = h[ptype][self._grp_len_name]
                    self._fof_group_offsets[ptype][curr_groups:curr_groups + len(offset)] = offset
                    self._fof_group_lengths[ptype][curr_groups:curr_groups + len(offset)] = length
                    curr_groups += len(offset)

                    # subfind subhalos
                    offset = h[ptype][self._sub_offset_name]
                    length = h[ptype][self._sub_len_name]
                    self._subfind_halo_offsets[ptype][curr_subhalos:curr_subhalos + len(offset)] = offset
                    self._subfind_halo_lengths[ptype][curr_subhalos:curr_subhalos + len(offset)] = length
                    curr_subhalos += len(offset)


    def _get_halo(self, i) :
        if self.base is None :
            raise RuntimeError("Parent SimSnap has been deleted")

        if i > len(self)-1 :
            raise RuntimeError("Group %d does not exist"%i)

        type_map = self.base._family_to_group_map

        # create the particle lists
        tot_len = 0
        for g_ptypes in list(type_map.values()) :
            for g_ptype in g_ptypes:
                tot_len += self._fof_group_lengths[g_ptype][i]

        plist = np.zeros(tot_len,dtype='int64')

        npart = 0
        for ptype in self.base._families_ordered():
            # family slice in the SubFindHDFSnap
            sl = self.base._family_slice[ptype]

            for g_ptype in type_map[ptype]:
                # add the particle indices to the particle list
                offset = self._fof_group_offsets[g_ptype][i]
                length = self._fof_group_lengths[g_ptype][i]
                ind = np.arange(sl.start + offset, sl.start + offset + length)
                plist[npart:npart+length] = ind
                npart += length

        return SubFindFOFGroup(i, self, self.base, plist)


    def __len__(self) :
        return self.base._hdf_files[0].attrs[self._numgrps_name]

    @property
    def base(self):
        return self._base()


class SubFindFOFGroup(Halo) :
    """
    SubFind FOF group class
    """

    def __init__(self, group_id, *args) :
        super(SubFindFOFGroup,self).__init__(group_id, *args)

        self._subhalo_catalogue = SubFindHDFSubhaloCatalogue(group_id, self._halo_catalogue)

        self._descriptor = "fof_group_"+str(group_id)

        # load properties
        for key in list(self._halo_catalogue._fof_properties.keys()) :
            self.properties[key] = array.SimArray(self._halo_catalogue._fof_properties[key][group_id],
                                            self._halo_catalogue._fof_properties[key].units)
            self.properties[key].sim = self.base


    def __getattr__(self, name):
        if name == 'sub':
            return self._subhalo_catalogue
        else :
            return super(SubFindFOFGroup,self).__getattr__(name)


class Gadget4SubfindHDFCatalogue(HaloCatalogue):

    """Class to handle catalogues produced by the SubFind halo finder from Gadget-4 and Arepo.

    By default, the FoF groups are imported, but the subhalos can also be imported by setting subs=True
    when constructing.

    """

    def __init__(self, sim, subs=False, grp_array=None, particle_type="dm", **kwargs):
        """Initialise a SubFind catalogue from Gadget-4/Arepo.

        *subs*: if True, load the subhalos; otherwise, load FoF groups (default)
        *grp_array*: if True, create a 'grp' array with the halo id to which each particle is assigned
        """

        self.ptype = config_parser.get('gadgethdf-type-mapping', particle_type)
        self.ptypenum = int(self.ptype[-1])
        self._base = weakref.ref(sim)
        self._subs = subs
        self._num_files = sim.properties['NumFilesPerSnapshot']

        self._halos = {}
        HaloCatalogue.__init__(self, sim)
        self.dtype_int = sim['iord'].dtype
        self.dtype_flt = sim['mass'].dtype
        self.halofilename = self._name_of_catalogue(sim)
        self._get_hdf_files_names()
        self.header = self._readheader()
        self.params = self._readparams()
        self.ids = self._read_ids()
        assert np.allclose(self.ids, self.base[[fi for fi in self.base.families() if fi.name == particle_type][0]]['iord']), \
            "Particle IDs in snapshot are not ordered by their haloID (i.e. first IDs should be those in halo 0, etc.)."
        self._keys = {}
        self._halodat, self._subhalodat = self._read_groups()
        if grp_array:
            self.make_grp()

    def make_grp(self, name='grp'):
        """ Creates a 'grp' array which labels each particle according to its parent halo. """
        if self._subs is True:
            name = 'subgrp'
        self.base[name] = self.get_group_array()

    def get_subhalo_offsets(self, halo_id=None):
        try:
            # gagdet-4
            if halo_id is not None:
                offs = self._subhalodat['SubhaloOffsetType'][halo_id, self.ptypenum]
            else:
                offs = self._subhalodat['SubhaloOffsetType'][:, self.ptypenum]

        except:
            # Arepo does not directly output offsets so we have to compute them here
            if halo_id is not None:
                parent_halo = self._subhalodat['SubhaloGrNr'][halo_id]
                offset_parent = self.get_halo_offsets(parent_halo)
                subhalos_in_parent = np.where(self._subhalodat['SubhaloGrNr'][:] == parent_halo)[0]
                length_subhalos = self._subhalodat['SubhaloLenType'][subhalos_in_parent]
                offs = offset_parent + np.sum(length_subhalos[subhalos_in_parent < halo_id, self.ptypenum])
            else:
                offs = np.zeros(len(self._subhalodat['SubhaloGrNr']), dtype="int")
                unique_groups = np.unique(self._subhalodat['SubhaloGrNr'])
                for parent_halo in unique_groups:
                    offset_parent = self.get_halo_offsets(parent_halo)
                    subhalos_in_parent = np.where(self._subhalodat['SubhaloGrNr'][:] == parent_halo)[0]
                    off = offset_parent + np.append(np.zeros(1, dtype="int"),
                                                    np.cumsum(self._subhalodat['SubhaloLenType'][subhalos_in_parent[:-1], self.ptypenum]))
                    offs[subhalos_in_parent] = off
        return offs

    def get_halo_offsets(self, halo_id=None):
        try:
            # gagdet-4
            if halo_id is not None:
                offs = self._halodat['GroupOffsetType'][halo_id, self.ptypenum]
            else:
                offs = self._halodat['GroupOffsetType'][:, self.ptypenum]
        except:
            # Arepo does not directly output offsets so we have to compute them here
            if halo_id is not None:
                offs = np.sum(self._halodat['GroupLenType'][:halo_id, self.ptypenum])
            else:
                offs = np.append(np.zeros(1, dtype=self.dtype_int),
                                 np.cumsum(self._halodat['GroupLenType'][:-1, self.ptypenum]))
        return offs

    def get_group_array(self):
        ar = np.zeros(len(self.base), dtype=int) - 1
        if self._subs is True:
            offs = self.get_subhalo_offsets()
            ls = self._subhalodat['SubhaloLenType'][:, self.ptypenum]
        else:
            offs = self.get_halo_offsets()
            ls = self._halodat['GroupLenType'][:, self.ptypenum]

        for i in range(len(offs)):
            ar[offs[i]:offs[i] + ls[i]] = i
        return ar

    def _get_halo(self, i):
        if self._subs is True:
            length = self._subhalodat['SubhaloLenType'][i, self.ptypenum]
            offset = self.get_subhalo_offsets(halo_id=i)
        else:
            length = self._halodat['GroupLenType'][i, self.ptypenum]
            offset = self.get_halo_offsets(halo_id=i)

        x = Halo(i, self, self.base, np.arange(offset, offset + length))
        x._descriptor = "halo_" + str(i)
        x.properties.update(self.get_halo_properties(i))
        return x

    def get_halo_properties(self, i):
        properties = {}
        if self._subs is False:
            for key in self._keys:
                properties[key] = self.extract_property_w_units(self._halodat[key], i)
                try:
                    # gadget-4
                    properties['children'] = np.where(self._subhalodat['SubhaloGroupNr'] == i)[0]
                except:
                    # arepo
                    properties['children'] = np.where(self._subhalodat['SubhaloGrNr'] == i)[0]

        else:
            for key in self._keys:
                properties[key] = self.extract_property_w_units(self._subhalodat[key], i)
        return properties

    @staticmethod
    def extract_property_w_units(simarray, elem):
        if type(simarray) == array.SimArray:
            if len(simarray.shape) > 1:
                return simarray[elem]
            else:
                simarray_i = array.SimArray([simarray[elem]], simarray.units)
                simarray_i.sim = simarray.sim
                return simarray_i
        else:
            return simarray[elem]

    def _readheader(self):
        """ Load the group catalog header. """
        with h5py.File(self._hdf_files[0], 'r') as f:
            header = dict(f['Header'].attrs.items())
        return header

    def _readparams(self):
        """ Load the group catalog parameters. """
        with h5py.File(self._hdf_files[0], 'r') as f:
            header = dict(f['Parameters'].attrs.items())
        return header

    def _read_ids(self):
        data_ids = np.array([], dtype=self.dtype_int)
        for n in range(self._num_files):
            ids = self.base._hdf_files[n][self.ptype]['ParticleIDs']
            data_ids = np.append(data_ids, ids)
        return data_ids

    def _read_groups(self):
        """ Read all halos/subhalos information from the group catalog. """
        halodat = {}
        subhalodat = {}
        for n in range(self._num_files):
            with h5py.File(self._hdf_files[n], 'r') as f:
                self._read_group_properties(halodat, f, 'Group')
                self._read_group_properties(subhalodat, f, 'Subhalo')

        self._keys = list(halodat.keys())
        if self._subs is True:
            self._keys = list(subhalodat.keys())

        self.add_units_to_properties(halodat, subhalodat)
        return halodat, subhalodat

    @staticmethod
    def _read_group_properties(groupdat, file, gname):
        """ Copy `file[gname]` dictionary into groupdat dictionary. """
        if groupdat:
            for key in list(groupdat.keys()):
                groupdat[key] = np.append(groupdat[key], file[gname][key][:], axis=0)
        else:
            for key in list(file[gname].keys()):
                groupdat[key] = file[gname][key][:]
        return groupdat

    def add_units_to_properties(self, halodat, subhalodat):
        allkeys = list(halodat.keys()) + list(subhalodat.keys())
        keyname, keydim = self._get_property_dimensions(allkeys)

        for name, dimension in zip(keyname, keydim):
            if name in halodat:
                halodat[name] = array.SimArray(halodat[name], self.base.infer_original_units(dimension))
                halodat[name].sim = self.base
            if name in subhalodat:
                subhalodat[name] = array.SimArray(subhalodat[name], self.base.infer_original_units(dimension))
                subhalodat[name].sim = self.base

    @staticmethod
    def _get_property_dimensions(property_names):
        ar_names = []
        ar_dimensions = []
        for key in property_names:
            if 'Mass' in key or '_M_' in key:
                ar_names.append(key)
                ar_dimensions.append('kg')
            elif 'Pos' in key or '_R_' in key or 'CM' in key:
                ar_names.append(key)
                ar_dimensions.append('m')
            elif 'Vel' in key:
                ar_names.append(key)
                ar_dimensions.append('m s^-1')
            else:
                pass

        ar_names.append('SubhaloVmax'); ar_dimensions.append('m s^-1')
        ar_names.append('SubhaloVmaxRad'); ar_dimensions.append('m')
        return ar_names, ar_dimensions

    def __len__(self):
        if self._subs:
            return len(self._subhalodat['SubhaloMass'])
        else:
            return len(self._halodat['GroupMass'])

    def _get_hdf_files_names(self):
        if self._num_files == 1:
            self._hdf_files = [self.halofilename]
        else:
            snapnum = self.halofilename.split("_")[-1]
            self._hdf_files = [self.halofilename + "/fof_subhalo_tab_" + snapnum + "." + str(n) + ".hdf5"
                               for n in range(self._num_files)]

    @staticmethod
    def _name_of_catalogue(sim):
        # multiple snapshot files
        snapnum = os.path.basename(os.path.dirname(sim.filename)).split("_")[-1]
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(sim.filename)))
        dir_path = os.path.join(parent_dir, "groups_" + snapnum)
        if os.path.exists(dir_path):
            return dir_path
        else:
            # single snapshot file
            snapnum = os.path.basename(sim.filename).split("_")[-1]
            parent_dir = os.path.dirname(os.path.abspath(sim.filename))
            return os.path.join(parent_dir, "fof_subhalo_tab_" + snapnum)

    @property
    def base(self):
        return self._base()

    @staticmethod
    def _can_load(sim, **kwargs):
        """ Check if Subfind HDF file exists. """
        file = Gadget4SubfindHDFCatalogue._name_of_catalogue(sim)
        if os.path.exists(file):
            if file.endswith(".hdf5"):
                return True
            elif os.listdir(file)[0].endswith(".hdf5"):
                return True
            else:
                return False
        else:
            return False