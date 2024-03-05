import os.path
import warnings
import weakref

import numpy as np

from .. import units
from ..array import SimArray
from . import HaloCatalogue
from .details import number_mapping, particle_indices


class SubfindCatalogue(HaloCatalogue):
    """Class to handle catalogues produced by the SubFind halo finder."""


    def __init__(self, sim, subs=False, ordered=None):
        """Initialise a SubFind catalogue

        By default, the FoF groups are imported, and subhalos are available via the 'subhalos' attribute of each
        halo object, e.g.

        >>> f = pynbody.load('path/to/snapshot')
        >>> h = f.halos()
        >>> h[1].subhalos[2] # returns the third subhalo of FoF group 1

        However by setting subs=True, the FoF groups are ignored and the catalogue is of all subhalos.

        **kwargs** :

        *subs*: if True, load the subhalos; otherwise, load FoF groups (default)
        *ordered*: if True, the snapshot must have sequential iords. If False, the iords might be out-of-sequence, and
                 therefore reordering will take place. If None (default), the iords are examined to check whether
                 re-ordering is required or not. Note that re-ordering can be undesirable because it also destroys
                 subfind's order-by-binding-energy
        """

        self._subs=subs

        if ordered is not None:
            self._ordered = ordered
        else:
            self._ordered = bool((sim['iord']==np.arange(len(sim))).all())

        self._halos = {}

        self.dtype_int = sim['iord'].dtype
        self.dtype_flt = 'float32' #SUBFIND data is apparently always single precision???
        self._subfind_dir = self._name_of_catalogue(sim)
        self.header = self._read_header()

        if subs is True:
            if self.header[6]==0:
                raise ValueError("This file does not contain subhalos")

        self._tasks = self.header[4]

        self._keys={}
        self._halodat, self._subhalodat=self._read_groups(sim)

        if subs:
            length = len(self._subhalodat['sub_off'])
        else:
            length = len(self._halodat['group_off'])

        super().__init__(sim, number_mapping.SimpleHaloNumberMapper(0, length))

    def _get_all_particle_indices(self):
        ids = self._read_ids()
        boundaries = np.empty((len(self), 2), dtype=self.dtype_int)
        if self._subs:
            boundaries[:, 0] = self._subhalodat['sub_off']
            boundaries[:, 1] = self._subhalodat['sub_off'] + self._subhalodat['sub_len']
        else:
            boundaries[:, 0] = self._halodat['group_off']
            boundaries[:, 1] = self._halodat['group_off'] + self._halodat['group_len']

        if not self._ordered:
            self._init_iord_to_fpos()
            for a, b in boundaries:
                ids[a:b] = self._iord_to_fpos.map_ignoring_order(ids[a:b])
                # must be done segmented in case iord_to_fpos doesn't preserve input order

        return particle_indices.HaloParticleIndices(ids, boundaries)

    def _get_properties_one_halo(self, i):

        extract = units.get_item_with_unit

        properties = {}
        halo_number_within_group = (self._subhalodat['sub_groupNr'][:i] == self._subhalodat['sub_groupNr'][i]).sum()
        if self._subs is False:
            for key in self._keys:
                properties[key] = extract(self._halodat[key], i)
            if self.header[6] > 0:
                properties['children'], = np.where(self._subhalodat['sub_groupNr'] == i)
                # this is the FIRST level of substructure, sub-subhalos (etc) can be accessed via the
                # subs=True output (below)
        else:
            for key in self._keys:
                properties[key] = extract(self._subhalodat[key], i)
            properties['children'], = np.where((self._subhalodat['sub_parent'] == halo_number_within_group) \
                                               & (self._subhalodat['sub_groupNr'] == properties['sub_groupNr']))
            # this goes down one level in the hierarchy, i.e. a subhalo will have all its sub-subhalos listed,
            # but not its sub-sub-subhalos (those will be listed in each sub-subhalo)
        return properties

    def _read_header(self):
        iout = self._subfind_dir.split("_")[-1]
        filename = os.path.join(
            self._subfind_dir,
            f"subhalo_tab_{iout}.0"
        )
        with open(filename, "rb") as fd:
            # read header: this is strange but it works: there is an extra value in
            # header which we delete in the next step
            header = np.fromfile(fd, dtype='int32', sep="", count=8)

        header = np.delete(header, 4)

        return header

    def _read_ids(self):
        data_ids = np.array([], dtype=self.dtype_int)
        iout = self._subfind_dir.split("_")[-1]
        for n in range(0, self._tasks):
            filename = os.path.join(
                self._subfind_dir,
                f"subhalo_ids_{iout}.{n}"
            )
            fd = open(filename, "rb")
            # for some reason there is an extra value in header which we delete
            # in the next step
            header1 = np.fromfile(fd, dtype='int32', sep="", count=7)
            header = np.delete(header1, 4)
            # TODO: include a check if both headers agree (they better)
            ids = np.fromfile(fd, dtype=self.dtype_int, sep="", count=-1)
            fd.close()
            data_ids = np.append(data_ids, ids)
        return data_ids

    def _read_groups(self, sim):
        halodat={}
        keys_flt=['mass', 'pos', 'mmean_200', 'rmean_200', 'mcrit_200', 'rcrit_200', 'mtop_200', 'rtop_200', 'contmass']
        keys_int=['group_len', 'group_off',  'first_sub', 'Nsubs', 'cont_count', 'mostboundID']
        for key in keys_flt:
            halodat[key]=np.array([], dtype=self.dtype_flt)
        for key in keys_int:
            halodat[key]=np.array([], dtype='int32')

        subhalodat={}
        subkeys_int=['sub_len', 'sub_off', 'sub_parent', 'sub_mostboundID', 'sub_groupNr']
        subkeys_flt=['sub_pos', 'sub_vel', 'sub_CM', 'sub_mass', 'sub_spin', 'sub_veldisp', 'sub_VMax', 'sub_VMaxRad', 'sub_HalfMassRad', ]
        for key in subkeys_int:
            subhalodat[key]=np.array([], dtype='int32')
        subhalodat['sub_mostboundID']=np.array([], dtype=self.dtype_int)
        #subhalodat['sub_groupNr']=np.array([], dtype=self.dtype_int) #these are special
        for key in subkeys_flt:
            subhalodat[key]=np.array([], dtype=self.dtype_flt)

        self._keys=keys_flt+keys_int
        if self._subs is True:
            self._keys=subkeys_flt+subkeys_int

        for n in range(0,self._tasks):
            iout = self._subfind_dir.split("_")[-1]
            filename = os.path.join(
                self._subfind_dir,
                f"subhalo_tab_{iout}.{n}"
            )
            fd=open(filename, "rb")
            header1=np.fromfile(fd, dtype='int32', sep="", count=8)
            header=np.delete(header1,4)
            #read groups
            if header[0]>0:
                halodat['group_len']=np.append(halodat['group_len'], np.fromfile(fd, dtype='int32', sep="", count=header[0]))
                halodat['group_off']=np.append(halodat['group_off'], np.fromfile(fd, dtype='int32', sep="", count=header[0]))
                halodat['mass']=np.append(halodat['mass'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=header[0]))
                halodat['pos']=np.append(halodat['pos'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=3*header[0]) )
                halodat['mmean_200']=np.append(halodat['mmean_200'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=header[0]))
                halodat['rmean_200']=np.append(halodat['rmean_200'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=header[0]))
                halodat['mcrit_200']=np.append(halodat['mcrit_200'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=header[0]))
                halodat['rcrit_200']=np.append(halodat['rcrit_200'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=header[0]))
                halodat['mtop_200']=np.append(halodat['mtop_200'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=header[0]))
                halodat['rtop_200']=np.append(halodat['rtop_200'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=header[0]))
                halodat['cont_count']=np.append(halodat['cont_count'], np.fromfile(fd, dtype='int32', sep="", count=header[0]))
                halodat['contmass']=np.append(halodat['contmass'],np.fromfile(fd, dtype=self.dtype_flt, sep="", count=header[0]))
                halodat['Nsubs']=np.append(halodat['Nsubs'],np.fromfile(fd, dtype='int32', sep="", count=header[0]))
                halodat['first_sub']=np.append(halodat['first_sub'],np.fromfile(fd, dtype='int32', sep="", count=header[0]))
            #read subhalos only if expected to exist from header
            if header[5]>0:
                subhalodat['sub_len']=np.append(subhalodat['sub_len'], np.fromfile(fd, dtype='int32', sep="", count=header[5]))
                subhalodat['sub_off']=np.append(subhalodat['sub_off'], np.fromfile(fd, dtype='int32', sep="", count=header[5]))
                subhalodat['sub_parent']=np.append(subhalodat['sub_parent'], np.fromfile(fd, dtype='int32', sep="", count=header[5]))
                subhalodat['sub_mass']=np.append(subhalodat['sub_mass'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=header[5]))
                subhalodat['sub_pos']=np.append(subhalodat['sub_pos'],np.fromfile(fd, dtype=self.dtype_flt, sep="", count=3*header[5]))
                subhalodat['sub_vel']=np.append(subhalodat['sub_vel'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=3*header[5]))
                subhalodat['sub_CM']=np.append(subhalodat['sub_CM'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=3*header[5]))
                subhalodat['sub_spin']=np.append(subhalodat['sub_spin'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=3*header[5]))
                subhalodat['sub_veldisp']=np.append(subhalodat['sub_veldisp'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=header[5]))
                subhalodat['sub_VMax']=np.append(subhalodat['sub_VMax'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=header[5]))
                subhalodat['sub_VMaxRad']=np.append(subhalodat['sub_VMaxRad'],np.fromfile(fd, dtype=self.dtype_flt, sep="", count=header[5]))
                subhalodat['sub_HalfMassRad']=np.append(subhalodat['sub_HalfMassRad'], np.fromfile(fd, dtype=self.dtype_flt, sep="", count=header[5]))
                subhalodat['sub_mostboundID']=np.append(subhalodat['sub_mostboundID'], np.fromfile(fd, dtype=self.dtype_int, sep="", count=header[5]))
                subhalodat['sub_groupNr']=np.append(subhalodat['sub_groupNr'], np.fromfile(fd, dtype='int32', sep="", count=header[5]))
            fd.close()

        halodat['pos']=np.reshape(halodat['pos'], (header[1],3))
        if header[6]>0:
            #some voodoo because some SubFind files may have (at least?) one extra entry which is not really a subhalo
            real_ones=np.where(halodat['first_sub']<header[6])[0]
            fake_ones=np.where(halodat['first_sub']>=header[6])[0]
            halodat['mostboundID']=np.zeros(len(halodat['Nsubs']),dtype=self.dtype_int)-1
            halodat['mostboundID'][real_ones]=subhalodat['sub_mostboundID'][halodat['first_sub'][real_ones]]  #useful for the case of unordered snapshot IDs

            subhalodat['sub_pos']=np.reshape(subhalodat['sub_pos'], (header[6],3))
            subhalodat['sub_vel']=np.reshape(subhalodat['sub_vel'], (header[6],3))
            subhalodat['sub_CM']=np.reshape(subhalodat['sub_CM'], (header[6],3))
            subhalodat['sub_spin']=np.reshape(subhalodat['sub_spin'], (header[6],3))

        ar_names = 'mass', 'pos', 'mmean_200', 'rmean_200', 'mcrit_200', 'rcrit_200', 'mtop_200', 'rtop_200', \
                   'sub_mass', 'sub_pos', 'sub_vel', 'sub_CM', 'sub_veldisp', 'sub_VMax', 'sub_VMaxRad', 'sub_HalfMassRad'
        ar_dimensions = 'kg', 'm', 'kg', 'm', 'kg', 'm', 'kg', 'm', \
                    'kg', 'm', 'm s^-1', 'm', 'm s^-1', 'm s^-1', 'm', 'm'

        for name, dimension in zip(ar_names, ar_dimensions):
            if name in halodat:
                halodat[name] = SimArray(halodat[name], sim.infer_original_units(dimension))
                halodat[name].sim = sim
            if name in subhalodat:
                subhalodat[name] = SimArray(subhalodat[name], sim.infer_original_units(dimension))
                subhalodat[name].sim = sim

        return halodat, subhalodat

    @staticmethod
    def _name_of_catalogue(sim):
        # standard path for multiple snapshot files
        snapnum = os.path.basename(
            os.path.dirname(sim.filename)).split("_")[-1]
        parent_dir = os.path.dirname(os.path.dirname(sim.filename))
        dir_path=os.path.join(parent_dir,"groups_" + snapnum)

        if os.path.exists(dir_path):
            return dir_path
        # alternative path if snapshot is single file
        else:
            snapnum = os.path.basename(sim.filename).split("_")[-1]
            parent_dir = os.path.dirname(sim.filename)
            return os.path.join(parent_dir,"groups_" + snapnum)

    @staticmethod
    def _can_load(sim, **kwargs):
        file = SubfindCatalogue._name_of_catalogue(sim)
        if os.path.exists(file):
            if os.path.exists(file):
                if file.endswith(".hdf5"):
                    return False
                elif os.listdir(file)[0].endswith(".hdf5"):
                    return False
                else:
                    return True
        else:
            return False
