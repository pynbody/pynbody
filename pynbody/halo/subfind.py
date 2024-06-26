"""Support for the SubFind halo finder"""

from __future__ import annotations

import os.path
import warnings

import numpy as np

from .. import units
from ..array import SimArray
from . import HaloCatalogue
from .details import number_mapping, particle_indices
from .subhalo_catalogue import SubhaloCatalogue


class SubfindCatalogue(HaloCatalogue):
    """Handles catalogues produced by the SubFind halo finder (old versions that do not use HDF5 outputs)."""

    def __init__(self, sim, filename=None, subs=None, subhalos=False,  ordered=None, _inherit_data_from=None):
        """Initialise a SubFind catalogue

        By default, the FoF groups are imported, and subhalos are available via the 'subhalos' attribute of each
        halo object, e.g.

        >>> snap = pynbody.load('path/to/snapshot')
        >>> halos = snap.halos()
        >>> halos[1].subhalos[2] # returns the third subhalo of FoF group 1

        However by setting ``subs=True``, the FoF groups are ignored and the catalogue is of all subhalos.

        Parameters
        ----------

        sim : ~pynbody.snapshot.simsnap.SimSnap
            The simulation snapshot to which this catalogue applies.

        filename : str, optional
            The path to the SubFind output(s). This is expected to be a folder named ``groups_XXX`` where XXX is the
            snapshot number. The code extracts the snapshot number from the folder name and uses it to construct
            the filename of the catalogue files, for example ``groups_XXX/subhalo_tab_XXX.0``. If no filename is
            provided, the code will attempt to find a suitable catalogue starting from the simulation's directory.

        subhalos : bool, optional
            If False (default), catalogue represents the FoF groups and subhalos are available through the
            :meth:`~pynbody.halo.Halo.subhalos` attribute of each group (see note above). If True, the catalogue
            represents the subhalos directly and FoF groups are not available.

        ordered : bool, optional
            If True, the snapshot must have sequential iords. If False, the iords might be out-of-sequence, and
            therefore reordering will take place. If None (default), the iords are examined to check whether
            re-ordering is required or not. Note that re-ordering can be undesirable because it also destroys
            subfind's order-by-binding-energy

        _inherit_data_from : SubfindCatalogue, optional
            For internal use only; allows subhalo catalogue to share data with its parent FOF catalogue


        """

        if subs is not None:
            warnings.warn("The 'subs' argument to SubfindCatalogue is deprecated; use 'subhalos' instead",
                          DeprecationWarning)
            subhalos = subs

        self._subs=subhalos

        if ordered is not None:
            self._ordered = ordered
        else:
            self._ordered = bool((sim['iord']==np.arange(len(sim))).all())

        self._halos = {}

        self.dtype_int = sim['iord'].dtype
        self.dtype_flt = 'float32' #SUBFIND data is apparently always single precision???
        self._subfind_dir = filename or self._name_of_catalogue(sim)

        self.header = self._read_header()
        self._tasks = self.header[4]

        if _inherit_data_from:
            self._inherit_data(_inherit_data_from)
        else:
            self._read_data(sim)



        if subhalos:
            if self.header[6]==0:
                raise ValueError("This file does not contain subhalos")
            length = len(self._subhalodat['sub_off'])
        else:
            length = len(self._halodat['group_off'])

        super().__init__(sim, number_mapping.SimpleHaloNumberMapper(0, length))

        if not subhalos:
            self._subhalo_catalogue = SubfindCatalogue(sim, subhalos=True, filename=filename,
                                                       ordered=self._ordered, _inherit_data_from=self)

    def _inherit_data(self, from_):
        inherit = ['_halodat', '_subhalodat', '_keys_halo', '_keys_subhalo']
        for k in inherit:
            setattr(self, k, getattr(from_, k))

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

    def get_properties_one_halo(self, i):

        extract = units.get_item_with_unit

        properties = {}

        if self._subs is False:
            for key in self._keys_halo:
                properties[key] = extract(self._halodat[key], i)
            properties['children'] = self._get_children_of_group(i)
        else:
            for key in self._keys_subhalo:
                properties[key] = extract(self._subhalodat[key], i)
            properties['children'] = self._get_children_of_sub(i)
        return properties

    def _get_children_of_group(self, group_nr):
        if self.header[6] > 0:
            return np.where(self._subhalodat['sub_groupNr'] == group_nr)[0]
        else:
            return []

    def _get_children_of_sub(self, subhalo_nr):
        # this goes down one level in the hierarchy, i.e. a subhalo will have all its sub-subhalos listed,
        # but not its sub-sub-subhalos (those will be listed in each sub-subhalo)
        halo_number_within_group = (self._subhalodat['sub_groupNr'][:subhalo_nr]
                                    == self._subhalodat['sub_groupNr'][subhalo_nr]).sum()
        return np.where((self._subhalodat['sub_parent'] == halo_number_within_group) &
                        (self._subhalodat['sub_groupNr'] == self._subhalodat['sub_groupNr'][subhalo_nr]))[0]


    def get_properties_all_halos(self, with_units=True) -> dict:
        properties = {}
        if self._subs:
            data = self._subhalodat
            properties['children'] = [self._get_children_of_sub(subhalo_nr)
                                      for subhalo_nr in self.number_mapper.all_numbers]
        else:
            data = self._halodat
            if self.header[6] > 0:
                properties['children'] = [self._get_children_of_group(group_nr)
                                          for group_nr in self.number_mapper.all_numbers]
            else:
                properties['children'] = []

        keys = self._keys_halo if not self._subs else self._keys_subhalo

        for key in keys:
            if with_units:
                properties[key] = data[key]
            else:
                properties[key] = data[key].view(np.ndarray)


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

    def _get_subhalo_catalogue(self, parent_halo_number):
        if self._subs:
            return SubhaloCatalogue(self, self._get_children_of_sub(parent_halo_number))
        else:
            return SubhaloCatalogue(self._subhalo_catalogue,
                                    self._get_children_of_group(parent_halo_number))
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

    def _read_data(self, sim):
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

        self._keys_subhalo = subkeys_flt+subkeys_int
        self._keys_halo = keys_flt + keys_int

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

        self._halodat, self._subhalodat = halodat, subhalodat

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
    def _can_load(sim, filename=None, **kwargs):
        if filename is not None:
            if str(filename)[:-3].endswith("groups_"):
                return True
        file = SubfindCatalogue._name_of_catalogue(sim)
        if os.path.exists(file):
            if os.path.exists(file):
                if file.endswith(".hdf5"):
                    return False
                elif os.path.isdir(file) and os.listdir(file)[0].endswith(".hdf5"):
                    return False
                else:
                    return True
        else:
            return False
