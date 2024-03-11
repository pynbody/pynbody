import glob
import os.path
import sys

import numpy as np

from .. import util
from . import HaloCatalogue
from .details import number_mapping


class RockstarFormatRevisionError(RuntimeError):
    pass


class RockstarCatalogue(HaloCatalogue):
    def __init__(self, sim, dummy=False, pathname=None, format_revision=None,
                 filenames=None, sort=False, **kwargs):
        """Initialize a RockstarCatalogue.

        **kwargs** :


        *dummy*: if True, the particle file is not loaded, and all
                 halos returned are just dummies (with the correct
                 properties dictionary loaded). Use load_copy to get
                 the actual data in this case.

        *sort*: if True, resort the halos into descending order of
                particle number. Otherwise, leave in RockStar output order.

        *filenames*: a list of filenames of each of the RockStar outputs.
                     You probably want to use pathname instead, which specifies
                     the path to the output folder.

        *pathname*: the path of the output folder with the individual RockStar outputs

        *format_revision*: Override the header's format revision information. Specify
                    1, 2, 'caterpillar', 'galaxies' for Rockstar prior to 2014, post 2014,
                    customized for the caterpillar project and for rockstar
                    galaxies respectively

        """

        if filenames is not None:
            self._files = filenames
        else:
            if pathname is None:
                pathname = os.path.dirname(sim.filename)
            self._files = glob.glob(os.path.join(pathname,'halos*.bin'))
            if len(self._files)==0 :
                self._files = glob.glob(os.path.join(pathname, 'halos*.boundbin'))
            self._files.sort()

        if len(self._files)==0:
            raise OSError("Could not find any Rockstar output. Try specifying pathname='/path/to/rockstar/outputfolder'")

        self._cpus = [_RockstarCatalogueOneCpu(file_i, format_revision=format_revision) for file_i in self._files]
        self._prune_files_from_wrong_scalefactor(sim)

        halomins = [cpu.halo_min_inclusive for cpu in self._cpus]
        # order self._cpus and self._files by halomins
        # this isn't technically necessary but it means that the internal halo indexing will be
        # in the same order as the halo numbering in the files. In fact, it probably means that
        # the internal halo indexing will be identical to the halo numbering in the files, but we
        # won't take that for granted anywhere.
        self._cpus = [x for _, x in sorted(zip(halomins, self._cpus))]
        self._files = [x for _, x in sorted(zip(halomins, self._files))]

        halo_numbers = np.empty(sum(len(x) for x in self._cpus), dtype=int)
        self._cpu_per_halo = np.empty(len(halo_numbers), dtype=int)
        i = 0
        for j, cat in enumerate(self._cpus):
            halo_numbers[i:i+len(cat)] = cat.get_halo_numbers()
            self._cpu_per_halo[i:i+len(cat)] = j
            i+=len(cat)
        assert i == len(halo_numbers)

        super().__init__(sim, number_mapping.create_halo_number_mapper(halo_numbers))



    def _prune_files_from_wrong_scalefactor(self, sim):
        new_cpus = []
        new_files = []
        for file,cpu in zip(self._files,self._cpus):
            if abs(sim.properties['a']-cpu._head['scale'][0])<1e-6:
                new_cpus.append(cpu)
                new_files.append(file)
        self._cpus = new_cpus
        self._files = new_files

    def _get_particle_indices_one_halo(self, halo_number):
        halo_index = self.number_mapper.number_to_index(halo_number)
        cpu = self._cpu_per_halo[halo_index]
        iords = self._cpus[cpu].read_iords_for_halo(halo_number)
        self._init_iord_to_fpos()
        return self._iord_to_fpos.map_ignoring_order(iords)

    def _get_all_particle_indices(self):
        iords = np.empty(0, dtype=int)
        boundaries = np.empty((0, 2), dtype=int)
        for cpu in self._cpus:
            iords_this_cpu, boundaries_this_cpu = cpu.read_iords_for_all_halos()
            boundaries = np.append(boundaries, boundaries_this_cpu + len(iords), axis=0)
            iords = np.append(iords, iords_this_cpu)

        self._init_iord_to_fpos()
        fpos = iords # nb this doesn't copy! but we won't use the iords again after this, so it's ok
        for a, b in boundaries:
            # iord_to_fpos may not retain ordering, so have to do this per halo
            fpos[a:b] = self._iord_to_fpos.map_ignoring_order(iords[a:b])
        return fpos, boundaries

    def get_properties_one_halo(self, halo_number):
        halo_index = self.number_mapper.number_to_index(halo_number)
        cpu = self._cpu_per_halo[halo_index]
        return self._cpus[cpu].read_properties_for_halo(halo_number)

    def get_properties_all_halos(self, with_units=True) -> dict:
        props = {}
        for cpu in self._cpus:
            props_this_cpu = cpu.read_properties_all_halos()
            # concatenate all the properties
            for k, v in props_this_cpu.items():
                if k in props:
                    props[k] = np.concatenate((props[k], v))
                else:
                    props[k] = v

        return props

    @classmethod
    def _can_load(cls, sim, **kwargs):
        return len(
            glob.glob(os.path.join(os.path.dirname(sim.filename), 'halos*.bin'))
        ) > 0


class _RockstarCatalogueOneCpu:
    """
    Low-level reader for single CPU output from Rockstar. Users should normally not use this class,
    rather using RockstarCatalogue which collates the multiple sub-files that Rockstar produces.
    """

    head_type = np.dtype([('magic',np.uint64),('snap',np.int64),
                          ('chunk',np.int64),('scale','f'),
                          ('Om','f'),('Ol','f'),('h0','f'),
                          ('bounds','f',6),('num_halos',np.int64),
                          ('num_particles',np.int64),('box_size','f'),
                          ('particle_mass','f'),('particle_type',np.int64),
                          ('format_revision',np.int32),
                          ('rockstar_version',np.str_,12)])

    halo_types = {
        1: np.dtype([('id', np.int64), ('pos', 'f', 3), ('vel', 'f', 3),
                  ('corevel', 'f', 3), ('bulkvel', 'f', 3), ('m', 'f'),
                  ('r', 'f'),
                  ('child_r', 'f'), ('vmax_r', 'f'), ('mgrav', 'f'),
                  ('vmax', 'f'), ('rvmax', 'f'), ('rs', 'f'),
                  ('klypin_rs', 'f'), ('vrms', 'f'), ('J', 'f', 3),
                  ('energy', 'f'), ('spin', 'f'), ('alt_m', 'f', 4),
                  ('Xoff', 'f'), ('Voff', 'f'), ('b_to_a', 'f'),
                  ('c_to_a', 'f'), ('A', 'f', 3), ('b_to_a2', 'f'),
                  ('c_to_a2', 'f'), ('A2', 'f', 3), ('bullock_spin', 'f'),
                  ('kin_to_pot', 'f'), ('m_pe_b', 'f'), ('m_pe_d', 'f'),
                  ('num_p', np.int64), ('num_child_particles', np.int64),
                  ('p_start', np.int64), ('desc', np.int64),
                  ('flags', np.int64), ('n_core', np.int64),
                  ('min_pos_err', 'f'), ('min_vel_err', 'f'),
                  ('min_bulkvel_err', 'f')], align=True)  # Rockstar format v1
        ,
        2: np.dtype([('id',np.int64),('pos','f',3),('vel','f',3),
                          ('corevel','f',3),('bulkvel','f',3),('m','f'),
                          ('r','f'),
                          ('child_r','f'),('vmax_r','f'),('mgrav','f'),
                          ('vmax','f'),('rvmax','f'),('rs','f'),
                          ('klypin_rs','f'),('vrms','f'),('J','f',3),
                          ('energy','f'),('spin','f'),('alt_m','f',4),
                          ('Xoff','f'),('Voff','f'),('b_to_a','f'),
                          ('c_to_a','f'),('A','f',3),('b_to_a2','f'),
                          ('c_to_a2','f'),('A2','f',3),('bullock_spin','f'),
                          ('kin_to_pot','f'),('m_pe_b','f'),('m_pe_d','f'),
                          ('halfmass_radius','f'),
                          ('num_p',np.int64),('num_child_particles',np.int64),
                          ('p_start',np.int64),('desc',np.int64),
                          ('flags',np.int64),('n_core',np.int64),
                          ('min_pos_err','f'),('min_vel_err','f'),
                          ('min_bulkvel_err','f')], align=True), # Rockstar format v2, includes halfmass_radius

        'caterpillar': np.dtype([('id',np.int64),
                                 ('pos','f',3),('vel','f',3),
                          ('corevel','f',3),('bulkvel','f',3),('m','f'),
                          ('r','f'),
                          ('child_r','f'),('vmax_r','f'),('mgrav','f'),
                          ('vmax','f'),('rvmax','f'),('rs','f'),
                          ('klypin_rs','f'),('vrms','f'),('J','f',3),
                          ('energy','f'),('spin','f'),('alt_m','f',4),
                          ('Xoff','f'),('Voff','f'),('b_to_a','f'),
                          ('c_to_a','f'),('A','f',3),('b_to_a2','f'),
                          ('c_to_a2','f'),('A2','f',3),('bullock_spin','f'),
                          ('kin_to_pot','f'),('m_pe_b','f'),('m_pe_d','f'),
                          ('halfmass_radius','f'),
                          ('num_p',np.int64),('num_child_particles',np.int64),

                          ('p_start',np.int64),('desc',np.int64),
                          ('flags',np.int64),('n_core',np.int64),
                          ('min_pos_err','f'),('min_vel_err','f'),
                          ('min_bulkvel_err','f'),
                          ('num_bound', 'i8'), ('num_iter', 'i8')]
                                , align=True), # Hacked rockstar from caterpillar project
        'galaxies':  np.dtype(
        [
                ("id", np.int64),
                ("pos", np.float32, 3),
                ("vel", np.float32, 3),
                ("corevel", np.float32, 3),
                ("bulkvel", np.float32, 3),
                ("m", np.float32),
                ("r", np.float32),
                ("child_r", np.float32),
                ("vmax_r", np.float32),
                ("mgrav", np.float32),
                ("vmax", np.float32),
                ("rvmax", np.float32),
                ("rs", np.float32),
                ("klypin_rs", np.float32),
                ("vrms", np.float32),
                ("J", np.float32, 3),
                ("energy", np.float32),
                ("spin", np.float32),
                ("alt_m", np.float32, 4),
                ("Xoff", np.float32),
                ("Voff", np.float32),
                ("b_to_a", np.float32),
                ("c_to_a", np.float32),
                ("A", np.float32, 3),
                ("b_to_a2", np.float32),
                ("c_to_a2", np.float32),
                ("A2", np.float32, 3),
                ("bullock_spin", np.float32),
                ("kin_to_pot", np.float32),
                ("m_pe_b", np.float32),
                ("m_pe_d", np.float32),
                ("num_p", np.int64),
                ("num_child_particles", np.int64),
                ("p_start", np.int64),
                ("desc", np.int64),
                ("flags", np.int64),
                ("n_core", np.int64),
                ("min_pos_err", np.float32),
                ("min_vel_err", np.float32),
                ("min_bulkvel_err", np.float32),
                ("type", np.int32),
                ("sm", np.float32),
                ("gas", np.float32),
                ("bh", np.float32),
                ("peak_density", np.float32),
                ("av_density", np.float32),
            ],
            align=True
        ), # Galaxy format from Rockstar
    }


    def __init__(self, filename=None, format_revision=None):

        self._rsFilename = filename


        if not os.path.exists(self._rsFilename):
            raise OSError(
                "Halo catalogue not found -- check the file name of catalogue data"
                " or try specifying a catalogue using the filename keyword"
            )

        with util.open_(self._rsFilename, 'rb') as f:
            self._head = np.frombuffer(f.read(self.head_type.itemsize),
                                       dtype=self.head_type)

            # Seek to absolute position
            f.seek(256)

            self._nhalos = self._head['num_halos'][0]

            self._load_rs_halos_with_format_detection(f, format_revision)

    def _load_rs_halos_with_format_detection(self, f, format_revision):
        if format_revision is None:
            # The 'galaxies' format can be either 1 or 2, so we need to try it
            # in both cases.
            format_revision_to_try = [self._head['format_revision'][0], "galaxies"]
        else:
            format_revision_to_try = [format_revision]

        current_pos = f.tell()
        for format_revision in format_revision_to_try:
            f.seek(current_pos)
            try:
                self.halo_type = self.halo_types[format_revision]
                self._load_rs_halos(f)
                return
            except RockstarFormatRevisionError:
                pass

        raise RockstarFormatRevisionError(
            "Could not detect the format revision of the Rockstar catalogue."
        )


    def __len__(self):
        return len(self._halo_lens)



    def read_properties_for_halo(self, n):
        if n<self.halo_min_inclusive or n>=self.halo_max_exclusive:
            raise KeyError("No such halo")

        with util.open_(self._rsFilename, 'rb') as f:
            f.seek(self._haloprops_offset + (n - self.halo_min_inclusive) * self.halo_type.itemsize)
            halo_data = np.fromfile(f, dtype=self.halo_type, count=1)

        # TODO: properties are in Msun / h, Mpc / h
        return dict(list(zip(halo_data.dtype.names,halo_data[0])))

    def read_properties_all_halos(self):
        with util.open_(self._rsFilename, 'rb') as f:
            f.seek(self._haloprops_offset)
            data = np.fromfile(f, dtype=self.halo_type, count=self.halo_max_exclusive - self.halo_min_inclusive)

        data_dict = {name: data[name] for name in data.dtype.names}

        return data_dict

    def _load_rs_halos(self, f):
        self._haloprops_offset = f.tell()
        self._halo_offsets = np.empty(self._head['num_halos'][0],dtype=np.int64)
        self._halo_lens = np.empty(self._head['num_halos'][0],dtype=np.int64)

        offset = self._haloprops_offset+self.halo_type.itemsize*self._head['num_halos'][0]

        self.halo_min_inclusive = int(np.fromfile(f, dtype=self.halo_type, count=1)['id'][0])
        self.halo_max_exclusive = int(self.halo_min_inclusive + self._head['num_halos'][0])

        f.seek(self._haloprops_offset)

        for this_id in range(self.halo_min_inclusive, self.halo_max_exclusive):
            halo_data = np.fromfile(f, dtype=self.halo_type, count=1)
            if halo_data['id'] != this_id:
                raise RockstarFormatRevisionError(
                    "Error while reading halo catalogue. Expected "
                    "halo ID %d, but got %d" % (this_id, halo_data['id'])
                )
            self._halo_offsets[this_id - self.halo_min_inclusive] = offset
            if 'num_bound' in self.halo_type.names:
                num_ptcls = int(halo_data['num_bound'][0])
            else:
                num_ptcls = int(halo_data['num_p'][0])
            self._halo_lens[this_id - self.halo_min_inclusive] = num_ptcls
            offset+=num_ptcls*np.dtype('int64').itemsize


    def get_halo_numbers(self):
        return np.arange(self.halo_min_inclusive, self.halo_max_exclusive)

    def read_iords_for_halo(self, num):
        if num<self.halo_min_inclusive or num>=self.halo_max_exclusive:
            raise KeyError("No such halo")

        with util.open_(self._rsFilename, 'rb') as f:
            f.seek(self._halo_offsets[num - self.halo_min_inclusive])
            return np.fromfile(f, dtype=np.int64, count=self._halo_lens[num - self.halo_min_inclusive])

    def read_iords_for_all_halos(self):
        """Returns an array with all halo iords, and the boundaries of each halo in the array."""
        # check the halo ids are contiguous as expected
        assert (np.diff(self._halo_offsets) == 8 * self._halo_lens[:-1]).all()
        with util.open_(self._rsFilename, 'rb') as f:
            f.seek(self._halo_offsets[0])
            iords = np.fromfile(f, dtype=np.int64, count=self._halo_lens.sum())

        boundaries = np.empty((len(self._halo_lens), 2), dtype=np.int64)
        boundaries[:,1] = np.cumsum(self._halo_lens)
        boundaries[0,0] = 0
        boundaries[1:,0] = boundaries[:-1,1]

        return iords, boundaries
