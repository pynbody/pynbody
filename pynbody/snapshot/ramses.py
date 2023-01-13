"""

ramses
======

Implements classes and functions for handling RAMSES files. AMR cells
are loaded as particles. You rarely need to access this module
directly as it will be invoked automatically via pynbody.load.


For a complete demo on how to use RAMSES outputs with pynbody, look at
the `ipython notebook demo
<http://nbviewer.ipython.org/github/pynbody/pynbody/blob/master/examples/notebooks/pynbody_demo-ramses.ipynb>`_

"""

  # for py2.5


import csv
import logging
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np

from .. import analysis, array, config, config_parser, family, units
from ..extern.cython_fortran_utils import FortranFile
from . import SimSnap, namemapper

logger = logging.getLogger('pynbody.snapshot.ramses')

from collections import OrderedDict

multiprocess_num = int(config_parser.get('ramses', "parallel-read"))
multiprocess = (multiprocess_num > 1)

issue_multiprocess_warning = False

if multiprocess:
    try:
        import multiprocessing

        import posix_ipc
        remote_exec = array.shared_array_remote
        remote_map = array.remote_map
    except ImportError:
        issue_multiprocess_warning = True
        multiprocess = False

if not multiprocess:
    def remote_exec(fn):
        def q(*args):
            t0 = time.time()
            r = fn(*args)
            return r
        return q

    def remote_map(*args, **kwargs):
        return list(map(*args[1:], **kwargs))


_float_type = 'd'
_int_type = 'i'


def _timestep_id(basename):
    try:
        return re.findall("output_([0-9]*)/*$", basename)[0]
    except IndexError:
        return None


def _cpu_id(i):
    return str(i).rjust(5, "0")


@remote_exec
def _cpui_count_particles_with_implicit_families(filename, distinguisher_field, distinguisher_type):

    with FortranFile(filename) as f:
        f.seek(0, 2)
        eof_fpos = f.tell()
        f.seek(0, 0)
        header = f.read_attrs(ramses_particle_header)
        npart_this = header['npart']
        f.skip(distinguisher_field)
        # Avoid end-of-file issues
        if f.tell() == eof_fpos:
            data = np.array([])
        else:
            data = f.read_vector(distinguisher_type)

        if len(data)>0:
            my_mask = np.array((data != 0), dtype=np.int8) # -> 0 for dm, 1 for star
        else:
            my_mask = np.zeros(npart_this, dtype=np.int8)
        nstar_this = (data != 0).sum()
        return npart_this, nstar_this, my_mask

@remote_exec
def _cpui_count_particles_with_explicit_families(filename, family_field, family_type):
    assert np.issubdtype(family_type, np.int8)
    counts_array = np.zeros(256,dtype=np.int64)
    with FortranFile(filename) as f:
        header = f.read_attrs(ramses_particle_header)
        npart_this = header['npart']

        f.skip(family_field)
        my_mask = f.read_vector(family_type)

        unique_mask_ids, counts = np.unique(my_mask, return_counts=True)
        counts_array[unique_mask_ids]=counts

        assert sum(counts)==npart_this

        return counts_array, my_mask

@remote_exec
def _cpui_load_particle_block(filename, arrays, offset, first_index, type_, family_mask):
    with FortranFile(filename) as f:
        header = f.read_attrs(ramses_particle_header)
        f.skip(offset)
        data = f.read_vector(type_)
        for fam_id, ar in enumerate(arrays):
            data_this_family = data[family_mask == fam_id]
            ind0 = first_index[fam_id]
            ind1 = ind0 + len(data_this_family)
            ar[ind0:ind1] = data_this_family


def _cpui_level_iterator(cpu, amr_filename, bisection_order, maxlevel, ndim):
    with FortranFile(amr_filename) as f:
        header = f.read_attrs(ramses_amr_header)
        f.skip(13)

        n_per_level = f.read_vector(_int_type).reshape((header['nlevelmax'], header['ncpu']))
        f.skip(1)
        if header['nboundary'] > 0:
            f.skip(2)
            n_per_level_boundary = f.read_vector(_int_type).reshape((header['nlevelmax'], header['nboundary']))

        f.skip(2)
        if bisection_order:
            f.skip(5)
        else:
            f.skip(1)
        f.skip(3)

        offset = np.array(header['ng'], dtype=_float_type) / 2
        offset -= 0.5

        coords = np.zeros(3, dtype=_float_type)

        for level in range(maxlevel or header['nlevelmax']):

            # loop through those CPUs with grid data (includes ghost regions)
            for cpuf in 1 + np.where(n_per_level[level, :] != 0)[0]:
                # print "CPU=",cpu,"CPU on
                # disk=",cpuf,"npl=",n_per_level[level,cpuf-1]

                if cpuf == cpu:

                    # this is the data we want
                    f.skip(3)  # grid, next, prev index

                    # store the coordinates in temporary arrays. We only want
                    # to copy it if the cell is not refined
                    coords = [
                        f.read_vector(_float_type) for ar in range(ndim)]

                    # stick on zeros if we're in less than 3D
                    coords += [np.zeros_like(coords[0]) for ar in range(3 - ndim)]

                    f.skip(1  # father index
                                 + 2 * ndim  # nbor index
                                 # son index,cpumap,refinement map
                                 + 2 * (2 ** ndim)
                                 )

                    refine = np.array(
                        [f.read_vector(_int_type) for i in range(2 ** ndim)])

                    if(level+1 == maxlevel or level+1==header['nlevelmax']):
                        refine[:] = 0

                    coords[0] -= offset[0]
                    coords[1] -= offset[1]
                    coords[2] -= offset[2]
                    # x0-=offset[0]; y0-=offset[1]; z0-=offset[2]

                    yield coords, refine, cpuf, level

                else:

                    # skip ghost regions from other CPUs
                    f.skip(3 + ndim + 1 + 2 * ndim + 3 * 2 ** ndim)

            if header['nboundary'] > 0:
                for boundaryf in np.where(n_per_level_boundary[level, :] != 0)[0]:

                    f.skip(3 + ndim + 1 + 2 * ndim + 3 * 2 ** ndim)


@remote_exec
def _cpui_count_gas_cells(level_iterator_args):
    ncell = 0
    for coords, refine, cpu, level in _cpui_level_iterator(*level_iterator_args):
        ncell += (refine == 0).sum()
    return ncell


@remote_exec
def _cpui_load_gas_pos(pos_array, smooth_array, ndim, boxlen, i0, level_iterator_args):
    dims = [pos_array[:, i] for i in range(ndim)]
    subgrid_index = np.arange(2 ** ndim)[:, np.newaxis]
    subgrid_z = np.floor((subgrid_index) / 4)
    subgrid_y = np.floor((subgrid_index - 4 * subgrid_z) / 2)
    subgrid_x = np.floor(subgrid_index - 2 * subgrid_y - 4 * subgrid_z)
    subgrid_x -= 0.5
    subgrid_y -= 0.5
    subgrid_z -= 0.5

    for (x0, y0, z0), refine, cpu, level in _cpui_level_iterator(*level_iterator_args):
        dx = boxlen * 0.5 ** (level + 1)

        x0 = boxlen * x0 + dx * subgrid_x
        y0 = boxlen * y0 + dx * subgrid_y
        z0 = boxlen * z0 + dx * subgrid_z

        mark = np.where(refine == 0)

        i1 = i0 + len(mark[0])
        for q, d in zip(dims, [x0, y0, z0][:ndim]):
            q[i0:i1] = d[mark]

        smooth_array[i0:i1] = dx
        i0 = i1

_gv_load_hydro = 0
_gv_load_gravity = 1
_gv_load_rt = 2


@remote_exec
def _cpui_load_gas_vars(dims, maxlevel, ndim, filename, cpu, lia, i1,
                        mode=_gv_load_hydro):

    logger.info("Loading data from CPU %d", cpu)

    nvar = len(dims)
    grid_info_iter = _cpui_level_iterator(*lia)

    with FortranFile(filename) as f:
        exact_nvar = False
        if mode is _gv_load_hydro:
            header = f.read_attrs(ramses_hydro_header)
            nvar_file = header['nvarh']
        elif mode is _gv_load_gravity:
            header = f.read_attrs(ramses_grav_header)
            nvar_file = 4
        elif mode is _gv_load_rt:
            header = f.read_attrs(ramses_rt_header)
            nvar_file = header['nrtvar']
            exact_nvar = True
        else:
            raise ValueError("Unknown RAMSES load mode")

        if nvar_file != nvar and exact_nvar:
            raise ValueError("Wrong number of variables in RAMSES dump")
        elif nvar_file < nvar:
            warnings.warn("Fewer hydro variables are in this RAMSES dump than are defined in config.ini (expected %d, got %d in file)" % (
                nvar, nvar_file), RuntimeWarning)
            nvar = nvar_file
            dims = dims[:nvar]
        elif nvar_file > nvar:
            warnings.warn("More hydro variables (%d) are in this RAMSES dump than are defined in config.ini (%d)" % (
                nvar_file, nvar), RuntimeWarning)

        for level in range(maxlevel or header['nlevelmax']):

            for cpuf in range(1, header['ncpu'] + 1):
                flevel = f.read_int()
                ncache = f.read_int()
                assert flevel - 1 == level

                if ncache > 0:
                    if cpuf == cpu:

                        coords, refine, gi_cpu, gi_level = next(grid_info_iter)
                        mark = np.where(refine == 0)

                        assert gi_level == level
                        assert gi_cpu == cpu

                    if cpuf == cpu and len(mark[0]) > 0:
                        for icel in range(2 ** ndim):
                            i0 = i1
                            i1 = i0 + (refine[icel] == 0).sum()
                            for ar in dims:
                                ar[i0:i1] = f.read_vector(
                                    _float_type)[(refine[icel] == 0)]

                            f.skip(nvar_file - nvar)

                    else:
                        f.skip((2 ** ndim) * nvar_file)

            for boundary in range(header['nboundary']):
                flevel = f.read_int()
                ncache = f.read_int()
                if ncache > 0:
                    f.skip((2 ** ndim) * nvar_file)


ramses_particle_header = (
    ('ncpu', 1, 'i'),
    ('ndim', 1, 'i'),
    ('npart', 1, 'i'),
    ('randseed', -1, 'i'),
    ('nstar', 1, 'i'),
    ('mstar', 1, 'd'),
    ('mstar_lost', 1, 'd'),
    ('nsink', 1, 'i')
)

ramses_amr_header = (
    ('ncpu', 1, 'i'),
    ('ndim', 1, 'i'),
    ('ng', 3, 'i'),
    ('nlevelmax', 1, 'i'),
    ('ngridmax', 1, 'i'),
    ('nboundary', 1, 'i'),
    ('ngrid', 1, 'i'),
    ('boxlen', 1, 'd')
)

ramses_hydro_header = (
    ('ncpu', 1, 'i'),
    ('nvarh', 1, 'i'),
    ('ndim', 1, 'i'),
    ('nlevelmax', 1, 'i'),
    ('nboundary', 1, 'i'),
    ('gamma', 1, 'd')
)

ramses_grav_header = (
    ('ncpu', 1, 'i'),
    ('ndim', 1, 'i'),
    ('nlevelmax', 1, 'i'),
    ('nboundary', 1, 'i')
)

ramses_rt_header = (
    ('ncpu', 1, 'i'),
    ('nrtvar', 1, 'i'),
    ('ndim', 1, 'i'),
    ('nlevelmax', 1, 'i'),
    ('nboundary', 1, 'i'),
    ('gamma', 1, 'd')
)

TYPE_MAP = {'i4': 'i',
            'i8': 'l',
            'f4': 'f',
            'f8': 'd'}
particle_blocks = list(map(
    str.strip, config_parser.get('ramses', "particle-blocks").split(",")))
particle_format = [TYPE_MAP[str.strip(e)] for e in config_parser.get('ramses', "particle-format").split(",")]

hydro_blocks = [_.strip() for _ in config_parser.get('ramses', "hydro-blocks").split(",")]
grav_blocks = [_.strip() for _ in config_parser.get('ramses', "gravity-blocks").split(",")]
rt_blocks = [_.strip() for _ in config_parser.get('ramses', 'rt-blocks', raw=True).split(",")]

particle_distinguisher = [_.strip() for _ in config_parser.get('ramses', 'particle-distinguisher').split(",")]
positive_typemap = [family.get_family(str.strip(x)) for x in config_parser.get('ramses', 'type-mapping-positive').split(",")]

negative_typemap = [family.get_family(str.strip(x)) for x in config_parser.get('ramses', 'type-mapping-negative').split(",")]

def read_descriptor(fname):
    description = []
    name_mapping = namemapper.AdaptiveNameMapper('ramses-name-mapping')
    with open(fname) as fd:
        if fd.readline() != "# version:  1\n":
            raise OSError("Wrong file format")
        fd.readline()  # ivar, variable_name, variable_type
        for line in fd.readlines():
            i, name, dtype = (_.strip() for _ in line.split(","))

            description.append(name_mapping(name, reverse=True))
    return description


class RamsesSnap(SimSnap):
    reader_pool = None

    def __init__(self, dirname, **kwargs):
        """Initialize a RamsesSnap. Extra kwargs supported:

         *cpus* : a list of the CPU IDs to load. If not set, load all CPU's data.
         *maxlevel* : the maximum refinement level to load. If not set, the deepest level is loaded.
         *with_gas* : if False, never load any gas cells (particles only) - default is True
         *force_gas* : if True, load the AMR cells as "gas particles" even if they don't actually contain gas in the run
         """

        global config
        super().__init__()

        self.__setup_parallel_reading()

        if isinstance(dirname, Path):
            dirname = str(dirname)

        self._timestep_id = _timestep_id(dirname)
        self._filename = dirname
        self._load_sink_data_to_temporary_store()
        self._load_infofile()
        self._load_namelistfile()
        self._setup_particle_descriptor()

        assert self._info['ndim'] <= 3
        if self._info['ndim'] < 3:
            warnings.warn(
                "Snapshots with less than three dimensions are supported only experimentally", RuntimeWarning)

        self._ndim = self._info['ndim']
        self.ncpu = self._info['ncpu']
        if 'cpus' in kwargs:
            self._cpus = kwargs['cpus']
        else:
            self._cpus = list(range(1, self.ncpu + 1))
        self._maxlevel = kwargs.get('maxlevel', None)

        type_map = self._count_particles()

        has_gas = os.path.exists(
            self._hydro_filename(1)) or kwargs.get('force_gas', False)

        if not kwargs.get('with_gas',True):
            has_gas = False

        ngas = self._count_gas_cells() if has_gas else 0

        if ngas>0:
            type_map[family.gas] = ngas

        count = 0
        for fam in type_map:
            self._family_slice[fam] = slice(count, count+type_map[fam])
            count+=type_map[fam]

        self._num_particles = count
        self._load_fluid_descriptors()
        self._load_rt_infofile()
        self._decorate()
        self._transfer_sink_data_to_family_array()

    def __setup_parallel_reading(self):
        if multiprocess:
            self._shared_arrays = True
            if (RamsesSnap.reader_pool is None):
                RamsesSnap.reader_pool = multiprocessing.Pool(multiprocess_num)
        elif issue_multiprocess_warning:
            warnings.warn(
                "RamsesSnap is configured to use multiple processes, but the posix_ipc module is missing. Reverting to single thread.",
                RuntimeWarning)

    def _load_fluid_descriptors(self):
        types = ["hydro", "grav"]
        default_blocks = [hydro_blocks, grav_blocks]
        descriptors_fnames = [os.path.join(self._filename, f"{ftype}_file_descriptor.txt") for ftype in types]

        for desc_type, default_block, descriptor_fname in zip(types, default_blocks, descriptors_fnames):
            try:
                block = read_descriptor(descriptor_fname)
            except (FileNotFoundError, OSError):
                block = default_block

            setattr(self, f"_{desc_type}_blocks", block)

    def _load_rt_infofile(self):
        self._rt_blocks = []
        self._rt_blocks_3d = set()
        try:
            f = open(os.path.join(self._filename, f"info_rt_{self._timestep_id}.txt"))
        except OSError:
            return

        self._load_info_from_specified_file(f)

        for group in range(self._info['nGroups']):
            for block in rt_blocks:
                self._rt_blocks.append(block%group)

        self._rt_unit = self._info['unit_pf']*units.Unit("cm^-2 s^-1")

        for block in self._rt_blocks:
            self._rt_blocks_3d.add(self._array_name_1D_to_ND(block) or block)

    def _load_info_from_specified_file(self, f):
        for l in f:
            if '=' in l:
                name, val = list(map(str.strip, l.split('=')))
                try:
                    if '.' in val:
                        self._info[name] = float(val)
                    else:
                        self._info[name] = int(val)
                except ValueError:
                    self._info[name] = val

    def _load_infofile(self):
        self._info = {}
        f = open(os.path.join(self._filename, f"info_{self._timestep_id}.txt"))

        self._load_info_from_specified_file(f)
        try:
            f = open(os.path.join(self._filename, f"header_{self._timestep_id}.txt"))
            # most of this file is unhelpful, but depending on the ramses
            # version, there may be information on the particle fields present
            for l in f:
                if "level" in l:
                    self._info['particle-blocks'] = l.split()
        except OSError:
            warnings.warn(
                "No header file found -- no particle block information available")

    def _load_namelist_from_specified_file(self, f):
        for l in f:
            l = l.split("!")[0]  # remove fortran comments
            if '=' in l:
                name, val = map(str.strip, l.split('='))
                if val == ".true.":
                    self._namelist[name] = True
                elif val == ".false.":
                    self._namelist[name] = False
                else:
                    try:
                        if '.' in val:
                            self._namelist[name] = float(val)
                        else:
                            self._namelist[name] = int(val)
                    except ValueError:
                        self._namelist[name] = val

    def _load_namelistfile(self):
        self._namelist = {}

        namelist_file = os.path.join(self._filename, "namelist.txt")

        if os.path.exists(namelist_file):
            f = open(namelist_file)
            try:
                self._load_namelist_from_specified_file(f)
            except ValueError:
                warnings.warn("Namelist found but unable to read.")

        else:
            warnings.warn("No namelist file found.")

    def _setup_particle_descriptor(self):
        try:
            self._load_particle_descriptor()
        except OSError:
            self._guess_particle_descriptor()
        self._has_explicit_particle_families = 'family' in self._particle_blocks

    def _load_particle_descriptor(self):
        with open(os.path.join(self._filename, "part_file_descriptor.txt")) as f:
            self._particle_blocks = []
            self._particle_types = []
            self._translate_array_name = namemapper.AdaptiveNameMapper('ramses-name-mapping')
            for l in f:
                if not l.startswith("#"):
                    ivar, name, dtype = list(map(str.strip,l.split(",")))
                    self._particle_blocks.append(self._translate_array_name(name, reverse=True))
                    self._particle_types.append(dtype)
            self._particle_blocks_are_explictly_known = True


    def _guess_particle_descriptor(self):
        # determine whether we have explicit information about
        # what particle blocks are present
        self._particle_blocks_are_explictly_known = False

        self._particle_types = particle_format
        if 'particle-blocks' in self._info:
            self._particle_blocks_are_explictly_known = True
            self._particle_blocks = self._info['particle-blocks']
            self._particle_blocks = ['x', 'y', 'z', 'vx', 'vy', 'vz'] + self._particle_blocks[2:]
        else:
            self._particle_blocks = particle_blocks

        if len(self._particle_types) < len(self._particle_blocks):
            warnings.warn("Some fields do not have format configured - assuming they are doubles", RuntimeWarning)
            type_ = 'd'
            self._particle_types += [type_] * (len(self._particle_blocks) - len(self._particle_types))

    def _particle_filename(self, cpu_id):
        return os.path.join(self._filename, f"part_{self._timestep_id}.out{_cpu_id(cpu_id)}")

    def _amr_filename(self, cpu_id):
        return os.path.join(self._filename, f"amr_{self._timestep_id}.out{_cpu_id(cpu_id)}")

    def _hydro_filename(self, cpu_id):
        return os.path.join(self._filename, f"hydro_{self._timestep_id}.out{_cpu_id(cpu_id)}")

    def _grav_filename(self, cpu_id):
        return os.path.join(self._filename, f"grav_{self._timestep_id}.out{_cpu_id(cpu_id)}")

    def _rt_filename(self, cpu_id):
        return os.path.join(self._filename, f"rt_{self._timestep_id}.out{_cpu_id(cpu_id)}")

    def _sink_filename(self):
        return os.path.join(self._filename, f"sink_{self._timestep_id}.csv")

    def _count_particles(self):
        if self._has_explicit_particle_families:
            return self._count_particles_using_explicit_families()
        else:
            ndm, nstar = self._count_particles_using_implicit_families()
            return OrderedDict([(family.dm, ndm), (family.star, nstar)])

    def _has_particle_file(self):
        """Check whether the output has a particle file available"""
        if len(self._cpus)>0 :
            return os.path.exists(self._particle_filename(self._cpus[0]))
        else:
            return False

    def _count_particles_using_explicit_families(self):
        """Returns an ordered dictionary of family types based on the new explicit RAMSES particle file format"""
        if not self._has_particle_file():
            return OrderedDict()
        family_block = self._particle_blocks.index('family')
        family_dtype = self._particle_types[family_block]
        self._particle_family_ids_on_disk = []
        self._particle_file_start_indices = []
        results = remote_map(self.reader_pool,
                             _cpui_count_particles_with_explicit_families,
                             [self._particle_filename(i) for i in self._cpus],
                             [family_block] * len(self._cpus),
                             [family_dtype] * len(self._cpus))

        aggregate_counts = np.zeros(256,dtype=np.int64)
        for counts, family_ids in results:
            aggregate_counts+=counts
            self._particle_family_ids_on_disk.append(family_ids)

        # The above family IDs are defined according to ramses' own internal system. We now need
        # to map them to pynbody family types

        nonzero_families = np.nonzero(aggregate_counts)[0]

        # map such that negative integers are correctly represented
        ramses_id_to_internal_id = np.zeros(256, dtype=np.uint8)
        internal_id_to_family = []
        aggregate_counts_remapped = np.zeros(256, dtype=np.int64)

        for ramses_family_id in nonzero_families:
            if ramses_family_id>128 or ramses_family_id == 0:
                neg_offset = (256 - ramses_family_id) % 256
                if neg_offset>len(negative_typemap):
                    pynbody_family = negative_typemap[-1]
                else:
                    pynbody_family = negative_typemap[neg_offset]
            else:
                if ramses_family_id>len(positive_typemap):
                    pynbody_family = positive_typemap[-1]
                else:
                    pynbody_family = positive_typemap[ramses_family_id-1]
            if pynbody_family in internal_id_to_family:
                internal_id = internal_id_to_family.index(pynbody_family)
            else:
                internal_id = len(internal_id_to_family)
                internal_id_to_family.append(pynbody_family)
            ramses_id_to_internal_id[ramses_family_id] = internal_id
            aggregate_counts_remapped[internal_id]+=aggregate_counts[ramses_family_id]

        # perform the remapping for our stored particle identifiers
        for fid in self._particle_family_ids_on_disk:
            fid[:] = ramses_id_to_internal_id[fid]

        return_d = OrderedDict()
        self._particle_file_start_indices = [ [] for x in results]
        for internal_family_id in range(256):
            if aggregate_counts_remapped[internal_family_id]>0:
                fam = internal_id_to_family[internal_family_id]
                count = aggregate_counts_remapped[internal_family_id]
                return_d[fam] = count
                startpoint = 0
                for i,fid in enumerate(self._particle_family_ids_on_disk):
                    self._particle_file_start_indices[i].append(startpoint)
                    startpoint+=(fid==internal_family_id).sum()
                assert startpoint==count

        n_sinks = self._count_sink_particles()

        if n_sinks>0:
            return_d[self._sink_family] = n_sinks

        return return_d


    def _load_sink_data_to_temporary_store(self):
        if not os.path.exists(self._sink_filename()):
            self._after_load_sink_data_failure(warn=False)
            return

        self._sink_family = family.get_family(config_parser.get('ramses', 'type-sink'))

        with open(self._sink_filename()) as sink_file:
            reader = csv.reader(sink_file, skipinitialspace=True)
            data = list(reader)

        if len(data)<2:
            self._after_load_sink_data_failure()
            return

        column_names = data[0]
        dimensions = data[1]
        data = np.array(data[2:], dtype=object)

        if column_names[0][0]!='#' or dimensions[0][0]!='#':
            self._after_load_sink_data_failure()
            return

        self._fix_fortran_missing_exponent(data)

        column_names[0] = column_names[0][1:].strip()
        dimensions[0] = dimensions[0][1:].strip()

        self._sink_column_names = column_names
        self._sink_dimensions = dimensions
        self._sink_data = data

    def _after_load_sink_data_failure(self, warn=True):
        if warn:
            warnings.warn("Unexpected format in file %s -- sink data has not been loaded" % self._sink_filename())

        self._sink_column_names = self._sink_dimensions = self._sink_data = []
        self._sink_family = None


    @staticmethod
    def _fix_fortran_missing_exponent(data_array):
        flattened_data = data_array.flat
        for i in range(len(flattened_data)):
            d = flattened_data[i]
            if "-" in d and "E" not in d:
                flattened_data[i] = "E-".join(d.split("-"))


    def _transfer_sink_data_to_family_array(self):
        if len(self._sink_data)==0:
            return

        target = self[self._sink_family]
        for column, dimension, data in zip(self._sink_column_names,
                                           self._sink_dimensions,
                                           self._sink_data.T):
            dtype = np.float64
            if column=="id":
                dtype = np.int64
            target[column] = data.astype(dtype)
            unit = dimension.replace("m","g").replace("l","cm").replace("t","yr")
            if unit=="1":
                target[column].units="1"
            else:
                target[column].set_units_like(unit)



    def _count_sink_particles(self):
        return len(self._sink_data)



    def _count_particles_using_implicit_families(self):
        """Returns ndm, nstar where ndm is the number of dark matter particles
        and nstar is the number of star particles."""

        npart = 0
        nstar = 0

        dm_i0 = 0
        star_i0 = 0

        self._particle_family_ids_on_disk = []
        self._particle_file_start_indices = []

        if not os.path.exists(self._particle_filename(1)):
            return 0, 0

        if not self._particle_blocks_are_explictly_known:
            distinguisher_field = int(particle_distinguisher[0])
            distinguisher_type = TYPE_MAP[particle_distinguisher[1]]
        else:
            # be more cunning about finding the distinguisher field (likely 'age') -
            # as it may have moved around in some patches of ramses

            distinguisher_name = particle_blocks[int(particle_distinguisher[0])]
            try:
                distinguisher_field = self._particle_blocks.index(distinguisher_name)
            except ValueError:
                # couldn't find the named distinguisher field. Fall back to using index.
                distinguisher_field = int(particle_distinguisher[0])
                if len(self._particle_blocks)>distinguisher_field:
                    pb_name = "%r"%self._particle_blocks[distinguisher_field]
                else:
                    pb_name = "at offset %d"%distinguisher_field
                warnings.warn("Using field %s to distinguish stars. If this is wrong, try editing your config.ini, section [ramses], entry particle-distinguisher."%pb_name)
            distinguisher_type = self._particle_types[distinguisher_field]

        results = remote_map(self.reader_pool,
                             _cpui_count_particles_with_implicit_families,
                             [self._particle_filename(i) for i in self._cpus],
                             [distinguisher_field] * len(self._cpus),
                             [distinguisher_type] * len(self._cpus))

        for npart_this, nstar_this, family_ids in results:
            self._particle_file_start_indices.append((dm_i0, star_i0))
            dm_i0 += (npart_this - nstar_this)
            star_i0 += nstar_this
            npart += npart_this
            nstar += nstar_this

            self._particle_family_ids_on_disk.append(family_ids)

        return npart - nstar, nstar

    def _count_gas_cells(self):
        ncells = remote_map(self.reader_pool, _cpui_count_gas_cells,
                            [self._cpui_level_iterator_args(xcpu) for xcpu in self._cpus])
        self._gas_i0 = np.cumsum([0] + ncells)[:-1]
        return np.sum(ncells)

    def _cpui_level_iterator_args(self, cpu=None):
        if cpu:
            return cpu, self._amr_filename(cpu), self._info['ordering type'] == 'bisection', self._maxlevel, self._ndim
        else:
            return [self._cpui_level_iterator_args(x) for x in self._cpus]

    def _level_iterator(self):
        """Walks the AMR grid levels on disk, yielding a tuplet of coordinates and
        refinement maps and levels working through the available CPUs and levels."""

        for cpu in self._cpus:
            yield from _cpui_level_iterator(*self._cpui_level_iterator_args(cpu))

    def _load_gas_pos(self):
        i0 = 0
        self.gas['pos'].set_default_units()
        smooth = self.gas['smooth']
        smooth.set_default_units()

        boxlen = self._info['boxlen']

        remote_map(self.reader_pool,
                   _cpui_load_gas_pos,
                   [self.gas['pos']] * len(self._cpus),
                   [self.gas['smooth']] * len(self._cpus),
                   [self._ndim] * len(self._cpus),
                   [boxlen] * len(self._cpus),
                   self._gas_i0,
                   self._cpui_level_iterator_args())

    def _load_gas_vars(self, mode=_gv_load_hydro):
        i1 = 0

        dims = []

        for i in [self._hydro_blocks, self._grav_blocks, self._rt_blocks][mode]:
            if i not in self.gas:
                self.gas._create_array(i)
            if self._ndim < 3 and i[-1] == 'z':
                continue
            if self._ndim < 2 and i[-1] == 'y':
                continue
            dims.append(self.gas[i])
            self.gas[i].set_default_units()


        if not os.path.exists(self._hydro_filename(1)):
            #Case where force_gas = True, make sure rho is non-zero and such that mass=1.
            # This does not keep track of units for mass or rho since their value is enforced.
            logger.info("No hydro file found, gas likely from force_gas=True => hard setting rho gas")
            self.gas['rho'][:] = 1.0
            self.gas['rho'] /= (np.array(self.gas['smooth']) ** 3)
            self.gas['rho'].set_default_units()



        nvar = len(dims)

        grid_info_iter = self._level_iterator()

        logger.info("Loading %s files", ['hydro', 'grav', 'rt'][mode])

        filenamer = [self._hydro_filename, self._grav_filename, self._rt_filename][mode]

        remote_map(self.reader_pool,
                   _cpui_load_gas_vars,
                   [dims] * len(self._cpus),
                   [self._maxlevel] * len(self._cpus),
                   [self._ndim] * len(self._cpus),
                   [filenamer(i) for i in self._cpus],
                   self._cpus,
                   self._cpui_level_iterator_args(),
                   self._gas_i0,
                   [mode] * len(self._cpus))

        if mode is _gv_load_gravity:
            # potential is awkwardly in expected units divided by box size
            self.gas['phi'] *= self._info['boxlen']

        logger.info("Done")

    def _iter_particle_families(self):
        for f in self.families():
            if f is not family.gas and f is not self._sink_family:
                yield f

    def _create_array_for_particles(self, name, type_):
        for f in self._iter_particle_families():
            if name not in list(self[f].keys()):
                for f in self._iter_particle_families():
                    self[f]._create_array(name, dtype=type_)


    def _load_particle_block(self, blockname):
        offset = self._particle_blocks.index(blockname)
        _type = self._particle_types[offset]

        self._create_array_for_particles(blockname, _type)

        arrays = []
        for f in self._iter_particle_families():
            arrays.append(self[f][blockname])


        try:
            remote_map(self.reader_pool,
                       _cpui_load_particle_block,
                       [self._particle_filename(i) for i in self._cpus],
                       [arrays] * len(self._cpus),
                       [offset] * len(self._cpus),
                       self._particle_file_start_indices,
                       [_type] * len(self._cpus),
                       self._particle_family_ids_on_disk
                       )
        except:
            warnings.warn("Exception encountered while reading %r; is there an incompatibility in your Ramses configuration?"%blockname)
            del self[blockname]
            raise

        # The potential is awkwardly not in physical units, but in
        # physical units divided by the box size. This was different
        # in an intermediate version, but then later made consistent
        # with the AMR phi output. So, we need to make a correction here
        # IF we are dealing with the latest ramses format.

        if self._particle_blocks_are_explictly_known and blockname == 'phi':
            for f in self._iter_particle_families():
                self[f]['phi'] *= self._info['boxlen']

    def _load_particle_cpuid(self):
        ind0_dm = 0
        ind0_star = 0
        for i, fam_mask in zip(self._cpus, self._particle_family_ids_on_disk):
            mask_dm = fam_mask == 0
            mask_st = fam_mask == 1

            ind1_dm = ind0_dm + mask_dm.sum()
            ind1_star = ind0_star + mask_st.sum()

            self.dm['cpu'][ind0_dm:ind1_dm] = i
            self.star['cpu'][ind0_star:ind1_star] = i
            ind0_dm, ind0_star = ind1_dm, ind1_star

    def _load_gas_cpuid(self):
        gas_cpu_ar = self.gas['cpu']
        i1 = 0
        for coords, refine, cpu, level in self._level_iterator():
            for cell in range(2 ** self._ndim):
                i0 = i1
                i1 = i0 + (refine[cell] == 0).sum()
                gas_cpu_ar[i0:i1] = cpu

    def loadable_keys(self, fam=None):

        if fam is None:
            keys = None
            for f0 in self.families():
                if keys:
                    keys.intersection_update(self.loadable_keys(f0))
                else:
                    keys = set(self.loadable_keys(f0))
        else:
            if fam is family.gas:
                keys = ['x', 'y', 'z', 'smooth'] + self._hydro_blocks + self._rt_blocks
            elif fam in self._iter_particle_families():
                keys = self._particle_blocks
            elif fam is self._sink_family:
                keys = set(self._sink_column_names)
            else:
                keys = set()

        keys_ND = set()
        for key in keys:
            keys_ND.add(self._array_name_1D_to_ND(key) or key)
        return list(keys_ND)

    def _not_cosmological(self):
        not_cosmological = True

        if "cosmo" in self._namelist and self._namelist["cosmo"]:
            not_cosmological = False

        if not self._namelist:
            warnings.warn("Namelist file either not found or unable to read. " +
                          "Guessing whether run is cosmological from cosmological parameters assuming flat LCDM.")
            not_cosmological = (self._info['omega_k'] == self._info['omega_l'] == 0)

        return not_cosmological

    def _convert_tform(self):
        # Copy the existing t array in weird Ramses format into a hidden raw array
        self.star['tform_raw'] = self.star['tform']
        self.star['tform_raw'].units = self._file_units_system[1]

        if self._is_using_proper_time:
            t0 = analysis.cosmology.age(self, z=0.0, unit="Gyr")
            birth_time = t0 + self.s["tform_raw"].in_units("Gyr") / self.properties["a"] ** 2
            birth_time[birth_time > t0] = t0 - 1e-7
            self.star['tform'] = birth_time
        elif not self._not_cosmological():
            # Only attempt tform conversion for cosmological runs. The built-in tforms for isolated runs
            # are actually meaningful (issue 554)
            from ..analysis import ramses_util

            # Replace the tform array by its usual meaning using the birth files
            ramses_util.get_tform(self)

    def _read_proper_time(self):
        try:
            self._is_using_proper_time = config_parser.getboolean("ramses", "proper_time")
        except:
            self._is_using_proper_time = False

    def _convert_metal_name(self):
        # Name of ramses metallicity has no 's' at the end, contrary to tipsy and gadget
        # Correcting this prevents analysis routines relying on 'metals' field from breaking.
        self.star['metals'] = self.star['metal']

    def _load_array(self, array_name, fam=None):
        # Framework always calls with 3D name. Ramses particle blocks are
        # stored as 1D slices.

        if array_name == 'cpu':
            self['cpu'] = np.zeros(len(self), dtype=int)
            self._load_particle_cpuid()
            self._load_gas_cpuid()

        elif fam is not family.gas and fam is not None:

            # Deal with tform for stars that require extra conversion
            if array_name == 'tform' or array_name == 'tform_raw':
                if 'tform' in self._particle_blocks: # Only attempt these conversion if tform is actually on disc (Issue #689)
                    self._load_particle_block('tform')
                    self._read_proper_time()
                    self._convert_tform()

            # Deal with metals for stars that have extra name mapping
            elif array_name == 'metals' or array_name == 'metal':
                if 'metal' in self._particle_blocks:
                    self._load_particle_block('metal')
                    self._convert_metal_name()

            elif array_name in self._split_arrays:
                for array_1D in self._array_name_ND_to_1D(array_name):
                    self._load_particle_block(array_1D)

            elif array_name in self._particle_blocks:
                self._load_particle_block(array_name)
            else:
                raise OSError("No such array on disk")

        elif fam is family.gas:

            if array_name == 'pos' or array_name == 'smooth':
                if 'pos' not in self.gas:
                    self.gas._create_array('pos', 3)
                if 'smooth' not in self.gas:
                    self.gas._create_array('smooth')
                self._load_gas_pos()
            elif array_name == 'vel' or array_name in self._hydro_blocks:
                self._load_gas_vars()
            elif array_name in self._grav_blocks:
                self._load_gas_vars(1)
            elif array_name in self._rt_blocks_3d:
                warnings.warn("Loading RT data from disk. Photon densities are stored in flux units by Ramses and need "
                              "to be multiplied by the reduced speed of light of the run to obtain a physical number. "
                              "This is currently left to the user, see issue 542 for more discussion.")
                self._load_gas_vars(_gv_load_rt)
                for u_block in self._rt_blocks_3d:
                    self[fam][u_block].units = self._rt_unit
            else:
                raise OSError("No such array on disk")
        elif fam is None and array_name in ['pos', 'vel']:
            # synchronized loading of pos/vel information
            if 'pos' not in self:
                self._create_array('pos', 3)
            if 'vel' not in self:
                self._create_array('vel', 3)
            if 'smooth' not in self.gas:
                self.gas._create_array('smooth')

            if len(self.gas) > 0:
                self._load_gas_pos()
                self._load_gas_vars()

            # the below triggers loading ALL particles, not just DM
            for name in 'x','y','z','vx','vy','vz':
                self._load_particle_block(name)

        elif fam is None and array_name == 'mass':
            self._create_array('mass')
            self._load_particle_block('mass')
            self['mass'].set_default_units()
            if len(self.gas) > 0:
                gasmass = mass(self.gas)
                gasmass.convert_units(self['mass'].units)
                self.gas['mass'] = gasmass
        else:
            raise OSError("No such array on disk")



    @staticmethod
    def _can_load(f):
        tsid = _timestep_id(f)
        if tsid:
            return os.path.isdir(f) and os.path.exists(os.path.join(f, f"info_{tsid}.txt"))
        return False


@RamsesSnap.decorator
def translate_info(sim):

    if sim._info['H0']>1e-3:
        sim.properties['a'] = sim._info['aexp']
        sim.properties['omegaM0'] = sim._info['omega_m']
        sim.properties['omegaL0'] = sim._info['omega_l']
        sim.properties['h'] = sim._info['H0'] / 100.

    # N.B. these conversion factors provided by ramses already have the
    # correction from comoving to physical units
    d_unit = sim._info['unit_d'] * units.Unit("g cm^-3")
    t_unit = sim._info['unit_t'] * units.Unit("s")
    l_unit = sim._info['unit_l'] * units.Unit("cm")

    sim.properties['boxsize'] = sim._info['boxlen'] * l_unit

    if sim._not_cosmological():
        sim.properties['time'] = sim._info['time'] * t_unit
    else:
        sim.properties['time'] = analysis.cosmology.age(
            sim) * units.Unit('Gyr')

    sim._file_units_system = [d_unit, t_unit, l_unit]


@RamsesSnap.derived_quantity
def mass(sim):
    return sim['rho'] * sim['smooth'] ** 3


@RamsesSnap.derived_quantity
def temp(sim):
    """ Gas temperature derived from pressure and density """
    # Has to be redefined and rederived here from Ramses native variables
    # to avoid running into circular dependencies with the traditional derived definition
    # Now uses the self-consistent molecular weight field from pynbody (issue 598)
    from ..derived import mu
    mu_est = array.SimArray(np.ones(len(sim)), units="1")
    for i in range(5):
        temp = ((sim['p'] / sim['rho']) * (mu_est * units.m_p / units.k)).in_units("K")
        mu_est = mu(sim, t0=temp)
    return temp
