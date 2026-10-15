"""
Implements classes and functions for handling RAMSES files.

Note that AMR cells are loaded as particles, although this works surprisingly well for most analysis purposes.
In particular SPH image generation routines automatically switch on an appropriate de-noising scheme to provide
effective interpolation between cells.

"""

import copy
import csv
import logging
import os
import re
import warnings
from pathlib import Path

import numpy as np

from ... import array, config_parser, family, filt, units
from ...analysis.cosmology import age
from .. import SimSnap, namemapper
from . import _execute, _hilbert, _layout, _plan
from ._selection import (
    SKIP,
    CellSelection,
    ParticleSelection,
    depends_only_on_position,
    in_region,
)

logger = logging.getLogger('pynbody.snapshot.ramses')

_float_type = 'd'
_int_type = 'i'


def _timestep_id(basename):
    try:
        return re.findall("output_([0-9]*)/*$", str(basename))[0]
    except IndexError:
        return None


def _cpu_id(i):
    return str(i).rjust(5, "0")


_gv_load_hydro = 0
_gv_load_gravity = 1
_gv_load_rt = 2


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
            'i8': 'q',
            'f4': 'f',
            'f8': 'd'}
particle_blocks = list(map(
    str.strip, config_parser.get('ramses', "particle-blocks").split(",")))
particle_format = [TYPE_MAP[str.strip(e)] for e in config_parser.get('ramses', "particle-format").split(",")]

hydro_blocks = [_.strip() for _ in config_parser.get('ramses', "hydro-blocks").split(",")]
grav_blocks = [_.strip() for _ in config_parser.get('ramses', "gravity-blocks").split(",")]
rt_blocks = [_.strip() for _ in config_parser.get('ramses', 'rt-blocks', raw=True).split(",")]

particle_distinguisher = [_.strip() for _ in config_parser.get('ramses', 'particle-distinguisher').split(",")]
positive_typemap = [family.get_family(str.strip(x)) for x in
                    config_parser.get('ramses', 'type-mapping-positive').split(",")]

negative_typemap = [family.get_family(str.strip(x)) for x in
                    config_parser.get('ramses', 'type-mapping-negative').split(",")]


def _read_descriptor(fname):
    """Read a RAMSES file descriptor and return a list of the variable names."""
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
    """Implements loading of Ramses snapshots.

    Note that AMR cells are loaded as particles, although this works surprisingly well for most analysis purposes.
    In particular SPH image generation routines automatically switch on an appropriate de-noising scheme to provide
    effective interpolation between cells.
    """

    _particle_families_by_id = ()  # (the family of each internal particle family id; none if there are no particles)
    def __init__(self, dirname, cpus=None, maxlevel=None, with_gas=True, force_gas=False, times_are_proper=None,
                 negative_iords_on_purpose=False, take_region=None, trim_to_region=True):
        """
        Initialize a RamsesSnap.

        Parameters
        ----------
        dirname : str
            Directory containing the RAMSES output
        cpus : list of int, optional
            A list of the CPU IDs to load. If not set, load all
            CPU's data.
        maxlevel : int, optional
            The maximum refinement level to load. If not set, the
            deepest level is loaded.
        with_gas : bool, optional
            If False, never load any gas cells (particles only).
            Default is True
        force_gas : bool, optional
            If True, load the AMR cells as "gas particles" even if
            they don't actually contain gas in the run. Default is
            False.
        times_are_proper : bool, optional
            If True, the times in the output are assumed to be proper
            times. If False, they are assumed to be conformal. If
            None (default), assume proper for non-cosmological simulations
            and conformal for cosmological ones.
        take_region : pynbody.filt.Filter, optional
            If given, load only the cells (by their centres) and particles
            (by their positions) that the filter selects, reading only the
            CPU files whose domains hold (or come near) its region, and
            within them only what lies in it (but see ``trim_to_region``).
            Combined with ``cpus``, only CPUs in both are read. Choosing
            CPU files needs a 3D output with Hilbert ordering of the domains
            (the default in RAMSES), and a Sphere or Cuboid, or a combination
            of them with ``&`` and ``|`` (sizes given with units are
            converted to code units; centres and corners given as numbers
            are in code units); with other filters, every file is read, with
            a warning.
        trim_to_region : bool, optional
            If True (default), load exactly what the take_region filter
            selects, which means reading the positions of every cell and
            particle in the CPU files chosen when the snapshot is opened. This
            needs a filter that selects by position alone (a Sphere, Cuboid,
            Disc, or BandPass, HighPass or LowPass on ``x``, ``y``, ``z``,
            ``r`` or ``rxy``, and combinations of them with ``&``, ``|`` and
            ``~``); with any other filter, or if False, every cell and
            particle in those files is loaded instead (a superset of the
            region, read only as arrays are needed).
        """

        global config
        super().__init__()

        if isinstance(dirname, Path):
            dirname = str(dirname)

        self._timestep_id = _timestep_id(dirname)
        self._filename = dirname
        if os.path.isdir(dirname):
            self._dirname = dirname
        else:
            self._dirname = os.path.split(dirname)[0]
        self._translate_array_name = namemapper.AdaptiveNameMapper('ramses-name-mapping')

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
        if cpus is not None:
            self._cpus = cpus
        else:
            self._cpus = list(range(1, self.ncpu + 1))
        self._maxlevel = maxlevel
        self._take_region = take_region
        if take_region is not None:
            in_region = set(self._cpus_in_region(take_region))
            self._cpus = [cpu for cpu in self._cpus if cpu in in_region]
            if not self._cpus:
                warnings.warn("take_region selects no CPU files, so the snapshot is empty")

        # ramses partial-loads by selecting whole cpu files; anything less than all of them means the
        # snapshot does not contain all the particles in the output
        self.partial_load = sorted(self._cpus) != list(range(1, self.ncpu + 1))
        self._trim_to_region = take_region is not None and trim_to_region
        if self._trim_to_region and not depends_only_on_position(take_region):
            warnings.warn(f"take_region keeps every cell and particle of the files it reads, as it can trim them only "
                          f"with a filter that selects by position alone, not {take_region!r}")
            self._trim_to_region = False
        if self._trim_to_region:
            self.partial_load = True
            self._trim_sinks()
        self._read_plans = _plan.PlanCache()  # plans of reads (see _plan), kept for loading arrays again
        self._nvar_warned = set()  # the kinds of hydro-like file whose number of variables has been warned about
        self._rt_units_warned = False

        type_map = self._count_particles()

        gas_on_disk = os.path.exists(
            self._hydro_filename(1)
        ) or force_gas

        has_gas = gas_on_disk and with_gas

        # Dropping the gas is also a partial load: the remaining particles no longer line up with the
        # positions a halo finder assigned within the full output. with_gas=False omits the cells entirely,
        # while maxlevel stops the AMR grid being descended to the finest ones. Neither matters if the
        # output has no gas in the first place.
        if gas_on_disk and (not has_gas or maxlevel is not None):
            self.partial_load = True

        if times_are_proper is not None:
            self.times_are_proper = times_are_proper

        if negative_iords_on_purpose is not None:
            self._negative_iords_on_purpose = negative_iords_on_purpose

        ngas = self._count_gas_cells() if has_gas else 0

        if ngas > 0:
            type_map[family.gas] = ngas

        count = 0
        for fam in type_map:
            self._family_slice[fam] = slice(count, count + type_map[fam])
            count += type_map[fam]

        self._num_particles = count
        self._load_fluid_descriptors()
        self._load_rt_infofile()
        self._decorate()
        self._transfer_sink_data_to_family_array()

    def _load_fluid_descriptors(self):
        types = ["hydro", "grav"]
        default_blocks = [hydro_blocks, grav_blocks]
        descriptors_fnames = [os.path.join(self._filename, f"{ftype}_file_descriptor.txt") for ftype in types]

        for desc_type, default_block, descriptor_fname in zip(types, default_blocks, descriptors_fnames):
            try:
                block = _read_descriptor(descriptor_fname)
            except (FileNotFoundError, OSError):
                block = default_block

            setattr(self, f"_{desc_type}_blocks", block)
        self._grav_blocks_3d = {self._array_name_1D_to_ND(b) for b in self._grav_blocks} - {None}

    def _load_rt_infofile(self):
        self._rt_blocks = []
        self._rt_blocks_3d = set()
        try:
            with open(os.path.join(self._filename, f"info_rt_{self._timestep_id}.txt")) as f:
                lines = f.readlines()
        except OSError:
            return

        self._load_info_from_specified_file(lines)

        for group in range(self._info['nGroups']):
            for block in rt_blocks:
                self._rt_blocks.append(block % group)

        self._rt_unit = self._info['unit_pf'] * units.Unit("cm^-2 s^-1")

        for block in self._rt_blocks:
            self._rt_blocks_3d.add(self._array_name_1D_to_ND(block) or block)

    def _load_info_from_specified_file(self, lines):
        for line in lines:
            if '=' in line:
                name, val = list(map(str.strip, line.split('=')))
                try:
                    if '.' in val:
                        self._info[name] = float(val)
                    else:
                        self._info[name] = int(val)
                except ValueError:
                    self._info[name] = val

    def _in_region(self, positions):
        """Which of *positions* (in code units) the take_region filter selects"""
        (_, _, length_unit), properties = _units_and_properties(self._info)
        return in_region(self._take_region, positions, length_unit, properties)

    def _particles_in_region(self, layouts):
        """For each particle file, which of its particles lie in the take_region filter's region (reading the
        positions of files a batch at a time, to bound the memory they take)"""
        records = [self._particle_blocks.index(c) for c in ('x', 'y', 'z')]
        dtype = np.dtype(self._particle_types[records[0]])
        keep = []
        for batch in _batches([layout.npart for layout in layouts]):
            npart = [layouts[i].npart for i in batch]
            first = np.cumsum([0] + npart)[:-1]
            selections = [ParticleSelection.all_particles(np.zeros(n, dtype=np.uint8), [f], 1)
                          for n, f in zip(npart, first)]
            plan = _plan.particle_fields([layouts[i] for i in batch], selections,
                                         [(r, dtype.itemsize) for r in records])
            positions = np.empty((sum(npart), 3), dtype=dtype)
            _plan.load_particle_fields(plan, selections, [(dtype, [positions[:, k]]) for k in range(3)])
            keep += np.split(self._in_region(positions), np.cumsum(npart)[:-1])
        return keep

    def _trim_sinks(self):
        """Keep only the sinks in the take_region filter's region"""
        if len(self._sink_data) == 0:
            return
        if not all(c in self._sink_column_names for c in ('x', 'y', 'z')):
            warnings.warn("take_region keeps every sink, as the sink file gives no positions")
            return
        columns = [self._sink_column_names.index(c) for c in ('x', 'y', 'z')]
        positions = self._sink_data[:, columns].astype(np.float64)
        self._sink_data = self._sink_data[self._in_region(positions)]

    def _region_in_code_units(self, region):
        """A copy of the filter *region* with any sizes given in units converted to code units, for spheres and
        cuboids (and combinations of them), so that the cells they intersect can be found; other filters as given"""
        (_, _, length_unit), properties = _units_and_properties(self._info)
        context = {k: properties[k] for k in ('a', 'h') if k in properties}

        def in_code_units(x):
            if units.is_unit_like(x):
                return x.in_units(length_unit, **context)
            return x

        region = copy.copy(region)
        if isinstance(region, (filt.And, filt.Or)):
            region.f1, region.f2 = self._region_in_code_units(region.f1), self._region_in_code_units(region.f2)
        elif isinstance(region, filt.Sphere):
            region.radius = in_code_units(region.radius)
        elif isinstance(region, filt.Cuboid):
            region.x1, region.y1, region.z1, region.x2, region.y2, region.z2 = (
                in_code_units(x) for x in (region.x1, region.y1, region.z1, region.x2, region.y2, region.z2))
        return region

    def _cpus_in_region(self, region):
        """The CPUs whose files hold (or may hold) cells and particles in the region of the filter *region*"""
        if self._info['ndim'] != 3:
            raise ValueError("take_region is supported only for 3D outputs")
        ncpu = self._info['ncpu']
        try:
            ordering = self._info.get('ordering type', 'hilbert').strip()
            if ordering != 'hilbert':
                raise NotImplementedError(f"its domains are ordered by {ordering!r}, not along a Hilbert curve")
            try:
                bound_keys = _hilbert.domain_keys(os.path.join(self._filename, f"info_{self._timestep_id}.txt"))
            except ValueError as e:
                raise NotImplementedError(str(e)) from e
            if (len(bound_keys) - 1) % ncpu:
                raise NotImplementedError("the info file lists %d Hilbert domains for %d CPUs"
                                          % (len(bound_keys) - 1, ncpu))
            # (cells coarser than levelmin are leaves if maxlevel stops short of it)
            coarsest = min(int(self._info['levelmin']), self._maxlevel or int(self._info['levelmin']))
            cpus = _hilbert.cpus_in_region(self._region_in_code_units(region), bound_keys, coarsest,
                                           int(self._info['levelmax']), float(self._info['boxlen']), ncpu)
        except NotImplementedError as e:
            warnings.warn(f"take_region loads every CPU file, as it cannot choose them for {region!r}: {e}")
            return list(range(1, ncpu + 1))
        logger.info("take_region loads %d of %d CPU files", len(cpus), self._info['ncpu'])
        return cpus

    def _load_infofile(self):
        self._info = {}
        info_fname = os.path.join(self._filename, f"info_{self._timestep_id}.txt")
        header_fname = os.path.join(self._filename, f"header_{self._timestep_id}.txt")
        with open(info_fname) as f:
            self._load_info_from_specified_file(f)
        try:
            # most of this file is unhelpful, but depending on the ramses
            # version, there may be information on the particle fields present
            with open(header_fname) as f:
                for line in f:
                    if "level" in line:
                        self._info['particle-blocks'] = line.split()
        except OSError:
            warnings.warn(
                "No header file found -- no particle block information available")

    def _load_namelist_from_specified_file(self, f):
        for line in f:
            line = line.split("!")[0]  # remove fortran comments
            if '=' in line:
                name, val = map(str.strip, line.split('='))
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
            with open(namelist_file) as f:
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
            for line in f:
                if not line.startswith("#"):
                    ivar, name, dtype = list(map(str.strip, line.split(",")))
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
            return {family.dm: ndm, family.star: nstar}

    def _has_particle_file(self):
        """Check whether the output has a particle file available"""
        if len(self._cpus) > 0:
            return os.path.exists(self._particle_filename(self._cpus[0]))
        else:
            return False

    def _count_particles_using_explicit_families(self):
        """Returns an ordered dictionary of family types based on the new explicit RAMSES particle file format"""
        if not self._has_particle_file():
            return {}
        family_block = self._particle_blocks.index('family')
        family_dtype = self._particle_types[family_block]
        if np.dtype(family_dtype).itemsize != 1:
            raise ValueError("The family field of RAMSES particle files is expected to hold one byte per particle")
        self._particle_family_ids_on_disk = []
        self._particle_file_start_indices = []
        results = _layout.scan_particle_files([self._particle_filename(i) for i in self._cpus],
                                              family_block, family_dtype)
        keep = self._particles_in_region([r[0] for r in results]) if self._trim_to_region else None

        aggregate_counts = np.zeros(256, dtype=np.int64)
        for i, (particle_file, family_ids) in enumerate(results):
            family_ids = family_ids.view(np.uint8)
            counts = np.bincount(family_ids if keep is None else family_ids[keep[i]], minlength=256)
            assert keep is not None or counts.sum() == particle_file.npart
            aggregate_counts += counts
            self._particle_family_ids_on_disk.append(family_ids)

        # The above family IDs are defined according to ramses' own internal system. We now need
        # to map them to pynbody family types

        nonzero_families = np.nonzero(aggregate_counts)[0]

        # map such that negative integers are correctly represented
        ramses_id_to_internal_id = np.zeros(256, dtype=np.uint8)
        internal_id_to_family = []
        aggregate_counts_remapped = np.zeros(256, dtype=np.int64)

        for ramses_family_id in nonzero_families:
            if ramses_family_id > 128 or ramses_family_id == 0:
                neg_offset = (256 - ramses_family_id) % 256
                if neg_offset > len(negative_typemap):
                    pynbody_family = negative_typemap[-1]
                else:
                    pynbody_family = negative_typemap[neg_offset]
            else:
                if ramses_family_id > len(positive_typemap):
                    pynbody_family = positive_typemap[-1]
                else:
                    pynbody_family = positive_typemap[ramses_family_id - 1]
            if pynbody_family in internal_id_to_family:
                internal_id = internal_id_to_family.index(pynbody_family)
            else:
                internal_id = len(internal_id_to_family)
                internal_id_to_family.append(pynbody_family)
            ramses_id_to_internal_id[ramses_family_id] = internal_id
            aggregate_counts_remapped[internal_id] += aggregate_counts[ramses_family_id]

        # perform the remapping for our stored particle identifiers
        self._particle_family_ids_on_disk = [ramses_id_to_internal_id[fid] for fid in self._particle_family_ids_on_disk]
        if keep is not None:
            for fid, keep_file in zip(self._particle_family_ids_on_disk, keep):
                fid[~keep_file] = SKIP

        return_d = {}
        self._particle_file_start_indices = [[] for x in results]
        for internal_family_id in range(256):
            if aggregate_counts_remapped[internal_family_id] > 0:
                fam = internal_id_to_family[internal_family_id]
                count = aggregate_counts_remapped[internal_family_id]
                return_d[fam] = count
                startpoint = 0
                for i, fid in enumerate(self._particle_family_ids_on_disk):
                    self._particle_file_start_indices[i].append(startpoint)
                    startpoint += (fid == internal_family_id).sum()
                assert startpoint == count

        self._set_particle_selections([r[0] for r in results], internal_id_to_family)

        n_sinks = self._count_sink_particles()

        if n_sinks > 0:
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

        if len(data) < 2:
            self._after_load_sink_data_failure()
            return

        column_names = data[0]
        dimensions = data[1]
        data = np.array(data[2:], dtype=object)

        if column_names[0][0] != '#' or dimensions[0][0] != '#':
            self._after_load_sink_data_failure()
            return

        self._fix_fortran_missing_exponent(data)

        column_names[0] = column_names[0][1:].strip()
        dimensions[0] = dimensions[0][1:].strip()

        self._sink_column_names = [self._translate_array_name(c, reverse=True)
                                   for c in column_names]
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
        if len(self._sink_data) == 0:
            return

        target = self[self._sink_family]
        for column, dimension, data in zip(self._sink_column_names,
                                           self._sink_dimensions,
                                           self._sink_data.T):
            dtype = np.float64
            if column == "id":
                dtype = np.int64
            target[column] = data.astype(dtype)
            unit = dimension.replace("m", "g").replace("l", "cm").replace("t", "yr")
            if unit == "1":
                target[column].units = "1"
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
                if len(self._particle_blocks) > distinguisher_field:
                    pb_name = "%r" % self._particle_blocks[distinguisher_field]
                else:
                    pb_name = "at offset %d" % distinguisher_field
                warnings.warn(
                    "Using field %s to distinguish stars. If this is wrong, try editing your config.ini, section [ramses], entry particle-distinguisher." % pb_name)
            distinguisher_type = self._particle_types[distinguisher_field]

        # (each file's distinguishing field is made a family id, 0 for dm and 1 for star, as it is read)
        results = _layout.scan_particle_files([self._particle_filename(i) for i in self._cpus],
                                              distinguisher_field, distinguisher_type,
                                              convert=lambda data: (data != 0).view(np.int8))
        keep = self._particles_in_region([r[0] for r in results]) if self._trim_to_region else None

        for i, (particle_file, family_ids) in enumerate(results):
            npart_this = particle_file.npart
            if family_ids is None or len(family_ids) == 0:
                family_ids = np.zeros(npart_this, dtype=np.int8)
            if keep is not None:
                family_ids.view(np.uint8)[~keep[i]] = SKIP
                npart_this = int(np.count_nonzero(keep[i]))
            nstar_this = int(np.count_nonzero(family_ids == 1))
            self._particle_file_start_indices.append((dm_i0, star_i0))
            dm_i0 += (npart_this - nstar_this)
            star_i0 += nstar_this
            npart += npart_this
            nstar += nstar_this

            self._particle_family_ids_on_disk.append(family_ids)

        self._set_particle_selections([r[0] for r in results], [family.dm, family.star])
        return npart - nstar, nstar

    def _set_particle_selections(self, layouts, families):
        """Select the particles of each file, from the (internal) family id of each, where each family's first goes,
        and the family of each id"""
        self._particle_layouts = layouts
        self._particle_families_by_id = list(families)
        self._particle_selections = [ParticleSelection.all_particles(fam, first, len(families)) for fam, first in
                                     zip(self._particle_family_ids_on_disk, self._particle_file_start_indices)]

    def _particle_destinations(self, name):
        """The array *name* of each particle family, in order of internal family id (None for a family the
        snapshot holds none of, as when a region holds stars but no dark matter)"""
        present = self.families()
        return [self[f][name] if f in present else None for f in self._particle_families_by_id]

    def _read_plan(self, key, build):
        """The plan of reads for *key*, built by calling *build* if it is not kept"""
        return self._read_plans.get_or_build(key, build)

    def _count_gas_cells(self):
        if len(self._cpus) == 0:
            self._gas_i0 = np.zeros(0, dtype=np.int64)
            return 0
        self._amr_geometry, self._amr_layouts, leaves = _layout.scan_amr(
            [self._amr_filename(cpu) for cpu in self._cpus], self._cpus,
            self._info['ordering type'] == 'bisection', self._maxlevel)
        self._cell_selections = [CellSelection.all_leaves(layout, leaf_cells)
                                 for layout, leaf_cells in zip(self._amr_layouts, leaves)]
        if self._trim_to_region:
            self._trim_cells()
        ncells = [selection.ncells for selection in self._cell_selections]
        self._gas_i0 = np.cumsum([0] + ncells)[:-1]
        return int(np.sum(ncells))

    def _trim_cells(self):
        """Restrict the cell selections to the cells whose centres lie in the take_region filter's region (reading
        the AMR files a batch at a time, to bound the memory the positions take)"""
        for batch in _batches([selection.ncells for selection in self._cell_selections]):
            selections = [self._cell_selections[i] for i in batch]
            ncells = [selection.ncells for selection in selections]
            first = np.cumsum([0] + ncells)[:-1]
            positions, sizes = np.empty((sum(ncells), 3)), np.empty(sum(ncells))
            plan = _plan.cell_positions([self._amr_layouts[i] for i in batch], selections, first,
                                        [self._amr_filename(self._cpus[i]) for i in batch])
            _plan.load_cell_positions(plan, selections, self._amr_geometry, self._info['boxlen'], positions, sizes)
            keep = self._in_region(positions)
            del positions, sizes
            for i, f, n in zip(batch, first, ncells):
                self._cell_selections[i] = self._cell_selections[i].restricted(keep[f:f + n])

    def _load_gas_pos(self):
        self.gas['pos'].set_default_units()
        smooth = self.gas['smooth']
        smooth.set_default_units()

        try:
            paths = [self._amr_filename(cpu) for cpu in self._cpus]
            plan = self._read_plan(('cell positions', _execute.read_gap(paths)), lambda: _plan.cell_positions(
                self._amr_layouts, self._cell_selections, self._gas_i0, paths))
            _plan.load_cell_positions(plan, self._cell_selections, self._amr_geometry, self._info['boxlen'],
                                      self.gas['pos'], self.gas['smooth'])
        except Exception:
            self._delete_gas_arrays(['pos', 'smooth'])
            raise

    def _delete_gas_arrays(self, names):
        """Delete gas arrays (and the components of vectors), after a failure to load them"""
        for name in dict.fromkeys(self._array_name_1D_to_ND(n) or n for n in names):
            if name in self.gas.keys():
                del self.gas[name]
        for name in names:  # (the components of a deleted vector are otherwise left behind)
            if name in self._family_arrays and family.gas in self._family_arrays[name]:
                del self._family_arrays[name][family.gas]
                if not self._family_arrays[name]:
                    del self._family_arrays[name]

    def _gas_blocks_to_load(self, mode, array_name):
        """The blocks of the hydro-like file of *mode* to load for *array_name*: every block, under the ``all``
        setting of ``hydro-loading`` (or if no name is given); otherwise the block itself, or each component of a
        vector"""
        blocks = [self._hydro_blocks, self._grav_blocks, self._rt_blocks][mode]
        if array_name is None or _execute.config['hydro-loading'] == 'all':
            return list(blocks)
        if array_name in blocks:
            return [array_name]
        return [b for b in self._array_name_ND_to_1D(array_name) if b in blocks]

    def _load_gas_vars(self, mode=_gv_load_hydro, array_name=None):
        """Load blocks of the hydro-like files of *mode* (see _gas_blocks_to_load), returning their names"""
        load = self._gas_blocks_to_load(mode, array_name)
        created = [i for i in load if i not in self.gas]
        try:
            self._load_gas_vars_into(mode, load)
        except Exception:
            self._delete_gas_arrays(created)
            raise
        return load

    def _load_gas_vars_into(self, mode, load):
        """Load the blocks *load* of the hydro-like files of *mode*"""
        blocks = [self._hydro_blocks, self._grav_blocks, self._rt_blocks][mode]
        # the variables in the file, in order
        file_blocks = [i for i in blocks if not (self._ndim < 3 and i[-1] == 'z') and
                       not (self._ndim < 2 and i[-1] == 'y')]

        created = [i for i in load if i not in self.gas]
        for i in created:
            if i not in self.gas:  # (creating one component of a vector creates the others)
                self.gas._create_array(i)
        for i in load:
            if i in file_blocks:
                self.gas[i].set_default_units()

        def zero_unread(read):
            """Zero the arrays just created that will not be read (as a block the file does not hold)"""
            for i in created:
                if i not in read:
                    self.gas[i].view(np.ndarray)[:] = 0

        if not os.path.exists(self._hydro_filename(1)) and mode is _gv_load_hydro:
            # Case where force_gas = True, make sure rho is non-zero and such that mass=1.
            # This does not keep track of units for mass or rho since their value is enforced.
            logger.info("No hydro file found, gas likely from force_gas=True => hard setting rho gas")
            zero_unread([])
            if 'rho' in load:
                self.gas['rho'][:] = 1.0
                self.gas['rho'] /= (np.array(self.gas['smooth']) ** 3)
                self.gas['rho'].set_default_units()
            return

        logger.info("Loading %s from %s files", ", ".join(load), ['hydro', 'grav', 'rt'][mode])

        filenamer = [self._hydro_filename, self._grav_filename, self._rt_filename][mode]
        paths = [filenamer(i) for i in self._cpus]
        if len(paths) == 0:
            return

        header_layout = [ramses_hydro_header, ramses_grav_header, ramses_rt_header][mode]
        header, header_nbytes = _layout.read_header(paths[0], header_layout)
        nvar_file = [header.get('nvarh'), 4, header.get('nrtvar')][mode]
        nvar = len(file_blocks)
        if nvar_file != nvar and mode is _gv_load_rt:
            raise ValueError("Wrong number of variables in RAMSES dump")
        elif nvar_file != nvar and mode not in self._nvar_warned:
            if nvar_file < nvar:
                warnings.warn("Fewer hydro variables are in this RAMSES dump than are defined in config.ini (expected "
                              "%d, got %d in file)" % (nvar, nvar_file), RuntimeWarning)
            else:
                warnings.warn("More hydro variables (%d) are in this RAMSES dump than are defined in config.ini (%d)"
                              % (nvar_file, nvar), RuntimeWarning)
            self._nvar_warned.add(mode)
        file_blocks = file_blocks[:nvar_file]

        names = list(dict.fromkeys(i for i in load if i in file_blocks))
        zero_unread(names)
        if not names:
            return
        kind = _layout.HydroLikeFile(header_nbytes=header_nbytes, nvar=int(nvar_file))
        variable = {name: i for i, name in enumerate(file_blocks)}  # (the last, if a name is repeated)
        wanted = [(variable[i], k) for k, i in enumerate(names)]
        plan = self._read_plan(('cell values', mode, kind, tuple(wanted), _execute.read_gap(paths)),
                               lambda: _plan.cell_values(self._amr_layouts, self._cell_selections, self._gas_i0, paths,
                                                         kind, wanted))
        _plan.load_cell_values(plan, self._cell_selections, [self.gas[i] for i in names])

        if mode is _gv_load_gravity and 'phi' in names:
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
        self._load_particle_blocks([blockname])

    def _load_particle_blocks(self, blocknames):
        """Load particle blocks, reading all of them in one pass through the particle files"""
        records, fields = [], []
        for blockname in blocknames:
            offset = self._particle_blocks.index(blockname)
            _type = self._particle_types[offset]
            self._create_array_for_particles(blockname, _type)
            records.append((offset, np.dtype(_type).itemsize))
            fields.append((_type, self._particle_destinations(blockname)))

        try:
            plan = self._read_plan(
                ('particle fields', tuple(records), _execute.read_gap([pl.path for pl in self._particle_layouts])),
                lambda: _plan.particle_fields(self._particle_layouts, self._particle_selections, records))
            _plan.load_particle_fields(plan, self._particle_selections, fields)
        except Exception:
            warnings.warn(
                "Exception encountered while reading %s; is there an incompatibility in your Ramses configuration?"
                % ", ".join(repr(b) for b in blocknames))
            for blockname in blocknames:
                del self[blockname]
            raise

        # The potential is awkwardly not in physical units, but in
        # physical units divided by the box size. This was different
        # in an intermediate version, but then later made consistent
        # with the AMR phi output. So, we need to make a correction here
        # IF we are dealing with the latest ramses format.

        if self._particle_blocks_are_explictly_known and 'phi' in blocknames:
            for f in self._iter_particle_families():
                self[f]['phi'] *= self._info['boxlen']

    def _load_particle_cpuid(self):
        destinations = self._particle_destinations('cpu')
        if all(destination is None for destination in destinations):
            return
        for cpu, family_ids, starts in zip(self._cpus, self._particle_family_ids_on_disk,
                                           self._particle_file_start_indices):
            counts = np.bincount(family_ids.view(np.uint8), minlength=256)
            for internal_id, destination in enumerate(destinations):
                if destination is not None:
                    destination[starts[internal_id]:starts[internal_id] + counts[internal_id]] = cpu

    def _load_gas_cpuid(self):
        if len(self.gas) == 0:
            return
        gas_cpu_ar = self.gas['cpu']
        for cpu, i0, selection in zip(self._cpus, self._gas_i0, self._cell_selections):
            gas_cpu_ar[i0:i0 + selection.ncells] = cpu

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

    @property
    def is_cosmological(self):
        return not self._not_cosmological()

    @property
    def is_not_cosmological(self):
        return self._not_cosmological()

    def _not_cosmological(self):
        not_cosmological = True

        if "cosmo" in self._namelist and self._namelist["cosmo"]:
            not_cosmological = False

        if not self._namelist:
            warnings.warn("Namelist file either not found or unable to read. " +
                          "Guessing whether run is cosmological from cosmological parameters assuming flat LCDM.")
            not_cosmological = (self._info['omega_k'] == self._info['omega_l'] == 0)

        return not_cosmological

    @property
    def has_negative_iords(self):
        return self._has_negative_iords()

    def _has_negative_iords(self):
        return self.dm['iord'].min() < 0

    @property
    def has_potential_negative_iords_bug(self):
        return self._has_potential_negative_iords_bug()

    def _has_potential_negative_iords_bug(self):
        if self._negative_iords_on_purpose:
            return False

        if self.is_cosmological and family.dm in self.families() and self.has_negative_iords:
            return True
        return False

    def _convert_tform(self):
        # Copy the existing t array in weird Ramses format into a hidden raw array
        self.star['tform_raw'] = self.star['tform']
        self.star['tform_raw'].units = self._file_units_system[1]

        if self.is_cosmological:
            # Only attempt tform conversion for cosmological runs. The built-in tforms for isolated runs
            # are actually meaningful (issue 554)
            from ...analysis import ramses_util

            # Replace the tform array by its usual meaning using the birth files
            ramses_util.get_tform(self, times_are_proper=self.times_are_proper)

    @property
    def times_are_proper(self) -> bool:
        """Best guess for whether the times in the output (e.g. star formation times) are proper times"""
        if hasattr(self, "_is_using_proper_time"):
            # Already set, skipping
            pass
        elif config_parser.has_option("ramses", "proper_time"):
            self.times_are_proper = config_parser.getboolean(
                "ramses", "proper_time"
            )
        elif self.is_not_cosmological:
            self.times_are_proper = True
        else:
            # If we detect an RT file, assume proper time
            iout = self._timestep_id
            rt_file_candidates = (
                os.path.join(
                    self._dirname,
                    f"rt_{iout}.out{icpu:05d}",
                )
                for icpu in self._cpus
            )
            for rt_file in rt_file_candidates:
                if not os.path.exists(rt_file):
                    continue
                choice = "proper"
                reason = f"one RT file was detected ({rt_file})"
                self.times_are_proper = True
                break
            else:
                choice = "conformal"
                reason = "no RT file was found"
                self.times_are_proper = False
            warnings.warn(
                f"Assumed times to be in {choice} units because {reason}. "
                "If this is incorrect, pass the `times_are_proper` keyword "
                "argument when loading the dataset, or set the option `proper_time` "
                "in your .pynbodyrc."
            )

        return self._is_using_proper_time

    @times_are_proper.setter
    def times_are_proper(self, value):
        self._is_using_proper_time = value

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
            if array_name in ('tform', 'tform_raw'):
                if 'tform' in self._particle_blocks:  # Only attempt these conversion if tform is actually on disc (Issue #689)
                    self._load_particle_block('tform')
                    self._convert_tform()

            # Deal with metals for stars that have extra name mapping
            elif array_name in ('metals', 'metal'):
                if 'metal' in self._particle_blocks:
                    self._load_particle_block('metal')
                    self._convert_metal_name()

            elif array_name in self._split_arrays:
                self._load_particle_blocks(self._array_name_ND_to_1D(array_name))

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
                self._load_gas_vars(_gv_load_hydro, array_name)
            elif array_name in self._grav_blocks or array_name in self._grav_blocks_3d:
                self._load_gas_vars(_gv_load_gravity, array_name)
            elif array_name in self._rt_blocks_3d:
                if not self._rt_units_warned:
                    self._rt_units_warned = True
                    warnings.warn("Loading RT data from disk. Photon densities are stored in flux units by Ramses and "
                                  "need to be multiplied by the reduced speed of light of the run to obtain a physical "
                                  "number. This is currently left to the user, see issue 542 for more discussion.")
                loaded = self._load_gas_vars(_gv_load_rt, array_name)
                for u_block in dict.fromkeys(self._array_name_1D_to_ND(b) or b for b in loaded):
                    self[fam][u_block].units = self._rt_unit
            elif array_name == 'mass':
                # Very special case. If 'mass' has been created already for particle blocks (or sink blocks), and the
                # user requests it across the whole simulation, the SimSnap framework will issue this request rather
                # than a request to load across all families (handled below). If we raise an exception, the user
                # will see a confusing message about mass being unavailable for gas. So instead we redirect this to
                # a derived array, which will result in a full mass array being returned (good) and a warning about
                # conjoining derived and non-derived arrays (logical, at least).
                self._derive_array('mass', fam)
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
                self._load_gas_vars(_gv_load_hydro, 'vel')

            # the below triggers loading ALL particles, not just DM
            self._load_particle_blocks(['x', 'y', 'z', 'vx', 'vy', 'vz'])

        elif fam is None and array_name == 'mass':
            # mass has to be handled separately because it is present as a particle block
            # but not as a gas block -- there it will be derived. It is also present as a sink block,
            # which is again handled differently. So somehow we need to bring all of this together in
            # a way which doesn't baffle users.

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


# The positions of at most this many cells or particles are read at once to trim them to a region (but always those
# of at least one file)
trim_batch = 2 ** 24


def _batches(sizes):
    """Split files of the given sizes into consecutive batches of at most trim_batch in all (or one file), as lists
    of indices"""
    batches, total = [], None
    for i, n in enumerate(sizes):
        if total is None or total + n > trim_batch:
            batches.append([])
            total = 0
        batches[-1].append(i)
        total += n
    return batches


def _units_and_properties(info):
    """The units of density, time and length of an output, and its properties (but for the time), from its info"""
    properties = {}
    if info['H0'] > 1e-3:
        properties['a'] = info['aexp']
        properties['omegaM0'] = info['omega_m']
        properties['omegaL0'] = info['omega_l']
        properties['h'] = info['H0'] / 100.

    # N.B. these conversion factors provided by ramses already have the
    # correction from comoving to physical units
    d_unit = info['unit_d'] * units.Unit("g cm^-3")
    t_unit = info['unit_t'] * units.Unit("s")
    l_unit = info['unit_l'] * units.Unit("cm")

    properties['boxsize'] = info['boxlen'] * l_unit
    return [d_unit, t_unit, l_unit], properties


@RamsesSnap.decorator
def _translate_info(sim):
    (d_unit, t_unit, l_unit), properties = _units_and_properties(sim._info)
    sim.properties.update(properties)

    if sim.is_not_cosmological:
        sim.properties['time'] = sim._info['time'] * t_unit
    else:
        sim.properties["time"] = age(sim, unit="Gyr") * units.Unit("Gyr")

    sim._file_units_system = [d_unit, t_unit, l_unit]


@RamsesSnap.derived_array
def mass(sim):
    """Gas mass estimated from cell size and density"""
    return sim['rho'] * sim['smooth'] ** 3


@RamsesSnap.derived_array
def temp(sim):
    """Gas temperature derived from pressure and density """

    # Has to be redefined and rederived here from Ramses native variables
    # to avoid running into circular dependencies with the traditional derived definition
    # Now uses the self-consistent molecular weight field from pynbody (issue 598)

    from ...derived import mu
    mu_est = array.SimArray(np.ones(len(sim)), units="1")
    for i in range(5):
        temp = ((sim['p'] / sim['rho']) * (mu_est * units.m_p / units.k)).in_units("K")
        mu_est = mu(sim, t0=temp)
    return temp
