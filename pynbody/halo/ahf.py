"""AHF (Amiga Halo Finder) support"""

from __future__ import annotations

import glob
import gzip
import os.path
import pathlib
import re
import warnings

import numpy as np

from .. import snapshot, util
from . import HaloCatalogue, HaloParticleIndices, logger
from .details.number_mapping import (
    NonMonotonicHaloNumberMapper,
    SimpleHaloNumberMapper,
    create_halo_number_mapper,
)


class AHFCatalogue(HaloCatalogue):
    """
    Class to handle catalogues produced by Amiga Halo Finder (AHF).
    """

    def __init__(self, sim, filename=None, make_grp=None, get_all_parts=None, use_iord=None, ahf_basename=None,
                 dosort=None, only_stat=None, write_fpos=True, halo_numbers='ahf',
                 ignore_missing_substructure=True,
                 **kwargs):
        """Initialize an AHFCatalogue.

        Parameters
        ----------

        sim: SimSnap
          the simulation snapshot to which this catalogue refers

        filename: str | pathlib.Path
          specify a path to an AHF halo catalog. Note that AHF actually outputs multiple files; you can specify the
          path to the ``AHF_halos`` or ``AHF_particles`` file and the code will infer the other filenames from this.
          Alternatively you can specify the path up to the ``AHF_`` prefix and the code will similarly infer the full
          set of filenames.

        halo_numbers: str, optional
          specify how to number the halos. Options are:

          * 'ahf' (default): use the halo numbers written in the AHF halos file if present, or a zero-based indexing
          * 'file-order': zero-based indexing of the halos
          * 'v1': one-based indexing of the halos, compatible with pynbody v1 default behaviour
          * 'length-order': sort by the number of particles in each halo, with the halo with most particles being halo 0
          * 'length-order-v1': as length-order, but indexing from halo 1, compatible with ``dosort=True`` in pynbody v1

        ignore_missing_substructure : bool, optional
          If True (default), the code will not raise an exception if the substructure file is missing or corrupt. If
          False, it will raise an exception.

        use_iord : bool, optional
          if True, the particle IDs in the Amiga catalogue are taken to refer to the iord array. If False,
          they are the particle offsets within the file. If None, the parameter defaults to True for GadgetSnap,
          False otherwise.

        write_fpos : bool, optional
            If True (default), the code will attempt to write a file containing the starting positions of each halo's
            particle information within the AHF_particles file. If False, it will not attempt to write this file. This
            file is used to speed up loading of particle information for individual halos when :meth:`load_all` is not
            called. If :meth`load_all` is called, there is no benefit to writing the file and better performance
            is obtained by using ``write_fpos=False``.

        ahf_basename : str, optional
          Deprecated way to specify the location of the catalogue

        make_grp :
          Deprecated. If True a 'grp' array is created in the underlying snapshot specifying the lowest level halo
          that any given particle belongs to. If it is False, no such array is created; if None, the behaviour is
          determined by the configuration system.

        get_all_parts :
          Deprecated; use the :meth:`load_all` method instead.

        dosort :
          Deprecated; equivalent to ``halo_numbers='length-order'``

        only_stat :
          Deprecated; this keyword is now ignored. To obtain halo information without loading the particles,
          use the methods :meth:`get_properties_one_halo` or :meth:`get_properties_all_halos`.

        """

        if use_iord is None:
            use_iord = isinstance(
                sim.ancestor,
                (snapshot.gadget.GadgetSnap, snapshot.gadgethdf.GadgetHDFSnap)
            )

        self._use_iord = use_iord
        self._only_stat = only_stat
        self._try_writing_fpos = write_fpos

        if only_stat:
            warnings.warn(DeprecationWarning("only_stat keyword is deprecated; instead, use the catalogue's get_dummy_halo method"))

        if filename is not None:
            self._ahfBasename = str(self._user_specified_filename_to_ahf_basename(filename))
        elif ahf_basename is not None:
            warnings.warn(DeprecationWarning("ahf_basename keyword is deprecated; use filename instead"))
            self._ahfBasename = ahf_basename
        else:
            self._ahfBasename = self._infer_ahf_basename(sim)

        self._determine_format_revision_from_filename()

        logger.info("AHFCatalogue loading halo properties")
        self._load_ahf_halo_properties(self._ahfBasename + 'halos')

        if dosort:
            warnings.warn(DeprecationWarning("dosort keyword is deprecated; instead pass halo_numbers='length-order-v1'"))
            halo_numbers = 'length-order-v1'

        number_mapper = self._setup_halo_numbering(halo_numbers)

        super().__init__(sim, number_mapper)

        self._remap_host_halo_property()

        if self._use_iord:
            self._init_iord_to_fpos()

        try:
            self._load_ahf_substructure(self._ahfBasename + 'substructure')
        except (KeyError, ValueError, FileNotFoundError, IndexError):
            if not ignore_missing_substructure:
                raise
            logger.error("Unable to load AHF substructure file; continuing without. To expose the underlying problem as an exception, pass ignore_missing_substructure=False to the AHFCatalogue constructor")

        if make_grp:
            warnings.warn(DeprecationWarning("make_grp keyword is deprecated; instead, use the catalogue's get_group_array method"))
            self.base['grp'] = self.get_group_array()

        if get_all_parts:
            warnings.warn(
                DeprecationWarning("get_all_parts keyword is deprecated; instead, use the catalogue's load_all method"))
            self.load_all()

        logger.info("AHFCatalogue loaded")

    def _setup_halo_numbering(self, halo_numbers):
        has_id = 'ID' in self._halo_properties
        if has_id and len(self._halo_properties['ID'])>0:
            self._ahf_own_number_mapper = create_halo_number_mapper(self._halo_properties['ID'])
        else:
            # if no explicit IDs, ahf implicitly numbers starting at 0 in file order
            self._ahf_own_number_mapper = SimpleHaloNumberMapper(0, self._num_halos)

        if halo_numbers == 'v1':
            number_mapper = SimpleHaloNumberMapper(1, self._num_halos)
        elif halo_numbers == 'file-order':
            number_mapper = SimpleHaloNumberMapper(0, self._num_halos)
        elif halo_numbers == 'length-order' or halo_numbers == 'length-order-v1':
            npart = self._halo_properties['npart']
            osort = np.argsort(-npart, kind='stable')  # this is better than argsort(npart)[::-1], because it's stable
            if halo_numbers.endswith('v1'):
                number_mapper = NonMonotonicHaloNumberMapper(osort, ordering=True, start_index=1)
            else:
                number_mapper = NonMonotonicHaloNumberMapper(osort, ordering=True, start_index=0)
        elif halo_numbers == 'ahf':
            number_mapper = self._ahf_own_number_mapper
        else:
            raise ValueError(f"halo_numbers keyword {halo_numbers} not recognised")

        return number_mapper

    def _remap_host_halo_property(self):
        """When loaded from the .halos file, hostHalo is the AHF halo number of the host halo.
        This maps it onto whatever halo number pynbody is using."""

        if 'hostHalo' in self._halo_properties:
            host_halo = self._halo_properties['hostHalo']

            # Below, we used to use:
            #   mask = host_halo != -1
            # but now it seems like specific runs (presumably MPI ones, where IDs are random?) use zero to indicate
            # 'no host'.
            #
            # Of course, in other runs, halo 0 might be a perfectly valid host halo, so we need to deal with this
            # by saying the mask is wherever host_halo is not a valid halo number. Rather than compare every entry
            # to all the halo numbers (which would be expensive), we look for the minimum valid

            if len(host_halo) > 0:
                mask = host_halo >= np.min(self._ahf_own_number_mapper.all_numbers)

                host_halo[mask] = self.number_mapper.index_to_number(
                    self._ahf_own_number_mapper.number_to_index(host_halo[mask])
                )

    def _determine_format_revision_from_filename(self):
        if self._ahfBasename.split("z")[-2][-1] == ".":
            self._is_new_format = True
        else:
            self._is_new_format = False

    @classmethod
    def _infer_ahf_basename(cls, sim):
        candidates = cls._list_candidate_ahf_basenames(sim)
        if len(candidates) == 1:
            candidate = candidates[0]
        elif len(candidates) == 0:
            raise FileNotFoundError("No candidate AHF catalogue found; try specifying a catalogue using the "
                                    "ahf_basename keyword")
        else:
            candidate = candidates[0]
            warnings.warn(f"Multiple candidate AHF catalogues found; using {candidate}. To specify a different "
                          "catalogue, use the ahf_basename keyword, or move the other catalogues.")
        return util.cutgz(candidate)[:-9]


    def _write_fpos(self):
        try:
            np.savetxt(self._ahfBasename+'fpos', self._fpos, fmt="%d")
        except OSError:
            warnings.warn("Unable to write AHF_fpos file; performance will be reduced. Pass write_fpos=False to halo constructor to suppress this message, or use load_all() method to remove need for storing fpos information.")
        self._try_writing_fpos = False

    @property
    def base(self):
        return self._base()

    def _get_file_positions(self):
        """Get the starting positions of each halo's particle information within the
        AHF_particles file for faster access later"""
        if not hasattr(self, "_fpos"):
            if os.path.exists(self._ahfBasename + 'fpos'):
                self._fpos = np.loadtxt(self._ahfBasename+'fpos', dtype=int)
            else:
                self._fpos = np.empty(len(self.number_mapper), dtype=int)
                with util.open_(self._ahfBasename + 'particles') as f:
                    nhalo = int(f.readline().strip())
                    assert nhalo == len(self.number_mapper)
                    for hnum in range(nhalo):
                        npart = int(f.readline().split()[0].strip())
                        assert npart == self._halo_properties['npart'][hnum]
                        self._fpos[hnum] = f.tell()
                        for i in range(npart):
                            f.readline()
                if self._try_writing_fpos:
                    if not os.path.exists(self._ahfBasename + 'fpos'):
                        self._write_fpos()

        return self._fpos

    def _load_ahf_particle_block(self, f, nparts):
        """Load the particles for the next halo described in particle file f"""
        if self._is_new_format:
            if not isinstance(f, gzip.GzipFile):
                data = np.fromfile(
                    f,
                    dtype=int,
                    sep=" ",
                    count=nparts * 2
                )[::2]
                data = np.ascontiguousarray(data)
            else:
                # unfortunately with gzipped files there does not
                # seem to be an efficient way to load nparts lines
                data = np.empty(nparts, dtype=int)
                for i in range(nparts):
                    data[i] = int(f.readline().split()[0])

            data = self._ahf_to_pynbody_particle_ids(data)
        else:
            if not isinstance(f, gzip.GzipFile):
                data = np.fromfile(f, dtype=int, sep=" ", count=nparts)
            else:
                # see comment above on gzipped files
                data = np.empty(nparts, dtype=int)
                for i in range(nparts):
                    data[i] = int(f.readline())
        data.sort()
        return data

    def _ahf_to_pynbody_particle_ids(self, data):
        ng = len(self.base.gas)
        nd = len(self.base.dark)
        ns = len(self.base.star)
        nds = nd + ns
        if self._use_iord:
            data = self._iord_to_fpos.map_ignoring_order(data)
        elif isinstance(self.base, snapshot.ramses.RamsesSnap):
            # AHF only expects three families, DM, star, gas in this order
            # and generates iords on disc according to this rule
            # For classical Ramses snapshots, this is perfectly adequate, but
            # for more modern outputs that have extra tracers, BHs families
            # we need to offset the ids to return the correct slicing
            # TODO These tests on snapshot type might not be necessary
            #  as using the family logic properly should be general.
            #  It is currently kept to ensure 100% backwards compatibility with previous behaviour,
            #  as this code is not explicitly checked by the pynbody test distribution

            if len(self.base) != nd + ns + ng:  # We have extra families to the base ones
                # First identify DM, star and gas particles in AHF
                ahf_dm_mask = data < nd
                ahf_star_mask = (data >= nd) & (data < nds)
                ahf_gas_mask = data >= nds

                # Then offset them by DM family start, to account for
                # additional families before it, e.g. gas tracers
                data[np.where(ahf_dm_mask)] += self.base._get_family_slice('dm').start

                # Star ids used to start at NDM, now they start with the star family slice
                offset = self.base._get_family_slice('star').start - nd
                data[np.where(ahf_star_mask)] += offset

                # Gas ids were greater than NDM + NSTAR, now they start with the gas slice
                offset = self.base._get_family_slice('gas').start - nds
                data[np.where(ahf_gas_mask)] += offset
        elif not isinstance(self.base, snapshot.nchilada.NchiladaSnap):
            hi_mask = data >= nds
            data[np.where(hi_mask)] -= nds
            data[np.where(~hi_mask)] += ng
        else:
            st_mask = (data >= nd) & (data < nds)
            g_mask = data >= nds
            data[np.where(st_mask)] += ng
            data[np.where(g_mask)] -= ns
        return data

    def _get_particle_indices_one_halo(self, halo_number):
        fpos = self._get_file_positions()
        file_index = self.number_mapper.number_to_index(halo_number)
        with util.open_(self._ahfBasename + 'particles') as f:
            f.seek(fpos[file_index],0)
            ids = self._load_ahf_particle_block(f, nparts=self._halo_properties['npart'][file_index])
        return ids

    def _get_all_particle_indices(self):
        boundaries = np.cumsum(np.concatenate(([0], self._halo_properties['npart'])))
        boundaries = np.vstack((boundaries[:-1], boundaries[1:])).T
        particle_ids = np.empty(boundaries[-1,1], dtype=int)
        with util.open_(self._ahfBasename + 'particles') as f:
            f.readline()
            for nparts, (start, end) in zip(self._halo_properties['npart'], boundaries):
                f.readline()
                particle_ids[start:end] = self._load_ahf_particle_block(f, nparts=nparts)


        return HaloParticleIndices(particle_ids=particle_ids, boundaries=boundaries)


    def get_properties_one_halo(self, i):
        index = self.number_mapper.number_to_index(i)
        return {key: self._halo_properties[key][index] for key in self._halo_properties}

    def get_properties_all_halos(self, with_units=True) -> dict:
        return self._halo_properties

    def _load_ahf_halo_properties(self, filename):
        # Note: we need to open in 'rt' mode in case the AHF catalogue
        # is gzipped.
        with util.open_(filename, "rt") as f:
            first_line = f.readline()
            lines = f.readlines()

        # get all the property names from the first, commented line
        # remove (#)
        fields = first_line.replace("#", "").split()
        keys = [re.sub(r'\([0-9]*\)', '', field)
                for field in fields]

        omit_first_column = False

        if self._is_new_format:
            # fix for column 0 being a non-column in some versions of the AHF
            # output
            if keys[0] == '':
                omit_first_column = True
                keys = keys[1:]

        self._halo_properties = {k: [] for k in keys}

        self._num_halos = len(lines)

        for line in lines:
            values = [
                float(x) if any(_ in x for _ in (".", "e", "nan", "inf"))
                else int(x)
                for x in line.split()
            ]
            # XXX Unit issues!  AHF uses distances in Mpc/h, possibly masses as
            # well
            if omit_first_column:
                values = values[1:]

            for key, value in zip(keys, values):
                self._halo_properties[key].append(value)

        for key in keys:
            self._halo_properties[key] = np.array(self._halo_properties[key])

    def _load_ahf_substructure(self, filename):
        with util.open_(filename) as f:
            lines = f.readlines()
        logger.info("AHFCatalogue loading substructure")

        if len(lines[0].split())!=2:
            # some variants of the format start with a line count, which we ignore
            lines = lines[1:]

        self._halo_properties['children'] = children = [[] for _ in range(self._num_halos)]
        self._halo_properties['parent'] = parent = np.empty(self._num_halos, dtype=int)
        parent.fill(-1)

        for halo_line, children_line in zip(lines[::2], lines[1::2]):
            haloid, _nsubhalos = (int(x) for x in halo_line.split())
            halo_index = self._ahf_own_number_mapper.number_to_index(haloid)
            halo_number = self.number_mapper.index_to_number(halo_index)
            children_ahf_numbering = [int(x) for x in children_line.split()]
            children_index = self._ahf_own_number_mapper.number_to_index(children_ahf_numbering)
            children_pynbody_numbering = self.number_mapper.index_to_number(children_index)

            children[halo_index] = children_pynbody_numbering
            for child_index in children_index:
                parent[child_index] = halo_number


    @staticmethod
    def _list_candidate_ahf_basenames(sim):
        candidates = set(glob.glob(f"{sim._filename}*z*particles*"))
        # use a set to ensure that no duplicates can be produced
        # This could arise in an edge case where _filename is a directory
        # and having the "/" at the end of it would lead to a first detection here
        # and a second one again below

        if os.path.isdir(sim._filename):
            candidates = candidates.union(glob.glob(os.path.join(sim._filename, "*z*particles*")))

        return list(candidates)

    @classmethod
    def _user_specified_filename_to_ahf_basename(cls, filename: str | pathlib.Path) -> pathlib.Path:
        allowed_endings = ["AHF_halos", "AHF_particles", "AHF_", "AHF"]
        filename = util.cutgz(filename)
        for ending in allowed_endings:
            if str(filename).endswith(ending):
                return pathlib.Path(str(filename)[:-len(ending)]+"AHF_")
        raise ValueError("Filename cannot be understood as an AHF catalogue basename. "
                         "For information about how to specify an AHF filename, see the class documentation for "
                         "AHFCatalogue.")

    @classmethod
    def _can_load(cls, sim, filename=None, **kwargs):
        if filename is not None:
            try:
                cls._user_specified_filename_to_ahf_basename(filename)
                return True
            except ValueError:
                return False
        else:
            candidates = cls._list_candidate_ahf_basenames(sim)
            number_ahf_file_candidates = len(candidates)
            return number_ahf_file_candidates > 0
