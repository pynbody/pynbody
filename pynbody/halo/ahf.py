import glob
import gzip
import os.path
import re
import warnings

import numpy as np

from .. import config_parser, snapshot, util
from . import DummyHalo, Halo, HaloCatalogue, HaloParticleIndices, logger
from .details.number_mapper import HaloNumberMapper, SimpleHaloNumberMapper


class AHFCatalogue(HaloCatalogue):

    """
    Class to handle catalogues produced by Amiga Halo Finder (AHF).
    """

    def __init__(self, sim, make_grp=None, get_all_parts=None, use_iord=None, ahf_basename=None,
                 dosort=None, only_stat=None, write_fpos=True, **kwargs):
        """Initialize an AHFCatalogue.

        **kwargs** :

        *make_grp*: if True a 'grp' array is created in the underlying
                    snapshot specifying the lowest level halo that any
                    given particle belongs to. If it is False, no such
                    array is created; if None, the behaviour is
                    determined by the configuration system.

        *get_all_parts*: if True, the particle file is loaded for all halos.
                    Suggested to keep this None, as this is memory intensive.
                    The default function is to load in this data as needed.

        *use_iord*: if True, the particle IDs in the Amiga catalogue
                    are taken to refer to the iord array. If False,
                    they are the particle offsets within the file. If
                    None, the parameter defaults to True for
                    GadgetSnap, False otherwise.

        *ahf_basename*: specify the basename of the AHF halo catalog
                        files - the code will append 'halos',
                        'particles', and 'substructure' to this
                        basename to load the catalog data.

        *dosort*: specify if halo catalog should be sorted so that
                  halo 1 is the most massive halo, halo 2 the
                  second most massive and so on.

        *only_stat*: specify that you only wish to collect the halo
                    properties stored in the AHF_halos file and not
                    worry about particle information [deprecated, use
                    get_dummy_halo instead]

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

        if ahf_basename is not None:
            self._ahfBasename = ahf_basename
        else:
            self._ahfBasename = self._infer_ahf_basename(sim)

        self._determine_format_revision_from_filename()

        logger.info("AHFCatalogue loading halo properties")
        self._load_ahf_halo_properties(self._ahfBasename + 'halos')

        # Now we know what halos we have, we can initialise the base class
        # TODO - here is where dosort should be implemented, and also where AHF's own halo numbering could be used
        if dosort is not None:
            warnings.warn(DeprecationWarning("dosort keyword is deprecated"))
            npart = np.array([properties['npart'] for properties in self._halo_properties])
            osort = np.argsort(-npart)  # this is better than argsort(npart)[::-1], because it's stable
            raise NotImplementedError("Need to do a different halo number mapper here")

        number_mapper = SimpleHaloNumberMapper(1, len(self._halo_properties))
        super().__init__(sim, number_mapper)

        if self._use_iord:
            self._init_iord_to_fpos()

        self._load_ahf_substructure(self._ahfBasename + 'substructure')


        if make_grp:
            warnings.warn(DeprecationWarning("make_grp keyword is deprecated; instead, use the catalogue's get_group_array method"))
            self.base['grp'] = self.get_group_array()

        if get_all_parts:
            warnings.warn(
                DeprecationWarning("get_all_parts keyword is deprecated; instead, use the catalogue's load_all method"))
            self.load_all()

        logger.info("AHFCatalogue loaded")

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
                self._fpos = np.empty(len(self._number_mapper), dtype=int)
                with util.open_(self._ahfBasename + 'particles') as f:
                    for hnum in range(len(self._number_mapper)):
                        if len(f.readline().split()) == 1:
                            f.readline()
                        self._fpos[hnum] = f.tell()
                        for i in range(self._halo_properties[hnum]['npart']):
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
            data = self._iord_to_fpos[data]
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

    def _get_index_list_one_halo(self, i):
        fpos = self._get_file_positions()
        file_index = self._number_mapper.number_to_index(i)
        with util.open_(self._ahfBasename + 'particles') as f:
            f.seek(fpos[file_index],0)
            ids = self._load_ahf_particle_block(f, nparts=self._halo_properties[file_index]['npart'])
        return ids

    def _get_all_particle_indices(self):
        boundaries = np.cumsum([0] + [properties['npart'] for properties in self._halo_properties])
        boundaries = np.vstack((boundaries[:-1], boundaries[1:])).T
        particle_ids = np.empty(boundaries[-1,1], dtype=int)
        with util.open_(self._ahfBasename + 'particles') as f:
            f.readline()
            for properties, (start, end) in zip(self._halo_properties, boundaries):
                f.readline()
                particle_ids[start:end] = self._load_ahf_particle_block(f, nparts=properties['npart'])


        return HaloParticleIndices(particle_ids=particle_ids,
                                   boundaries=boundaries,
                                   halo_number_mapper=self._number_mapper)


    def _get_properties_one_halo(self, i):
        return self._halo_properties[self._number_mapper.number_to_index(i)]

    def _load_ahf_halo_properties(self, filename):
        # Note: we need to open in 'rt' mode in case the AHF catalogue
        # is gzipped.
        with util.open_(filename, "rt") as f:
            first_line = f.readline()
            lines = f.readlines()

        self._halo_properties = [{} for h in range(len(lines))]

        # get all the property names from the first, commented line
        # remove (#)
        fields = first_line.replace("#", "").split()
        keys = [re.sub(r'\([0-9]*\)', '', field)
                for field in fields]

        if self._is_new_format:
            # fix for column 0 being a non-column in some versions of the AHF
            # output
            if keys[0] == '':
                keys = keys[1:]

        for properties, line in zip(self._halo_properties, lines):
            values = [
                float(x) if any(_ in x for _ in (".", "e", "nan", "inf"))
                else int(x)
                for x in line.split()
            ]
            # XXX Unit issues!  AHF uses distances in Mpc/h, possibly masses as
            # well
            for i, key in enumerate(keys):
                if self._is_new_format:
                    properties[key] = values[i]
                else:
                    properties[key] = values[i - 1]

    def _load_ahf_substructure(self, filename):
        try:
            with util.open_(filename) as f:
                lines = f.readlines()
        except FileNotFoundError:
            return
        logger.info("AHFCatalogue loading substructure")

        # In the substructure catalog, halos are either referenced by their index
        # or by their ID (if they have one).
        ID2index = {self._halo_properties[i].get("ID", i): i for i in range(len(self._halo_properties))}

        for line in lines:
            try:
                haloid, _nsubhalos = (int(x) for x in line.split())
                halo_index = ID2index[haloid]
                children = [
                    ID2index[int(x)] for x in f.readline().split()
                ]
            except ValueError:
                logger.error(
                    "An error occurred while reading substructure file. "
                )
                break
            except KeyError:
                logger.error(
                    (
                        "Could not identify some substructure of "
                        "halo %s. Ignoring"
                    ),
                    haloid + 1
                )
                break

            self._halo_properties[halo_index]['children'] = children
            for ichild in children:
                self._halo_properties[ichild]['parent_id'] = halo_index


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
    def _can_load(cls, sim, ahf_basename=None, **kwargs):
        if ahf_basename:
            return True
        candidates = cls._list_candidate_ahf_basenames(sim)
        number_ahf_file_candidates = len(candidates)
        return number_ahf_file_candidates > 0
