import glob
import gzip
import os.path
import re
import warnings

import numpy as np

from .. import config_parser, snapshot, util
from . import DummyHalo, Halo, HaloCatalogue, logger


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

        import os.path
        if not self._can_load(sim, ahf_basename):
            self._run_ahf(sim)

        HaloCatalogue.__init__(self, sim)

        if use_iord is None:
            use_iord = isinstance(
                sim.ancestor,
                (snapshot.gadget.GadgetSnap, snapshot.gadgethdf.GadgetHDFSnap)
            )

        self._use_iord = use_iord

        self._all_parts = get_all_parts

        self._only_stat = only_stat
        self._try_writing_fpos = write_fpos

        if only_stat:
            raise DeprecationWarning("only_stat keyword is deprecated; instead, use the catalogue's get_dummy_halo method")
        if get_all_parts:
            raise DeprecationWarning("get_all_parts keyword is deprecated; instead, use the catalogue's load_all method")
        if make_grp:
            raise DeprecationWarning("make_grp keyword is deprecated; instead, use the catalogue's get_group_array method")

        if ahf_basename is not None:
            self._ahfBasename = ahf_basename
        else:
            self._ahfBasename = self._infer_ahf_basename()

        logger.info("AHFCatalogue loading halo properties")
        self._load_ahf_halo_properties(self._ahfBasename + 'halos')

        logger.info("AHFCatalogue loading particles")
        self._load_ahf_particles(self._ahfBasename + 'particles')

        if dosort is not None:
            npart = np.array([properties['npart'] for properties in self._halo_properties])
            osort = np.argsort(-npart) # this is better than argsort(npart)[::-1], because it's stable
            self._halo_ids = osort + 1

        self._load_ahf_substructure(self._ahfBasename + 'substructure')

        if make_grp is None:
            make_grp = config_parser.getboolean('AHFCatalogue', 'AutoGrp')

        if make_grp:
            self.make_grp()

        if config_parser.getboolean('AHFCatalogue', 'AutoPid'):
            sim['pid'] = np.arange(0, len(sim))



        logger.info("AHFCatalogue loaded")

    def _infer_ahf_basename(self):
        candidates = self._list_candidate_ahf_basenames(sim)
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

    def __getitem__(self,item):
        """
        get the appropriate halo if dosort is on
        """
        if self._dosort is not None:
            i = self._sorted_indices[item-1]
        else:
            i = item
        return super().__getitem__(i)

    def make_grp(self, name='grp'):
        """
        Creates a 'grp' array which labels each particle according to
        its parent halo.
        """
        self.base[name] = self.get_group_array()

    def _write_fpos(self):
        try:
            with open(self._ahfBasename + 'fpos', 'w') as f:
                for i in range(self._nhalos):
                    if i < self._nhalos - 1:
                        f.write(str(self._halos[i+1].properties['fstart'])+'\n')
                    else:
                        f.write(str(self._halos[i+1].properties['fstart']))
        except OSError:
            warnings.warn("Unable to write AHF_fpos file; performance will be reduced. Pass write_fpos=False to halo constructor to suppress this message.")
        self._try_writing_fpos = False

    def _setup_children(self):
        """
        Creates a 'children' array inside each halo's 'properties'
        listing the halo IDs of its children. Used in case the reading
        of substructure data from the AHF-supplied _substructure file
        fails for some reason.
        """

        for i in range(self._nhalos):
            self._halos[i + 1].properties['children'] = []

        for i in range(self._nhalos):
            host = self._halos[i + 1].properties.get('hostHalo', -2)
            if host > -1:
                try:
                    self._halos[host + 1].properties['children'].append(i + 1)
                except KeyError:
                    pass

    def _get_halo(self, i):
        if self.base is None:
            raise RuntimeError("Parent SimSnap has been deleted")
        if self._all_parts is not None:
            return self._halos[i]

        with util.open_(self._ahfBasename+'particles') as f:
            fpos = self._halos[i].properties['fstart']
            f.seek(fpos,0)
            return Halo(
                i,
                self,
                self.base,
                self._load_ahf_particle_block(f, self._halos[i].properties['npart'])
            )



    @property
    def base(self):
        return self._base()

    def load_copy(self, i):
        """Load the a fresh SimSnap with only the particle in halo i"""

        from .. import load

        if self._dosort is not None:
            i = self._sorted_indices[i-1]

        with util.open_(self._ahfBasename + 'particles') as f:
            fpos = self._halos[i].properties['fstart']
            f.seek(fpos,0)
            ids = self._load_ahf_particle_block(f, nparts=self._halos[i].properties['npart'])

        return load(self.base.filename, take=ids)

    def _get_file_positions(self, filename):
        """Get the starting positions of each halo's particle information within the
        AHF_particles file for faster access later"""
        if os.path.exists(self._ahfBasename + 'fpos'):
            with util.open_(self._ahfBasename + 'fpos') as f:
                for i in range(self._nhalos):
                    self._halos[i+1].properties['fstart'] = int(f.readline())
        else:
            with util.open_(filename) as f:
                for h in range(self._nhalos):
                    if len(f.readline().split()) == 1:
                        f.readline()
                    self._halos[h+1].properties['fstart'] = f.tell()
                    for i in range(self._halos[h+1].properties['npart']):
                        f.readline()
            if self._try_writing_fpos:
                if not os.path.exists(self._ahfBasename + 'fpos'):
                    self._write_fpos()

    def _load_ahf_particle_block(self, f, nparts=None):
        """Load the particles for the next halo described in particle file f"""
        ng = len(self.base.gas)
        nd = len(self.base.dark)
        ns = len(self.base.star)
        nds = nd+ns

        if nparts is None:
            startline = f.readline()
            if len(startline.split())==1:
                startline = f.readline()
            nparts = int(startline.split()[0])

        if self.isnew:
            if not isinstance(f, gzip.GzipFile):
                data = np.fromfile(
                    f,
                    dtype=int,
                    sep=" ",
                    count=nparts * 2
                ).reshape(nparts, 2)[:, 0]
                data = np.ascontiguousarray(data)
            else:
                # unfortunately with gzipped files there does not
                # seem to be an efficient way to load nparts lines
                data = np.zeros(nparts, dtype=int)
                for i in range(nparts):
                    data[i] = int(f.readline().split()[0])

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

                if len(self.base) != nd + ns + ng:                      # We have extra families to the base ones
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
        else:
            if not isinstance(f, gzip.GzipFile):
                data = np.fromfile(f, dtype=int, sep=" ", count=nparts)
            else:
                # see comment above on gzipped files
                data = np.zeros(nparts, dtype=int)
                for i in range(nparts):
                    data[i] = int(f.readline())
        data.sort()
        return data

    def _load_ahf_particles(self, filename):
        if self._use_iord:
            self._init_iord_to_fpos()


        if filename.split("z")[-2][-1] == ".":
            self.isnew = True
        else:
            self.isnew = False

        if self._all_parts:
            self.load_all()

    def load_all(self):
        with util.open_(filename) as f:
            for h in range(self._nhalos):
                self._halos[h + 1] = Halo(
                    h + 1, self, self.base, self._load_ahf_particle_block(f))
                self._halos[h + 1]._descriptor = "halo_" + str(h + 1)

    def _load_ahf_halo_properties(self, filename):
        # Note: we need to open in 'rt' mode in case the AHF catalogue
        # is gzipped.
        with util.open_(filename, "rt") as f:
            first_line = f.readline()
            lines = f.readlines()

        self._halo_ids = np.arange(1, len(lines) + 1)
        self._halo_properties = [{} for i in self._halo_ids]

        # get all the property names from the first, commented line
        # remove (#)
        fields = first_line.replace("#", "").split()
        keys = [re.sub(r'\([0-9]*\)', '', field)
                for field in fields]
        # provide translations
        for i, key in enumerate(keys):
            if self.isnew:
                if(key == '#npart'):
                    keys[i] = 'npart'
            else:
                if(key == '#'):
                    keys[i] = 'dumb'
            if(key == 'a'):
                keys[i] = 'a_axis'
            if(key == 'b'):
                keys[i] = 'b_axis'
            if(key == 'c'):
                keys[i] = 'c_axis'
            if(key == 'Mvir'):
                keys[i] = 'mass'

        if self.isnew:
            # fix for column 0 being a non-column in some versions of the AHF
            # output
            if keys[0] == '#':
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
                if self.isnew:
                    properties[key] = values[i]
                else:
                    properties[key] = values[i - 1]

    def _load_ahf_substructure(self, filename):
        try:
            with util.open_(filename) as f:
                lines = f.readlines()
        except FileNotFoundError:
            self._setup_children()
            return
        logger.info("AHFCatalogue loading substructure")

        # In the substructure catalog, halos are either referenced by their index
        # or by their ID (if they have one).
        ID2index = {}
        for i, halo in self._halos.items():
            # If the "ID" property doesn't exist, use pynbody's internal index
            id = halo.properties.get("ID", i)
            ID2index[id] = i

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
                    "Falling back to using the halo info."
                )
                self._setup_children()
                break
            except KeyError:
                logger.error(
                    (
                        "Could not identify some substructure of "
                        "halo %s. Ignoring"
                    ),
                    haloid + 1
                )
                children = []

            self._halos[halo_index].properties['children'] = children
            for ichild in children:
                self._halos[ichild].properties['parent_id'] = halo_index


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

