from os.path import isfile

import h5py
import numpy as np

import pynbody


def setup_module():
    global snap, halos, subhalos, htest, snap_arepo, halos_arepo, subhalos_arepo, htest_arepo
    snap = pynbody.load('testdata/testL10N64/snapshot_000.hdf5')
    halos = pynbody.halo.Gadget4SubfindHDFCatalogue(snap)
    subhalos = pynbody.halo.Gadget4SubfindHDFCatalogue(snap, subs=True)
    htest = Halos('testdata/testL10N64/', 0)
    snap_arepo = pynbody.load('testdata/arepo/cosmobox_015.hdf5')
    halos_arepo = pynbody.halo.ArepoSubfindHDFCatalogue(snap_arepo)
    subhalos_arepo = pynbody.halo.ArepoSubfindHDFCatalogue(snap_arepo, subs=True)
    htest_arepo = Halos('testdata/arepo/', 15)


def teardown_module():
    global snap, halos, subhalos, htest, snap_arepo, halos_arepo, subhalos_arepo, htest_arepo
    del snap, halos, subhalos, htest, snap_arepo, halos_arepo, subhalos_arepo, htest_arepo


def test_catalogue():
    _h_nogrp = snap.halos(grp_array=True)
    _subh_nogrp = snap.halos(subs=True, grp_array=True)
    _harepo_nogrp = snap_arepo.halos(grp_array=True)
    _subharepo_nogrp = snap_arepo.halos(subs=True, grp_array=True)
    for h in [halos, subhalos, _h_nogrp, _subh_nogrp, halos_arepo, subhalos_arepo, _harepo_nogrp, _subharepo_nogrp]:
        assert(isinstance(h, pynbody.halo.subfindhdf.Gadget4SubfindHDFCatalogue)), \
            "Should be a Gadget4SubfindHDFCatalogue catalogue but instead it is a " + str(type(h))

def test_lengths():
    assert len(halos)==299
    assert len(subhalos)==343
    assert len(halos_arepo)==447
    assert len(subhalos_arepo)==475


def _test_halo_or_subhalo_properties(comparison_catalogue, pynbody_catalogue):

    np.random.seed(1)
    hids = np.random.choice(range(len(pynbody_catalogue)), 20)

    for hid in hids:
        for key in list(comparison_catalogue.keys()):
            props = pynbody_catalogue.get_halo_properties(hid, with_unit=False)
            if key in list(props.keys()):
                np.testing.assert_allclose(props[key], comparison_catalogue[key][hid])

def test_halo_properties():
    for htest_file, halocatalogue in [(htest, halos), (htest_arepo, halos_arepo)]:
        _test_halo_or_subhalo_properties(htest_file.load()['halos'], halocatalogue)


def test_subhalo_properties():
    for htest_file, halocatalogue in [(htest, subhalos), (htest_arepo, subhalos_arepo)]:
        _test_halo_or_subhalo_properties(htest_file.load()['subhalos'], halocatalogue)



def test_halo_loading() :
    """ Check that halo loading works """
    # check that data loading for individual fof groups works
    _ = halos[0]['pos']
    _ = halos[1]['pos']
    _ = halos[0]['mass'].sum()
    _ = halos[1]['mass'].sum()
    _ = halos_arepo[0]['pos']
    _ = halos_arepo[1]['pos']
    _ = halos_arepo[0]['mass'].sum()
    _ = halos_arepo[1]['mass'].sum()
    assert(len(halos[0]['iord']) == len(halos[0]) == htest.load()['halos']['GroupLenType'][0, 1])
    arepo_halos = htest_arepo.load()['halos']
    assert(len(halos_arepo[0]['iord']) == len(halos_arepo[0]) == np.sum(arepo_halos['GroupLenType'][0, :], axis=-1))


def test_particle_data():
    hids = np.random.choice(range(len(halos)), 5)
    for hid in hids:
        assert(np.allclose(halos[hid].dm['iord'], htest[hid]['iord']))


class Halos:
    """
    This class extracts halo/subhalo information directly from the group catalog HDF5 file. It is used here to
    test that the group catalogue properties match those obtained via the pynbody interface.

    It can load all halo/subhalo information from the entire group catalog for one snapshot or
    return group catalog information for one halo/subhalo.

    """
    def __init__(self, basePath, snapNum):
        self.basePath = basePath
        self.snapNum = snapNum

        self.gcfile = self.gcPath(basePath, snapNum)
        self.snapfile = self.snapPath(basePath, snapNum)

        self.properties = self.load()

    def __getitem__(self, haloID):
        r = {}
        fields = ['GroupPos', 'Group_R_Mean200']
        keynames = ['cen', 'r200']
        for i in range(len(fields)):
            r[keynames[i]] = self.properties['halos'][fields[i]][haloID]

        r['iord'] = self.loadHalo(haloID, "dm", fields=['ParticleIDs'])['ParticleIDs']
        return r

    def gcPath(self, basePath, snapNum):
        """ Return absolute path to a group catalog HDF5 file. """
        filePath1 = basePath + 'groups_%03d.hdf5' % snapNum
        filePath2 = basePath + 'fof_subhalo_tab_%03d.hdf5' % snapNum

        if isfile(filePath1):
            return filePath1
        return filePath2

    def snapPath(self, basePath, snapNum):
        """ Return absolute path to a snapshot HDF5 file. """
        filePath1 = basePath + 'snapshot_' + str(snapNum).zfill(3) + '.hdf5'
        return filePath1

    def loadSubhalos(self, fields=None):
        """ Load all subhalo information from the entire group catalog for one snapshot
           (optionally restrict to a subset given by fields). """
        try:
            return self.loadObjects("Subhalo", "subhalos", fields)
        except:
            return self.loadObjects("Subhalo", "subgroups", fields)

    def loadHalos(self, fields=None):
        """ Load all halo information from the entire group catalog for one snapshot
           (optionally restrict to a subset given by fields). """

        return self.loadObjects("Group", "groups", fields)

    def loadHeader(self):
        """ Load the group catalog header. """
        with h5py.File(self.gcfile, 'r') as f:
            header = dict(f['Header'].attrs.items())

        return header

    def loadParams(self):
        """ Load the group catalog header. """
        with h5py.File(self.gcfile, 'r') as f:
            params = dict(f['Parameters'].attrs.items())

        return params

    def load(self, fields=None):
        """ Load complete group catalog all at once. """
        r = {}
        r['subhalos'] = self.loadSubhalos(fields)
        r['halos'] = self.loadHalos(fields)
        r['header'] = self.loadHeader()
        return r

    def loadObjects(self, gName, nName, fields):
        """ Load either halo or subhalo information from the group catalog. """
        result = {}

        # make sure fields is not a single element
        if isinstance(fields, str):
            fields = [fields]

        # load header from first chunk
        with h5py.File(self.gcfile, 'r') as f:

            header = dict(f['Header'].attrs.items())
            result['count'] = f['Header'].attrs['N' + nName + '_Total']

            if not result['count']:
                print('warning: zero groups, empty return (snap=' + str(self.snapNum) + ').')
                return result

            # if fields not specified, load everything
            if not fields:
                fields = list(f[gName].keys())

            for field in fields:
                # verify existence
                if field not in f[gName].keys():
                    raise Exception("Group catalog does not have requested field [" + field + "]!")

                # replace local length with global
                shape = list(f[gName][field].shape)
                shape[0] = result['count']

                # allocate within return dict
                result[field] = np.zeros(shape, dtype=f[gName][field].dtype)

        # loop over chunks
        wOffset = 0

        with h5py.File(self.gcfile, 'r') as f:

            # loop over each requested field
            for field in fields:
                if field not in f[gName].keys():
                    raise Exception("Group catalog does not have requested field [" + field + "]!")

                # shape and type
                shape = f[gName][field].shape

                # read data local to the current file
                if len(shape) == 1:
                    result[field][wOffset:wOffset + shape[0]] = f[gName][field][0:shape[0]]
                else:
                    result[field][wOffset:wOffset + shape[0], :] = f[gName][field][0:shape[0], :]

            wOffset += shape[0]

        # only a single field? then return the array instead of a single item dict
        if len(fields) == 1:
            return result[fields[0]]

        return result

    def loadSingle(self, haloID=-1, subhaloID=-1):
        """ Return complete group catalog information for one halo or subhalo. """
        if (haloID < 0 and subhaloID < 0) or (haloID >= 0 and subhaloID >= 0):
            raise Exception("Must specify either haloID or subhaloID (and not both).")

        gName = "Subhalo" if subhaloID >= 0 else "Group"
        searchID = subhaloID if subhaloID >= 0 else haloID

        # load halo/subhalo fields into a dict
        result = {}

        with h5py.File(self.gcfile, 'r') as f:
            for haloProp in f[gName].keys():
                result[haloProp] = f[gName][haloProp][searchID]

        return result

    def getSnapOffsets(self, searchID, type):
        """ Compute offsets within snapshot for a particular group/subgroup. """

        r = {}
        # load the length (by type) of this group/subgroup from the group catalog
        with h5py.File(self.gcfile, 'r') as f:
            r['lenType'] = f[type][type + 'LenType'][searchID, :]
            r['offsetType'] = f[type][type + 'OffsetType'][searchID, :]
            r['indices'] = (r['offsetType'], r['offsetType'] + r['lenType'])

        return r

    def getNumPart(self, file):
        """ Calculate number of particles of all types given a snapshot header. """
        nTypes = int(file['Config'].attrs['NTYPES'])

        nPart = np.zeros(nTypes, dtype=np.int64)
        for j in range(nTypes):
            nPart[j] = file['Header'].attrs['NumPart_Total'][j]

        return nPart

    def loadSubset(self, partType="dm", fields=None, subset=None, float32=False):
        """ Load a subset of fields for all particles/cells of a given partType.
            If offset and length specified, load only that subset of the partType.
            If mdi is specified, must be a list of integers of the same length as fields,
            giving for each field the multi-dimensional index (on the second dimension) to load.
              For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
              of y-Coordinates only, together with Masses.
            If sq is True, return a numpy array instead of a dict if len(fields)==1.
            If float32 is True, load any float64 datatype arrays directly as float32 (save memory). """
        result = {}

        ptNum = self.partTypeNum(partType)
        gName = "PartType" + str(ptNum)

        lengroup = subset['lenType'][ptNum]
        idx0, idx1 = subset['indices'][0][ptNum], subset['indices'][1][ptNum]
        result['count'] = lengroup

        with h5py.File(self.snapfile, 'r') as f:
            # if fields not specified, load everything
            if not fields:
                fields = list(f[gName].keys())

            for i, field in enumerate(fields):
                # verify existence
                if field not in f[gName].keys():
                    raise Exception("Particle type [" + str(ptNum) + "] does not have field [" + field + "]")

                # replace local length with global
                shape = list(f[gName][field].shape)
                shape[0] = lengroup

                # allocate within return dict
                dtype = f[gName][field].dtype
                if dtype == np.float64 and float32: dtype = np.float32
                result[field] = np.zeros(shape, dtype=dtype)
                result[field] = f[gName][field][idx0:idx1]

        return result

    def loadSubhalo(self, haloid, partType, fields=None):
        """ Load all particles/cells of one type for a specific subhalo
            (optionally restricted to a subset fields). """
        subset = self.getSnapOffsets(haloid, "Subhalo")
        return self.loadSubset(partType, fields, subset=subset)

    def loadHalo(self, haloid, partType, fields=None):
        """ Load all particles/cells of one type for a specific halo
            (optionally restricted to a subset fields). """
        subset = self.getSnapOffsets(haloid, "Group")
        return self.loadSubset(partType, fields, subset=subset)

    def partTypeNum(self, partType):
        """ Mapping between common names and numeric particle types. """
        if str(partType).isdigit():
            return int(partType)

        if str(partType).lower() in ['gas', 'cells']:
            return 0
        if str(partType).lower() in ['dm', 'darkmatter']:
            return 1
        if str(partType).lower() in ['tracer', 'tracers', 'tracermc', 'trmc']:
            return 3
        if str(partType).lower() in ['star', 'stars', 'stellar']:
            return 4  # only those with GFM_StellarFormationTime>0
        if str(partType).lower() in ['wind']:
            return 4  # only those with GFM_StellarFormationTime<0
        if str(partType).lower() in ['bh', 'bhs', 'blackhole', 'blackholes']:
            return 5

        raise Exception("Unknown particle type name.")
