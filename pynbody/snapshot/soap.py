"""
Implements reading SOAP halo catalogues, as produced from SWIFT simulations (e.g. FLAMINGO, COLIBRE).

SOAP (Spherical Overdensity and Aperture Processor) catalogues are tables of properties, one row per subhalo, computed
in a number of ways: for example the bound subhalo (``BoundSubhalo/...``), spheres of fixed overdensity
(``SO/200_crit/...``), and fixed apertures (``ExclusiveSphere/50kpc/...``). They can be opened without the snapshot
from which they were computed, so pynbody presents them as a snapshot whose "particles" are the subhalos, all in the
family ``halo``:

>>> cat = pynbody.load("halo_properties_0077.hdf5")
>>> cat.halo
<FamilySubSnap "halo_properties_0077.hdf5::halo" len=952>
>>> cat['SO/200_crit/TotalMass'].in_units('Msol')
SimArray([4.155e+12, 1.8475e+12, ...], dtype=float32, 'Msol')

A few properties are given pynbody's standard names, as specified by the config.ini section ``[soap-name-mapping]``:
by default ``pos`` is the halo centre (``InputHalos/HaloCentre``), ``vel`` and ``mass`` are those of the bound
subhalo, and ``iord`` is the HBT+ ``TrackId``, which identifies a subhalo consistently across snapshots. All other
properties are available under their path in the HDF5 file. The subhalos are given in the order they appear in the
file, so that the indices stored in ``SOAP/HostHaloIndex`` remain valid (unless only part of the catalogue is loaded).
Note that SOAP computes some properties (such as those in ``SO/...``) only for central subhalos, giving zero for
satellites; ``InputHalos/IsCentral`` says which is which.

As for SWIFT snapshots, ``take_region`` loads only the subhalos in cells (of the catalogue's spatial index) that
overlap a region; and ``remote_dir`` reads a catalogue from an hdfstream server.

.. versionadded:: 2.8.0

"""

from .. import family
from ..util import hdf_bulk_read
from .swift import SwiftMultiFileManager, SwiftSnap

_halo_group = "/"
"""The 'particle group' in which the properties of all subhalos lie, which is the root of the file"""


class SOAPMultiFileManager(SwiftMultiFileManager):
    _size_from_hdf5_key = "InputHalos/HaloCatalogueIndex"

    def __init__(self, filename, mode='r', **kwargs):
        super().__init__(filename, mode, **kwargs)
        # SOAP compresses almost all its datasets with filters that only libhdf5 can decode, so reading directly
        # (see pynbody.util.hdf_bulk_read) would never be possible, and would only warn that it is not
        self._bulk_reader = hdf_bulk_read.BulkReader(enabled=False)

    def _read_cell_metadata(self, h1):
        num_halos = int(h1["Header"].attrs["NumSubhalos_Total"][0])
        self._cells = {_halo_group: self._read_cell_metadata_for_group(h1, "Subhalos", num_halos,
                                                                        self._get_num_files(h1))}
        self._cell_centres = h1["Cells/Centres"][...]

    def get_unit_attrs(self):
        return self[0].parent['Units'].attrs

    def get_header_attrs(self):
        return self[0].parent['Header'].attrs


class SOAPSnap(SwiftSnap):
    """Reads a SOAP halo catalogue, in which each "particle" is a subhalo, in the family ``halo``.

    See the documentation of :mod:`pynbody.snapshot.soap` for more information."""

    _multifile_manager_class = SOAPMultiFileManager
    _readable_hdf5_test_key = "Header"
    _readable_hdf5_test_attr = "Header", "OutputType", "SOAP"
    _namemapper_config_section = 'soap-name-mapping'
    _position_hdf_name = "InputHalos/HaloCentre"

    _metadata_groups = {'Cells', 'Code', 'Cosmology', 'Header', 'Parameters', 'PhysicalConstants', 'SWIFT',
                        'SubgridScheme', 'Units'}
    """Groups in a SOAP file that do not hold properties of the subhalos"""

    def _init_family_map(self):
        self._family_to_group_map = {family.get_family('halo'): [_halo_group]}

    def _all_hdf_groups(self):
        yield from self._hdf_files.iter_particle_groups_with_name(_halo_group)

    def _have_softening_for_particle_group(self, particle_group):
        return False

    def _get_hdf_dataset(self, particle_group, hdf_name):
        # SOAP files have no header masses or softenings to emulate, unlike the snapshots for which the base class
        # is written; and both h5py and hdfstream resolve a path to a nested dataset
        return particle_group[hdf_name]

    def _get_hdf_allarray_keys(self, group):
        """Return the paths of all datasets in *group* (the root of a SOAP file) that hold a property of each subhalo"""
        num_halos = group[self._hdf_files._size_from_hdf5_key].shape[0]
        keys = []

        def _append_if_property(path, obj):
            if not hasattr(obj, 'keys') and 1 <= len(obj.shape) <= 2 and obj.shape[0] == num_halos:
                keys.append(path)

        for name in group.keys():
            if name in self._metadata_groups:
                continue
            obj = group[name]
            if hasattr(obj, 'keys'):
                obj.visititems(lambda path, sub_obj: _append_if_property(name + "/" + path, sub_obj))
            else:
                _append_if_property(name, obj)
        return keys

    def _get_swift_header_attrs(self):
        # SOAP keeps a copy of the header of the snapshot from which it was computed, which (unlike its own header)
        # gives the time
        return self._hdf_files.get_file0_root()['SWIFT/Header'].attrs

    def halos(self, *args, **kwargs):
        raise NotImplementedError("A SOAP catalogue is itself a catalogue of halos; halos() is not yet supported")
