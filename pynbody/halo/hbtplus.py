import h5py
import numpy as np
import pathlib
import re
from numpy._typing import NDArray

from . import HaloCatalogue, HaloParticleIndices
from .details import number_mapping

class HBTPlusCatalogue(HaloCatalogue):

    def __init__(self, sim, halo_numbers=None, hbt_filename=None):
        """Initialize a HBTPlusCatalogue object.

        Parameters
        ----------
        sim : SimSnap
            The simulation snapshot to which this catalogue applies.
        halo_numbers : str, optional
            How to number the halos. If None (default), use a zero-based indexing.
            If 'track', use the TrackId from the catalogue.
            If 'length-order', order by Nbound (descending), similar to the AHF option of the same name
        hbt_filename : str, optional
            The filename of the HBTPlus catalogue. If None (default), attempt to find
            the file automatically.
        """
        if hbt_filename is None:
            hbt_filename = self._infer_hbt_filename(sim)

        self._file = h5py.File(hbt_filename, 'r')

        num_halos = int(self._file["NumberOfSubhalosInAllFiles"][0])
        if int(self._file["NumberOfFiles"][0])>1:
            raise ValueError("HBTPlusCatalogue does not support multi-file catalogues")

        if halo_numbers is None:
            number_mapper = number_mapping.SimpleHaloNumberMapper(0, num_halos)
        elif halo_numbers == 'track':
            number_mapper = number_mapping.create_halo_number_mapper(self._file["Subhalos"]["TrackId"])
        elif halo_numbers == 'length-order':
            osort = np.argsort(np.argsort(-self._file["Subhalos"]["Nbound"][:]))
            number_mapper = number_mapping.NonMonotonicHaloNumberMapper(osort, ordering=True, start_index=0)
        else:
            raise ValueError(f"Invalid value for halo_numbers: {halo_numbers}")
        super().__init__(sim, number_mapper)

    @classmethod
    def _infer_hbt_filename(cls, sim):
        sim_filename: pathlib.Path  = sim.filename
        try:
            snap_num = int(re.search(r'_(\d+)', sim_filename.name).group(1))
        except AttributeError:
            raise FileNotFoundError(f'Could not infer HBT filename from {sim_filename}. '
                                    f'Try passing hbt_filename explicitly.') from None

        candidate_paths = [sim_filename.with_name(f'SubSnap_{snap_num:03d}.0.hdf5'),
                            sim_filename.parent / f'{snap_num:03d}' / f'SubSnap_{snap_num:03d}.0.hdf5']

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path

        raise FileNotFoundError(f'Could not find HBTPlus catalogue for {sim_filename}. Try passing hbt_filename explicitly.')


    def _get_particle_indices_one_halo(self, halo_number) -> NDArray[int]:
        self._init_iord_to_fpos()

        iords = self._file["SubhaloParticles"][self.number_mapper.number_to_index(halo_number)]

        return self._iord_to_fpos.map_ignoring_order(iords)

    #def _get_all_particle_indices(self) -> HaloParticleIndices | tuple[np.ndarray, np.ndarray]:
    #    pass

    def get_properties_one_halo(self, halo_number) -> dict:
        index = self.number_mapper.number_to_index(halo_number)
        result = {}
        for k in self._file["Subhalos"].dtype.names:
            result[k] = self._file["Subhalos"][k][index]
        return result

    def get_properties_all_halos(self, with_units=True) -> dict:
        return super().get_properties_all_halos(with_units)



    @classmethod
    def _can_load(cls):
        return False






