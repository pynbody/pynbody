"""Support for subhalo catalogues, which are effectively views on a parent halo catalogue"""

import weakref

from numpy.typing import NDArray

from . import HaloCatalogue
from .details import number_mapping


class SubhaloCatalogue(HaloCatalogue):
    """A class that represents the subhalos of a single halo, retrievable in a zero-indexed manner"""
    def __init__(self, full_halo_catalogue: HaloCatalogue, subhalo_numbers: NDArray[int]):
        self._full_halo_catalogue_weakref = weakref.ref(full_halo_catalogue)
        self._subhalo_numbers = subhalo_numbers
        super().__init__(full_halo_catalogue.base,
                         number_mapping.SimpleHaloNumberMapper(0, len(subhalo_numbers)))

    @property
    def _full_halo_catalogue(self) -> HaloCatalogue:
        hc = self._full_halo_catalogue_weakref()
        if hc is None:
            raise ValueError("The underlying halo catalogue has been deleted")
        return hc

    def load_all(self):
        self._full_halo_catalogue.load_all()

    def get_group_array(self, family=None, use_index=False, fill_value=-1):
        raise RuntimeError("It is not possible to retrieve the group array of a subhalo catalogue")

    def physical_units(self, distance='kpc', velocity='km s^-1', mass='Msol', persistent=True, convert_parent=False):
        self._full_halo_catalogue.physical_units(distance, velocity, mass, persistent, convert_parent)

    def _get_halo(self, i):
        return self._full_halo_catalogue._get_halo(self._subhalo_numbers[i])

    def load_copy(self, i):
        return self._full_halo_catalogue.load_copy(self._subhalo_numbers[i])
