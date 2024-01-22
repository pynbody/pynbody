from typing import Union

import numpy as np
from numpy import typing as npt


class HaloNumberMapper:
    def __init__(self, halo_numbers: npt.NDArray[int]):
        """A HaloNumberMapper maps halo numbers (arbitrary unique integers) to zero-based indices within a halo catalogue.

        The halo_numbers array must be monotonically increasing.
        """
        halo_numbers = np.asarray(halo_numbers)
        assert np.all(np.diff(halo_numbers) > 0)
        self.halo_numbers = halo_numbers

    def number_to_index(self, halo_number: Union[int, npt.NDArray[int]]) -> Union[int, npt.NDArray[int]]:
        """Convert a halo number to a zero-based index """
        halo_number = np.asarray(halo_number)
        halo_index = np.searchsorted(self.halo_numbers, halo_number)
        if isinstance(halo_index, np.ndarray):
            missing_halo_mask = (halo_index >= len(self.halo_numbers)) | (self.halo_numbers[halo_index] != halo_number)
            missing_halo_numbers = halo_number[missing_halo_mask]

            if missing_halo_numbers.size > 0:
                raise KeyError(f"No such halos: {missing_halo_numbers}")
        else:
            if halo_index >= len(self.halo_numbers) or self.halo_numbers[halo_index] != halo_number:
                raise KeyError(f"No such halo {halo_number}")
        return halo_index

    def index_to_number(self, halo_index: Union[int, npt.NDArray[int]]) -> Union[int, npt.NDArray[int]]:
        """Convert a zero-based offset to a halo number"""
        return self.halo_numbers[halo_index]

    def __len__(self):
        """Returns the number of halos in the catalogue"""
        return len(self.halo_numbers)

    def __iter__(self):
        """Iterates over the available halo numbers"""
        yield from self.halo_numbers

    def __repr__(self):
        return f"<{self.__class__.__name__} len={len(self)}>"

    @property
    def all_numbers(self):
        return self.halo_numbers


class SimpleHaloNumberMapper(HaloNumberMapper):
    def __init__(self, zero_offset, num_halos):
        """A HaloNumberMapper where the relationship between halo numbers and indices is simply an offset"""
        self.zero_offset = zero_offset
        self.num_halos = num_halos

    def number_to_index(self, halo_number: Union[int, npt.NDArray[int]]) -> Union[int, npt.NDArray[int]]:
        if hasattr(halo_number, "__len__"):
            halo_number = np.asarray(halo_number)
            index = halo_number - self.zero_offset
            missing_halo_mask = (index < 0) | (index >= self.num_halos)
            missing_halo_numbers = np.asarray(halo_number)[missing_halo_mask]

            if missing_halo_numbers.size > 0:
                raise KeyError(f"No such halos: {missing_halo_numbers}")
        else:
            index = halo_number - self.zero_offset
            if index< 0 or index >= self.num_halos:
                raise KeyError(f"No such halo {halo_number}")
        return index

    def index_to_number(self, halo_index: Union[int, npt.NDArray[int]]) -> Union[int, npt.NDArray[int]]:
        if hasattr(halo_index, "__len__"):
            halo_index = np.asarray(halo_index)
            missing_halo_mask = (halo_index < 0) | (halo_index >= self.num_halos)
            missing_halo_index = halo_index[missing_halo_mask]

            if missing_halo_index.size > 0:
                raise IndexError(f"No such halos: {missing_halo_index}")
        else:
            if halo_index < 0 or halo_index >= self.num_halos:
                raise IndexError(f"No such halo {halo_index}")

        return halo_index + self.zero_offset

    def __len__(self):
        return self.num_halos

    def __iter__(self):
        yield from range(self.zero_offset, self.zero_offset + self.num_halos)

    @property
    def all_numbers(self):
        return np.arange(self.zero_offset, self.zero_offset + self.num_halos)
