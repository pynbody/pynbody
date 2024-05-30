"""
Supports loading of ASCII snapshots. The format is a simple text file with a header line defining columns.

"""

import os
import pathlib

import numpy as np

from .. import chunk, family, units
from . import SimSnap

_max_buf = 1024 * 512


class AsciiSnap(SimSnap):
    """Supports loading of ASCII snapshots. The format is a simple text file with a header line defining columns.

    For an ASCII snapshot to be identified automatically using :func:`pynbody.load`, the file must have a `.txt`
    extension.

    All particles are assumed to be dark matter.
    """

    def _setup_slices(self, num_lines, take=None):
        disk_family_slice = {family.dm: slice(0,num_lines)}
        self._load_control = chunk.LoadControl(
            disk_family_slice, _max_buf, take)
        self._family_slice = self._load_control.mem_family_slice
        self._num_particles = self._load_control.mem_num_particles

    def __init__(self, filename, take=None, header=None):
        super().__init__()

        num_particles = 0

        self._header = header

        with open(filename) as f:
            for l in f:
                if not header:
                    header = l
                else:
                    num_particles+=1

        self._loadable_keys = header.split()
        self._filename = filename
        self._file_units_system = [units.Unit("G"), units.Unit("kpc"), units.Unit("Msol")]
        self._setup_slices(num_particles,take=take)
        self._decorate()

    def loadable_keys(self, fam=None):
        return self._loadable_keys

    def _load_arrays(self, array_name_list):
        with open(self._filename) as f:

            if not self._header: f.readline()
            rs=[]
            for array_name in array_name_list:
                self._create_array(array_name, ndim=1)

            rs = [self[array_name] for array_name in array_name_list]
            cols = [self._loadable_keys.index(array_name) for array_name in array_name_list]
            ncols = len(self._loadable_keys)

            buf_shape = _max_buf

            b = np.empty(buf_shape)

            for readlen, buf_index, mem_index in self._load_control.iterate([family.dm],[family.dm]):
                b = np.fromfile(f, count=readlen*ncols, sep=' ').reshape((-1,ncols))

                if mem_index is not None:
                    for r,col in zip(rs,cols):
                        r[mem_index] = b[buf_index,col]

    def _load_array(self, array_name, fam=None):

        if fam is not None:
            raise OSError("Arrays only loadable at snapshot, not family level")

        ars = [array_name]

        if array_name not in self._loadable_keys:
            ars =  self._array_name_ND_to_1D(array_name)
            for array_name_i in ars:
                if array_name_i not in self._loadable_keys:
                    raise OSError("No such array on disk")

        self._load_arrays(ars)

    @classmethod
    def _can_load(cls, f: pathlib.Path):
        return f.exists() and f.suffix == '.txt'
