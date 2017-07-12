"""

nchilada
========

Implements classes and functions for handling nchilada files.  You rarely
need to access this module directly as it will be invoked
automatically via pynbody.load.

"""

from __future__ import with_statement  # for py2.5
from __future__ import division

from .. import array, util
from .. import family
from .. import units
from .. import config, config_parser
from .. import chunk
from . import SimSnap
from . import namemapper

import struct
import os
import numpy as np
import gzip
import sys
import warnings
import copy
import types
import math
import xml.dom.minidom
import xdrlib

_name_map, _rev_name_map = namemapper.setup_name_maps('nchilada-name-mapping')
_translate_array_name = namemapper.name_map_function(_name_map, _rev_name_map)

_type_codes = map(np.dtype, [None, 'int8', 'uint8', 'int16', 'uint16',
                             'int32', 'uint32', 'int64', 'uint64',
                             'float32', 'float64'])


_max_buf = 1024 * 512


class NchiladaSnap(SimSnap):

    def _load_header(self, f):
        u = xdrlib.Unpacker(f.read(28))
        assert u.unpack_int() == 1062053
        t = u.unpack_double()
        highword, nbod, ndim, code = [u.unpack_int() for i in range(4)]
        return t, nbod, ndim, _type_codes[code]

    def _update_loadable_keys(self):
        self._loadable_keys_registry = {}
        d = self._loadable_keys_registry
        fams = self._dom_sim.getElementsByTagName('family')
        for f in sorted(fams, key=lambda x: str(x)):
            fam_name = f.attributes['name'].value
            our_fam = family.get_family(fam_name)
            d_f = {}
            for a in f.getElementsByTagName('attribute'):
                our_name = _translate_array_name(
                    a.attributes['name'].value, reverse=True)
                filename = os.path.join(
                    self._filename, a.attributes['link'].value)
                d_f[our_name] = filename
            d[our_fam] = d_f

    def _setup_slices(self, take=None):
        disk_family_slice = {}
        i = 0
        # for each family, find an array (any array) to determine length and
        # set up a logical map of particles on disk
        for f in sorted(self._loadable_keys_registry.keys()):
            ars = self._loadable_keys_registry[f]
            tf = open(ars.values()[0], 'rb')
            header_time, nbod, _, _ = self._load_header(tf)
            disk_family_slice[f] = slice(i, i + nbod)
            i += nbod
            self.properties['a'] = header_time
        self._load_control = chunk.LoadControl(
            disk_family_slice, _max_buf, take)
        self._family_slice = self._load_control.mem_family_slice
        self._num_particles = self._load_control.mem_num_particles

    def __init__(self, filename, take=None, paramfile=None):
        super(NchiladaSnap, self).__init__()
        self._dom_sim = xml.dom.minidom.parse(
            os.path.join(filename, "description.xml")).getElementsByTagName('simulation')[0]
        self._filename = filename
        self._update_loadable_keys()
        self._setup_slices(take=take)
        self._paramfilename = paramfile
        self._decorate()

    def loadable_keys(self, fam=None):
        if fam is not None:
            return self._loadable_keys_registry[fam].keys()
        else:
            loadable = None
            for f in self._loadable_keys_registry.itervalues():
                if loadable is None:
                    loadable = set(f.iterkeys())
                else:
                    loadable = loadable.intersection(f.iterkeys())
            return list(loadable)

    def _open_file_for_array(self, fam, array_name):
        fname = self._loadable_keys_registry[fam].get(array_name, None)
        if not fname:
            raise IOError, "No such array on disk"
        f = open(fname, 'rb')
        return f

    def _attempt_load_all_families(self, array_name):
        fams = self.families()
        universal_dtype = None
        universal_ndim = None
        if fams==[]:
            return
        for fam in fams:
            # this will raise an IOError propagating upwards if any family doesn't have the appropriate array
            _, nbod, ndim, dtype = self._load_header(self._open_file_for_array(fam, array_name))
            if universal_dtype is not None:
                if ndim!=universal_ndim:
                    raise IOError, "Mismatching dimensions for array"
                if dtype!=universal_dtype:
                    raise IOError, "Mismatching data type for array"
            universal_ndim, universal_dtype = ndim, dtype

        self._create_array(array_name,universal_ndim,universal_dtype,False)

        for fam in fams:
            self._load_array(array_name, fam)


    def _load_array(self, array_name, fam=None):
        if fam is None:
            self._attempt_load_all_families(array_name)
            return

        f = self._open_file_for_array(fam, array_name)

        _, nbod, ndim, dtype = self._load_header(f)
        if array_name not in self.keys():
            self._create_family_array(array_name, fam, ndim=ndim,dtype=dtype)
        r = self[fam][array_name]
        if units.has_units(r):
            r.convert_units(self._default_units_for(array_name))
        else:
            r.set_default_units()

        disk_dtype = dtype.newbyteorder('>')

        buf_shape = _max_buf
        if ndim > 1:
            buf_shape = (_max_buf, ndim)

        b = np.empty(buf_shape)

        # skip over min and max values (see issue #211)
        np.fromfile(f, dtype=disk_dtype, count=2 * ndim)

        for readlen, buf_index, mem_index in self._load_control.iterate(fam, fam):
            b = np.fromfile(f, dtype=disk_dtype, count=readlen * ndim)
            if ndim > 1:
                b = b.reshape((readlen, ndim))

            if mem_index is not None:
                r[mem_index] = b[buf_index]

    """
    def _write_array(self, array_name, fam=None) :
        if fam is None :
            fam = self.families()
        
        for f in fam :
            fname = self._loadable_keys_registry[fam][array_name]
            # to do: sort out what happens when this doesn't exist
            ar = self[fam][array_name]
            
            _, nbod, ndim, dtype = self._load_header(f)
            for readlen, buf_index, mem_index in self._load_control.iterate(fam, fam) :
                b = np.fromfile(f, dtype=disk_dtype, count=readlen*ndim)
                if ndim>1 : b = b.reshape((readlen, ndim))
                if mem_index is not None :
                    b[buf_index] = ar[mem_index]
                    f.seek(-readlen*ndim*disk_dtype.itemsize,1)
    """

    @staticmethod
    def _can_load(f):
        return os.path.isdir(f) and os.path.exists(os.path.join(f, "description.xml"))
