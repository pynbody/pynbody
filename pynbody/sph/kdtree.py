"""

KDTree
======

Provides access to nearest neighbour lists and smoothing lengths.

"""
from . import kdmain
from .. import config
from .. import array as ar
import numpy as np
import time
import logging
import weakref
import warnings

logger = logging.getLogger('pynbody.sph.kdtree')


class KDTree(object):
    PROPID_HSM = 1
    PROPID_RHO = 2
    PROPID_QTYMEAN_1D = 3
    PROPID_QTYMEAN_ND = 4
    PROPID_QTYDISP_1D = 5
    PROPID_QTYDISP_ND = 6

    def __init__(self, pos, mass, leafsize=32, boxsize=None):
        self.kdtree = kdmain.init(pos, mass, int(leafsize))
        self.derived = True
        self.boxsize=boxsize
        self._pos = pos
        self.s_len = len(pos)
        self.flags = {'WRITEABLE': False}

    def nn(self, nn=None):
        if nn is None:
            nn = 64

        smx = kdmain.nn_start(self.kdtree, int(nn), 1, self.boxsize)

        while True:
            nbr_list = kdmain.nn_next(self.kdtree, smx)
            if nbr_list is None:
                break
            yield nbr_list

        kdmain.nn_stop(self.kdtree, smx)

    def all_nn(self, nn=None):
        return [x for x in self.nn(nn)]

    @staticmethod
    def array_name_to_id(name):
        if name=="smooth":
            return 0
        elif name=="rho":
            return 1
        elif name=="mass":
            return 2
        elif name=="qty":
            return 3
        elif name=="qty_sm":
            return 4
        else :
            raise ValueError, "Unknown KDTree array"

    def set_array_ref(self, name, ar) :
        if self.array_name_to_id(name)<3:
            if not np.issubdtype(self._pos.dtype, ar.dtype):
                raise TypeError("KDTree requires matching dtypes for %s (%s) and pos (%s) arrays"%(name, ar.dtype, self._pos.dtype))
        kdmain.set_arrayref(self.kdtree,self.array_name_to_id(name),ar)
        assert self.get_array_ref(name) is ar

    def get_array_ref(self, name) :
        return kdmain.get_arrayref(self.kdtree,self.array_name_to_id(name))


    def smooth_operation_to_id(self,name):
        if name=="hsm":
            return self.PROPID_HSM
        elif name=="rho":
            return self.PROPID_RHO
        elif name=="qty_mean":
            input_array = self.get_array_ref('qty')
            if len(input_array.shape)==1:
                return self.PROPID_QTYMEAN_1D
            elif len(input_array.shape)==2:
                if input_array.shape[1]!=3:
                    raise ValueError, "Currently only able to smooth 3D or 1D arrays"
                return self.PROPID_QTYMEAN_ND
        elif name=="qty_disp":
            input_array = self.get_array_ref('qty')
            if len(input_array.shape)==1:
                return self.PROPID_QTYDISP_1D
            elif len(input_array.shape)==2:
                if input_array.shape[1]!=3:
                    raise ValueError, "Currently only able to smooth 3D or 1D arrays"
                return self.PROPID_QTYDISP_ND
        else:
            raise ValueError, "Unknown smoothing request %s"%name



    def populate(self, mode, nn):
        from . import _thread_map

        n_proc=config['number_of_threads']

        if kdmain.has_threading() is False and n_proc>1:
            n_proc=1
            warnings.warn("Pynbody is configured to use threading for the KDTree, but pthread support was not available during compilation. Reverting to single thread.", RuntimeWarning)

        if nn is None:
            nn = 64

        smx = kdmain.nn_start(self.kdtree, int(nn), self.boxsize)

        propid = self.smooth_operation_to_id(mode)

        if propid==self.PROPID_HSM:
            kdmain.domain_decomposition(self.kdtree,n_proc)

        if n_proc==1 :
            kdmain.populate(self.kdtree,smx,propid,0)
        else :
            _thread_map(kdmain.populate,[self.kdtree]*n_proc,[smx]*n_proc,[propid]*n_proc,range(0,n_proc))

        kdmain.nn_stop(self.kdtree, smx)

    def sph_mean(self, array, nsmooth=64):
        """Calculate the SPH mean of a simulation array.
        """
        output=np.empty_like(array)

        if hasattr(array,'units'):
            output = output.view(ar.SimArray)
            output.units=array.units

        self.set_array_ref('qty', array)
        self.set_array_ref('qty_sm', output)

        logger.info("Smoothing array with %d nearest neighbours"%nsmooth)
        start = time.time()
        self.populate('qty_mean',nsmooth)
        end = time.time()

        logger.info('SPH smooth done in %5.3g s' % (end - start))

        return output

    def sph_dispersion(self, array, nsmooth=64):
        output=np.empty_like(array)
        if hasattr(array,'units'):
            output = output.view(ar.SimArray)
            output.units=array.units

        self.set_array_ref('qty', array)
        self.set_array_ref('qty_sm', output)

        logger.info("Getting dispersion of array with %d nearest neighbours"%nsmooth)
        start = time.time()
        self.populate('qty_disp',nsmooth)
        end = time.time()

        logger.info('SPH dispersion done in %5.3g s' % (end - start))

        return output


    def __del__(self):
        if hasattr(self, 'kdtree'):
            kdmain.free(self.kdtree)
