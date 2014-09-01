"""

KDTree
======

Provides access to nearest neighbour lists and smoothing lengths.

"""
from . import kdmain
from .. import config

class KDTree(object):
    PROPID_HSM = 1
    PROPID_RHO = 2
    PROPID_VMEAN = 3
    PROPID_VDISP = 4

    def __init__(self, pos, mass, leafsize=32):
        self.kdtree = kdmain.init(pos, mass, int(leafsize))
        self.propid = {'hsm': self.PROPID_HSM,
                       'rho': self.PROPID_RHO,
                       'v_mean': self.PROPID_VMEAN,
                       'v_disp': self.PROPID_VDISP}
        self.derived = True
        self.s_len = len(pos)
        self.flags = {'WRITEABLE': False}

    def nn(self, nn=None):
        if nn is None:
            nn = 64

        smx = kdmain.nn_start(self.kdtree, int(nn), 1)

        while True:
            nbr_list = kdmain.nn_next(self.kdtree, smx)
            if nbr_list is None:
                break
            yield nbr_list

        kdmain.nn_stop(self.kdtree, smx)

    def all_nn(self, nn=None):
        return [x for x in self.nn(nn)]

    def set_array_ref(self, name, ar) :
        if name=="smooth":
            kdmain.set_arrayref(self.kdtree,0,ar)
        elif name=="rho":
            kdmain.set_arrayref(self.kdtree,1,ar)
        elif name=="mass":
            kdmain.set_arrayref(self.kdtree,2,ar)
        else :
            raise ValueError, "Unknown KDtree array "+name

    def populate(self, mode, nn):
        from . import _thread_map

        n_proc=config['number_of_threads']

        if nn is None:
            nn = 64

        smx = kdmain.nn_start(self.kdtree, int(nn))

        propid = int(self.propid[mode])

        if propid==self.PROPID_HSM:
            kdmain.domain_decomposition(self.kdtree,n_proc)

        if n_proc==1 :
            kdmain.populate(self.kdtree,smx,propid,0)
        else :
            _thread_map(kdmain.populate,[self.kdtree]*n_proc,[smx]*n_proc,[propid]*n_proc,range(0,n_proc))

        kdmain.nn_stop(self.kdtree, smx)

    def __del__(self):
        if hasattr(self, 'kdtree'):
            kdmain.free(self.kdtree)
