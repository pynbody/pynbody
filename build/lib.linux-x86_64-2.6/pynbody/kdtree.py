"""

KDTree 
======

Provides access to nearest neighbour lists and smoothing lengths.

"""
from . import kdmain

class KDTree:
    PROPID_HSM = 1
    PROPID_RHO = 2
    PROPID_VMEAN = 3
    PROPID_VDISP = 4
    
    def __init__(self, pos, vel, mass, leafsize=32):
        self.kdtree = kdmain.init(pos, vel, mass, int(leafsize))
        self.propid = {'hsm': self.PROPID_HSM,
                       'rho': self.PROPID_RHO,
                       'v_mean': self.PROPID_VMEAN,
                       'v_disp': self.PROPID_VDISP}
        self.derived = True
        self.flags = {'WRITEABLE':False}
        
    def nn(self, nn=None):
        if nn is None: nn = 64

        smx = kdmain.nn_start(self.kdtree, int(nn))

        while True:
            nbr_list = kdmain.nn_next(self.kdtree, smx)
            if nbr_list is None: break
            yield nbr_list

        kdmain.nn_stop(self.kdtree, smx)

    def all_nn(self, nn=None):
        return [x for x in self.nn(nn)]

    def populate(self, dest, property, nn=None, smooth=None, rho=None):
        if nn is None: nn = 64
        if (smooth is not None) and (rho is not None):
            smx = kdmain.nn_start(self.kdtree, int(nn), smooth, rho)
        elif smooth is not None:
            smx = kdmain.nn_start(self.kdtree, int(nn), smooth)
        else :
            smx = kdmain.nn_start(self.kdtree, int(nn))
            
        kdmain.populate(self.kdtree, smx, dest, int(self.propid[property]))
        kdmain.nn_stop(self.kdtree, smx)
        
        
    def __del__(self):
        if hasattr(self, 'kdtree'):
            kdmain.free(self.kdtree)
        
