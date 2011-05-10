"""KDTree. Provides access to nearest neighbour lists and smoothing lengths."""

import kdmain

class KDTree:
    PROPID_HSM = 1

    def __init__(self, pos, leafsize=32):
        self.kdtree = kdmain.init(pos, int(leafsize))
        self.propid = {'hsm': self.PROPID_HSM}

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

    def populate(self, dest, property, nn=None):
        if nn is None: nn = 64
        smx = kdmain.nn_start(self.kdtree, int(nn))
        kdmain.populate(self.kdtree, smx, dest, int(self.propid[property]))
        kdmain.nn_stop(self.kdtree, smx)

    def __del__(self):
        if hasattr(self, 'kdtree'):
            kdmain.free(self.kdtree)
        
