"""Gravity Tree. Builds tree based on pkdgrav2"""

from .. import grav
from .. import config
import numpy as np

class GravTree:
    
    def __init__(self, pos, mass, eps=None, leafsize=16):
        if config['tracktime']:
            import time
            start = time.clock()
            self.tree = grav.treeinit(pos, mass, int(leafsize),int(len(mass)),np.float32(eps))
            end = time.clock()
            if config['verbose']: print 'Tree build done in %5.3g s'%(end-start)
        else:
            self.tree = grav.treeinit(pos, mass, int(leafsize),int(len(mass)),np.float32(eps))

        self.derived = True
        self.flags = {'WRITEABLE':False}
        
    def calc(self, vec_pos, eps=None):
        accel = np.zeros((len(vec_pos),3))
        pot = np.zeros(len(vec_pos))
        if config['verbose']: print 'Calculating Gravity'
        if config['tracktime']:
            import time
            start = time.clock()
            grav.calculate(self.tree, np.array(vec_pos), accel, pot, eps, len(vec_pos))
            end = time.clock()
            if config['verbose']: print 'Gravity calculated in %5.3g s'%(end-start)
        else:
            grav.calculate(self.tree, np.array(vec_pos), accel, pot, eps)
        return accel, pot
        
    def __del__(self):
        if hasattr(self, 'tree'):
            grav.free(self.tree)
        
