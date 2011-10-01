"""Gravity Tree. Builds tree based on pkdgrav2"""

try :
    from .. import grav
except ImportError :
    import warnings
    warnings.warn("Unable to import PKDGrav gravity solver. Most likely this means either that your installation is broken, or that you are running python inside the pynbody distribution directory, in which case python cannot see the installed version. However, it also doesn't matter unless you want to use the tree gravity solver.",RuntimeWarning)
    
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
        
