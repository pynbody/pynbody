"""Gravity Tree. Builds tree based on pkdgrav2"""

try :
    from .. import pkdgrav
except ImportError :
    pass # pkdgrav2 never works at the moment, and the warning below confuses/annoys people
    #import warnings
    #warnings.warn("Unable to import PKDGrav gravity solver. Most likely this means either that your installation is broken, or that you are running python inside the pynbody distribution directory, in which case python cannot see the installed version. However, it also doesn't matter unless you want to use the tree gravity solver.",RuntimeWarning)
    
from .. import config
import numpy as np

class GravTree:
    
    def __init__(self, pos, mass, rs, eps=None, leafsize=16):
        import pdb; pdb.set_trace()
        if config['tracktime']:
            import time
            start = time.clock()
            self.tree = pkdgrav.pkdPythonInitialize(pos, mass, eps, rs, int(leafsize))
            end = time.clock()
            if config['verbose']: print 'Tree build done in %5.3g s'%(end-start)
        else:
            self.tree = pkdgrav.pkdPythonInitialize(pos, mass, eps, rs, int(leafsize))

        self.derived = True
        self.flags = {'WRITEABLE':False}
        
    def calc(self, vec_pos, eps=None):
        accel = np.zeros((len(vec_pos),3))
        pot = np.zeros(len(vec_pos))
        if config['verbose']: print 'Calculating Gravity'
        if config['tracktime']:
            import time
            start = time.clock()
            pkdgrav.pkdPythonDoGravity(self.tree, accel, pot, theta=0.55)
            end = time.clock()
            if config['verbose']: print 'Gravity calculated in %5.3g s'%(end-start)
        else:
            pkdgrav.pkdPythonDoGravity(self.tree, accel, pot, theta=0.55)
        return accel, pot
        
    def __del__(self):
        if hasattr(self, 'tree'):
            pkdgrav.free(self.tree)
        
