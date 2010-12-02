import numpy as np
import weakref

class HaloCatalogue(object) :
    pass


class AmigaGrpCatalogue(HaloCatalogue) :
    def __init__(self, f) :
	f['amiga.grp'] # trigger lazy-loading and/or kick up a fuss if unavailable
	self._base = weakref.ref(f)
	
    def __getitem__(self, i) :
	if self.base is None :
	    raise RuntimeError, "Parent SimSnap has been deleted"
	
	x = self.base[np.where(self.base['amiga.grp']==i)]
	x._descriptor = "halo_"+str(i)
	return x

    @property
    def base(self) :
	return self._base()
    
    @staticmethod
    def _can_load(f) :
	try :
	    f['amiga.grp']
	    return True
	except KeyError :
	    return False
    

_halo_classes = [AmigaGrpCatalogue]
