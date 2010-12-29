import numpy as np
import weakref

class HaloCatalogue(object) :
    def __init__(self) :
	self._halos = {}
	
    def __getitem__(self, i) :
	if self._halos.has_key(i) and self._halos[i]() is not None :
	    return self._halos[i]()
	else :
	    h = self._get_halo(i)
	    self._halos[i] = weakref.ref(h)
	    return h


class AmigaGrpCatalogue(HaloCatalogue) :
    def __init__(self, f) :
	f['amiga.grp'] # trigger lazy-loading and/or kick up a fuss if unavailable
	self._base = weakref.ref(f)
	HaloCatalogue.__init__(self)
	
    def _get_halo(self, i) :
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
