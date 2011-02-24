import numpy as np
import weakref
import os.path
import glob

class HaloCatalogue(object) :
    def __init__(self) :
        self._halos = {}

    def __getitem__(self, i) :
        if self._halos.has_key(i) :  # and self._halos[i]() is not None :
            return self._halos[i] # ()
        else :
            h = self._get_halo(i)
            self._halos[i] = h # weakref.ref(h)
            return h


class AmigaCatalogue(HaloCatalogue) :
    def __init__(self, f) :
        import os.path
        self._load_amiga_particles(glob.glob(f._filename+'*z*particles')[0])
        
        self._base = weakref.ref(f)
        HaloCatalogue.__init__(self)

    def _get_halo(self, i) :
        if self.base is None :
            raise RuntimeError, "Parent SimSnap has been deleted"

        x = self.base[self._halos_particles[i]]
        x._descriptor = "halo_"+str(i)
        return x

    @property
    def base(self) :
        return self._base()

    def _load_amiga_particles(self,filename) :
        f = open(filename)
        self._halos_particles = []
        while f :
            try:
                n = int(f.readline())  # number of particles in halo
            except ValueError:
                break
            
            #temph = []#np.array([])
            temph = [ int(f.readline().split()[0]) for i in xrange(n) ]
            #for i in xrange(n) :
             #   temph.append(int(f.readline().split()[0]))
            self._halos_particles.append(sorted(temph))
    
    @staticmethod
    def _can_load(f) :
        if os.path.exists(glob.glob(f._filename+'*z*particles')[0]) :
            return True
        else :
            return False

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


_halo_classes = [AmigaCatalogue,AmigaGrpCatalogue]
