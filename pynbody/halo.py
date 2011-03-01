import numpy as np
import weakref
import os.path
import glob
import re
from . import snapshot

class Halo(snapshot.IndexedSubSnap) :
    def __init__(self, halo_id, halo_catalogue, *args) :
        super(Halo, self).__init__(*args)
        self._halo_catalogue = halo_catalogue
        self._halo_id = halo_id
        self._descriptor = "halo_"+str(halo_id)
        self.props = {}

    def is_subhalo(self, otherhalo):
        return self._halo_catalogue.is_subhalo(self._halo_id, otherhalo._halo_id)
        
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

    def is_subhalo(self,childid,parentid) :
        if (childid in self._halos[parentid].props['children']) :
            return True
        else : return False

class AHFCatalogue(HaloCatalogue) :
    def __init__(self, f) :
        import os.path
        self._base = weakref.ref(f)
        HaloCatalogue.__init__(self)
        self._load_ahf_halos(glob.glob(f._filename+'*z*halos')[0])
        self._load_ahf_substructure(glob.glob(f._filename+'*z*substructure')[0])
        self._particles_loaded = False

    def _get_halo(self, i) :
        if self.base is None :
            raise RuntimeError, "Parent SimSnap has been deleted"

        # XXX how do we lazy load particles (and only do it once)?
        # in other words, WHEN do we lazy load particles?
        if (not self._particles_loaded) :
            self._load_ahf_particles(glob.glob(self.base._filename+'*z*particles')[0])
            self._particles_loaded = True
            
        x = self.base[self._halos[i]]
        x._descriptor = "halo_"+str(i)
        return x

    @property
    def base(self) :
        return self._base()

    def _load_ahf_particles(self,filename) :
        f = open(filename)
        for h in len(self._halos) :
            n = int(f.readline())  # number of particles in halo
            self._halos[h+1]._slice = sorted([ int(f.readline().split()[0]) for i in xrange(n) ])

    def _load_ahf_halos(self,filename) :
        f = open(filename)
        # get all the property names from the first, commented line
        keys = [re.sub('\([0-9]*\)','',field) for field in f.readline().split()[1:]]
        for h, line in enumerate(f) :
            # create halos, one off to leave 0 for everything that's not part of any halo
            self._halos[h+1] = Halo( h+1, self, self.base,[])
            values = [float(x) for x in line.split()]
            # XXX Unit issues!  AHF uses distances in Mpc/h, possibly masses as well
            # XXX trouble with 'a':  both expansion factor and longest axis
            # XXX need to alias 'Mvir' to 'mass'?
            for i,key in enumerate(keys) : self._halos[h+1].props[key] = values[i]

    def _load_ahf_substructure(self,filename) :
        f = open(filename)
        nhalos = int(f.readline())  # number of halos?  no, some crazy number
                                    # that we will ignore
        for i in xrange(len(self._halos)) :
            haloid, nsubhalos = [int(x) for x in f.readline().split()]
            self._halos[haloid+1].props['children'] = [int(x) for x in f.readline().split()]

    @staticmethod
    def _can_load(f) :
        if os.path.exists(glob.glob(f._filename+'*z*particles')[0]) :
            return True
        else :
            return False

    def _run_ahf(self) :
        #build units file
        print "not ready"
        # make input file

        # determine parallel possibilities

        # run it

class AmigaGrpCatalogue(HaloCatalogue) :
    def __init__(self, f) :
        f['amiga.grp'] # trigger lazy-loading and/or kick up a fuss if unavailable
        self._base = weakref.ref(f)
        self._halos = {}
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


_halo_classes = [AHFCatalogue,AmigaGrpCatalogue]
