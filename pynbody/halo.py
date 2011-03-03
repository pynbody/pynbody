import numpy as np
import weakref
import os.path
import glob
import re
import copy
from . import snapshot

class Halo(snapshot.IndexedSubSnap) :
    def __init__(self, halo_id, halo_catalogue, *args) :
        super(Halo, self).__init__(*args)
        self._halo_catalogue = halo_catalogue
        self._halo_id = halo_id
        self._descriptor = "halo_"+str(halo_id)
        self.properties = copy.copy(self.properties)

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
        if (childid in self._halos[parentid].properties['children']) :
            return True
        else : return False

class AHFCatalogue(HaloCatalogue) :
    def __init__(self, sim) :
        import os.path
        if not self._can_load(sim) :
            self._run_ahf(sim)
        self._base = weakref.ref(sim)
        HaloCatalogue.__init__(self)
        self._load_ahf_particles(glob.glob(self.base._filename+'*z*particles')[0])
        self._load_ahf_halos(glob.glob(sim._filename+'*z*halos')[0])
        self._load_ahf_substructure(glob.glob(sim._filename+'*z*substructure')[0])

    def _get_halo(self, i) :
        if self.base is None :
            raise RuntimeError, "Parent SimSnap has been deleted"

        # XXX how do we lazy load particles (and only do it once)?
        # in other words, WHEN do we lazy load particles?
        if (not self._particles_loaded) :
            self._particles_loaded = True
            
        x = self.base[self._halos[i]]
        x._descriptor = "halo_"+str(i)
        return x

    @property
    def base(self) :
        return self._base()

    def _load_ahf_particles(self,filename) :
        f = open(filename)
        # tried readlines, which is fast, but the time is spent in the
        # for loop below, so sticking with this (hopefully) more readable 
        nhalos=int(f.readline())
        for h in xrange(nhalos) :
            nparts = int(f.readline())
            keys = {}
            for i in xrange(nparts) : keys[int(f.readline().split()[0])]=1
            self._halos[h+1] = Halo( h+1, self, self.base, sorted(keys.keys()))
        f.close()

    def _load_ahf_halos(self,filename) :
        f = open(filename)
        # get all the property names from the first, commented line
        # remove (#)
        keys = [re.sub('\([0-9]*\)','',field)
                for field in f.readline().split()]
        # provide translations
        for i,key in enumerate(keys) :
            if(key == '#npart') : keys[i] = 'npart'
            if(key == 'a') : keys[i] = 'a_axis'
            if(key == 'b') : keys[i] = 'b_axis'
            if(key == 'c') : keys[i] = 'c_axis'
            if(key == 'Mvir') : keys[i] = 'mass'
        for h, line in enumerate(f) :
            values = [float(x) for x in line.split()]
            # XXX Unit issues!  AHF uses distances in Mpc/h, possibly masses as well
            for i,key in enumerate(keys) :
                self._halos[h+1].properties[key] = values[i]
        f.close()

    def _load_ahf_substructure(self,filename) :
        f = open(filename)
        nhalos = int(f.readline())  # number of halos?  no, some crazy number
                                    # that we will ignore
        for i in xrange(len(self._halos)) :
            try:
                haloid, nsubhalos = [int(x) for x in f.readline().split()]
                self._halos[haloid+1].properties['children'] = [int(x) for x in f.readline().split()]
            except ValueError:
                break
        f.close()

    @staticmethod
    def _can_load(sim) :
        for file in glob.glob(sim._filename+'*z*particles') :
            if os.path.exists(file) :
                return True
        return False

    def _run_ahf(self, sim) :
        # if (sim is pynbody.tipsy.TipsySnap) :
        import pynbody.units as units
        #build units file
        f = open('tipsy.info','w')
        f.write(str(sim.properties['omegaM0'])+"\n")
        f.write(str(sim.properties['omegaL0'])+"\n")
        f.write(str(float(sim._paramfile['dKpcUnit'])/1000.0 * sim.properties['h'])+"\n")
        f.write(str(sim['vel'].units.ratio(units.km/units.s))+"\n")
        f.write(str(sim['mass'].units.ratio(units.Msol))+"\n")
        f.close()
        typecode='90'
        #elif (sim is pynbody.gadget.GadgetSnap):
        #   typecode = '60' or '61'
        # make input file
        f = open('AHFstep.in','w')
        f.write(sim._filename+" "+typecode+" 1\n")
        f.write(sim._filename+"\n256\n5\n5\n0\n0\n0\n0\n")
        f.close()

        # determine parallel possibilities
        # find AHFstep
        for directory in os.environ["PATH"].split(os.pathsep) :
            groupfinder = os.path.join(directory,"AHFstep")
            if os.path.exists(groupfinder) :
                # run it
                os.system(groupfinder+" AHFstep.in")

    @staticmethod
    def _can_run(sim) :
        for directory in os.environ["PATH"].split(os.pathsep) :
            if os.path.exists(os.path.join(directory,"AHFstep")) :
                return True
        return False

class AmigaGrpCatalogue(HaloCatalogue) :
    def __init__(self, sim) :
        sim['amiga.grp'] # trigger lazy-loading and/or kick up a fuss if unavailable
        self._base = weakref.ref(sim)
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
    def _can_load(sim) :
        try :
            sim['amiga.grp']
            return True
        except KeyError :
            return False


_halo_classes = [AHFCatalogue,AmigaGrpCatalogue]
_runable_halo_classes = [AHFCatalogue]

