"""

halo
====

Implements halo catalogue functions. If you have a supported halo
catalogue on disk or a halo finder installed and correctly configured,
you can access a halo catalogue through f.halos() where f is a
SimSnap.  
<http://code.google.com/p/pynbody/wiki/HaloCatalogue>

"""

import numpy as np
import weakref
import os.path
import glob
import re
import copy
import sys
from . import snapshot, util, config, config_parser

class DummyHalo(object) :
    def __init__(self) :
        self.properties = {}


class Halo(snapshot.IndexedSubSnap) :
    def __init__(self, halo_id, halo_catalogue, *args) :
        super(Halo, self).__init__(*args)
        self._halo_catalogue = halo_catalogue
        self._halo_id = halo_id
        self._descriptor = "halo_"+str(halo_id)
        self.properties = copy.copy(self.properties)
        self.properties['halo_id'] = halo_id

    def is_subhalo(self, otherhalo):
        return self._halo_catalogue.is_subhalo(self._halo_id, otherhalo._halo_id)
        
class HaloCatalogue(object) :
    def __init__(self) :
        self._halos = {}

    def calc_item(self, i):
        if self._halos.has_key(i) :  # and self._halos[i]() is not None :
            return self._halos[i] # ()
        else :
            h = self._get_halo(i)
            self._halos[i] = h # weakref.ref(h)
            return h

    def __getitem__(self, item) :
        if isinstance(item, slice):
            indices = item.indices(len(self._halos))
            [self.calc_item(i+1) for i in range(*indices)]
            return self._halos[item]
        else:
            return self.calc_item(item)

    def is_subhalo(self,childid,parentid) :
        if (childid in self._halos[parentid].properties['children']) :
            return True
        else : return False

    def contains(self,haloid):
        if (haloid in self._halos) : return True
        else: return False

    @staticmethod
    def _can_load(self):
        return False

    @staticmethod
    def _can_run(self):
        return False

    def writegrp(self,snapshot,halos,grpoutfile):
        """          
        simply write a skid style .grp file from ahf_particles
        file. header = total number of particles, then each line is
        the halo id for each particle (0 means free).
        """
        outfile=grpoutfile
        print "write grp file to ",grpoutfile
        fpout=open(outfile,"w")
        grparray=snapshot['grp']
        print >> fpout, len(grparray)

        ## writing 1st to a string sacrifices memory for speed.
        ##  but this is much faster than numpy.savetxt (could make an option).
        ##  it is assumed that max halo id <= nhalos (i.e.length of string is set len(str(nhalos))
        stringarray=grparray.astype('|S'+str(len(str(halos._nhalos))))  
        outstring = "\n".join(stringarray)
        print >> fpout, outstring  
        fpout.close()
        return 1


    def writestat(self,snapshot,halos,statoutfile,hubble=None):
        """
        write a condensed skid.stat style ascii file from ahf_halos
        file.  header + 1 halo per line. should reproduce `Alyson's
        idl script' except does not do last 2 columns (Is it a
        satellite?) and (Is central halo is `false'ly split?).  output
        units are set to Mpc Msun, km/s.
        
        user can specify own hubble constant hubble=(H0/(100
        km/s/Mpc)), ignoring the snaphot arg for hubble constant
        (which sometimes has a large roundoff error).
        """
        s=snapshot
        mindarkmass=min(s.dark['mass'])

        if hubble is None:
            hubble = s.properties['h']

        outfile=statoutfile
        print "write stat file to ",statoutfile
        fpout=open(outfile,"w")  
        header = "#Grp  N_tot     N_gas      N_star    N_dark    Mvir(M_sol)       Rvir(kpc)       GasMass(M_sol) StarMass(M_sol)  DarkMass(M_sol)  V_max  R@V_max  VelDisp    Xc   Yc   Zc   VXc   VYc   VZc   Contam   Satellite?   False?   ID_A"
        print >> fpout, header
        nhalos = halos._nhalos
        for ii in xrange(nhalos):
            h=halos[ii+1].properties  ## halo index starts with 1 not 0        
##  'Contaminated'? means multiple dark matter particle masses in halo)" 
            icontam=np.where(halos[ii+1].dark['mass'] > mindarkmass)
            if (len(icontam[0]) > 0):
                contam="contam"
            else:    
                contam="clean"
## may want to add implement satellite test and false central breakup test.

            n_dark = h['npart'] - h['n_gas'] - h['n_star']
            M_dark = h['mass'] - h['M_gas'] - h['M_star']
            ss="     " ## can adjust column spacing
            outstring = str(int(h['halo_id']))+ss 
            outstring += str(int(h['npart']))+ss+str(int(h['n_gas']))+ss
            outstring += str(int(h['n_star'])) +ss+str(int(n_dark))+ss
            outstring += str(h['mass']/hubble)+ss+str(h['Rvir']/hubble)+ss
            outstring += str(h['M_gas']/hubble)+ss+str(h['M_star']/hubble)+ss
            outstring += str(M_dark/hubble)+ss
            outstring += str(h['Vmax'])+ss+str(h['Rmax']/hubble)+ss 
            outstring += str(h['sigV'])+ss
        ## pos: convert kpc/h to mpc (no h).
            outstring += str(h['Xc']/hubble/1000.)+ss
            outstring += str(h['Yc']/hubble/1000.)+ss
            outstring += str(h['Zc']/hubble/1000.)+ss
            outstring += str(h['VXc'])+ss+str(h['VYc'])+ss+str(h['VZc'])+ss
            outstring += contam+ss 
            outstring +="unknown"+ss ## unknown means sat. test not implemented.
            outstring +="unknown"+ss ## false central breakup.
            print >> fpout, outstring
        fpout.close()
        return 1


    def writetipsy(self,snapshot,halos,tipsyoutfile,hubble=None):
        """
        write halos to tipsy file (write as stars) from ahf_halos
        file.  returns a shapshot where each halo is a star particle.

        user can specify own hubble constant hubble=(H0/(100
        km/s/Mpc)), ignoring the snaphot arg for hubble constant
        (which sometimes has a large roundoff error).
        """
        from . import analysis
        from . import tipsy
        import analysis.cosmology
        from snapshot import _new as new
        import math
        s=snapshot
        outfile=tipsyoutfile
        print "write tipsy file to ",tipsyoutfile
        nhalos = halos._nhalos
        nstar=nhalos
        sout = new(star=nstar) ## create new tipsy snapshot written as halos.
        sout.properties['a'] = s.properties['a']
        sout.properties['z'] = s.properties['z']
        sout.properties['boxsize'] = s.properties['boxsize']
        if hubble is None:
            hubble = s.properties['h']
        sout.properties['h'] = hubble
    ### ! dangerous -- rho_crit function and unit conversions needs simplifying
        rhocrithhco = analysis.cosmology.rho_crit(s,z=0,unit="Msol Mpc^-3 h^2")
        lboxkpc =  sout.properties['boxsize'].ratio("kpc a")
        lboxkpch = lboxkpc*sout.properties['h']
        lboxmpch = lboxkpc*sout.properties['h']/1000.
        tipsyvunitkms = lboxmpch * 100./ (math.pi * 8./3.)**.5 
        tipsymunitmsun = rhocrithhco * lboxmpch**3 / sout.properties['h'] 

        print "transforming ",nhalos," halos into tipsy star particles"
        for ii in xrange(nhalos):
            h=halos[ii+1].properties
            sout.star[ii]['mass'] = h['mass']/hubble / tipsymunitmsun
            ## tipsy units: box centered at 0. (assume 0<=x<=1)
            sout.star[ii]['x'] = h['Xc']/lboxkpch - 0.5 
            sout.star[ii]['y'] = h['Yc']/lboxkpch - 0.5
            sout.star[ii]['z'] = h['Zc']/lboxkpch - 0.5
            sout.star[ii]['vx'] =  h['VXc']/tipsyvunitkms
            sout.star[ii]['vy'] =  h['VYc']/tipsyvunitkms
            sout.star[ii]['vz'] =  h['VZc']/tipsyvunitkms
            sout.star[ii]['eps'] = h['Rvir']/lboxkpch
            sout.star[ii]['metals'] = 0.
            sout.star[ii]['phi'] = 0.
            sout.star[ii]['tform'] = 0.
        print "writing tipsy outfile",outfile
        sout.write(fmt=tipsy.TipsySnap,filename=outfile)
        return sout


    def writehalos(self,snapshot,halos,hubble=None,outfile=None):
        """ Write the (ahf) halo catalog to disk.  This is really a
        wrapper that calls writegrp, writetipsy, writestat.  Writes
        .amiga.grp file (ascii group ids), .amiga.stat file (ascii
        halo catalog) and .amiga.gtp file (tipsy halo catalog).
        default outfile base simulation is same as snapshot s.
        function returns simsnap of halo catalog.
        """
        s=snapshot
        grpoutfile = s.filename+".amiga.grp"
        statoutfile = s.filename+".amiga.stat"
        tipsyoutfile = s.filename+".amiga.gtp"
        halos.writegrp(s,halos,grpoutfile)
        halos.writestat(s,halos,statoutfile,hubble=hubble)
        shalos = halos.writetipsy(s,halos,tipsyoutfile,hubble=hubble)
        return shalos



class AHFCatalogue(HaloCatalogue) :
    def __init__(self, sim, make_grp=None, dummy=False) :
        """Initialize an AHFCatalogue.

        **kwargs** :
         *make_grp*: if True a 'grp' array is created in the
           underlying snapshot specifying the lowest level halo that any
           given particle belongs to. If it is False, no such array is created;
           if None, the behaviour is determined by the configuration system.

         *dummy*: if True, the particle file is not loaded, and all halos
           returned are just dummies (with the correct properties dictionary
           loaded). Use load_copy to get the actual data in this case. """
 
        import os.path
        if not self._can_load(sim) :
            self._run_ahf(sim)
        self._base = weakref.ref(sim)
        HaloCatalogue.__init__(self)

        self._dummy = dummy
        
        self._ahfBasename = util.cutgz(glob.glob(sim._filename+'*z*halos*')[0])[:-5]
        
        f = util.open_(self._ahfBasename+'halos')
        for i, l in enumerate(f):
            pass
        self._nhalos=i
        f.close()
        
        if config['verbose']: print "AHFCatalogue: loading particles...",
        sys.stdout.flush()
        
        self._load_ahf_particles(self._ahfBasename+'particles')
        
        if config['verbose']: print "halos...",
        sys.stdout.flush()
        
        self._load_ahf_halos(self._ahfBasename+'halos')


        if os.path.isfile(self._ahfBasename+'substructure') :
            if config['verbose']: print "substructure...",
            sys.stdout.flush()
            self._load_ahf_substructure(self._ahfBasename+'substructure')
        else :
            self._setup_children()

        if make_grp is None :
            make_grp = config_parser.getboolean('AHFCatalogue', 'AutoGrp') 

        if make_grp :
            self.make_grp()

        if config_parser.getboolean('AHFCatalogue', 'AutoPid') :
            sim['pid'] = np.arange(0,len(sim))

        if config['verbose']: print "done!"
 

        
    def make_grp(self, name='grp') :
        for halo in self._halos.values():
            halo[name] = np.repeat([halo._halo_id],len(halo))

    def _setup_children(self) :
        for i in xrange(self._nhalos) :
            self._halos[i+1].properties['children'] = []

        for i in xrange(self._nhalos) :
            host =  self._halos[i+1].properties.get('hostHalo',-2)
            if host>-1 :
                try:
                    self._halos[host+1].properties['children'].append(i+1)
                except KeyError :
                    pass
        
    def _get_halo(self, i) :
        if self.base is None :
            raise RuntimeError, "Parent SimSnap has been deleted"

        return self._halos[i]
    

    @property
    def base(self) :
        return self._base()


    def load_copy(self, i) :
        """Load the a fresh SimSnap with only the particle in halo i"""

        from . import load
        
        f = util.open_(self._ahfBasename+'particles')
      
        if self.isnew:
            nhalos=int(f.readline())
        else: 
            nhalos = self._nhalos


        for h in xrange(i) :
            ids = self._load_ahf_particle_block(f)
            
        f.close()

        return load(self.base.filename, take=ids)

    def _load_ahf_particle_block(self, f) :
        """Load the particles for the next halo described in particle file f"""
        ng = len(self.base.gas)
        nds = len(self.base.dark) + len(self.base.star)
        nparts = int(f.readline())
        if self.isnew :
            if isinstance(f,file) :
                data = (np.fromfile(f, dtype=int, sep=" ", count = nparts*2).reshape(nparts,2))[:,0]
            else :
                # unfortunately with gzipped files there does not
                # seem to be an efficient way to load nparts lines
                data = np.zeros(nparts, dtype=int)
                for i in xrange(nparts) :
                    data[i] = int(f.readline().split()[0])

            hi_mask = data>=nds
            data[np.where(hi_mask)]-=nds
            data[np.where(~hi_mask)]+=ng
        else :
            if isinstance(f, file) :
                data = np.fromfile(f, dtype=int, sep=" ", count=nparts)
            else :
                # see comment above on gzipped files
                data = np.zeros(nparts, dtype=int)
                for i in xrange(nparts) :
                    data[i] = int(f.readline())
        data.sort()
        return data
            
    def _load_ahf_particles(self,filename) :
        f = util.open_(filename)
        if filename.split("z")[-2][-1] is "." : self.isnew = True
        else : self.isnew = False
      
        if self.isnew:
            nhalos=int(f.readline())
        else: 
            nhalos = self._nhalos

        if not self._dummy :
            for h in xrange(nhalos) :
                self._halos[h+1] = Halo( h+1, self, self.base, self._load_ahf_particle_block(f))
                self._halos[h+1]._descriptor = "halo_"+str(h+1)
        else :
            for h in xrange(nhalos) :
                self._halos[h+1] = DummyHalo()

        f.close()

    def _load_ahf_halos(self,filename) :
        f = util.open_(filename)
        # get all the property names from the first, commented line
        # remove (#)
        keys = [re.sub('\([0-9]*\)','',field)
                for field in f.readline().split()]
        # provide translations
        for i,key in enumerate(keys) :
            if self.isnew:
                if(key == '#npart') : keys[i] = 'npart'
            else: 
                if(key == '#') : keys[i] = 'dumb'
            if(key == 'a') : keys[i] = 'a_axis'
            if(key == 'b') : keys[i] = 'b_axis'
            if(key == 'c') : keys[i] = 'c_axis'
            if(key == 'Mvir') : keys[i] = 'mass'

        if self.isnew :
            # fix for column 0 being a non-column in some versions of the AHF output
            if keys[0]=='#' : keys=keys[1:]
                
        for h, line in enumerate(f) :
            values = [float(x) if '.' in x or 'e' in x or 'nan' in x else int(x) for x in line.split()]
            # XXX Unit issues!  AHF uses distances in Mpc/h, possibly masses as well
            for i,key in enumerate(keys) :
                if self.isnew:
                    self._halos[h+1].properties[key] = values[i]
                else :
                    self._halos[h+1].properties[key] = values[i-1]
        f.close()

    def _load_ahf_substructure(self,filename) :
        f = util.open_(filename)
        nhalos = int(f.readline())  # number of halos?  no, some crazy number
                                    # that we will ignore
        #import pdb; pdb.set_trace()
        for i in xrange(len(self._halos)) :
            try:
                haloid, nsubhalos = [int(x) for x in f.readline().split()]
                self._halos[haloid+1].properties['children'] = [int(x)+1 for x in f.readline().split()]
            except KeyError:
                pass
            except ValueError:
                break
        f.close()

    @staticmethod
    def _can_load(sim) :
        for file in glob.glob(sim._filename+'*z*particles*') :
            if os.path.exists(file) :
                return True
        return False

    def _run_ahf(self, sim) :
        # if (sim is pynbody.tipsy.TipsySnap) :
	typecode=90
        #elif (sim is pynbody.gadget.GadgetSnap):                               
        #   typecode = '60' or '61'                                             
        import pynbody.units as units
        # find AHFstep
        
        groupfinder = config_parser.get('AHFCatalogue','Path')

        if groupfinder=='None' :
            for directory in os.environ["PATH"].split(os.pathsep) :
                ahfs = glob.glob(os.path.join(directory,"AHF*"))
                for iahf, ahf in enumerate(ahfs):
                    # if there are more AHF*'s than 1, it's not the last one, and
                    # it's AHFstep, then continue, otherwise it's OK.
                    if ((len(ahfs)>1) & (iahf != len(ahfs)-1) & 
                        (os.path.basename(ahf) == 'AHFstep')): 
                        continue
                    else: 
                        groupfinder=ahf
                        break

        if not os.path.exists(groupfinder) :
            raise RuntimeError, "Path to AHF (%s) is invalid"%groupfinder

        if (os.path.basename(groupfinder) == 'AHFstep'):  isAHFstep=True
        else:  isAHFstep=False
        #build units file
        if isAHFstep:
            f = open('tipsy.info','w')
            f.write(str(sim.properties['omegaM0'])+"\n")
            f.write(str(sim.properties['omegaL0'])+"\n")
            f.write(str(sim['pos'].units.ratio(units.kpc,a=1)/1000.0 * sim.properties['h'])+"\n")
            f.write(str(sim['vel'].units.ratio(units.km/units.s,a=1))+"\n")
            f.write(str(sim['mass'].units.ratio(units.Msol))+"\n")
            f.close()
            # make input file
            f = open('AHF.in','w')
            f.write(sim._filename+" "+str(typecode)+" 1\n")
            f.write(sim._filename+"\n256\n5\n5\n0\n0\n0\n0\n")
            f.close()
        else:
            # make input file
            f = open('AHF.in','w')
            
            lgmax = np.min([int(2**np.floor(np.log2(1.0 / np.min(sim['eps'])))),131072])
            # hardcoded maximum 131072 might not be necessary
            
            print >>f, config_parser.get('AHFCatalogue', 'Config', vars = {
                'filename': sim._filename,
                'typecode': typecode,
                'gridmax': lgmax
                })

            print >>f, config_parser.get('AHFCatalogue', 'ConfigTipsy', vars = {
                'omega0': sim.properties['omegaM0'],
                'lambda0': sim.properties['omegaL0'],
                'boxsize': sim['pos'].units.ratio('Mpc a h^-1', **sim.conversion_context()),
                'vunit': sim['vel'].units.ratio('km s^-1 a', **sim.conversion_context()),
                'munit': sim['mass'].units.ratio('Msol h^-1', **sim.conversion_context()),
                'eunit': 0.03 # surely this can't be right?
                })
                
            f.close()

        if (not os.path.exists(sim._filename)):
            os.system("gunzip "+sim._filename+".gz")
        # determine parallel possibilities


        if os.path.exists(groupfinder) :
            # run it
            os.system(groupfinder+" AHF.in")
            return

    @staticmethod
    def _can_run(sim) :
        if config_parser.getboolean('AHFCatalogue', 'AutoRun') :
            if config_parser.get('AHFCatalogue', 'Path')=='None' :
                for directory in os.environ["PATH"].split(os.pathsep) :
                    if (len(glob.glob(os.path.join(directory,"AHF*"))) > 0):
                        return True
            else :
                path = config_parser.get('AHFCatalogue', 'Path')
                return os.path.exists(path) 
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

        x = Halo(i, self, self.base, np.where(self.base['amiga.grp']==i))
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


# AmigaGrpCatalogue MUST be scanned first, because if it exists we probably
# want to use it, but an AHFCatalogue will probably be on-disk too.

_halo_classes = [AmigaGrpCatalogue, AHFCatalogue]
_runable_halo_classes = [AHFCatalogue]

