"""

halo
====

Implements halo catalogue functions. If you have a supported halo
catalogue on disk or a halo finder installed and correctly configured,
you can access a halo catalogue through f.halos() where f is a
SimSnap.

See the `halo tutorial
<http://pynbody.github.io/pynbody/tutorials/halos.html>`_ for some
examples.

"""

import numpy as np
import weakref
import os.path
import glob
import re
import copy
import sys
from . import snapshot, util, config, config_parser, gadget, units
from array import SimArray


class DummyHalo(object):
    def __init__(self):
        self.properties = {}


class Halo(snapshot.IndexedSubSnap):
    """
    Generic class representing a halo.
    """

    def __init__(self, halo_id, halo_catalogue, *args):
        super(Halo, self).__init__(*args)
        self._halo_catalogue = halo_catalogue
        self._halo_id = halo_id
        self._descriptor = "halo_"+str(halo_id)
        self.properties = copy.copy(self.properties)
        self.properties['halo_id'] = halo_id

    def is_subhalo(self, otherhalo):
        """
        Convenience function that calls the corresponding function in 
        a halo catalogue.
        """

        return self._halo_catalogue.is_subhalo(self._halo_id, otherhalo._halo_id)


# ----------------------------#
# General HaloCatalogue class # 
#-----------------------------#

class HaloCatalogue(object):
    """
    Generic halo catalogue object. 
    """

    def __init__(self):
        self._halos = {}
        self.lazy_off = util.ExecutionControl()

    def calc_item(self, i):
        if i in self._halos:  # and self._halos[i]() is not None :
            return self._halos[i]  # ()
        else:
            h = self._get_halo(i)
            self._halos[i] = h  # weakref.ref(h)
            return h

    def __len__(self) : 
        return len(self._halos)

    def __iter__(self) : 
        if not self.lazy_off :
            return self._halo_generator()
        else : 
            return iter(self._halos.values())

    def __getitem__(self, item):
        if isinstance(item, slice):
            for x in self._halo_generator(item.start,item.stop) : pass
            indices = item.indices(len(self._halos))
            res = [self.calc_item(i) for i in range(*indices)]
            return res
        else:
            return self.calc_item(item)
    
    def _halo_generator(self, i_start=None, i_stop=None) : 
        if len(self) == 0 : return
        if i_start is None :
            try : 
                self[0]
                i = 0
            except KeyError :
                i = 1
        else : 
            i = i_start
            
        if i_stop is None : 
            i_stop = len(self)

        while True:
            try : 
                yield self[i]
                i+=1
                if len(self[i]) == 0: continue
            except RuntimeError: 
                break
            if i == i_stop: raise StopIteration

    def is_subhalo(self, childid, parentid):
        """
        Checks whether the specified 'childid' halo is a subhalo 
        of 'parentid' halo. 
        """
        if (childid in self._halos[parentid].properties['children']):
            return True
        else:
            return False

    def contains(self, haloid):
        if (haloid in self._halos):
            return True
        else:
            return False

    @staticmethod
    def _can_load(self):
        return False

    @staticmethod
    def _can_run(self):
        return False

    

#-------------------------------#
# Rockstar Halo Catalogue class #
#-------------------------------#

class RockstarCatalogue(HaloCatalogue):
    """
    Class to handle catalogues produced by Rockstar (by Peter Behroozi).
    """

    head_type = np.dtype([('magic',np.uint64),('snap',np.int64),
                          ('chunk',np.int64),('scale','f'),
                          ('Om','f'),('Ol','f'),('h0','f'),
                          ('bounds','f',6),('num_halos',np.int64),
                          ('num_particles',np.int64),('box_size','f'),
                          ('particle_mass','f'),('particle_type',np.int64),
                          ('format_revision',np.int32),
                          ('rockstar_version',np.str_,12)])

    halo_type = np.dtype([('id',np.int64),('pos','f',3),('vel','f',3),
                          ('corevel','f',3),('bulkvel','f',3),('m','f'),
                          ('r','f'),
                          ('child_r','f'),('vmax_r','f'),('mgrav','f'),
                          ('vmax','f'),('rvmax','f'),('rs','f'),
                          ('klypin_rs','f'),('vrms','f'),('J','f',3),
                          ('energy','f'),('spin','f'),('alt_m','f',4),
                          ('Xoff','f'),('Voff','f'),('b_to_a','f'),
                          ('c_to_a','f'),('A','f',3),('b_to_a2','f'),
                          ('c_to_a2','f'),('A2','f',3),('bullock_spin','f'),
                          ('kin_to_pot','f'),('m_pe_b','f'),('m_pe_d','f'),
                          ('dum',np.str_,4),
                          ('num_p',np.int64),('num_child_particles',np.int64),
                          ('p_start',np.int64),('desc',np.int64),
                          ('flags',np.int64),('n_core',np.int64),
                          ('min_pos_err','f'),('min_vel_err','f'),
                          ('min_bulkvel_err','f'),('type',np.int32),
                          ('sm','f'),('gas','f'),('bh','f'),
                          ('peak_density','f'),('av_density','f'),
                          ('odum',np.str_,4)])


    def __init__(self, sim, make_grp=None, dummy=False, use_iord=None, filename=None):
        """Initialize a RockstarCatalogue.

        **kwargs** :
         
        *make_grp*: if True a 'grp' array is created in the underlying
                    snapshot specifying the lowest level halo that any
                    given particle belongs to. If it is False, no such
                    array is created; if None, the behaviour is
                    determined by the configuration system.

        *dummy*: if True, the particle file is not loaded, and all
                 halos returned are just dummies (with the correct
                 properties dictionary loaded). Use load_copy to get
                 the actual data in this case.

        *use_iord*: if True, the particle IDs in the Amiga catalogue
                    are taken to refer to the iord array. If False,
                    they are the particle offsets within the file. If
                    None, the parameter defaults to True for
                    GadgetSnap, False otherwise.

        *basename*: specify the basename of the halo catalog
                        file - the code will load the catalog data from the
                        binary file.

        """

        import os.path
        if not self._can_load(sim):
            self._run_rockstar(sim)
        self._base = weakref.ref(sim)
        HaloCatalogue.__init__(self)

        if use_iord is None :
            use_iord = isinstance(sim.ancestor, gadget.GadgetSnap)

        self._use_iord = use_iord

        self._dummy = dummy
        
        if filename is not None: self._rsFilename = filename
        else: 
            self._rsFilename = util.cutgz(glob.glob('halos*.bin')[0])
        
        try : 
            f = util.open_(self._rsFilename)
        except IOError: 
            raise IOError("Halo catalogue not found -- check the file name of catalogue data or try specifying a catalogue using the filename keyword")

        self._head = np.fromstring(f.read(self.head_type.itemsize),
                                   dtype=self.head_type)
        unused = f.read(256 - self._head.itemsize)

        self._nhalos = self._head['num_halos'][0]

        if config['verbose']:
            print "RockstarCatalogue: loading halos...",
            sys.stdout.flush()
        
        self._load_rs_halos(f,sim)

        if not dummy:
            if config['verbose']:
                print " particles..."
                
            self._load_rs_particles(f,sim)

        f.close()
        
        if make_grp is None:
            make_grp = config_parser.getboolean('RockstarCatalogue', 'AutoGrp')

        if make_grp:
            self.make_grp()

        if config_parser.getboolean('RockstarCatalogue', 'AutoPid'):
            sim['pid'] = np.arange(0, len(sim))

        if config['verbose']:
            print "done!"

    def make_grp(self):
        """
        Creates a 'grp' array which labels each particle according to
        its parent halo. 
        """
        for halo in self._halos.values(): 
            halo['grp'] = np.repeat([halo._halo_id], len(halo))

        if config['verbose']:  print "writing %s"%(self._base().filename+'.grp')
        self._base().write_array('grp',overwrite=True,binary=False)

    def _setup_children(self):
        """
        Creates a 'children' array inside each halo's 'properties'
        listing the halo IDs of its children. Used in case the reading
        of substructure data from the AHF-supplied _substructure file
        fails for some reason.
        """

        for i in xrange(self._nhalos):
            self._halos[i+1].properties['children'] = []

        for i in xrange(self._nhalos):
            host = self._halos[i+1].properties.get('hostHalo', -2)
            if host > -1:
                try:
                    self._halos[host+1].properties['children'].append(i+1)
                except KeyError:
                    pass

    def _get_halo(self, i):
        if self.base is None:
            raise RuntimeError("Parent SimSnap has been deleted")

        return self._halos[i]

    @property
    def base(self):
        return self._base()

    def _load_rs_halos(self, f, sim):
        # want to read in halo properties first so that we can sort them
        # by particle number
        haloprops = []
        for h in xrange(self._head['num_halos']):
            haloprops.append(np.fromstring(f.read(self.halo_type.itemsize),dtype=self.halo_type))

        self._haloprops = np.array(haloprops)
        # sort by number of particles to make compatible with AHF
        self._num_p_rank = np.flipud(self._haloprops[:]['num_p'].argsort(axis=0))

        for h in xrange(self._head['num_halos']):
            hn = np.where(self._num_p_rank==h)[0][0]+1
            self._halos[hn] = DummyHalo()
            # properties are in Msun / h, Mpc / h
            self._halos[hn].properties = self._haloprops[h]

    def load_copy(self, i):
        """Load a fresh SimSnap with only the particle in halo i"""

        from . import load

        f = util.open_(self._rsFilename)
        f.seek(self._head.itemsize + self.halo_type*self._head['num_halos'])

        h = 0
        while h != i: 
            num_p = self._haloprops[h]['num_p'][0]
            f.seek(num_p*8,1)
            h=h+1

        num_p = self._haloprops[h]['num_p'][0]
        h_i=sorted(np.fromstring(f.read(num_p*8),dtype=np.int64))
        
        f.close()

        return load(self.base.filename, take=self._iord_to_fpos[h_i])

    def _load_rs_particles(self, f, sim):
        self._iord_to_fpos = np.zeros(self._base()['iord'].max()+1,dtype=int)
        self._iord_to_fpos[self._base()['iord']] = np.arange(len(self._base()))
            
        for h in xrange(self._head['num_halos']):
            num_p = self._haloprops[h]['num_p'][0]
            h_i=sorted(np.fromstring(f.read(num_p*8),dtype=np.int64))
            # ugly, but works
            hn = np.where(self._num_p_rank==h)[0][0]+1
            self._halos[hn]=Halo(hn, self, self.base,self._iord_to_fpos[h_i])
            self._halos[hn]._descriptor = "halo_"+str(hn)
            # properties are in Msun / h, Mpc / h
            self._halos[hn].properties = self._haloprops[h]


    def _load_ahf_substructure(self, filename):
        f = util.open_(filename)
        #nhalos = int(f.readline())  # number of halos?  no, some crazy number
                                    # that we will ignore
        nhalos = f.readline()  # Some crazy number, just need to skip it
        for i in xrange(len(self._halos)):
            try:
                haloid, nsubhalos = [int(x) for x in f.readline().split()]
                self._halos[haloid+1].properties['children'] = [
                    int(x)+1 for x in f.readline().split()]
            except KeyError:
                pass
            except ValueError:
                break
        f.close()

    @staticmethod
    def _can_load(sim):
        for file in glob.glob('halos*.bin'):
            if os.path.exists(file):
                return True
        return False

    def _run_rockstar(self, sim):
        import pynbody
        fileformat = 'TIPSY'
        if (sim is pynbody.gadget.GadgetSnap):
            fileformat = 'GADGET'
        import pynbody.units as units

        # find AHFstep
        groupfinder = config_parser.get('RockstarCatalogue', 'Path')

        if groupfinder == 'None':
            for directory in os.environ["PATH"].split(os.pathsep):
                ahfs = glob.glob(os.path.join(directory, "rockstar-galaxies"))
                for iahf, ahf in enumerate(ahfs):
                    if ((len(ahfs) > 1) & (iahf != len(ahfs)-1) &
                            (os.path.basename(ahf) == 'rockstar')):
                        continue
                    else:
                        groupfinder = ahf
                        break

        if not os.path.exists(groupfinder):
            raise RuntimeError("Path to Rockstar (%s) is invalid" % groupfinder)

        f = open('quickstart.cfg', 'w')
        print >>f, config_parser.get('RockstarCatalogue', 'Config', vars={
                'format': fileformat,
                'partmass': sim.d['mass'].in_units('Msol h^-1',**sim.conversion_context()).min(),
                'expfac': sim.properties['a'],
                'hub': sim.properties['h'],
                'omega0': sim.properties['omegaM0'],
                'lambda0': sim.properties['omegaL0'],
                'boxsize': sim['pos'].units.ratio('Mpc a h^-1', **sim.conversion_context()),
                'vunit': sim['vel'].units.ratio('km s^-1 a', **sim.conversion_context()),
                'munit': sim['mass'].units.ratio('Msol h^-1', **sim.conversion_context()),
                'softening': sim.s['eps'].in_units('Mpc a h^-1', **sim.conversion_context()).min()
            })

        f.close()

        if (not os.path.exists(sim._filename)):
            os.system("gunzip "+sim._filename+".gz")
        # determine parallel possibilities

        if os.path.exists(groupfinder):
            # run it
            if config['verbose']:
                print "RockstarCatalogue: running %s"%groupfinder
            os.system(groupfinder+" -c quickstart.cfg "+sim._filename)
            return

    @staticmethod
    def _can_run(sim):
        if config_parser.getboolean('RockstarCatalogue', 'AutoRun'):
            if config_parser.get('RockstarCatalogue', 'Path') == 'None':
                for directory in os.environ["PATH"].split(os.pathsep):
                    if (len(glob.glob(os.path.join(directory, "rockstar-galaxies"))) > 0):
                        return True
            else:
                path = config_parser.get('RockstarCatalogue', 'Path')
                return os.path.exists(path)
        return False

    def writestat(self, outfile=None, hubble=None):
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
        s = self._base()
        mindarkmass = min(s.dark['mass'])

        if hubble is None:
            hubble = s.properties['h']

        if outfile is None: outfile = self._base().filename+'.stat'
        print "write stat file to ", outfile
        fpout = open(outfile, "w")
        header = "#Grp  N_tot     N_gas      N_star    N_dark    Mvir(M_sol)       Rvir(kpc)       GasMass(M_sol) StarMass(M_sol)  DarkMass(M_sol)  V_max  R@V_max  VelDisp    Xc   Yc   Zc   VXc   VYc   VZc   Contam   Satellite?   False?   ID_A"
        print >> fpout, header
        for ii in np.arange(self._nhalos)+1:
            print '%d '%ii,
            sys.stdout.flush()
            h = self[ii].properties  # halo index starts with 1 not 0
##  'Contaminated'? means multiple dark matter particle masses in halo)"
            icontam = np.where(self[ii].dark['mass'] > mindarkmass)
            if (len(icontam[0]) > 0):
                contam = "contam"
            else:
                contam = "clean"
## may want to add implement satellite test and false central breakup test.
            ss = "     "  # can adjust column spacing
            outstring = str(ii)+ss
            outstring += str(len(self[ii]))+ss+str(len(self[ii].g))+ss
            outstring += str(len(self[ii].s)) + ss+str(len(self[ii].dark))+ss
            outstring += str(h['m']/hubble)+ss+str(h['r']/hubble)+ss
            outstring += str(self[ii].g['mass'].in_units('Msol').sum())+ss
            outstring += str(self[ii].s['mass'].in_units('Msol').sum())+ss
            outstring += str(self[ii].d['mass'].in_units('Msol').sum())+ss
            outstring += str(h['vmax'])+ss+str(h['vmax_r']/hubble)+ss
            outstring += str(h['vrms'])+ss
        ## pos: convert kpc/h to mpc (no h).
            outstring += str(h['pos'][0][0]/hubble)+ss
            outstring += str(h['pos'][0][1]/hubble)+ss
            outstring += str(h['pos'][0][2]/hubble)+ss
            outstring += str(h['vel'][0][0])+ss+str(h['vel'][0][1])+ss
            outstring += str(h['vel'][0][2])+ss
            outstring += contam+ss
            outstring += "unknown" + \
                ss  # unknown means sat. test not implemented.
            outstring += "unknown"+ss  # false central breakup.
            print >> fpout, outstring
        fpout.close()

    def writetipsy(self, outfile=None, hubble=None):
        """
        write halos to tipsy file (write as stars) from ahf_halos
        file.  returns a shapshot where each halo is a star particle.

        user can specify own hubble constant hubble=(H0/(100
        km/s/Mpc)), ignoring the snaphot arg for hubble constant
        (which sometimes has a large roundoff error).
        """
        from . import analysis
        from . import tipsy
        from .analysis import cosmology
        from snapshot import _new as new
        import math
        s = self._base()
        if outfile is None: outfile = s.filename+'.gtp'
        print "write tipsy file to ", outfile
        sout = new(star=self._nhalos)  # create new tipsy snapshot written as halos.
        sout.properties['a'] = s.properties['a']
        sout.properties['z'] = s.properties['z']
        sout.properties['boxsize'] = s.properties['boxsize']
        if hubble is None: hubble = s.properties['h']
        sout.properties['h'] = hubble
    ### ! dangerous -- rho_crit function and unit conversions needs simplifying
        rhocrithhco = cosmology.rho_crit(s, z=0, unit="Msol Mpc^-3 h^2")
        lboxkpc = sout.properties['boxsize'].ratio("kpc a")
        lboxkpch = lboxkpc*sout.properties['h']
        lboxmpch = lboxkpc*sout.properties['h']/1000.
        tipsyvunitkms = lboxmpch * 100. / (math.pi * 8./3.)**.5
        tipsymunitmsun = rhocrithhco * lboxmpch**3 / sout.properties['h']

        print "transforming ", self._nhalos, " halos into tipsy star particles"
        for ii in xrange(self._nhalos):
            h = self[ii+1].properties
            sout.star[ii]['mass'] = h['m']/hubble / tipsymunitmsun
            ## tipsy units: box centered at 0. (assume 0<=x<=1)
            sout.star[ii]['x'] = h['pos'][0][0]/lboxmpch - 0.5
            sout.star[ii]['y'] = h['pos'][0][1]/lboxmpch - 0.5
            sout.star[ii]['z'] = h['pos'][0][2]/lboxmpch - 0.5
            sout.star[ii]['vx'] = h['vel'][0][0]/tipsyvunitkms
            sout.star[ii]['vy'] = h['vel'][0][1]/tipsyvunitkms
            sout.star[ii]['vz'] = h['vel'][0][2]/tipsyvunitkms
            sout.star[ii]['eps'] = h['r']/lboxkpch
            sout.star[ii]['metals'] = 0.
            sout.star[ii]['phi'] = 0.
            sout.star[ii]['tform'] = 0.
        print "writing tipsy outfile %s"%outfile
        sout.write(fmt=tipsy.TipsySnap, filename=outfile)
        return sout

    def writehalos(self, hubble=None, outfile=None):
        s = self._base()
        if outfile is None:
            statoutfile = s.filename+".rockstar.stat"
            tipsyoutfile = s.filename+".rockstar.gtp"
        else:
            statoutfile = outfile+'.stat'
            gtpoutfile = outfile+'.gtp'
        self.make_grp()
        self.writestat(statoutfile, hubble=hubble)
        shalos = self.writetipsy(gtpoutfile, hubble=hubble)
        return shalos


#--------------------------#
# AHF Halo Catalogue class #
#--------------------------#

class AHFCatalogue(HaloCatalogue):
    """
    Class to handle catalogues produced by Amiga Halo Finder (AHF).
    """

    def __init__(self, sim, make_grp=None, dummy=False, use_iord=None, ahf_basename=None):
        """Initialize an AHFCatalogue.

        **kwargs** :
         
        *make_grp*: if True a 'grp' array is created in the underlying
                    snapshot specifying the lowest level halo that any
                    given particle belongs to. If it is False, no such
                    array is created; if None, the behaviour is
                    determined by the configuration system.

        *dummy*: if True, the particle file is not loaded, and all
                 halos returned are just dummies (with the correct
                 properties dictionary loaded). Use load_copy to get
                 the actual data in this case.

        *use_iord*: if True, the particle IDs in the Amiga catalogue
                    are taken to refer to the iord array. If False,
                    they are the particle offsets within the file. If
                    None, the parameter defaults to True for
                    GadgetSnap, False otherwise.

        *ahf_basename*: specify the basename of the AHF halo catalog
                        files - the code will append 'halos',
                        'particles', and 'substructure' to this
                        basename to load the catalog data.

        """

        import os.path
        if not self._can_load(sim):
            self._run_ahf(sim)
        self._base = weakref.ref(sim)
        HaloCatalogue.__init__(self)

        if use_iord is None :
            use_iord = isinstance(sim.ancestor, gadget.GadgetSnap)

        self._use_iord = use_iord

        self._dummy = dummy
        
        if ahf_basename is not None: self._ahfBasename = ahf_basename
        else: 
            self._ahfBasename = util.cutgz(glob.glob(sim._filename+'*z*halos*')[0])[:-5]
        
        try : 
            f = util.open_(self._ahfBasename+'halos')
        except IOError: 
            raise IOError("Halo catalogue not found -- check the base name of catalogue data or try specifying a catalogue using the ahf_basename keyword")

        for i, l in enumerate(f):
            pass
        self._nhalos = i
        f.close()

        if config['verbose']:
            print "AHFCatalogue: loading particles...",
        sys.stdout.flush()

        self._load_ahf_particles(self._ahfBasename+'particles')

        if config['verbose']:
            print "halos...",
        sys.stdout.flush()

        self._load_ahf_halos(self._ahfBasename+'halos')

        if os.path.isfile(self._ahfBasename+'substructure'):
            if config['verbose']:
                print "substructure...",
            sys.stdout.flush()
            self._load_ahf_substructure(self._ahfBasename+'substructure')
        else:
            self._setup_children()

        if make_grp is None:
            make_grp = config_parser.getboolean('AHFCatalogue', 'AutoGrp')

        if make_grp:
            self.make_grp()

        if config_parser.getboolean('AHFCatalogue', 'AutoPid'):
            sim['pid'] = np.arange(0, len(sim))

        if config['verbose']:
            print "done!"

    def make_grp(self, name='grp'):
        """
        Creates a 'grp' array which labels each particle according to
        its parent halo. 
        """
        for halo in self._halos.values(): 
            halo[name] = np.repeat([halo._halo_id], len(halo))

    def _setup_children(self):
        """
        Creates a 'children' array inside each halo's 'properties'
        listing the halo IDs of its children. Used in case the reading
        of substructure data from the AHF-supplied _substructure file
        fails for some reason.
        """

        for i in xrange(self._nhalos):
            self._halos[i+1].properties['children'] = []

        for i in xrange(self._nhalos):
            host = self._halos[i+1].properties.get('hostHalo', -2)
            if host > -1:
                try:
                    self._halos[host+1].properties['children'].append(i+1)
                except KeyError:
                    pass

    def _get_halo(self, i):
        if self.base is None:
            raise RuntimeError("Parent SimSnap has been deleted")

        return self._halos[i]

    @property
    def base(self):
        return self._base()

    def load_copy(self, i):
        """Load the a fresh SimSnap with only the particle in halo i"""

        from . import load

        f = util.open_(self._ahfBasename+'particles')

        if self.isnew:
            nhalos = int(f.readline())
        else:
            nhalos = self._nhalos

        for h in xrange(i):
            ids = self._load_ahf_particle_block(f)

        f.close()

        return load(self.base.filename, take=ids)

    def _load_ahf_particle_block(self, f):
        """Load the particles for the next halo described in particle file f"""
        ng = len(self.base.gas)
        nds = len(self.base.dark) + len(self.base.star)
        nparts = int(f.readline().split()[0])

        if self.isnew:
            if isinstance(f, file):
                data = (np.fromfile(
                    f, dtype=int, sep=" ", count=nparts*2).reshape(nparts, 2))[:, 0]
            else:
                # unfortunately with gzipped files there does not
                # seem to be an efficient way to load nparts lines
                data = np.zeros(nparts, dtype=int)
                for i in xrange(nparts):
                    data[i] = int(f.readline().split()[0])

            if self._use_iord :
                data = self._iord_to_fpos[data]
            else :
                hi_mask = data >= nds
                data[np.where(hi_mask)] -= nds
                data[np.where(~hi_mask)] += ng
        else:
            if isinstance(f, file):
                data = np.fromfile(f, dtype=int, sep=" ", count=nparts)
            else:
                # see comment above on gzipped files
                data = np.zeros(nparts, dtype=int)
                for i in xrange(nparts):
                    data[i] = int(f.readline())
        data.sort()
        return data

    def _load_ahf_particles(self, filename):
        if self._use_iord :
            iord = self._base()['iord']
            assert len(iord)==iord.max(), "Missing iord values - in principle this can be corrected for, but at the moment no code is implemented to do so"
            self._iord_to_fpos = iord.argsort()
            
            
        f = util.open_(filename)
        if filename.split("z")[-2][-1] is ".":
            self.isnew = True
        else:
            self.isnew = False

        if self.isnew:
            nhalos = int(f.readline())
        else:
            nhalos = self._nhalos

        if not self._dummy:
            for h in xrange(nhalos):
                self._halos[h+1] = Halo(
                    h+1, self, self.base, self._load_ahf_particle_block(f))
                self._halos[h+1]._descriptor = "halo_"+str(h+1)
        else:
            for h in xrange(nhalos):
                self._halos[h+1] = DummyHalo()

        f.close()

    def _load_ahf_halos(self, filename):
        f = util.open_(filename)
        # get all the property names from the first, commented line
        # remove (#)
        keys = [re.sub('\([0-9]*\)', '', field)
                for field in f.readline().split()]
        # provide translations
        for i, key in enumerate(keys):
            if self.isnew:
                if(key == '#npart'):
                    keys[i] = 'npart'
            else:
                if(key == '#'):
                    keys[i] = 'dumb'
            if(key == 'a'):
                keys[i] = 'a_axis'
            if(key == 'b'):
                keys[i] = 'b_axis'
            if(key == 'c'):
                keys[i] = 'c_axis'
            if(key == 'Mvir'):
                keys[i] = 'mass'

        if self.isnew:
            # fix for column 0 being a non-column in some versions of the AHF
            # output
            if keys[0] == '#':
                keys = keys[1:]

        for h, line in enumerate(f):
            values = [float(x) if '.' in x or 'e' in x or 'nan' in x else int(
                x) for x in line.split()]
            # XXX Unit issues!  AHF uses distances in Mpc/h, possibly masses as
            # well
            for i, key in enumerate(keys):
                if self.isnew:
                    self._halos[h+1].properties[key] = values[i]
                else:
                    self._halos[h+1].properties[key] = values[i-1]
        f.close()

    def _load_ahf_substructure(self, filename):
        f = util.open_(filename)
        #nhalos = int(f.readline())  # number of halos?  no, some crazy number
                                    # that we will ignore
        nhalos = f.readline()  # Some crazy number, just need to skip it
        for i in xrange(len(self._halos)):
            try:
                haloid, nsubhalos = [int(x) for x in f.readline().split()]
                self._halos[haloid+1].properties['children'] = [
                    int(x)+1 for x in f.readline().split()]
            except KeyError:
                pass
            except ValueError:
                break
        f.close()

    def writegrp(self, grpoutfile=False):
        """
        simply write a skid style .grp file from ahf_particles
        file. header = total number of particles, then each line is
        the halo id for each particle (0 means free).
        """
        snapshot = self[1].ancestor
        try:
            snapshot['grp']
        except:
            self.make_grp()
        if not grpoutfile: grpoutfile=snapshot.filename+'.grp'
        print "write grp file to ", grpoutfile
        fpout = open(grpoutfile, "w")
        print >> fpout, len(snapshot['grp'])

        ## writing 1st to a string sacrifices memory for speed.
        ##  but this is much faster than numpy.savetxt (could make an option).
        # it is assumed that max halo id <= nhalos (i.e.length of string is set
        # len(str(nhalos))
        stringarray = snapshot['grp'].astype('|S'+str(len(str(self._nhalos))))
        outstring = "\n".join(stringarray)
        print >> fpout, outstring
        fpout.close()

    def writestat(self, snapshot, halos, statoutfile, hubble=None):
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
        s = snapshot
        mindarkmass = min(s.dark['mass'])

        if hubble is None:
            hubble = s.properties['h']

        outfile = statoutfile
        print "write stat file to ", statoutfile
        fpout = open(outfile, "w")
        header = "#Grp  N_tot     N_gas      N_star    N_dark    Mvir(M_sol)       Rvir(kpc)       GasMass(M_sol) StarMass(M_sol)  DarkMass(M_sol)  V_max  R@V_max  VelDisp    Xc   Yc   Zc   VXc   VYc   VZc   Contam   Satellite?   False?   ID_A"
        print >> fpout, header
        nhalos = halos._nhalos
        for ii in xrange(nhalos):
            h = halos[ii+1].properties  # halo index starts with 1 not 0
##  'Contaminated'? means multiple dark matter particle masses in halo)"
            icontam = np.where(halos[ii+1].dark['mass'] > mindarkmass)
            if (len(icontam[0]) > 0):
                contam = "contam"
            else:
                contam = "clean"
## may want to add implement satellite test and false central breakup test.

            n_dark = h['npart'] - h['n_gas'] - h['n_star']
            M_dark = h['mass'] - h['M_gas'] - h['M_star']
            ss = "     "  # can adjust column spacing
            outstring = str(int(h['halo_id']))+ss
            outstring += str(int(h['npart']))+ss+str(int(h['n_gas']))+ss
            outstring += str(int(h['n_star'])) + ss+str(int(n_dark))+ss
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
            outstring += "unknown" + \
                ss  # unknown means sat. test not implemented.
            outstring += "unknown"+ss  # false central breakup.
            print >> fpout, outstring
        fpout.close()
        return 1

    def writetipsy(self, snapshot, halos, tipsyoutfile, hubble=None):
        """
        write halos to tipsy file (write as stars) from ahf_halos
        file.  returns a shapshot where each halo is a star particle.

        user can specify own hubble constant hubble=(H0/(100
        km/s/Mpc)), ignoring the snaphot arg for hubble constant
        (which sometimes has a large roundoff error).
        """
        from . import analysis
        from . import tipsy
        from .analysis import cosmology
        from snapshot import _new as new
        import math
        s = snapshot
        outfile = tipsyoutfile
        print "write tipsy file to ", tipsyoutfile
        nhalos = halos._nhalos
        nstar = nhalos
        sout = new(star=nstar)  # create new tipsy snapshot written as halos.
        sout.properties['a'] = s.properties['a']
        sout.properties['z'] = s.properties['z']
        sout.properties['boxsize'] = s.properties['boxsize']
        if hubble is None:
            hubble = s.properties['h']
        sout.properties['h'] = hubble
    ### ! dangerous -- rho_crit function and unit conversions needs simplifying
        rhocrithhco = cosmology.rho_crit(s, z=0, unit="Msol Mpc^-3 h^2")
        lboxkpc = sout.properties['boxsize'].ratio("kpc a")
        lboxkpch = lboxkpc*sout.properties['h']
        lboxmpch = lboxkpc*sout.properties['h']/1000.
        tipsyvunitkms = lboxmpch * 100. / (math.pi * 8./3.)**.5
        tipsymunitmsun = rhocrithhco * lboxmpch**3 / sout.properties['h']

        print "transforming ", nhalos, " halos into tipsy star particles"
        for ii in xrange(nhalos):
            h = halos[ii+1].properties
            sout.star[ii]['mass'] = h['mass']/hubble / tipsymunitmsun
            ## tipsy units: box centered at 0. (assume 0<=x<=1)
            sout.star[ii]['x'] = h['Xc']/lboxkpch - 0.5
            sout.star[ii]['y'] = h['Yc']/lboxkpch - 0.5
            sout.star[ii]['z'] = h['Zc']/lboxkpch - 0.5
            sout.star[ii]['vx'] = h['VXc']/tipsyvunitkms
            sout.star[ii]['vy'] = h['VYc']/tipsyvunitkms
            sout.star[ii]['vz'] = h['VZc']/tipsyvunitkms
            sout.star[ii]['eps'] = h['Rvir']/lboxkpch
            sout.star[ii]['metals'] = 0.
            sout.star[ii]['phi'] = 0.
            sout.star[ii]['tform'] = 0.
        print "writing tipsy outfile", outfile
        sout.write(fmt=tipsy.TipsySnap, filename=outfile)
        return sout

    def writehalos(self, snapshot, halos, hubble=None, outfile=None):
        """ Write the (ahf) halo catalog to disk.  This is really a
        wrapper that calls writegrp, writetipsy, writestat.  Writes
        .amiga.grp file (ascii group ids), .amiga.stat file (ascii
        halo catalog) and .amiga.gtp file (tipsy halo catalog).
        default outfile base simulation is same as snapshot s.
        function returns simsnap of halo catalog.
        """
        s = snapshot
        grpoutfile = s.filename+".amiga.grp"
        statoutfile = s.filename+".amiga.stat"
        tipsyoutfile = s.filename+".amiga.gtp"
        halos.writegrp(s, halos, grpoutfile)
        halos.writestat(s, halos, statoutfile, hubble=hubble)
        shalos = halos.writetipsy(s, halos, tipsyoutfile, hubble=hubble)
        return shalos
        
    @staticmethod
    def _can_load(sim):
        for file in glob.glob(sim._filename+'*z*particles*'):
            if os.path.exists(file):
                return True
        return False

    def _run_ahf(self, sim):
        # if (sim is pynbody.tipsy.TipsySnap) :
        typecode = 90
        # elif (sim is pynbody.gadget.GadgetSnap):
        #   typecode = '60' or '61'
        import pynbody.units as units
        # find AHFstep

        groupfinder = config_parser.get('AHFCatalogue', 'Path')

        if groupfinder == 'None':
            for directory in os.environ["PATH"].split(os.pathsep):
                ahfs = glob.glob(os.path.join(directory, "AHF*"))
                for iahf, ahf in enumerate(ahfs):
                    # if there are more AHF*'s than 1, it's not the last one, and
                    # it's AHFstep, then continue, otherwise it's OK.
                    if ((len(ahfs) > 1) & (iahf != len(ahfs)-1) &
                            (os.path.basename(ahf) == 'AHFstep')):
                        continue
                    else:
                        groupfinder = ahf
                        break

        if not os.path.exists(groupfinder):
            raise RuntimeError("Path to AHF (%s) is invalid" % groupfinder)

        if (os.path.basename(groupfinder) == 'AHFstep'):
            isAHFstep = True
        else:
            isAHFstep = False
        # build units file
        if isAHFstep:
            f = open('tipsy.info', 'w')
            f.write(str(sim.properties['omegaM0'])+"\n")
            f.write(str(sim.properties['omegaL0'])+"\n")
            f.write(str(sim['pos'].units.ratio(
                units.kpc, a=1)/1000.0 * sim.properties['h'])+"\n")
            f.write(str(sim['vel'].units.ratio(units.km/units.s, a=1))+"\n")
            f.write(str(sim['mass'].units.ratio(units.Msol))+"\n")
            f.close()
            # make input file
            f = open('AHF.in', 'w')
            f.write(sim._filename+" "+str(typecode)+" 1\n")
            f.write(sim._filename+"\n256\n5\n5\n0\n0\n0\n0\n")
            f.close()
        else:
            # make input file
            f = open('AHF.in', 'w')

            lgmax = np.min([int(2**np.floor(np.log2(
                1.0 / np.min(sim['eps'])))), 131072])
            # hardcoded maximum 131072 might not be necessary

            print >>f, config_parser.get('AHFCatalogue', 'Config', vars={
                'filename': sim._filename,
                'typecode': typecode,
                'gridmax': lgmax
            })

            print >>f, config_parser.get('AHFCatalogue', 'ConfigTipsy', vars={
                'omega0': sim.properties['omegaM0'],
                'lambda0': sim.properties['omegaL0'],
                'boxsize': sim['pos'].units.ratio('Mpc a h^-1', **sim.conversion_context()),
                'vunit': sim['vel'].units.ratio('km s^-1 a', **sim.conversion_context()),
                'munit': sim['mass'].units.ratio('Msol h^-1', **sim.conversion_context()),
                'eunit': 0.03  # surely this can't be right?
            })

            f.close()

        if (not os.path.exists(sim._filename)):
            os.system("gunzip "+sim._filename+".gz")
        # determine parallel possibilities

        if os.path.exists(groupfinder):
            # run it
            os.system(groupfinder+" AHF.in")
            return

    @staticmethod
    def _can_run(sim):
        if config_parser.getboolean('AHFCatalogue', 'AutoRun'):
            if config_parser.get('AHFCatalogue', 'Path') == 'None':
                for directory in os.environ["PATH"].split(os.pathsep):
                    if (len(glob.glob(os.path.join(directory, "AHF*"))) > 0):
                        return True
            else:
                path = config_parser.get('AHFCatalogue', 'Path')
                return os.path.exists(path)
        return False


#-----------------------------#
# General Grp Catalogue class #
#-----------------------------#

class GrpCatalogue(HaloCatalogue) :
    """
    A generic catalogue using a .grp file to specify which particles
    belong to which group. 
    """
    def __init__(self, sim, arr_name='grp'): 
        sim[arr_name]
    # trigger lazy-loading and/or kick up a fuss if unavailable
        self._base = weakref.ref(sim)
        self._halos = {}
        self._arr_name = arr_name
        HaloCatalogue.__init__(self)

    def _get_halo(self, i):
        if self.base is None:
            raise RuntimeError("Parent SimSnap has been deleted")

        x = Halo(i, self, self.base, np.where(self.base[self._arr_name] == i))
        if len(x) == 0 : 
            raise RuntimeError("Halo %s does not exist"%(str(i)))
        x._descriptor = "halo_"+str(i)
        return x

    @property
    def base(self):
        return self._base()

    @staticmethod
    def _can_load(sim,arr_name='grp'):
        if (arr_name in sim.loadable_keys()) or (arr_name in sim.keys()) : 
            return True
        else : 
            return False

class AmigaGrpCatalogue(GrpCatalogue):
    def __init__(self, sim, arr_name='amiga.grp'):
        GrpCatalogue.__init__(self,sim,arr_name)

    @staticmethod
    def _can_load(sim,arr_name='amiga.grp'):
        return GrpCatalogue._can_load(sim,arr_name)


#-----------------------------------------------------------------------#
# SubFind Catalogue classes -- including classes for handing HDF format #
#-----------------------------------------------------------------------#

class SubfindCatalogue(HaloCatalogue):
    """
        Class to handle catalogues produced by the SubFind halo finder. 
        Currently only imports groups (top level), no specific subhalos.
        Groups are sorted by mass (descending), most massive group is halo[0].
    """
    
    def __init__(self,sim):
        self._base=weakref.ref(sim)
        self._halos= {}
        HaloCatalogue.__init__(self)
        self.dtype_int=sim['iord'].dtype
        #self.dtype_flt=sim['x'].dtype #currently not used, but relevant for double precision Subfind output
        self.halodir=self._name_of_catalogue(sim)
        self.header=self._readheader()
        self.tasks=self.header[4]
        self.ids=self._read_ids()
        self.data_len, self.data_off=self._read_groups()

    def _get_halo(self,i):
        x = Halo(i, self, self.base, np.where(np.in1d(self.base['iord'],self.ids[self.data_off[i]:self.data_off[i]+self.data_len[i]])))
        x._descriptor = "halo_"+str(i)
        return x
    
    def _readheader(self):
        header=np.array([], dtype='int32')
        filename=self.halodir+"/subhalo_tab_"+self.halodir.split("_")[-1]+ ".0"
        fd=open(filename, "rb")
        #read header: this is strange but it works: there is an extra value in header which we delete in the next step
        header1=np.fromfile(fd, dtype='int32', sep="", count=8)
        header=np.delete(header1,4)
        fd.close()
        return header#[4]

    def _read_ids(self):
        data_ids=np.array([], dtype=self.dtype_int)
        for n in range(0,self.tasks):
            filename=self.halodir+"/subhalo_ids_"+self.halodir.split("_")[-1]+ "."+str(n)
            fd=open(filename,"rb")
            #for some reason there is an extra value in header which we delete in the next step
            header1=np.fromfile(fd, dtype='int32', sep="", count=7)
            header=np.delete(header1,4)
            #optional: include a check if both headers agree (they better)
            ids=np.fromfile(fd, dtype=self.dtype_int, sep="", count=-1 )
            fd.close()
            data_ids=np.append(data_ids, ids)
        return data_ids
            
    def _read_groups(self, ReadSubs=False, ReadAll=False):
        data_len=np.array([], dtype='int32')
        data_off=np.array([], dtype='int32')
        for n in range(0,self.tasks):
            filename=self.halodir+"/subhalo_tab_"+self.halodir.split("_")[-1]+"." +str(n)
            fd=open(filename, "rb")
            #read header (because header[0,5] changes between different files, header[4] does not) [same issue as with other headers]
            header1=np.fromfile(fd, dtype='int32', sep="", count=8)
            header=np.delete(header1,4)
            #read groups
            if header[0]>0:
                len=np.fromfile(fd, dtype='int32', sep="", count=header[0])
                offset=np.fromfile(fd, dtype='int32', sep="", count=header[0])
                #the following is for completeness, none of this information is currently used
                if ReadAll:
                    mass=np.fromfile(fd, dtype='float32', sep="", count=header[0])
                    pos=np.fromfile(fd, dtype='float32', sep="", count=3*header[0])
                    mmean200=np.fromfile(fd, dtype='float32', sep="", count=header[0])
                    rmean200=np.fromfile(fd, dtype='float32', sep="", count=header[0])
                    mcrit200=np.fromfile(fd, dtype='float32', sep="", count=header[0])
                    rcrit200=np.fromfile(fd, dtype='float32', sep="", count=header[0])
                    mtop200=np.fromfile(fd, dtype='float32', sep="", count=header[0])
                    rtop200=np.fromfile(fd, dtype='float32', sep="", count=header[0])
                    contcount=np.fromfile(fd, dtype='int32', sep="", count=header[0])
                    contmass=np.fromfile(fd, dtype='float32', sep="", count=header[0])
                    nsubs=np.fromfile(fd, dtype='int32', sep="", count=header[0])
                    firstsub=np.fromfile(fd, dtype='int32', sep="", count=header[0])
            #read subhalos only if expected to exist from header AND if ReadSubs==True (default False), because this info is not used yet
            if header[5]>0 and ReadSubs:
                print 'reading subs'
                sublen=np.fromfile(fd, dtype='int32', sep="", count=header[5])
                suboff=np.fromfile(fd, dtype='int32', sep="", count=header[5])
                if ReadAll:
                    subparent=np.fromfile(fd, dtype='int32', sep="", count=header[5])
                    submass=np.fromfile(fd, dtype='float32', sep="", count=header[5])
                    subpos=np.fromfile(fd, dtype='float32', sep="", count=3*header[5])
                    subvel=np.fromfile(fd, dtype='float32', sep="", count=3*header[5])
                    subCM=np.fromfile(fd, dtype='float32', sep="", count=3*header[5])
                    subspin=np.fromfile(fd, dtype='float32', sep="", count=3*header[5])
                    subVelDisp=np.fromfile(fd, dtype='float32', sep="", count=header[5])
                    subVMax=np.fromfile(fd, dtype='float32', sep="", count=header[5])
                    subVMaxRad=np.fromfile(fd, dtype='float32', sep="", count=header[5])
                    subHalfMassRad=np.fromfile(fd, dtype='float32', sep="", count=header[5])
                    subMostBound=np.fromfile(fd, dtype='int32', sep="", count=header[5])
                    subVMaxRad=np.fromfile(fd, dtype='float32', sep="", count=header[5])
                    #FIX: reformat 3D arrays (reshape ?) if they are supposed to be used
            fd.close()
            data_len=np.append(data_len, len)
            data_off=np.append(data_off, offset)
           
        return data_len, data_off
            
    @staticmethod
    def _name_of_catalogue(sim) :
        #standard path for multiple snapshot files
        snapnum = os.path.basename(os.path.dirname(sim.filename)).split("_")[-1]
        parent_dir = os.path.dirname(os.path.dirname(sim.filename))
        if os.path.exists(parent_dir+"/groups_"+snapnum):
            return parent_dir+"/groups_"+snapnum
        #alternative path if snapshot is single file
        else:
            snapnum = os.path.basename(sim.filename).split("_")[-1]
            parent_dir = os.path.dirname(sim.filename)
            return parent_dir+"/groups_"+snapnum

    @property
    def base(self):
        return self._base()

    @staticmethod
    def _can_load(sim):
        if os.path.exists(SubfindCatalogue._name_of_catalogue(sim)):
            return True
        else:
            return False
            

class SubFindHDFHaloCatalogue(HaloCatalogue) : 
    """
    Gadget's SubFind Halo catalogue -- used in concert with :class:`~SubFindHDFSnap`
    """

    def __init__(self, sim) : 
        super(SubFindHDFHaloCatalogue,self).__init__()
    
        self._base = weakref.ref(sim)

        # Read in the attribute units from SubFind
        try : 
            atr = sim._hdf[0]['Units'].attrs
        except KeyError : 
            warnings.warn("No unit information found: using defaults.",RuntimeWarning)
            sim._file_units_system = [units.Unit(x) for x in ('G', '1 kpc', '1e10 Msol')]
            return

        # Define the SubFind units, we will parse the attribute VarDescriptions for these
        vel_unit = atr['UnitVelocity_in_cm_per_s']*units.cm/units.s
        dist_unit = atr['UnitLength_in_cm']*units.cm
        mass_unit = atr['UnitMass_in_g']*units.g
        time_unit = atr['UnitTime_in_s']*units.s

        # Create a dictionary for the units, this will come in handy later
        unitvar = {'U_V' : vel_unit, 'U_L' : dist_unit, 'U_M' : mass_unit, 
                   'U_T' : time_unit, '[K]' : units.K, 
                   'SEC_PER_YEAR' : units.yr, 'SOLAR_MASS' : units.Msol}
        # Last two units are to catch occasional arrays like StarFormationRate which don't 
        # follow the patter of U_ units unfortunately

        # set up particle fof and subhalo group offsets
        self._fof_group_offsets = {}
        self._fof_group_lengths = {}
        self._subfind_halo_offsets = {}
        self._subfind_halo_lengths = {}


        self.ngroups = sim._hdf[0]['FOF'].attrs['Total_Number_of_groups']
        self.nsubhalos = sim._hdf[0]['FOF'].attrs['Total_Number_of_subgroups']

        self._subfind_halo_parent_groups = np.empty(self.nsubhalos)
        self._fof_group_first_subhalo = np.empty(self.ngroups)

        for ptype in sim._my_type_map.values() : 
            ptype = ptype[0] 
            self._fof_group_offsets[ptype] = np.empty(self.ngroups,dtype='int64')
            self._fof_group_lengths[ptype] = np.empty(self.ngroups,dtype='int64')
            self._subfind_halo_offsets[ptype] = np.empty(self.ngroups,dtype='int64')
            self._subfind_halo_lengths[ptype] = np.empty(self.ngroups,dtype='int64')

            curr_groups = 0
            curr_subhalos = 0

            for h in sim._hdf : 
                # fof groups
                offset = h['FOF'][ptype]['Offset']
                length = h['FOF'][ptype]['Length']
                self._fof_group_offsets[ptype][curr_groups:curr_groups + len(offset)] = offset
                self._fof_group_lengths[ptype][curr_groups:curr_groups + len(offset)] = length
                curr_groups += len(offset)

                # subfind subhalos
                offset = h['FOF'][ptype]['SUB_Offset']
                length = h['FOF'][ptype]['SUB_Length']
                self._subfind_halo_offsets[ptype][curr_subhalos:curr_subhalos + len(offset)] = offset
                self._subfind_halo_lengths[ptype][curr_subhalos:curr_subhalos + len(offset)] = length
                curr_subhalos += len(offset)
                            
        # get all the parent groups for all the subhalos
        nsub = 0
        nfof = 0
        for h in sim._hdf : 
            parent_groups = h['SUBFIND']['GrNr']
            self._subfind_halo_parent_groups[nsub:nsub+len(parent_groups)] = parent_groups
            nsub += len(parent_groups)

            first_groups = h['SUBFIND']['FirstSubOfHalo']
            self._fof_group_first_subhalo[nfof:nfof+len(first_groups)] = first_groups
            nfof += len(first_groups)

        # get the properties of fof groups and subhalos calculated by subfind
        fof_properties = {}

        fof_ignore = ['SF', 'NSF', 'Stars']

        sub_properties = {}
       
        sub_ignore = ['GrNr', 'FirstSubOfHalo', 'SubParentHalo', 'SubMostBoundID', 'InertiaTensor', 
                  'SF', 'NSF', 'NsubPerHalo', 'Stars']

        for t in sim._my_type_map.values() :
            sub_ignore.append(t[0]) # Don't add SubFind particles ever as this list is actually spherical overdensity 
            fof_ignore.append(t[0]) # Ignore here, will read in from gadgethdf 

        for key in sim._hdf[0]['FOF'].keys() :
            if key not in fof_ignore : 
                fof_properties[key] = np.array([])

        for key in sim._hdf[0]['SUBFIND'].keys() :
            if key not in sub_ignore : 
                sub_properties[key] = np.array([])

        for h in sim._hdf : 
            for key in fof_properties.keys() :
                fof_properties[key] = np.append(fof_properties[key],h['FOF'][key].value)

            for key in sub_properties.keys() :
                sub_properties[key] = np.append(sub_properties[key],h['SUBFIND'][key].value)

        for key in sub_properties.keys() : 
            arr_units = units.NoUnit()

            VarDescription = sim._hdf[0]['SUBFIND'][key].attrs['VarDescription']
            aexp = sim._hdf[0]['SUBFIND'][key].attrs['aexp-scale-exponent']
            hexp = sim._hdf[0]['SUBFIND'][key].attrs['h-scale-exponent']

            if key not in sub_ignore :
                for unitname in unitvar :
                    power = 1.
                    if unitname in VarDescription : 
                        sstart = VarDescription.find(unitname)
                        if sstart > 0 :
                            if VarDescription[sstart-1] == "/" :
                                power *= -1.
                        if len(VarDescription) > sstart+len(unitname):
                            if VarDescription[sstart+len(unitname)] == '^' :
                                ## Has an index, check if this is negative
                                if VarDescription[sstart+len(unitname)+1] == "-" :
                                    power *= -1.
                                    power *= float(VarDescription[sstart+len(unitname)+2:-1].split()[0]) ## Search for the power
                                else:
                                    power *= float(VarDescription[sstart+len(unitname)+1:-1].split()[0]) ## Search for the power

                        if hasattr(arr_units, '_no_unit'):
                            arr_units = unitvar[unitname]**power ## Set the new units
                        else:
                            arr_units *= unitvar[unitname]**power ## Combine the units
                ## Now the cosmological units
                arr_units *= (((units.a)**aexp) * (units.h)**hexp)

            try : 
                fof_properties[key] = fof_properties[key].view(SimArray)
                fof_properties[key].units = arr_units
            except KeyError :
                pass

            sub_properties[key] = sub_properties[key].view(SimArray)
            sub_properties[key].units = arr_units

        # set the sim 
        for arr in fof_properties.values() + sub_properties.values() : 
            try : 
                arr.sim = sim
            except AttributeError : 
                pass

        # reshape multi-D arrays
        for key in sub_properties.keys() : 
            # Test if there are no remainders, i.e. array is multiple of halo length
            # then solve for the case where this is 1, 2 or 3 dimension
            if len(sub_properties[key]) % self.nsubhalos == 0:
                ndim = len(sub_properties[key])/self.nsubhalos
                if ndim > 1 : 
                    sub_properties[key] = sub_properties[key].reshape(self.nsubhalos,ndim)

            try: 
                # The case fof FOF 
                if len(fof_properties[key]) % self.ngroups == 0:
                    ndim = len(fof_properties[key])/self.ngroups
                    if ndim > 1 :
                        fof_properties[key] = fof_properties[key].reshape(self.ngroups,ndim)
            except KeyError :
                pass
                        
        self._fof_properties = fof_properties
        self._sub_properties = sub_properties
                
    def _get_halo(self, i) : 
        if self.base is None : 
            raise RuntimeError("Parent SimSnap has been deleted")
        
        if i > len(self)-1 : 
            raise RuntimeError("Group %d does not exist"%i)

        type_map = self.base._my_type_map

        # create the particle lists
        tot_len = 0
        for g_ptype in type_map.values() : 
            g_ptype = g_ptype[0]
            tot_len += self._fof_group_lengths[g_ptype][i]

        plist = np.zeros(tot_len,dtype='int64')

        npart = 0
        for ptype in type_map.keys() : 
            # family slice in the SubFindHDFSnap 
            sl = self.base._family_slice[ptype]
            
            # gadget ptype
            g_ptype = type_map[ptype][0]

            # add the particle indices to the particle list
            offset = self._fof_group_offsets[g_ptype][i]
            length = self._fof_group_lengths[g_ptype][i]
            ind = np.arange(sl.start + offset, sl.start + offset + length) 
            plist[npart:npart+length] = ind
            npart += length
            
        return SubFindFOFGroup(i, self, self.base, plist)
        

    def __len__(self) : 
        return self.base._hdf[0]['FOF'].attrs['Total_Number_of_groups']
        
        
    @property
    def base(self):
        return self._base()



class SubFindFOFGroup(Halo) : 
    """
    SubFind FOF group class
    """

    def __init__(self, group_id, *args) : 
        super(SubFindFOFGroup,self).__init__(group_id, *args)

        self._subhalo_catalogue = SubFindHDFSubhaloCatalogue(group_id, self._halo_catalogue)
        
        self._descriptor = "fof_group_"+str(group_id)

        # load properties
        for key in self._halo_catalogue._fof_properties.keys() : 
            self.properties[key] = SimArray(self._halo_catalogue._fof_properties[key][group_id],
                                            self._halo_catalogue._fof_properties[key].units)
            self.properties[key].sim = self.base
            

    def __getattr__(self, name):
        if name == 'sub':
            return self._subhalo_catalogue
        else : 
            return super(SubFindFOFGroup,self).__getattr__(name)

        
class SubFindHDFSubhaloCatalogue(HaloCatalogue) : 
    """
    Gadget's SubFind HDF Subhalo catalogue. 

    Initialized with the parent FOF group catalogue and created
    automatically when an fof group is created
    """

    def __init__(self, group_id, group_catalogue) : 
        super(SubFindHDFSubhaloCatalogue,self).__init__()

        self._base = weakref.ref(group_catalogue.base)
        
        self._group_id = group_id
        self._group_catalogue = group_catalogue



    def __len__(self): 
        if self._group_id == (len(self._group_catalogue._fof_group_first_subhalo)-1) : 
            return self._group_catalogue.nsubhalos - self._group_catalogue._fof_group_first_subhalo[self._group_id]
        else: 
            return (self._group_catalogue._fof_group_first_subhalo[self._group_id + 1] - 
                    self._group_catalogue._fof_group_first_subhalo[self._group_id])
        
    def _get_halo(self, i):
        if self.base is None : 
            raise RuntimeError("Parent SimSnap has been deleted")
        
        if i > len(self)-1 : 
            raise RuntimeError("FOF group %d does not have subhalo %d"%(self._group_id, i))

        # need this to index the global offset and length arrays
        absolute_id = self._group_catalogue._fof_group_first_subhalo[self._group_id] + i
                               
        # now form the particle IDs needed for this subhalo
        type_map = self.base._my_type_map

        halo_lengths = self._group_catalogue._subfind_halo_lengths
        halo_offsets = self._group_catalogue._subfind_halo_offsets

        # create the particle lists
        tot_len = 0
        for g_ptype in type_map.values() : 
            g_ptype = g_ptype[0]
            tot_len += halo_lengths[g_ptype][absolute_id]

        plist = np.zeros(tot_len,dtype='int64')

        npart = 0
        for ptype in type_map.keys() : 
            # family slice in the SubFindHDFSnap 
            sl = self.base._family_slice[ptype]
            
            # gadget ptype
            g_ptype = type_map[ptype][0]

            # add the particle indices to the particle list
            offset = halo_offsets[g_ptype][absolute_id]
            length = halo_lengths[g_ptype][absolute_id]
            ind = np.arange(sl.start + offset, sl.start + offset + length) 
            plist[npart:npart+length] = ind
            npart += length
            
        return SubFindHDFSubHalo(i, self._group_id, self, self.base, plist)
        
        
    @property
    def base(self) : 
        return self._base()
        
class SubFindHDFSubHalo(Halo) : 
    """
    SubFind subhalo class
    """

    def __init__(self,halo_id, group_id, *args) : 
        super(SubFindHDFSubHalo,self).__init__(halo_id, *args)
    
        self._group_id = group_id
        self._descriptor = "fof_group_%d_subhalo_%d"%(group_id,halo_id)

        # need this to index the global offset and length arrays
        absolute_id = self._halo_catalogue._group_catalogue._fof_group_first_subhalo[self._group_id] + halo_id
        
        # load properties
        sub_props = self._halo_catalogue._group_catalogue._sub_properties
        for key in sub_props : 
            self.properties[key] = SimArray(sub_props[key][absolute_id], sub_props[key].units)
            self.properties[key].sim = self.base
        


# AmigaGrpCatalogue MUST be scanned first, because if it exists we probably
# want to use it, but an AHFCatalogue will probably be on-disk too.
_halo_classes = [GrpCatalogue, AmigaGrpCatalogue, AHFCatalogue, RockstarCatalogue, SubfindCatalogue, SubFindHDFHaloCatalogue]
_runable_halo_classes = [AHFCatalogue, RockstarCatalogue]

