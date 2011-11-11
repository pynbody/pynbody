"""

tipsy
=====

Implements classes and functions for handling tipsy files.  You rarely
need to access this module directly as it will be invoked
automatically via pynbody.load.

"""

from __future__ import with_statement # for py2.5

from . import snapshot, array, util
from . import family
from . import units
from . import config, config_parser

import struct, os
import numpy as np
import gzip
import sys
import warnings

class TipsySnap(snapshot.SimSnap) :
    def __init__(self, filename, only_header=False, must_have_paramfile=False) :

        global config

        super(TipsySnap,self).__init__()
        
        self._filename = util.cutgz(filename)
    
        f = util.open_(filename)
    
        if config['verbose'] : print>>sys.stderr, "TipsySnap: loading ",filename

        t, n, ndim, ng, nd, ns = struct.unpack("diiiii", f.read(28))
        if (ndim > 3 or ndim < 1):
            self._byteswap=True
            f.seek(0)
            t, n, ndim, ng, nd, ns = struct.unpack(">diiiii", f.read(28))
        else :
            self._byteswap=False
            
        # In non-cosmological simulations, what is t? Is it physical
        # time?  In which case, we need some way of telling what we
        # are dealing with and setting properties accordingly.
        self.properties['a'] = t
        try:
            self.properties['z'] = 1.0/t - 1.0
        except ZeroDivisionError:
            self.properties['z'] = None

        assert ndim==3

        self._num_particles = ng+nd+ns
        f.read(4)

        # Store slices corresponding to different particle types
        self._family_slice[family.gas] = slice(0,ng)
        self._family_slice[family.dm] = slice(ng, nd+ng)
        self._family_slice[family.star] = slice(nd+ng, ng+nd+ns)

        self._create_arrays(["pos","vel"],3)
        self._create_arrays(["mass","eps","phi"])
        self.gas._create_arrays(["rho","temp","metals"])
        self.star._create_arrays(["metals","tform"])

        self.gas["temp"].units = "K" # we know the temperature is always in K
        # Other units will be set by the decorators later


        # Load in the tipsy file in blocks.  This is the most
        # efficient balance I can see for keeping IO contiguuous, not
        # wasting memory, but also not having too much python <-> C
        # interface overheads

        max_block_size = 1024 # particles

        # describe the file structure as list of (num_parts, [list_of_properties]) 
        file_structure = ((ng,family.gas,["mass","x","y","z","vx","vy","vz","rho","temp","eps","metals","phi"]),
                  (nd,family.dm,["mass","x","y","z","vx","vy","vz","eps","phi"]),
                  (ns,family.star,["mass","x","y","z","vx","vy","vz","metals","tform","eps","phi"]))

        self._decorate()

        if  (not self._paramfile.has_key('achOutName')) :
            if must_have_paramfile :
                raise RuntimeError, "Could not find .param file for this run. Place it in the run's directory or parent directory."
            else :
                warnings.warn("No readable param file in the run directory or parent directory: using defaults.",RuntimeWarning)

        time_unit = None
        try :
            time_unit = self.infer_original_units('yr')
        except units.UnitsException :
            pass

        if self._paramfile.has_key('bComove') and int(self._paramfile['bComove'])!=0 :
            from . import analysis
            import analysis.cosmology
            self.properties['a'] = t
            try :
                self.properties['z'] = 1.0/t - 1.0
            except ZeroDivisionError :
                # no sensible redshift
                pass

            if self.properties['z'] is not None and self._paramfile.has_key('dMsolUnit') and self._paramfile.has_key('dKpcUnit'):
                self.properties['time'] =  analysis.cosmology.age(self, unit=time_unit)
            else :
                # something has gone wrong with the cosmological side of
                # things
                warnings.warn("Paramfile suggests time is cosmological, but header values are not sensible in this context.", RuntimeWarning)
                self.properties['time'] = t

        else :
            # Assume a non-cosmological run
            self.properties['time'] =  t

        if time_unit is not None :
            self.properties['time']*=time_unit

        if only_header == True:
            return

        for n_left, type, st in file_structure :
            n_done = 0
            self_type = self[type]
            while n_left>0 :
                n_block = min(n_left,max_block_size)


                # Read in the block
                if(self._byteswap):
                    g = np.fromstring(f.read(len(st)*n_block*4),'f').byteswap().reshape((n_block,len(st)))
                else:
                    g = np.fromstring(f.read(len(st)*n_block*4),'f').reshape((n_block,len(st)))

                # Copy into the correct arrays
                for i, name in enumerate(st) :
                    self_type[name][n_done:n_done+n_block] = g[:,i]

                # Increment total ptcls read in, decrement ptcls left of this type
                n_left-=n_block
                n_done+=n_block


    def _update_loadable_keys(self)  :
        def is_readable_array(x) :
            try:
                f = util.open_(x)
                return int(f.readline()) == len(self)
            except ValueError :
                # could be a binary file
                f.seek(0)

                if hasattr(f,'fileobj') :
                    # Cludge to get un-zipped length
                    fx = f.fileobj
                    fx.seek(-4, 2)
                    buflen = gzip.read32(fx)
                    ourlen_1 = ((len(self)*4)+4) & 0xffffffffL
                    ourlen_3 = ((len(self)*3*4)+4) & 0xffffffffL
                    
                else :
                    buflen = os.path.getsize(x)
                    ourlen_1 = ((len(self)*4)+4)
                    ourlen_3 = ((len(self)*4)+4)

                if buflen==ourlen_1 : # it's a vector
                    return True
                elif buflen==ourlen_3 : # it's an array
                    return True
                else :
                    return False
                
            except IOError :
                return False

        import glob

        fs = map(util.cutgz,glob.glob(self._filename+".*"))
        res =  map(lambda q: q[len(self._filename)+1:], 
                   filter(is_readable_array, fs))

        # Create an empty dictionary of sets to store the loadable
        # arrays for each family
        rdict = dict([(x, set()) for x in self.families()])

        # Now work out which families can load which arrays
        # according to the stored metadata
        for r in res :
            fams = self._get_loadable_array_metadata(r)[1]
            for x in fams or self.families() :
                rdict[x].add(r)

        self._loadable_keys_registry = rdict
            
    def loadable_keys(self, fam=None) :
        """Produce and return a list of loadable arrays for this TIPSY file."""
        if len(self._loadable_keys_registry) is 0 :
            self._update_loadable_keys()

        if fam is not None :
            # Return what is loadable for this family
            return list(self._loadable_keys_registry[fam])
        else :
            # Return what is loadable to all families
            return list(set.intersection(*self._loadable_keys_registry.values()))

    
    @staticmethod
    def _write(self, filename=None) :
        """Write a TIPSY file.  Just the reverse of reading a file. """

        global config
        
        with self.lazy_off : # prevent any lazy reading or evaluation
        
            if filename is None :
                filename = self._filename

            if config['verbose'] : print>>sys.stderr, "TipsySnap: writing main file as",filename

            f = util.open_(filename, 'w')

            try:
                t = self.properties['a']
            except KeyError :
                warnings.warn("Time is unknown: writing zero in header",RuntimeWarning)
                t = 0

            n = len(self)
            ndim = 3
            ng = len(self.gas)
            nd = len(self.dark)
            ns = len(self.star)


            byteswap = getattr(self, "_byteswap", None)

            if byteswap: 
                f.write(struct.pack(">diiiiii", t,n,ndim,ng,nd,ns,0))
            else:
                f.write(struct.pack("diiiiii", t,n,ndim,ng,nd,ns,0))

                
            # needs to be done in blocks like reading
            # describe the file structure as list of (num_parts, [list_of_properties]) 
            file_structure = ((ng,family.gas,["mass","x","y","z","vx","vy","vz","rho","temp","eps","metals","phi"]),
                              (nd,family.dm,["mass","x","y","z","vx","vy","vz","eps","phi"]),
                              (ns,family.star,["mass","x","y","z","vx","vy","vz","metals","tform","eps","phi"]))
            
            max_block_size = 1024 # particles
            for n_left, type, st in file_structure :
                n_done = 0
                self_type = self[type]
                while n_left>0 :
                    n_block = min(n_left,max_block_size)                   

                    g = np.zeros((n_block,len(st)),dtype=np.float32)

                    # Copy from the correct arrays
                    for i, name in enumerate(st) :
                        try:
                            g[:,i] =np.float32(self_type[name][n_done:n_done+n_block])
                        except KeyError :
                            pass

                    # Write out the block
                    if byteswap :
                        g.byteswap().tofile(f)
                    else:
                        g.tofile(f)

                    # Increment total ptcls written, decrement ptcls left of this type
                    n_left-=n_block
                    n_done+=n_block

            f.close()
            
            if config['verbose'] : print>>sys.stderr, "TipsySnap: writing auxiliary arrays"

            for x in set(self.keys()).union(self.family_keys()) :
                if not self.is_derived_array(x) and x not in ["mass","pos","x","y","z","vel","vx","vy","vz","rho","temp",
                                                              "eps","metals","phi", "tform"]  :
                    TipsySnap._write_array(self, x, filename=filename+"."+x)
    

    @staticmethod
    def __write_block(f,ar,binary, byteswap) :
        if issubclass(ar.dtype.type, np.integer) :
            ar = np.asarray(ar, dtype=np.int32)
        else :
            ar = np.asarray(ar, dtype=np.float32)

        if binary :
            if byteswap:
                ar.byteswap().tofile(f)
            else :
                ar.tofile(f)

        else :
            if issubclass(ar.dtype.type, np.integer) :
                fmt = "%d"
            else :
                fmt = "%g"
            np.savetxt(f, ar, fmt=fmt)


    def _get_loadable_array_metadata(self, array_name) :
        """Given an array name, returns the metadata consisting of
        the tuple units, families.

        Returns:
         *units*: the units of this data on disk
         *families*: a list of family objects for which data is on disk
                     for this array, or None if this cannot be determined"""

        try:
            f = open(self.filename+"."+array_name+".pynbody-meta")
        except IOError :
            u = units._default_units.get(array_name,None)
            if u is not None :
                u = self.infer_original_units(u)
            return u, None
        
        res = {}
        for l in f :
            X = l.split(":")
            if len(X)==2 :
                res[X[0].strip()] = X[1].strip()

        try:
            u = units.Unit(res['units'])
        except :
            u = None
        try:
            fams = [family.get_family(x.strip()) for x in res['families'].split(",")]
        except :
            fams = None
            
        return u,fams
        
        
    @staticmethod
    def _write_array_metafile(self, filename, units, families) :
        
        f = open(filename+".pynbody-meta","w")
        print>>f, "# This file automatically created by pynbody"
        if not hasattr(units,"_no_unit") :
            print>>f, "units:",units
        print>>f, "families:",
        for x in families :
            print>>f,x.name,
        
    @staticmethod
    def _write_array(self, array_name, fam_out=None, 
                     filename=None, binary=None, byteswap=None) :
        """Write a TIPSY-ASCII file."""

        units_out = None
        
        if binary is None :
            binary = getattr(self.ancestor,"_tipsy_arrays_binary",False)
            
        if binary and ( byteswap is None ) :
            byteswap = getattr(self.ancestor, "_byteswap", False)

        if array_name in ["mass","pos","x","y","z","vel","vx","vy","vz","rho","temp",
              "eps","metals","phi"] :
            raise RuntimeError, "Cannot write back into TIPSY file." 

        with self.lazy_off : # prevent any lazy reading or evaluation
            if filename is None :
                if self._filename[-3:] == '.gz' :
                    filename = self._filename[:-3]+"."+array_name+".gz"
                else :
                    filename = self._filename+"."+array_name

            # if the aux file already exists for this array, read it in
            # and replace only the families that are needed

            try : 
                f = util.open_(filename, 'r')
                data = self.__read_array_from_disk(array_name, filename = filename)
                replace = True
                f.close()

            except IOError:
                replace = False

            if binary :
                f = util.open_(filename, 'wb')
                    
                if byteswap :
                    f.write(struct.pack(">i", len(self)))
    
                else :
                    f.write(struct.pack("i", len(self)))
                
            else :
                f = util.open_(filename, 'w')
                print>>f, str(len(self))
        

            if array_name in self.family_keys() or fam_out is not None :

                # if the aux file for this array already exists, get the families and units
                if replace :
                    aux_u, aux_f = self._get_loadable_array_metadata(array_name)

                for fam in [family.gas, family.dm, family.star] :
                    try:
                        ar = self[fam][array_name]
                        units_out = ar.units
                           
                    except KeyError :
                        ar = np.zeros(len(self[fam]), dtype=int)
                    
                    if replace :
                        # if we are replacing data and the current family is *not* requested
                        # for output, then use the data from the auxiliary array on disk
                        if fam not in fam_out :
                            ar = data[self._get_family_slice(fam)]
                        
                        # if the current family array is written, then make sure it's 
                        # in the same units as the existing auxiliary array
                        else : 
                            try : 
                                ar.convert_units(aux_u)
                                units_out = aux_u
                            except units.UnitsException : 
                                raise IOError("Units must match the existing auxiliary array on disk.")
                    TipsySnap.__write_block(f, ar, binary, byteswap)
                    
            else :
                ar = self[array_name]
                units_out = ar.units
                TipsySnap.__write_block(f, ar, binary, byteswap)

        f.close()
        families = [xfam for xfam in self.families() if array_name in self[xfam]]
        TipsySnap._write_array_metafile(self,filename, units_out, families)
            

    def _load_array(self, array_name, fam=None, filename = None,
                    packed_vector = None) :

        fams = self._get_loadable_array_metadata(array_name)[1]
        if fams is not None and (fam or self.families())  not in fams :
            raise IOError, "This array is marked as available only for families %s"%fams
        
        data = self.__read_array_from_disk(array_name, fam=fam,
                                           filename=filename,
                                           packed_vector=packed_vector)

        
        if fam is None: 
            self[array_name] = data
        else : 
            self[fam][array_name] = data
        

    def __read_array_from_disk(self, array_name, fam=None, filename = None, 
                               packed_vector = None) :
        """Read a TIPSY-ASCII or TIPSY-BINARY auxiliary file with the
        specified name. If fam is not None, read only the particles of
        the specified family."""

        if filename is None and array_name in ['massform', 'rhoform', 'tempform','phiform','nsmooth'] :
            self.read_starlog()
            return

        import sys, os
	
	# N.B. this code is a bit inefficient for loading
	# family-specific arrays, because it reads in the whole array
	# then slices it.  But of course the memory is only wasted
	# while still inside this routine, so it's only really the
	# wasted time that's a problem.
	
        # determine whether the array exists in a file
        
        if filename is None :
            if self._filename[-3:] == '.gz' :
                filename = self._filename[:-3]+"."+array_name
            else :
                filename = self._filename+"."+array_name
                
        f = util.open_(filename)

        if config['verbose'] : print>>sys.stderr, "TipsySnap: attempting to load auxiliary array",filename
        # if we get here, we've got the file - try loading it
  
	try :
	    l = int(f.readline())
            binary = False
            if l!=len(self) :
                raise IOError, "Incorrect file format"
	
            # Inspect the first line to see whether it's float or int
            l = "0\n"
            while l=="0\n" : l = f.readline()
            if "." in l or "e" in l :
                tp = float
            else :
                tp = int

            # Restart at head of file
            f.seek(0)

            f.readline()
            data = np.fromfile(f, dtype=tp, sep="\n")
	except ValueError :
            # this is probably a binary file
            binary = True
            f.seek(0)

            # Read header and check endianness
            if self._byteswap:
                l = struct.unpack(">i",f.read(4))[0]
            else:
                l = struct.unpack("i", f.read(4))[0]

            if l!=len(self) :
                raise IOError, "Incorrect file format"

            # Set data format to be read (float or int) based on config
            int_arrays = map(str.strip, config_parser.get('tipsy', 'binary-int-arrays').split(","))
            if array_name in int_arrays : fmt = 'i'
            else: fmt = 'f'

            # Read longest data array possible.  
            # Assume byteswap since most will be.
            if self._byteswap:
                data = np.fromstring(f.read(3*len(self)*4),fmt).byteswap()
            else:
                data = np.fromstring(f.read(3*len(self)*4),fmt)
            if len(f.read(4))!= 0:
                raise IOError, "Incorrect file format"
            
        ndim = len(data)/len(self)

        if ndim*len(self)!=len(data) :
            raise IOError, "Incorrect file format"

        if ndim>1 :
            dims = (len(self),ndim)

            # check whether the vector format is specified in the param file
            # this only matters for binary because ascii files use packed vectors by default
            if (binary) and (packed_vector == None) :
                # default bPackedVector = 0, so must use column-major format when reshaping
                v_order = 'F'
                if self._paramfile != "" :
                    try:
                        if int(self._paramfile.get("bPackedVector",0)) :
                            v_order="C"
                    except ValueError :
                        pass
                    
            elif ((packed_vector is True) or (binary is False)) and (packed_vector is None) :
                if config['verbose']:
                    print>>sys.stderr, 'Warning: assuming packed vector format!'
                    print>>sys.stderr, 'Packed vector means values are in order x1, y1, z1... xn, yn, zn'
                v_order = 'C'
            else :
                v_order = 'F'
        else :
            dims = len(self)
            v_order = 'C'

        self.ancestor._tipsy_arrays_binary = binary

        if fam is None :
            r = data.reshape(dims, order=v_order).view(array.SimArray)
        else :
            r = data.reshape(dims,order=v_order).view(array.SimArray)[self._get_family_slice(fam)]

        u, f = self._get_loadable_array_metadata(array_name)
        if u is not None :
            r.units = u

        return r
        
    def read_starlog(self, fam=None) :
	"""Read a TIPSY-starlog file."""
	
        import sys, os, glob, pynbody.bridge
        x = os.path.abspath(self._filename)
        done = False
        filename=None
        x = os.path.dirname(x)
        # Attempt the loading of information
        l = glob.glob(os.path.join(x,"*.starlog"))
        if (len(l)) :
            for filename in l :
                sl = StarLog(filename)
        else:
            l = glob.glob(os.path.join(x,"../*.starlog"))
            if (len(l) == 0): raise IOError, "Couldn't find starlog file"
            for filename in l:
                sl = StarLog(filename)

        if config['verbose'] : print "Bridging starlog into SimSnap"
        b = pynbody.bridge.OrderBridge(self,sl)
        b(sl).star['iorderGas'] = sl.star['iorderGas'][:len(self.star)]
        b(sl).star['massform'] = sl.star['massform'][:len(self.star)]
        b(sl).star['rhoform'] = sl.star['rhoform'][:len(self.star)]
        b(sl).star['tempform'] = sl.star['tempform'][:len(self.star)]
        b(sl)['posform'] = sl['pos'][:len(self.star),:]
        b(sl)['velform'] = sl['vel'][:len(self.star),:]
                
	
    @staticmethod
    def _can_load(f) :
	# to implement!
	return True

# caculate the number fraction YH, YHe as a function of metalicity. Cosmic 
# production rate of helium relative to metals (in mass)  
# delta Y/delta Z = 2.1 and primordial He Yp = 0.236 (Jimenez et al. 2003, 
# Science 299, 5612. 
#  piecewise linear
#  Y = Yp + dY/dZ*ZMetal up to ZMetal = 0.1, then linear decrease to 0 at Z=1)  

#  SUM_Metal = sum(ni/nH *mi),it is a fixed number for cloudy abundance. 
#  Massfraction fmetal = Z*SUM_metal/(1 + 4*nHe/nH + Z*SUM_metal) (1)
#  4*nHe/nH = mHe/mH = fHe/fH 
#  also fH + fHe + fMetal = 1  (2)
#  if fHe specified, combining the 2 eq above will solve for 
#  fH and fMetal 
        
def _abundance_estimator(metal) :

    Y_He = ((0.236+2.1*metal)/4.0)*(metal<=0.1)
    Y_He+= ((-0.446*(metal-0.1)/0.9+0.446)/4.0)*(metal>0.1)
    Y_H = 1.0-Y_He*4. - metal

    return Y_H, Y_He

@TipsySnap.derived_quantity
def HII(sim) :
    """Number of HII ions per proton mass"""
    Y_H, Y_He = _abundance_estimator(sim["metals"])
    return Y_H - sim["HI"]

@TipsySnap.derived_quantity
def HeIII(sim) :
    """Number of HeIII ions per proton mass"""
    Y_H, Y_He = _abundance_estimator(sim["metals"])
    return Y_He-sim["HeII"]-sim["HeI"]

@TipsySnap.derived_quantity
def ne(sim) :
    """Number of electrons per proton mass"""
    return sim["HII"] + sim["HeII"] + 2*sim["HeIII"]
    
@TipsySnap.derived_quantity
def mu(sim) :
    """Relative atomic mass, i.e. number of particles per
    proton mass, ignoring metals (since we generally only know the
    mass fraction of metals, not their specific atomic numbers)"""
    
    x =  sim["HI"]+2*sim["HII"]+sim["HeI"]+2*sim["HeII"]+3*sim["HeIII"]
    
    x.units = 1/units.m_p
    return x
    
@TipsySnap.derived_quantity
def u(sim) :
    """Specific Internal energy"""
    u = (3./2) * units.k * sim["temp"] # per particle
    u=u*sim["mu"] # per unit mass
    u.convert_units("eV m_p^-1")
    return u

@TipsySnap.derived_quantity
def p(sim) :
    """Pressure"""
    p = sim["u"]*sim["rho"]*(2./3)
    p.convert_units("dyn")
    return p

@TipsySnap.derived_quantity
def hetot(self) :
    return 0.236+(2.1*self['metals'])

@TipsySnap.derived_quantity
def hydrogen(self) :
    return 1.0-self['metals']-self['hetot']

#from .tipsy import TipsySnap
# Asplund et al (2009) ARA&A solar abundances (Table 1)
# m_frac = 10.0^([X/H] - 12)*M_X/M_H*0.74
# OR
# http://en.wikipedia.org/wiki/Abundance_of_the_chemical_elements      
# puts stuff more straighforwardly cites Arnett (1996)
# A+G from http://www.t4.lanl.gov/opacity/grevand1.html
# Anders + Grev (1989)    Asplund
XSOLFe=0.125E-2         # 1.31e-3
# Looks very wrong ([O/Fe] ~ 0.2-0.3 higher than solar), 
# probably because SN ejecta are calculated with
# Woosley + Weaver (1995) based on Anders + Grevesse (1989)
# XSOLO=0.59E-2           # 5.8e-2
XSOLO=0.84E-2
XSOLH=0.706             # 0.74
XSOLC=3.03e-3           # 2.39e-3
XSOLN=9.2e-4          # 7e-4
XSOLNe=1.66e-3          # 1.26e-3
XSOLMg=6.44e-4          # 7e-4
XSOLSi=7e-4          # 6.7e-4


@TipsySnap.derived_quantity
def feh(self) :
    minfe = np.amin(self['FeMassFrac'][np.where(self['FeMassFrac'] > 0)])
    self['FeMassFrac'][np.where(self['FeMassFrac'] == 0)]=minfe
    return np.log10(self['FeMassFrac']/self['hydrogen']) - np.log10(XSOLFe/XSOLH)

@TipsySnap.derived_quantity
def ofe(self) :
    minox = np.amin(self['OxMassFrac'][np.where(self['OxMassFrac'] > 0)])
    self['OxMassFrac'][np.where(self['OxMassFrac'] == 0)]=minox
    minfe = np.amin(self['FeMassFrac'][np.where(self['FeMassFrac'] > 0)])
    self['FeMassFrac'][np.where(self['FeMassFrac'] == 0)]=minfe
    return np.log10(self['OxMassFrac']/self['FeMassFrac']) - np.log10(XSOLO/XSOLFe)

@TipsySnap.derived_quantity
def mgfe(sim) :
    minmg = np.amin(sim['MgMassFrac'][np.where(sim['MgMassFrac'] > 0)])
    sim['MgMassFrac'][np.where(sim['MgMassFrac'] == 0)]=minmg
    minfe = np.amin(sim['FeMassFrac'][np.where(sim['FeMassFrac'] > 0)])
    sim['FeMassFrac'][np.where(sim['FeMassFrac'] == 0)]=minfe
    return np.log10(sim['MgMassFrac']/sim['FeMassFrac']) - np.log10(XSOLMg/XSOLFe)

@TipsySnap.derived_quantity
def nefe(sim) :
    minne = np.amin(sim['NeMassFrac'][np.where(sim['NeMassFrac'] > 0)])
    sim['NeMassFrac'][np.where(sim['NeMassFrac'] == 0)]=minne
    minfe = np.amin(sim['FeMassFrac'][np.where(sim['FeMassFrac'] > 0)])
    sim['FeMassFrac'][np.where(sim['FeMassFrac'] == 0)]=minfe
    return np.log10(sim['NeMassFrac']/sim['FeMassFrac']) - np.log10(XSOLNe/XSOLFe)

@TipsySnap.derived_quantity
def sife(sim) :
    minsi = np.amin(sim['SiMassFrac'][np.where(sim['SiMassFrac'] > 0)])
    sim['SiMassFrac'][np.where(sim['SiMassFrac'] == 0)]=minsi
    minfe = np.amin(sim['FeMassFrac'][np.where(sim['FeMassFrac'] > 0)])
    sim['FeMassFrac'][np.where(sim['FeMassFrac'] == 0)]=minfe
    return np.log10(sim['SiMassFrac']/sim['FeMassFrac']) - np.log10(XSOLSi/XSOLFe)

@TipsySnap.derived_quantity
def c_s(self) :
    """Ideal gas sound speed"""
    #x = np.sqrt(5./3.*units.k*self['temp']*self['mu'])
    x = np.sqrt(5./3.*self['p']/self['rho'])
    x.convert_units('km s^-1')
    return x

@TipsySnap.derived_quantity
def c_s_turb(self) :
    """Turbulent sound speed (from Mac Low & Klessen 2004)"""
    x = np.sqrt(self['c_s']**2+1./3*self['v_disp']**2)
    x.convert_units('km s^-1')
    return x

@TipsySnap.derived_quantity
def mjeans(self) :
    """Classical Jeans mass"""
    x = np.pi**(5./2.)*self['c_s']**3/(6.*units.G**(3,2)*self['rho']**(1,2))
    x.convert_units('Msol')
    return x

@TipsySnap.derived_quantity
def mjeans_turb(self) :
    """Turbulent Jeans mass"""
    x = np.pi**(5./2.)*self['c_s_turb']**3/(6.*units.G**(3,2)*self['rho']**(1,2))
    x.convert_units('Msol')
    return x

@TipsySnap.derived_quantity
def ljeans(self) :
    """Jeans length"""
    x = self['c_s']*np.sqrt(np.pi/(units.G*self['rho']))
    x.convert_units('kpc')
    return x

@TipsySnap.derived_quantity
def ljeans_turb(self) :
    """Jeans length"""
    x = self['c_s_turb']*np.sqrt(np.pi/(units.G*self['rho']))
    x.convert_units('kpc')
    return x

class StarLog(snapshot.SimSnap):
    def __init__(self, filename):
        import os
        super(StarLog,self).__init__()
        self._filename = filename

        f = util.open_(filename)
        self.properties = {}
        bigstarlog = False
        
        file_structure = np.dtype({'names': ("iord","iorderGas","tform","x","y","z","vx","vy","vz","massform","rhoform","tempform"),'formats':('i4','i4','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8')})

        size = struct.unpack("i", f.read(4))
        if (size[0]> 1000 or size[0]<  10):
            self._byteswap=True
            f.seek(0)
            size = struct.unpack(">i", f.read(4))
        iSize = size[0]

        if (iSize > file_structure.itemsize) :
            file_structure = np.dtype({'names': ("iord","iorderGas","tform","x","y","z","vx","vy","vz","massform","rhoform","tempform","phiform","nsmooth"),'formats':('i4','i4','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','i4')})
            if (iSize != file_structure.itemsize and iSize != 104):
                raise ValueError, "Unknown starlog structure iSize:"+str(iSize)+", file_structure itemsize:"+str(file_structure.itemsize)
            else : bigstarlog = True
            
        datasize = os.path.getsize(filename)-f.tell()
        if config['verbose'] : print "Reading "+filename
        if(self._byteswap):
            g = np.fromstring(f.read(datasize),dtype=file_structure).byteswap()
        else:
            g = np.fromstring(f.read(datasize),dtype=file_structure)

        # hoping to provide backward compatibility for np.unique by
        # copying relavent part of current numpy source:
        # numpy/lib/arraysetops.py:192 (on 22nd March 2011)
        tmp = g['iord'].flatten()
        perm = tmp.argsort()
        aux = tmp[perm]
        flag = np.concatenate(([True],aux[1:]!=aux[:-1]))
        iord = aux[flag]; indices = perm[flag]

        self._num_particles = len(indices)

        self._family_slice[family.star] = slice(0, len(indices))
        self._create_arrays(["pos","vel"],3)
        self._create_arrays(["iord"])
        self._create_arrays(["iorderGas","massform","rhoform","tempform","metals","tform"])
        if bigstarlog :
            self._create_arrays(["phiform","nsmooth"])

        self._decorate()
        for name in file_structure.fields.keys() :
            self.star[name][:] = g[name][indices]

        #self._decorate()


@TipsySnap.decorator
@StarLog.decorator
def load_paramfile(sim) :
    import sys, os, glob
    x = os.path.abspath(sim._filename)
    done = False
    sim._paramfile = {}
    filename=None
    for i in xrange(2) :
        x = os.path.dirname(x)
	l = glob.glob(os.path.join(x,"*.param"))

	for filename in l :
	    # Attempt the loading of information
	    try :
		f = file(filename)
	    except IOError :
                l = glob.glob(os.path.join(x,"../*.param"))
                try : 
                    for filename in l:
                        f = file(filename)
                except IOError:
                    continue
	    
            
            for line in f :
		try :
                    if line[0]!="#" :
                        s = line.split("#")[0].split()
                        sim._paramfile[s[0]] = " ".join(s[2:])
                                
		except IndexError, ValueError :
		    pass

            if len(sim._paramfile)>1 :
                sim._paramfile["filename"] = filename
                done = True
                
        if done : break

            
@TipsySnap.decorator
@StarLog.decorator
def param2units(sim) :
    with sim.lazy_off :
        import sys, math, os, glob

        munit = dunit = hub = None

        try :
            hub = float(sim._paramfile["dHubble0"])
            sim.properties['omegaM0'] = float(sim._paramfile["dOmega0"])
            sim.properties['omegaL0'] = float(sim._paramfile["dLambda"])
        except KeyError :
            pass

        try :
            munit_st = sim._paramfile["dMsolUnit"]+" Msol"
            munit = float(sim._paramfile["dMsolUnit"])
            dunit_st = sim._paramfile["dKpcUnit"]+" kpc"
            dunit = float(sim._paramfile["dKpcUnit"])
        except KeyError :
            pass


        if munit is None or dunit is None :
            if hub!=None:
                sim.properties['h'] = hub
            return


        denunit = munit/dunit**3
        denunit_st = str(denunit)+" Msol kpc^-3"

        #
        # the obvious way:
        #
        #denunit_cgs = denunit * 6.7696e-32
        #kpc_in_km = 3.0857e16
        #secunit = 1./math.sqrt(denunit_cgs*6.67e-8)
        #velunit = dunit*kpc_in_km/secunit

        # the sensible way:
        # avoid numerical accuracy problems by concatinating factors:
        velunit = 8.0285 * math.sqrt(6.67e-8*denunit) * dunit
        velunit_st = ("%.5g"%velunit)+" km s^-1"

        #You have: kpc s / km
        #You want: Gyr
        #* 0.97781311
        timeunit = dunit / velunit * 0.97781311
        timeunit_st = ("%.5g"%timeunit)+" Gyr"

        #  Assuming G=1 in code units, phi is actually vel^2/a^3.
        # See also gasoline's master.c:5511.
        # Or should we be calculating phi as GM/R units (which
        # is the same for G=1 runs)?
        potunit_st = "%.5g km^2 s^-2"%(velunit**2)
        
        if hub!=None:
            hubunit = 10. * velunit / dunit
            hubunit_st = ("%.3f"%(hubunit*hub))
            sim.properties['h'] = hub*hubunit

            # append dependence on 'a' for cosmological runs
            dunit_st += " a"
            denunit_st += " a^-3"
            velunit_st += " a"
            potunit_st += " a^-1"
        
            # Assume the box size is equal to the length unit
            sim.properties['boxsize'] = units.Unit(dunit_st)

        sim["vel"].units = velunit_st            
        
        try :
            sim["phi"].units = potunit_st
            sim["eps"].units = dunit_st
        except KeyError :
            pass

        sim["pos"].units = dunit_st
        try :
            sim.gas["rho"].units = denunit_st
        except KeyError :
            pass

        try :
            sim["mass"].units = munit_st
        except KeyError :
            pass

        sim.star["tform"].units = timeunit_st

        try :
            sim.gas["metals"].units = ""
        except KeyError :
            pass

        sim.star["metals"].units = ""

        try :
            sim._file_units_system = [sim["vel"].units, sim["mass"].units, sim["pos"].units, units.K]
        except KeyError :
            try :
                sim._file_units_system = [sim["vel"].units, sim.star["massform"].units, sim["pos"].units, units.K]
            except : pass


@StarLog.decorator
def slparam2units(sim) :
    with sim.lazy_off :
        import sys, math, os, glob

        munit = dunit = hub = None

        try :
            munit_st = sim._paramfile["dMsolUnit"]+" Msol"
            munit = float(sim._paramfile["dMsolUnit"])
            dunit_st = sim._paramfile["dKpcUnit"]+" kpc"
            dunit = float(sim._paramfile["dKpcUnit"])
            hub = float(sim._paramfile["dHubble0"])
        except KeyError :
            pass

        if munit is None or dunit is None :
            return

        denunit = munit/dunit**3
        denunit_st = str(denunit)+" Msol kpc^-3"

        if hub!=None:
            # append dependence on 'a' for cosmological runs
            dunit_st += " a"
            denunit_st += " a^-3"
            
        sim.star["rhoform"].units = denunit_st
        sim.star["massform"].units = munit_st
