"""tipsy
=====

Implements classes and functions for handling tipsy files.  You rarely
need to access this module directly as it will be invoked
automatically via pynbody.load.

**Input**:

*filename*: file name string

**Optional Keywords**:

*paramfile*: string specifying the parameter file to load. If not
specified, the loader will look for a file `*.param` in the current and
parent directories.

"""

from __future__ import with_statement # for py2.5
from __future__ import division

from . import snapshot, array, util
from . import family
from . import units
from . import config, config_parser
from . import chunk
from . import nchilada

import struct, os
import numpy as np
import gzip
import sys
import warnings
import copy
import types
import math

class TipsySnap(snapshot.SimSnap) :
    _basic_loadable_keys = {family.dm: set(['phi', 'pos', 'eps', 'mass', 'vel']),
                            family.gas: set(['phi', 'temp', 'pos', 'metals', 'eps',
                                             'mass', 'rho', 'vel']),
                            family.star: set(['phi', 'tform', 'pos', 'metals',
                                              'eps','mass', 'vel']),
                            None: set(['phi', 'pos', 'eps', 'mass', 'vel']) }


    def __init__(self, filename, **kwargs):

        global config

        super(TipsySnap,self).__init__()

        only_header = kwargs.get('only_header', False)
        if only_header :
            warnings.warn("only_header kwarg is deprecated: all loading in TipsySnap is now lazy by default", RuntimeWarning)
            
        must_have_paramfile = kwargs.get('must_have_paramfile', False)
        take = kwargs.get('take', None)
        verbose = kwargs.get('verbose', config['verbose'])

        self.partial_load = take is not None

        self._filename = util.cutgz(filename)
    
        f = util.open_(filename,'rb')
    
        if verbose : print>>sys.stderr, "TipsySnap: loading ",filename

        t, n, ndim, ng, nd, ns = struct.unpack("diiiii", f.read(28))
        if (ndim > 3 or ndim < 1):
            self._byteswap=True
            f.seek(0)
            t, n, ndim, ng, nd, ns = struct.unpack(">diiiii", f.read(28))
        else :
            self._byteswap=False

        assert ndim==3
            
        self._header_t = t

        f.read(4)

        disk_family_slice = dict({family.gas : slice(0, ng),
                                   family.dm  : slice(ng, nd+ng),
                                   family.star: slice(nd+ng, ng+nd+ns)})

        self._load_control = chunk.LoadControl(disk_family_slice, 10240, take)
        
        self._family_slice = self._load_control.mem_family_slice
        self._num_particles = self._load_control.mem_num_particles
   
        self._paramfilename = kwargs.get('paramfile', None)

        self._decorate()

        # describe the file structure as list of (num_parts, [list_of_properties]) 
        # by default all fields are floats -- we look at the param file to determine
        # whether we should expect some doubles 

        if self._paramfile.get('bDoublePos',0) : 
            ptype = 'd'
        else : 
            ptype = 'f'

        if self._paramfile.get('bDoubleVel',0) : 
            vtype = 'd'
        else : 
            vtype = 'f'

        
        self._g_dtype = np.dtype({'names': ("mass","x","y","z","vx","vy","vz","rho","temp","eps","metals","phi"),
                                  'formats': ('f',ptype,ptype,ptype,vtype,vtype,vtype,'f','f','f','f','f')})
        self._d_dtype = np.dtype({'names': ("mass","x","y","z","vx","vy","vz","eps","phi"),
                                  'formats': ('f',ptype,ptype,ptype,vtype,vtype,vtype,'f','f')})
        self._s_dtype = np.dtype({'names': ("mass","x","y","z","vx","vy","vz","metals","tform","eps","phi"),
                                  'formats': ('f',ptype,ptype,ptype,vtype,vtype,vtype,'f','f','f','f')})

        
        if not self._paramfile.has_key('dKpcUnit'):
            if must_have_paramfile :
                raise RuntimeError, "Could not find .param file for this run. Place it in the run's directory or parent directory."
            else :
                warnings.warn("No readable param file in the run directory or parent directory: using defaults.",RuntimeWarning)
                self._file_units_system = [units.Unit(x) for x in ('G', '1 kpc', '1e10 Msol')]
        
        time_unit = None
        try :
            time_unit = self.infer_original_units('yr')
        except units.UnitsException :
            pass


        if time_unit is not None :
            self.properties['time']*=time_unit

        del f



    def _load_main_file(self) :

        if config['verbose'] : print>>sys.stderr, "TipsySnap: loading data from main file"
            
        f = util.open_(self._filename, 'rb')
        f.seek(32)

        write = []

        for w, ndim in ("pos", 3), ("vel", 3), ("mass", 1), ("eps", 1), ("phi", 1) :
            if w not in self.keys() :
                self._create_array(w, ndim, zeros=False)
                write.append(w)
                
        for w in "rho", "temp" :
            if w not in self.gas.keys() :
                self.gas._create_array(w, zeros=False)
                write.append(w)

        if ("metals" not in self.gas.keys()) and ("metals" not in self.star.keys()) :
            self.gas._create_array("metals", zeros=False)
            self.star._create_array("metals", zeros=False)
            write.append("metals")

        if "tform" not in self.star.keys() :
            self.star._create_array("tform", zeros=False)
            write.append("tform")
            
        if "temp" in write :
            self.gas["temp"].units = "K"

        for k in "pos", "vel", "mass", "eps", "phi" :
            if k in write :
                self[k].set_default_units(quiet=True)

        if "phi" in write and self.properties.has_key('h') : # only do this for cosmo runs
            self['phi'].units=self['phi'].units*units.a**-3 # messy :-(
            
        for k in "rho", "temp", "metals":
            if k in write :
                self.gas[k].set_default_units(quiet=True)

        for k in "metals", "tform":
            if k in write :
                self.star[k].set_default_units(quiet=True)

        if "pos" in write :
            write+=['x','y','z']

        if "vel" in write :
            write+=['vx','vy','vz']

        max_item_size = max([q.itemsize for q in self._g_dtype, self._d_dtype, self._s_dtype])
        tbuf = bytearray(max_item_size*10240)
        
        for fam, dtype in ((family.gas, self._g_dtype), (family.dm, self._d_dtype), (family.star, self._s_dtype)) :
            self_fam = self[fam]
            st_len = dtype.itemsize
            for readlen, buf_index, mem_index in self._load_control.iterate([fam], [fam], multiskip=True) :
                # Read in the block
                
                if mem_index is None :
                    f.seek(st_len*readlen,1)
                    continue
                
                buf = np.fromstring(f.read(st_len*readlen),dtype=dtype)

                if self._byteswap:
                    buf = buf.byteswap()

                if mem_index is not None :
                    # Copy into the correct arrays
                    for name in dtype.names :
                        if name in write :
                            self_fam[name][mem_index] = buf[name][buf_index]
                       


    def _update_loadable_keys(self)  :
        def is_readable_array(x) :
            try:
                f = util.open_(x,'r')
                return int(f.readline()) == len(self)
            except ValueError :
                # could be a binary file
                f = util.open_(x,'rb')

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
        rdict.update(dict([a, copy.copy(b)] for a,b in self._basic_loadable_keys.iteritems() if a is not None ))
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

    
    def _update_snapshot(self, arrays, filename=None, fam_out=[family.gas, family.dm, family.star]) :
        """
        Write a TIPSY file, but only updating the requested information and leaving the 
        rest intact on disk.
        """

        if self.partial_load :
            raise RuntimeError, "Writing back to partially loaded files not yet supported"
        
        global config
        
        # make arrays be a list
        if isinstance(arrays,str) : arrays = [arrays]

        # check if arrays includes a 3D array
        if 'pos' in arrays :
            arrays.remove('pos')
            for arr in ['x','y','z'] :
                arrays.append(arr)
        if 'vel' in arrays :
            arrays.remove('vel')
            for arr in ['vx','vy','vz'] :
                arrays.append(arr)


        with self.lazy_off : 
            fin  = util.open_(self.filename, "rb")
            fout = util.open_(self.filename+".tmp", "wb")

            if self._byteswap: 
                t, n, ndim, ng, nd, ns = struct.unpack(">diiiii", fin.read(28))
                fout.write(struct.pack(">diiiiii", t,n,ndim,ng,nd,ns,0))
            else: 
                t, n, ndim, ng, nd, ns = struct.unpack("diiiii", fin.read(28))
                fout.write(struct.pack("diiiiii", t,n,ndim,ng,nd,ns,0))

            fin.read(4)

            if family.gas   in fam_out: assert(ng == len(self[family.gas]))
            if family.dm    in fam_out: assert(nd == len(self[family.dm]))
            if family.star  in fam_out: assert(ns == len(self[family.star]))

            max_block_size = 1024**2 # particles

            # describe the file structure as list of (num_parts, [list_of_properties]) 
            file_structure = ((ng,family.gas,["mass","x","y","z","vx","vy","vz","rho","temp","eps","metals","phi"]),
                              (nd,family.dm,["mass","x","y","z","vx","vy","vz","eps","phi"]),
                              (ns,family.star,["mass","x","y","z","vx","vy","vz","metals","tform","eps","phi"]))


            # do the read/write -- at each block, replace the relevant array

            for n_left, fam, st in file_structure :
                n_done = 0
                self_fam = self[fam]
                while n_left>0 :
                    n_block = min(n_left,max_block_size)

                    # Read in the block
                    if(self._byteswap):
                        g = np.fromstring(fin.read(len(st)*n_block*4),'f').byteswap().reshape((n_block,len(st)))
                    else:
                        g = np.fromstring(fin.read(len(st)*n_block*4),'f').reshape((n_block,len(st)))

                    if fam in fam_out :
                        self_sub = self[fam][n_done:n_done+n_block]
                        # write over the relevant data
                        with self_sub.immediate_mode :
                            for i, name in enumerate(st) :

                                if name in arrays:
                                    ar = self_sub[name]
                                    try:
                                        if ar.units != 1 and ar.units != units.NoUnit(): 
                                            g[:,i] = ar.in_original_units().view(np.ndarray)
                                        else : 
                                            g[:,i] = ar.view(np.ndarray)
                                    except KeyError:
                                        pass

                    # Write out the block
                    if self._byteswap :
                        g.byteswap().tofile(fout)
                    else:
                        g.tofile(fout)

                    # Increment total ptcls read in, decrement ptcls left of this type
                    n_left-=n_block
                    n_done+=n_block

            fin.close()
            fout.close()
            
            os.system("mv " + self.filename + ".tmp " + self.filename)

    @staticmethod
    def _write(self, filename=None, double_pos = None, double_vel = None, binary_aux_arrays = None) :
        """

        Write a TIPSY (standard) formatted file.   
        
        Additionally, you can specify whether you want position and/or
        velocity arrays written out in double precision. If you are
        writing out a snapshot that was originally in tipsy format and
        the bDoublePos/bDoubleVel flags are set in the parameter file,
        then the write routine will follow those choices. If you are
        writing a snapshot other than a tipsy snapshot, then you have
        to specify these by hand.
        
        **Optional Keywords**

        *filename* (None): name of the file to be written out. If
                           None, the original file is overwritten.

        *double_pos* (False): set to 'True' if you want to write out positions as doubles

        *double_vel* (False): set to 'True' if you want to write out velocities as doubles

        *binary_aux_arrays* (None): set to 'True' to write auxiliary
                                    arrays in binary format; if left 'None', the preference is
                                    taken from the param file

        """

        global config
        
        if filename is None :
            filename = self._filename

        if config['verbose'] : print>>sys.stderr, "TipsySnap: writing main file as",filename

        f = util.open_(filename, 'wb')

        t = 0
        try:
            t = self.properties['a']
        except KeyError :
            warnings.warn("Time is unknown: writing zero in header",RuntimeWarning)
        

        n = len(self)
        ndim = 3
        ng = len(self.gas)
        nd = len(self.dark)
        ns = len(self.star)


        byteswap = getattr(self, "_byteswap", sys.byteorder=="little")

        if byteswap: 
            f.write(struct.pack(">diiiiii", t,n,ndim,ng,nd,ns,0))
        else:
            f.write(struct.pack("diiiiii", t,n,ndim,ng,nd,ns,0))

                
        # needs to be done in blocks like reading
        # describe the file structure as list of (num_parts, [list_of_properties]) 
            
        if type(self) is not TipsySnap : 
            if double_pos is None: double_pos = False
            if double_vel is None: double_vel = False
            ptype = 'd' if double_pos else 'f'
            vtype = 'd' if double_vel else 'f'

        else :
            dpos_param = self._paramfile.get('bDoublePos',False)
            dvel_param = self._paramfile.get('bDoubleVel',False)

            if double_pos: ptype = 'd'
            elif not double_pos: ptype = 'f'
            else : ptype = 'd' if dpos_param else 'f'
            
            if double_vel: vtype = 'd'
            elif not double_vel: vtype = 'f'
            else : vtype = 'd' if dvel_param else 'f'
            
        g_dtype = np.dtype({'names': ("mass","x","y","z","vx","vy","vz","rho","temp","eps","metals","phi"),
                            'formats': ('f',ptype,ptype,ptype,vtype,vtype,vtype,'f','f','f','f','f')})
        d_dtype = np.dtype({'names': ("mass","x","y","z","vx","vy","vz","eps","phi"),
                            'formats': ('f',ptype,ptype,ptype,vtype,vtype,vtype,'f','f')})
        s_dtype = np.dtype({'names': ("mass","x","y","z","vx","vy","vz","metals","tform","eps","phi"),
                            'formats': ('f',ptype,ptype,ptype,vtype,vtype,vtype,'f','f','f','f')})
        
            
        file_structure = ((ng,family.gas,g_dtype),
                          (nd,family.dm,d_dtype),
                          (ns,family.star,s_dtype))

        max_block_size = 1024**2 # particles

        with self.lazy_derive_off : 
            for n_left, fam, dtype in file_structure :
                n_done = 0
                self_type = self[fam]
                while n_left>0 :
                    n_block = min(n_left,max_block_size)                   
                
                #g = np.zeros((n_block,len(st)),dtype=np.float32)
                
                    g = np.empty(n_block,dtype=dtype)
                
                    self_type_block = self_type[n_done:n_done+n_block]
                
                    with self_type_block.immediate_mode :
                        # Copy from the correct arrays
                        for i, name in enumerate(dtype.names) :
                            try:
                                g[name] = self_type_block[name]
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

        with self.lazy_off : # prevent any lazy reading or evaluation
        
            for x in set(self.keys()).union(self.family_keys()) :
                if not self.is_derived_array(x) and x not in ["mass","pos","x","y","z","vel","vx","vy","vz","rho","temp",
                                                              "eps","metals","phi", "tform"]  :
                    TipsySnap._write_array(self, x, filename=filename+"."+x, binary=binary_aux_arrays)
    

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
                fmt = "%e"
            np.savetxt(f, ar, fmt=fmt)


    def _get_loadable_array_metadata(self, array_name) :
        """Given an array name, returns the metadata consisting of
        the tuple units, families.

        Returns:
         *units*: the units of this data on disk
         *families*: a list of family objects for which data is on disk
                     for this array, or None if this cannot be determined"""

        try:
            f = open(self.filename+"."+array_name+".pynbody-meta",'r')
        except IOError :
            return self._default_units_for(array_name), None
        
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
            fams = [family.get_family(x.strip()) for x in res['families'].split(" ")]
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
        print >>f
        f.close()
        
        if isinstance(self, TipsySnap) :
            # update the loadable keys if this operation is likely to have
            # changed them
            self._update_loadable_keys()

    @staticmethod
    def _families_in_main_file(array_name, fam=None) :
        fam_for_default = [fX for fX, ars in TipsySnap._basic_loadable_keys.iteritems() if array_name in ars and fX in fam]
        return fam_for_default
    
    def _update_array(self, array_name, fam=None,
                      filename=None, binary=None, byteswap=None) :
        assert fam is not None

        if self.partial_load :
            raise RuntimeError, "Writing back to partially loaded files not yet supported"
        
        fam_in_main = self._families_in_main_file(array_name, fam)
        if len(fam_in_main)>0 :
            self._update_snapshot(array_name, fam_out=fam_in_main)
            fam = list(set(fam).difference(fam_in_main))
            if len(fam)==0 : return
            
        
        # If we have disk units for this array, check we can convert into them
        
        aux_u, aux_f = self._get_loadable_array_metadata(array_name)
        if aux_f is None :
            aux_f = self.families()
            
        if aux_u is not None :
            for f in fam :
                if array_name in self[f] :
                    # check convertible
                    try:
                        self[f][array_name].units.in_units(aux_u)
                    except units.UnitsException :
                        raise IOError("Units must match the existing auxiliary array on disk.")

                    
       
        try:            
            data = self.__read_array_from_disk(array_name, filename = filename)
        except IOError :
            # doesn't really exist, probably because the other data on disk was
            # in the main snapshot
            self._write_array(self, array_name, fam, filename=filename, binary=binary,
                              byteswap=byteswap)
            return

            
        for f in fam :
            if aux_u is not None :
                data[self._get_family_slice(f)] = self[f][array_name].in_units(aux_u)
            else :
                data[self._get_family_slice(f)] = self[f][array_name]
                
        fam = list(set(aux_f).union(fam))
        
        self._write_array(self, array_name, fam=fam, contents = data, binary=binary,
                          byteswap=byteswap, filename=filename)

        
    @staticmethod
    def _write_array(self, array_name, fam=None, contents=None,
                     filename=None, binary=None, byteswap=None) :
        """Write the array to file."""

        fam_in_main = TipsySnap._families_in_main_file(array_name, fam)
        if len(fam_in_main)>0 :
            if isinstance(self,TipsySnap) :
                self._update_snapshot(array_name, fam_out=fam_in_main)
                fam = list(set(fam).difference(fam_in_main))
                if len(fam)==0 : return
            else :
                raise RuntimeError, "Cannot call static _write_array to write into main tipsy file."
                          
        units_out = None
        
        if binary is None :
            binary = getattr(self.ancestor,"_tipsy_arrays_binary",False)
            
        if binary and ( byteswap is None ) :
            byteswap = getattr(self.ancestor, "_byteswap", False)

            
        with self.lazy_off : # prevent any lazy reading or evaluation
            if filename is None :
                if self._filename[-3:] == '.gz' :
                    filename = self._filename[:-3]+"."+array_name+".gz"
                else :
                    filename = self._filename+"."+array_name

            if binary :
                fhand = util.open_(filename, 'wb')
                    
                if byteswap :
                    fhand.write(struct.pack(">i", len(self)))
    
                else :
                    fhand.write(struct.pack("i", len(self)))
                
            else :
                fhand = util.open_(filename, 'w')
                print>>fhand, str(len(self))
        

            if contents is None :
                if array_name in self.family_keys() :
                    for f in [family.gas, family.dm, family.star] :
                        try:
                            ar = self[f][array_name]
                            units_out = ar.units

                        except KeyError :
                            ar = np.zeros(len(self[f]), dtype=int)
                            
                        TipsySnap.__write_block(fhand, ar, binary, byteswap)

                else :
                    ar = self[array_name]
                    units_out = ar.units
                    TipsySnap.__write_block(fhand, ar, binary, byteswap)

            else :
                TipsySnap.__write_block(fhand, contents, binary, byteswap)
                units_out = contents.units

        fhand.close()

        if fam is None :
            fam = [family.gas, family.dm, family.star]
            
        TipsySnap._write_array_metafile(self,filename, units_out, fam)
        
            

    def _load_array(self, array_name, fam=None, filename = None,
                    packed_vector = None) :

        if array_name in self._basic_loadable_keys[fam] :
            self._load_main_file()
            return

        fams = self._get_loadable_array_metadata(array_name)[1] or self.families()
        
        if (fam is None and fams is not None and len(fams)!=len(self.families())) or \
               (fam is not None and fam not in fams) :
            # Top line says 'you requested all families but at least one isn't available'
            # Bottom line says 'you requested one family, but that one's not available'
            
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

        if filename is None and array_name in ['massform', 'rhoform', 'tempform','phiform','nsmooth', 
                                               'xform', 'yform', 'zform', 'vxform', 'vyform', 'vzform', 
                                               'posform', 'velform'] :

            try : 
                self.read_starlog()
                if fam is not None : return self[fam][array_name]
                else : return self[array_name]
            except IOError: 
                pass

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
                
        f = util.open_(filename,'r')

        if config['verbose'] : print>>sys.stderr, "TipsySnap: attempting to load auxiliary array",filename
        # if we get here, we've got the file - try loading it
  
        try :
            l = int(f.readline())
            binary = False
            if l!=self._load_control.disk_num_particles :
                raise IOError, "Incorrect file format"

            dtype = self._get_preferred_dtype(array_name)
            if not dtype :
                # Inspect the first line to see whether it's float or int
                l = "0\n"
                while l=="0\n" : l = f.readline()
                if "." in l or "e" in l or l[:-1]=="inf" :
                    dtype = float
                else :
                    dtype = int

                # Restart at head of file
                f.seek(0)
                f.readline()

            loadblock = lambda count : np.fromfile(f, dtype=dtype, sep="\n", count=count)
            # data = np.fromfile(f, dtype=tp, sep="\n")
        except ValueError :
            # this is probably a binary file
            binary = True
            f= util.open_(filename,'rb')

            # Read header and check endianness
            if self._byteswap:
                l = struct.unpack(">i",f.read(4))[0]
            else:
                l = struct.unpack("i", f.read(4))[0]

            if l!=self._load_control.disk_num_particles :
                raise IOError, "Incorrect file format"

            # Set data format to be read (float or int) based on config
            int_arrays = map(str.strip, config_parser.get('tipsy', 'binary-int-arrays').split(","))
            if array_name in int_arrays : dtype = 'i'
            else: dtype = 'f'

            # Read longest data array possible.  
            # Assume byteswap since most will be.
            if self._byteswap:
                loadblock = lambda count : np.fromstring(f.read(count*4), dtype=dtype, count=count).byteswap()
                # data = np.fromstring(f.read(3*len(self)*4),dtype).byteswap()
            else:
                loadblock = lambda count : np.fromstring(f.read(count*4), dtype=dtype, count=count)
                # data = np.fromstring(f.read(3*len(self)*4),dtype)
           

        """
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

        if fam is None :
            r = data.reshape(dims, order=v_order).view(array.SimArray)
        else :
            r = data.reshape(dims,order=v_order).view(array.SimArray)[self._get_family_slice(fam)]
        """
        ndim = 1

        self.ancestor._tipsy_arrays_binary = binary

        all_fam = [family.dm, family.gas, family.star]
        if fam is None :
            fam = all_fam
            r = np.empty(len(self), dtype=dtype).view(array.SimArray)
        else :
            r = np.empty(len(self[fam]), dtype=dtype).view(array.SimArray)
            
        for readlen, buf_index, mem_index in self._load_control.iterate(all_fam, fam) :
            buf = loadblock(readlen)
            if mem_index is not None :
                r[mem_index] = buf[buf_index]
     

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
        for i,x in enumerate(['x','y','z']): 
            self._arrays[x+'form'] = self['posform'][:,i]
        for i,x in enumerate(['vx','vy','vz']): 
            self._arrays[x+'form'] = self['velform'][:,i]
    
    @staticmethod
    def _can_load(f) :
        try:
            check = TipsySnap(f, verbose=False)
            del check
        except :
            return False
        
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
def oxh(self) :
    minox = np.amin(self['OxMassFrac'][np.where(self['OxMassFrac'] > 0)])
    self['OxMassFrac'][np.where(self['OxMassFrac'] == 0)]=minox
    return np.log10(self['OxMassFrac']/self['hydrogen']) - np.log10(XSOLO/XSOLH)

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
    def __init__(self, filename, sort=True, paramfile = None):
        import os
        super(StarLog,self).__init__()
        self._filename = filename
        self._paramfilename = paramfile

        f = util.open_(filename,"rb")
        self.properties = {}
        bigstarlog = False
        
        file_structure = np.dtype({'names': ("iord","iorderGas","tform",
                                             "x","y","z",
                                             "vx","vy","vz",
                                             "massform","rhoform","tempform"),
                                   'formats':('i4','i4','f8',
                                              'f8','f8','f8',
                                              'f8','f8','f8',
                                              'f8','f8','f8')})

        size = struct.unpack("i", f.read(4))
        if (size[0]> 1000 or size[0]<  10):
            self._byteswap=True
            f.seek(0)
            size = struct.unpack(">i", f.read(4))
        iSize = size[0]

        if (iSize > file_structure.itemsize) :
            file_structure = np.dtype({'names': ("iord","iorderGas","tform",
                                                 "x","y","z",
                                                 "vx","vy","vz",
                                                 "massform","rhoform","tempform",
                                                 "phiform","nsmooth"),
                                       'formats':('i4','i4','f8',
                                                  'f8','f8','f8',
                                                  'f8','f8','f8',
                                                  'f8','f8','f8',
                                                  'f8','i4')})

            if (iSize != file_structure.itemsize and iSize != 104):
                raise IOError, "Unknown starlog structure iSize:"+str(iSize)+", file_structure itemsize:"+str(file_structure.itemsize)
            else : bigstarlog = True
            
        datasize = os.path.getsize(filename)-f.tell()

        # check whether datasize is a multiple of iSize. If it is not,
        # the starlog is likely corrupted, but try to read it anyway

        if (datasize%iSize > 0) and (iSize != 104): 
            warnings.warn("The size of the starlog file does not make sense -- it is likely corrupted. Pynbody will read it anyway, but use with caution.")
            datasize -= datasize%iSize

        if config['verbose'] : print "Reading "+filename
        if(self._byteswap):
            g = np.fromstring(f.read(datasize),dtype=file_structure).byteswap()
        else:
            g = np.fromstring(f.read(datasize),dtype=file_structure)

        # hoping to provide backward compatibility for np.unique by
        # copying relavent part of current numpy source:
        # numpy/lib/arraysetops.py:192 (on 22nd March 2011)

        if sort : 
            tmp = g['iord'].flatten()
            perm = tmp.argsort()
            aux = tmp[perm]
            flag = np.concatenate(([True],aux[1:]!=aux[:-1]))
            iord = aux[flag]; indices = perm[flag]
            self._num_particles = len(indices)
        else : 
            self._num_particles = len(g)

        self._family_slice[family.star] = slice(0, self._num_particles)
        self._create_arrays(["pos","vel"],3)
        self._create_arrays(["iord"],dtype='int32')
        self._create_arrays(["iorderGas","massform","rhoform","tempform","metals","tform"])
        if bigstarlog :
            self._create_arrays(["phiform","nsmooth"])

        self._decorate()
        
        if sort:
            for name in file_structure.fields.keys() :
                self.star[name][:] = g[name][indices]
        else : 
            for name in file_structure.fields.keys() :
                self.star[name] = g[name]

    @staticmethod
    def _write(self, filename=None) :
        """Write the starlog file. """

        global config

        with self.lazy_off : 
            
            if filename is None : 
                filename = self._filename
                
            if config['verbose'] : print>>sys.stderr, "StarLog: writing starlog file as",filename

            f = util.open_(filename, 'wb')

            if 'phiform' in self.keys() :  # long starlog format
                file_structure = np.dtype({'names': ("iord","iorderGas","tform",
                                                     "x","y","z",
                                                     "vx","vy","vz",
                                                     "massform","rhoform","tempform",
                                                     "phiform","nsmooth"),
                                           'formats':('i4','i4','f8',
                                                      'f8','f8','f8',
                                                      'f8','f8','f8',
                                                      'f8','f8','f8',
                                                      'f8','i4')})
            else : # short (old) starlog format
                file_structure = np.dtype({'names': ("iord","iorderGas","tform",
                                                     "x","y","z",
                                                     "vx","vy","vz",
                                                     "massform","rhoform","tempform"),
                                           'formats':('i4','i4','f8',
                                                      'f8','f8','f8',
                                                      'f8','f8','f8',
                                                      'f8','f8','f8')})
        
            if self._byteswap : f.write(struct.pack(">i", file_structure.itemsize))
            else :              f.write(struct.pack("i", file_structure.itemsize))
            
    
            max_block_size = 1024 # particles
        
            n_left = len(self)
            n_done = 0

            while n_left > 0 : 
                n_block = min(n_left, max_block_size)

                g = np.zeros(n_block, dtype=file_structure)

                for arr in file_structure.names : 
                    g[arr] = self[arr][n_done:n_done+n_block]
                
                if self._byteswap : 
                    g.byteswap().tofile(f)
                else : 
                    g.tofile(f)
                    
                n_left -= n_block
                n_done += n_block

        f.close()

        
            

@TipsySnap.decorator
@StarLog.decorator
@nchilada.NchiladaSnap.decorator
def load_paramfile(sim) :
    import sys, os, glob
    x = os.path.abspath(sim._filename)
    done = False
    sim._paramfile = {}
    f = None
    if sim._paramfilename is None: 
        for i in xrange(2) :
            x = os.path.dirname(x)
            l = [x for x in glob.glob(os.path.join(x,"*.param")) if "mpeg" not in x]
            
            for filename in l :
                # Attempt the loading of information
                try :
                    f = open(filename)
                    done = True
                    break # the file is there, break out of the loop
                except IOError :
                    l = glob.glob(os.path.join(x,"../*.param"))
                    if l==[] :
                        continue
                    try : 
                        for filename in l:
                            f = open(filename)
                            break # the file is there, break out of the loop
                    except IOError:
                        continue
            if done: break

    else : 
        filename = sim._paramfilename
        try : 
            f = open(filename)
        except IOError : 
            raise IOError("The parameter filename you supplied is invalid")            

    if f is None :
        return
    
    for line in f :
        try :
            if line[0]!="#" :
                s = line.split("#")[0].split()
                sim._paramfile[s[0]] = " ".join(s[2:])
                                    
        except IndexError, ValueError :
            pass

        if len(sim._paramfile)>1 :
            sim._paramfile["filename"] = filename


            
@TipsySnap.decorator
@StarLog.decorator
@nchilada.NchiladaSnap.decorator
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
        velunit = 8.0285 * math.sqrt(6.6743e-8*denunit) * dunit
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

        if sim._paramfile.has_key('bComove') and int(sim._paramfile['bComove'])!=0 :
            hubunit = 10. * velunit / dunit
            hubunit_st = ("%.3f"%(hubunit*hub))
            sim.properties['h'] = hub*hubunit

            if isinstance(sim,StarLog) :
                a = "aform"
            else :
                a = "a"
                
            # append dependence on 'a' for cosmological runs
            dunit_st += " "+a
            denunit_st += " "+a+"^-3"
            velunit_st += " "+a
            potunit_st += " "+a+"^-1"
        
            # Assume the box size is equal to the length unit
            sim.properties['boxsize'] = units.Unit(dunit_st)

        try :
            sim["vel"].units = velunit_st            
        except KeyError :
            pass
        
        try :
            sim["phi"].units = potunit_st
            sim["eps"].units = dunit_st
        except KeyError :
            pass

        try :
            sim["pos"].units = dunit_st
        except KeyError :
            pass
        
        try :
            sim.gas["rho"].units = denunit_st
        except KeyError :
            pass

        try :
            sim["mass"].units = munit_st
        except KeyError :
            pass

        try :
            sim.star["tform"].units = timeunit_st
        except KeyError :
            pass
        
        try :
            sim.gas["metals"].units = ""
        except KeyError :
            pass

        try :
            sim.star["metals"].units = ""
        except KeyError :
            pass
        
        try :
            sim._file_units_system = [sim["vel"].units, sim.star["mass"].units, sim["pos"].units, units.K]
        except KeyError :
            try :
                sim._file_units_system = [sim["vel"].units, sim.star["massform"].units, sim["pos"].units, units.K]
            except KeyError:
                try:
                    sim._file_units_system = [units.Unit(velunit_st), 
                                              units.Unit(munit_st), 
                                              units.Unit(dunit_st), units.K]
                except : pass



@TipsySnap.decorator
def settime(sim) :
    if sim._paramfile.has_key('bComove') and int(sim._paramfile['bComove'])!=0 :
        from . import analysis
        from .analysis import cosmology 
        t = sim._header_t
        sim.properties['a'] = t
        try :
            sim.properties['z'] = 1.0/t - 1.0
        except ZeroDivisionError :
            # no sensible redshift
            pass

        if (sim.properties['z'] is not None and 
            sim._paramfile.has_key('dMsolUnit') and 
            sim._paramfile.has_key('dKpcUnit')):
            sim.properties['time'] =  analysis.cosmology.age(sim, unit=sim.infer_original_units('yr'))
        else :
            # something has gone wrong with the cosmological side of
            # things
            warnings.warn("Paramfile suggests time is cosmological, but header values are not sensible in this context.", RuntimeWarning)
            sim.properties['time'] = t

        sim.properties['a'] = t
        
    else :
        # Assume a non-cosmological run
            sim.properties['time'] =  sim._header_t

    
            
    
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
            dunit_st += " aform"
            
            # denunit_st += " a^-3"
            # N.B. density comoving -> physical conversion is done by Gasoline itself
            
        sim.star["rhoform"].units = denunit_st
        sim.star["massform"].units = munit_st
