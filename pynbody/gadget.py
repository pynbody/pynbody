from . import snapshot,array
from . import family
import numpy as np
#Needed to unpack things
import struct
import sys
import copy
import os.path as path

#Symbolic constants for the particle types
GAS_TYPE=0
DM_TYPE = 1
NEUTRINO_TYPE = 2
DISK_TYPE = 2
BULGE_TYPE = 3
STAR_TYPE = 4
BNDRY_TYPE = 5
N_TYPE = 6

def gadget_type(fam) :
    #-1 is "all types"
    if fam == None:
        return -1
    else :
        if fam == family.gas :
            return GAS_TYPE
        if fam == family.dm:
            return DM_TYPE
        if fam == family.disk or fam == family.neutrino:
            return NEUTRINO_TYPE
        if fam == family.bulge:
            return BULGE_TYPE
        if fam == family.stars:
            return STAR_TYPE
        if fam == family.bndry:
            return BNDRY_TYPE
    raise KeyError, "No particle of type"+fam.name

class GadgetBlock : 
    """Class to describe each block.
    Each block has a start, a length, and a length-per-particle"""
    def __init__(self) :
        #Start of block in file
        self.start=0
        #Length of block in file
        self.length=0
        #Bytes per particle in file
        self.partlen=0
        #Data type of block
        self.data_type = np.float32
        #Types of particle this block contains
        self.p_types = np.zeros(N_TYPE,bool)

class GadgetHeader :
    """Describes the header of gadget class files; this is all our metadata, so we are going to store it inline"""
    def __init__(self, data, endian='=') :
        """Takes a byte range, read from a file, creates a header"""
        #Number of particles
        self.npart = np.zeros(N_TYPE, dtype=np.uint32)
        # Mass of each particle type in this file. If zero, 
        # particle mass stored in snapshot.
        self.mass = np.zeros(N_TYPE)
        # Time of snapshot
        self.time = 0.
        # Redshift of snapshot
        self.redshift = 0.
        # Boolean to test the presence of star formation
        self.flag_sfr=False
        # Boolean to test the presence of feedback
        self.flag_feedback=False
        # First 32-bits of total number of particles in the simulation
        self.npartTotal=np.zeros(N_TYPE,dtype=np.int32)
        # Boolean to test the presence of cooling 
        self.flag_cooling=False
        # Number of files expected in this snapshot
        self.num_files=0
        # Box size of the simulation
        self.BoxSize=0.
        # Omega_Matter. Note this is Omega_DM + Omega_Baryons
        self.Omega0=0.
        # Dark energy density
        self.OmegaLambda=0.
        # Hubble parameter, in units where it is around 70. 
        self.HubbleParam=0.
        # Boolean to test whether stars have an age
        self.flag_stellarage=False
        # Boolean to test the presence of metals
        self.flag_metals=False
        # Long word of the total number of particles in the simulation. 
        # At least one version of N-GenICs sets this to something entirely different. 
        self.NallHW=np.zeros(N_TYPE,dtype=np.int32)
        self.flag_entropy_instead_u=False	# flags that IC-file contains entropy instead of u 
        self.flag_doubleprecision=False	 # flags that snapshot contains double-precision instead of single precision 

        self.flag_ic_info=False 
        # flag to inform whether IC files are generated with Zeldovich approximation,
        # or whether they contain 2nd order lagrangian perturbation theory ICs.
        #    FLAG_ZELDOVICH_ICS     (1)   - IC file based on Zeldovich
        #    FLAG_SECOND_ORDER_ICS  (2)   - Special IC-file containing 2lpt masses
        #    FLAG_EVOLVED_ZELDOVICH (3)   - snapshot evolved from Zeldovich ICs
        #    FLAG_EVOLVED_2LPT      (4)   - snapshot evolved from 2lpt ICs
        #    FLAG_NORMALICS_2LPT    (5)   - standard gadget file format with 2lpt ICs
        # All other values, including 0 are interpreted as "don't know" for backwards compatability.
        self.lpt_scalingfactor=0.      # scaling factor for 2lpt initial conditions 
    
        if data == '':
            return
        fmt= endian+"IIIIIIddddddddiiIIIIIIiiddddiiIIIIIIiiif48s"
        if struct.calcsize(fmt) != 256:
            raise Exception, "There is a bug in gadget.py; the header format string is not 256 bytes"
        (self.npart[0], self.npart[1],self.npart[2],self.npart[3],self.npart[4],self.npart[5],
        self.mass[0], self.mass[1],self.mass[2],self.mass[3],self.mass[4],self.mass[5],
        self.time, self.redshift,  self.flag_sfr, self.flag_feedback, 
        self.npartTotal[0], self.npartTotal[1],self.npartTotal[2],self.npartTotal[3],self.npartTotal[4],self.npartTotal[5],
        self.flag_cooling, self.num_files, self.BoxSize, self.Omega0, self.OmegaLambda, self.HubbleParam,self.flag_stellarage, self.flag_metals, 
        self.NallHW[0], self.NallHW[1],self.NallHW[2],self.NallHW[3],self.NallHW[4],self.NallHW[5],
        self.flag_entropy_instead_u, self.flag_doubleprecision, self.flag_ic_info, self.lpt_scalingfactor,fill) = struct.unpack(fmt, data) 
        return

class GadgetFile :
    """This is a helper class. 
    Should only be called by GadgetSnap.
    Contains the block location dictionary for each file.
    To read Gadget 1 format files, put a text file called blocks.txt 
    in the same directory as the snapshot containing a newline separated 
    list of the blocks in each snapshot file."""
    def __init__(self, filename) :
        self._filename=filename
        self.blocks = {}
        self.header=GadgetHeader('')
        self.endian=''
        self.format2=True
        t_part = 0
        fd=open(filename, "rb")
        self.check_format(fd)
        #If format 1, load the block definition file.
        if not self.format2 :
            try:
               self.block_names=np.loadtxt(path.join(path.dirname(filename),"/blocks.txt"))
               for n in self.block_names:
                   n = (n.upper().ljust(4," "))[0:4]
            #Sane defaults
            except IOError:
               self.block_names = np.array(["HEAD","POS ","VEL ","ID  ","MASS","U   ","RHO ","NE  ","NH  ","NHE ","HSML","SFR "])
        while True:
            block=GadgetBlock()
            (name, block.length) = self.read_block_head(fd)
            if block.length == 0 :
                break
            #Do special things for the HEAD block
            if name[0:4] == "HEAD" :
                if block.length != 256:
                    raise IOError, "Mis-sized HEAD block in "+filename
                self.header=fd.read(256)
                if len(self.header) != 256 :
                    raise IOError, "Could not read HEAD block in "+filename
                self.header=GadgetHeader(self.header, self.endian)
                record_size = self.read_block_foot(fd)
                if record_size != 256 :
                    raise IOError, "Bad record size for HEAD in "+filename
                t_part = self.header.npart.sum()
                continue
            #Set the partlen, using our amazing heuristics
            if name[0:4] == "POS " or name[0:4] == "VEL " :
                block.partlen = 12
            elif name[0:4] == "ID  ":
                #Heuristic for long (64-bit) IDs
                if block.length == t_part * 4 :
                    block.partlen = 4
                    block.data_type = np.int32
                else :
                    block.partlen = 8
                    block.data_type = np.int64
            else :
                block.partlen = 4
            block.start = fd.tell()
            # Check for the case where the record size overflows an int.
            # If this is true, we can't get record size from the length and we just have to guess
            # At least the record sizes at either end should be consistently wrong. 
            # Better hope this only happens for blocks where all particles are present.
            extra_len = t_part *block.partlen
            if extra_len >= 2**32 : 
                fd.seek(extra_len,1)
            else :
                fd.seek(block.length,1)
            record_size = self.read_block_foot(fd)
            if record_size != block.length :
                raise IOError, "Corrupt record in "+filename+" footer for block "+name
            if extra_len >= 2**32 : 
                block.length = extra_len
            # Set up the particle types in the block. This also is a heuristic,
            # which assumes that blocks are either fully present or not for a given particle type
            block.p_types = self.get_block_types(block, self.header.npart)
            self.blocks[name[0:4]] = block

        #and we're done.
        fd.close()
        return

    def get_block_types(self,block, npart):
        """ Set up the particle types in the block, with a heuristic,
        which assumes that blocks are either fully present or not for a given particle type"""
        #This function is horrible.
        p_types = np.zeros(N_TYPE,bool)
        if block.length == npart.sum()*block.partlen:
            p_types= np.ones(N_TYPE, bool)
            return p_types
        #Blocks which contain a single particle type
        for n in np.arange(0,N_TYPE) :
            if block.length == npart[n]*block.partlen :
                p_types[n] = True
                return p_types
        #Blocks which contain two particle types
        for n in np.arange(0,N_TYPE) :
            for m in np.arange(0,N_TYPE) :
                if block.length == (npart[n]+npart[m])*block.partlen :
                    p_types[n] = True
                    p_types[m] = True
                    return p_types
        #Blocks which contain three particle types
        for n in np.arange(0,N_TYPE) :
            for m in np.arange(0,N_TYPE) :
                for l in np.arange(0,N_TYPE) :
                    if block.length == (npart[n]+npart[m]+npart[l])*block.partlen :
                        p_types[n] = True
                        p_types[m] = True
                        p_types[l] = True
                        return p_types
        #Blocks which contain four particle types
        for n in np.arange(0,N_TYPE) :
            for m in np.arange(0,N_TYPE) :
                if block.length == (npart.sum() - npart[n]-npart[m])*block.partlen :
                    p_types = np.ones(N_TYPE, bool)
                    p_types[n] = False
                    p_types[m] = False
                    return p_types
        #Blocks which contain five particle type
        for n in np.arange(0,N_TYPE) :
            if block.length == (npart.sum() -npart[n])*block.partlen :
                p_types = np.ones(N_TYPE, bool)
                p_types[n] = False
                return p_types
        raise ValueError, "Could not determine particle types for block"



    def check_format(self, fd):
        """This function reads the first character of a file and, depending on its value, determines
        whether we have a format 1 or 2 file, and whether the endianness is swapped. For the endianness, 
        it then determines the correct byteorder string to pass to struct.unpack. There is not string
        for 'not native', so this is more complex than it needs to be"""
        fd.seek(0,0)
        (r,) = struct.unpack('=I',fd.read(4))
        if r == 8 :
            self.endian = '='
            self.format2 = True
        elif r == 134217728 : 
            if sys.byteorder == 'little':
                self.endian = '>'
            else :
                self.endian = '<'
            self.format2 = True
        elif r == 65536 :
            if sys.byteorder == 'little':
                self.endian = '>'
            else :
                self.endian = '<'
            self.format2 = False
        elif r == 256 :
            self.endian = '='
            self.format2 = False
        else :
            raise IOError, "File corrupt. First integer is: "+str(r)
        fd.seek(0,0)
        return 

    def read_block_foot(self, fd):
        """Unpacks the block footer, into a single integer"""
        record_size = fd.read(4)
        if len(record_size) != 4 :
            raise IOError, "Could not read block footer"
        (record_size,)= struct.unpack(self.endian+'I',record_size)
        return record_size


    def read_block_head(self, fd) :
        """Read the Gadget 2 "block header" record, ie, 8 name, length, 8.
           Takes an open file and returns a (name, length) tuple """
        if self.format2 :
            head=fd.read(5*4)
            #If we have run out of file, we don't want an exception, 
            #we just want a zero length empty block
            if len(head) != 5*4 :
                return ("    ",0)
            head=struct.unpack(self.endian+'I4sIII',head)
            if head[0] != 8 or head[3] != 8 or head[4] != head[2]-8 :
                raise IOError, "Corrupt header record. Possibly incorrect file format"
            #Don't include the two "record_size" indicators in the total length count
            return (head[1], head[2]-8)
        else :
            record_size = fd.read(4)
            if len(record_size) != 4 :
                raise IOError, "Ran out of data in "+filename+" before block "+name+" started"
            (record_size,)=struct.unpack(self.endian+'I', record_size)
            name = self.block_names[0]
            self.block_names = self.block_names[1:]
            return (name, record_size)
        
    def get_block(self, name, p_type, p_toread) :
        """Get a particle range from this file, starting at p_start, 
        and reading a maximum of p_toread particles"""
        p_read = 0
        cur_block = self.blocks[name]
        parts = self.get_block_parts(name, p_type)
        p_start = self.get_start_part(name, p_type)
        if p_toread > parts :
            p_toread = parts
        fd=open(self._filename, 'rb')
        fd.seek(cur_block.start+cur_block.partlen*p_start,0)
        #This is just so that we can get a size for the type
        dt = np.dtype(cur_block.data_type)
        n_type = p_toread*cur_block.partlen/dt.itemsize
        data=np.fromfile(fd, dtype=cur_block.data_type, count=n_type, sep = '')
        if self.endian != '=' :
            data=data.byteswap(True)
        return (p_toread, data)

    def get_block_parts(self, name, p_type):
        """Get the number of particles present in a block in this file"""
        if not self.blocks.has_key(name) :
                return 0
        cur_block = self.blocks[name]
        if p_type == -1 :
            return cur_block.length/cur_block.partlen
        else :
            return self.header.npart[p_type]*cur_block.p_types[p_type]

    def get_start_part(self, name, p_type) :
        """Find particle to skip to before starting, if reading particular type"""
        if p_type == -1:
            return 0
        else :
            if not self.blocks.has_key(name) :
                    return 0
            cur_block = self.blocks[name]
            return (cur_block.p_types*self.header.npart)[0:p_type].sum()

    def get_block_dims(self, name):
        """Get the dimensionality of the block, eg, 3 for POS, 1 for most other things"""
        if not self.blocks.has_key(name) :
                return 0
        cur_block = self.blocks[name]
        dt = np.dtype(cur_block.data_type)
        return cur_block.partlen/dt.itemsize


class GadgetSnap(snapshot.SimSnap):
    """Main class for reading Gadget-2 snapshots. The constructor makes a map of the locations 
    of the blocks, which are then read by _read_array"""
    def __init__(self, filename, only_header=False, must_have_paramfile=False) : 
        super(GadgetSnap,self).__init__()
        self._files = []
        self._filename=""
        npart = np.empty(N_TYPE)
        #Check whether the file exists, and get the ".0" right
        try:
            fd=open(filename)
        except IOError:
            fd=open(filename+".0")
            #The second time if there is an exception we let it go through
            filename = filename+".0"
        fd.close()
        if filename[-2:] == ".0" :
                self._filename = filename [:-2]
        else:
            self._filename=filename
        #Not sure why the class has self.filename, but everything seems to manipulate self._filename?
        self.filename = self._filename
        #Read the first file and use it to get an idea of how many files we are expecting.
        first_file = GadgetFile(filename)
        self._files.append(first_file)
        files_expected = self._files[0].header.num_files
        npart = np.array(self._files[0].header.npart)
        for i in np.arange(1, files_expected):
            filename = filename[:-1]+str(i)
            tmp_file=GadgetFile(filename)
            if not self.check_headers(tmp_file.header, self._files[0].header) :
                print "WARNING: file "+str(i)+" is not part of this snapshot set!"
                continue
            self._files.append(tmp_file)
            npart=npart+tmp_file.header.npart
        #Set up things from the parent class
        self._num_particles = npart.sum()
        #Set up global header
        self.header=copy.deepcopy(self._files[0].header)
        self.header.npart = npart
        #Set up _family_slice
        self._family_slice[family.gas] = slice(npart[0:GAS_TYPE].sum(),npart[GAS_TYPE+1].sum())
        self._family_slice[family.dm] = slice(npart[0:DM_TYPE].sum(), npart[0:DM_TYPE+1].sum())
        self._family_slice[family.neutrino] = slice(npart[0:NEUTRINO_TYPE].sum(), npart[0:NEUTRINO_TYPE+1].sum())
        self._family_slice[family.star] = slice(npart[0:STAR_TYPE].sum(),npart[0:STAR_TYPE+1].sum() )
        #Delete any arrays the parent class may have made
        self._arrays = {}
        #TODO: Set up file_units_system
        return

    def get_block_list():
        """Get list of unique blocks in snapshot (most of the time these should be the same 
        in each file), with the types they refer to"""
        b_list = {}
        for f in self._files:
            b_list.update(f.blocks)
        #Setup array references
        #Make all array names lower-case and trim trailing spaces, to match the names 
        #used for tipsy snapshots
        out_list={}
        for k,b in b_list.iteritems() :
            b_name = k.lower().rstrip()
            b_types = ()
            if b.p_types[GAS_TYPE]:
                b_types.append(family.gas)
            if b.p_types[DM_TYPE]:
                b_types.append(family.dm)
            if b.p_types[NEUTRINO_TYPE]:
                b_types.append(family.neutrino)
            if b.p_types[STAR_TYPE]:
                b_types.append(family.star)
            out_list[b_name] = b_types
        return out_list

    def get_block_parts(self, name, p_type) :
        """Get the number of particles present in a block, of a given type"""
        total=0
        for f in self._files:
            total+=f.get_block_parts(name, p_type)
        return total

    def check_headers(self, head1, head2) :
        """Check two headers for consistency"""
        if ( head1.time != head2.time or head1.redshift!= head2.redshift or 
           head1.flag_sfr != head2.flag_sfr or
           head1.flag_feedback != head2.flag_feedback or
           head1.num_files != head2.num_files or
           head1.BoxSize != head2.BoxSize or
           head1.Omega0 != head2.Omega0 or
           head1.OmegaLambda != head2.OmegaLambda or
           head1.HubbleParam != head2.HubbleParam  or 
           head1.flag_stellarage != head2.flag_stellarage or
           head1.flag_metals != head2.flag_metals) :
            return False;
        #Check array quantities
        if((head1.mass != head2.mass).any() or  (head1.npartTotal != head2.npartTotal).any()) :
        #        or head1.NallHW != head2.NallHW)
            return False;
        #  At least one version of N-GenICs writes a header file which 
        #  ignores everything past flag_metals (!), leaving it uninitialised. 
        #  Therefore, we can't check them.
        return True
    
    def _read_array(self, name, fam=None) :
        """Read in data from a Gadget file. 
        If fam != None, loads only data for that particle family"""
        #Make the name a four-character upper case name, possibly with trailing spaces
        g_name = (name.upper().ljust(4," "))[0:4]
        p_read = 0
        p_start = 0
        data = array.SimArray([])
        p_type = gadget_type(fam)
        ndim = self._files[0].get_block_dims(g_name)
        dims = (self.get_block_parts(g_name, p_type), ndim)
        for f in self._files:
            f_read = 0 
            f_parts = f.get_block_parts(g_name, p_type)
            if f_parts == 0:
                continue
            (f_read, f_data) = f.get_block(g_name, p_type, f_parts)
            p_read+=f_read
            data=np.append(data, f_data)
        if np.size(data) == 0:
                raise KeyError, "Block "+name+" not in snapshot for family "+fam.name
        #TODO: Add some logic. If we have already got the data from a family, make
        #the family array a pointer to the main array so we don't load things twice.
        if fam is None :
            self._arrays[name] = data.reshape(dims, order='C').view(array.SimArray)
            self._arrays[name].sim = self
        else :
            self._create_family_array(name, fam, ndim, data.dtype)
            self._get_family_array(name, fam)[:] = \
                  data.reshape(dims,order='C').view(array.SimArray)
            self._get_family_array(name, fam).sim = self


    def _write_array(self, name, p_toread,p_start, p_type) :
        raise Exception, "Not yet implemented"

    @staticmethod
    def _can_load(f) :
	"""Check whether we can load the file as Gadget format by reading 
        the first 4 bytes"""
        try:
            fd=open(f)
        except IOError:
            try:
                fd=open(f+".0")
            except:
                return False
                #If we can't open the file, we certainly can't load it...
        (r,) = struct.unpack('=I',fd.read(4))
        fd.close()
        #First int32 is 8 for a Gadget 2 file, or 256 for Gadget 1, or the byte swapped equivalent.
        if r == 8 or r == 134217728 or r == 65536 or r == 256 :
            return True
        else :
            return False
