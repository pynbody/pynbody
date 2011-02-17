from . import snapshot,array

import numpy as np
#Needed to unpack things
import struct
import sys

#Symbolic constants for the particle types
BARYON_TYPE = 0
DM_TYPE = 1
DISK_TYPE = 2
NEUTRINO_TYPE = 2
BULGE_TYPE = 3
STARS_TYPE = 4
BNDRY_TYPE = 5
N_TYPE = 6

class GadgetBlock : 
    """Class to describe each block.
    Each block has a start, a length, and a length-per-particle"""
    def __init__(self) :
        self.start=0
        self.length=0
        self.partlen=0
        self.p_type = np.float32

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
    Contains the block location dictionary for each file."""
    def __init__(self, filename, BlockNames={}) :
        """BlockNames is not used at the moment; it is intended for supporting Gadget-I format files"""
        self._filename=filename
        self.blocks = {}
        self.header=GadgetHeader('')
        self.endian=''
        self.format2=True
        t_part = 0
        fd=open(filename, "rb")
        self.check_format(fd)
        while True:
            block=GadgetBlock()
            if self.format2 :
                (name, block.length) = self.read_G2_block_head(fd)
                if block.length == 0 :
                    break
                record_size = fd.read(4)
                if len(record_size) != 4 :
                    raise IOError, "Ran out of data in "+filename+" before block "+name+" started"
                (record_size,)=struct.unpack(self.endian+'I', record_size)
                if block.length != record_size :
                    raise IOError, "Corrupt record in "+filename+" for block "+name+" ( "+str(block.length)+" vs "+str(record_size)
            else :
                raise Exception, "Reading Gadget 1 files not implemented"
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
                    block.p_type = np.int32
                else :
                    block.partlen = 8
                    block.p_type = np.int64
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
            self.blocks[name[0:4]] = block

        #and we're done.
        fd.close()
        return

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


    def read_G2_block_head(self, fd) :
        """Read the Gadget 2 "block header" record, ie, 8 name, length, 8.
           Takes an open file and returns a (name, length) tuple """
        head=fd.read(4*4)
        #If we have run out of file, we don't want an exception, we just want a zero length empty block
        if len(head) != 4*4 :
            return ("    ",0)
        head=struct.unpack(self.endian+'I4sII',head)
        if head[0] != 8 or head[3] != 8 :
            raise IOError, "Corrupt header record. Possibly incorrect file format"
        #Don't include the two "record_size" indicators in the total length count
        return (head[1], head[2]-8)
        
    def GetBlock(self, name, p_start, p_toread) :
        """Get a particle range from this file, starting at p_start, 
        and reading a maximum of p_toread particles"""
        p_read = 0
        cur_block = self.blocks[name]
        parts = self.GetBlockParts(name)
        if p_start >= parts :
            return (p_start - parts, np.empty([]))
        if p_start+p_toread > parts :
            p_toread = parts - p_start
        fd=open(self._filename, 'rb')
        fd.seek(cur_block.start+cur_block.partlen*p_start,0)
        #This is just so that we can get a size for the type
        dt = np.dtype(cur_block.p_type)
        n_type = p_toread*cur_block.partlen/dt.itemsize
        data=np.fromfile(fd, dtype=cur_block.p_type, count=n_type, sep = '')
        if self.endian != '=' :
            data=data.byteswap(True)
        return (p_toread, data)

    def GetBlockParts(self, name):
        """Get the number of particles present in a block in this file"""
        if not self.blocks.has_key(name) :
                return 0
        cur_block = self.blocks[name]
        return cur_block.length/cur_block.partlen
    
    def GetBlockDims(self, name):
        """Get the dimensionality of the block, eg, 3 for POS, 1 for most other things"""
        if not self.blocks.has_key(name) :
                return 0
        cur_block = self.blocks[name]
        dt = np.dtype(cur_block.p_type)
        return cur_block.partlen/dt.itemsize


class GadgetSnap(snapshot.SimSnap):
    """Main class for reading Gadget-2 snapshots. The constructor makes a map of the locations 
    of the blocks, which are then read by _read_array"""
    def __init__(self, filename, only_header=False, must_have_paramfile=False) : 
        super(GadgetSnap,self).__init__()
        self.files = []
        self._filename=""
        self.npart = np.empty(N_TYPE)
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
        self.files.append(first_file)
        files_expected = self.files[0].header.num_files
        self.npart = np.array(self.files[0].header.npart)
        for i in np.arange(1, files_expected):
            filename = filename[:-1]+str(i)
            tmp_file=GadgetFile(filename)
            if not self.check_headers(tmp_file.header, self.files[0].header) :
                print "WARNING: file "+str(i)+" is not part of this snapshot set!"
                continue
            self.files.append(tmp_file)
            self.npart=self.npart+tmp_file.header.npart
        #Set up things from the parent class
        self._num_particles = self.npart.sum()
        #List the blocks in the snapshot
        b_list = [] 
        for f in self.files:
            b_list = set(b_list) | set( f.blocks.keys() )
        #Make all array names lower-case and trin the trailing " "'s, to match the names 
        #used for tipsy snapshots
        b_list = [b.lower().rstrip() for b in b_list]
        self._arrays = {}
        for k in b_list :
            self._arrays[k] = array.SimArray([])
        #TODO: Set up file_units_system
        return

    def GetHeader(self, i) :
        """Get a header from a file"""
        if len(self.files) <= i:
            return GadgetHeader()
        return self.files[i].header

    def GetBlockParts(self, name) :
        """Get the number of particles present in a block"""
        total=0
        for f in self.files:
            total+=f.GetBlockParts(name)
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
        If fam != None, loads only data for that particle family (not implemented)"""
        #Make the name a four-character upper case name, possibly with trailing spaces
        g_name = (name.upper().ljust(4," "))[0:4]
        p_read = 0
        p_start = 0
        data = array.SimArray([])
        dims = (self.GetBlockParts(g_name), self.files[0].GetBlockDims(g_name))
        #TODO Need some mapping between fam and integers. 
        if fam != None :
            #Do something
            pass
        for f in self.files:
            f_read = 0 
            f_parts = f.GetBlockParts(g_name)
            if f_parts == 0:
                continue
            (f_read, f_data) = f.GetBlock(g_name, 0, f_parts)
            p_read+=f_read
            data=np.append(data, f_data)
        if np.size(data) == 0:
                raise KeyError, "Block "+name+" not found in snapshot"
        self._arrays[name] = data.reshape(dims, order='C').view(array.SimArray)
        self._arrays[name].sim = self

    def _write_array(self, name, p_toread,p_start, p_type) :
        raise Exception, "Not yet implemented"

    @staticmethod
    def _can_load(f) :
	# to implement!
        return True
