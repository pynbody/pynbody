from . import snapshot, array, util
from . import decorate, family

import struct
import numpy as np

class TipsySnap(snapshot.SimSnap) :
    def __init__(self, filename) :
	super(TipsySnap,self).__init__()
	
	self._filename = filename
	
	f = util.open_(filename)
	# factoring out gzip logic opens up possibilities for bzip2,
	# or other more advanced filename -> file object mapping later,
	# carrying this across the suite

	t, n, ndim, ng, nd, ns = struct.unpack(">dlllll", f.read(28))

        # In non-cosmological simulations, what is t? Is it physical
        # time?  In which case, we need some way of telling what we
        # are dealing with and setting properties accordingly.
        self.properties['a'] = t
        self.properties['z'] = 1.0/t - 1.0

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
	
	# ["rho","temp","eps","metals","phi","tform"])

	

	# Load in the tipsy file in blocks.  This is the most
	# efficient balance I can see for keeping IO contiguuous, not
	# wasting memory, but also not having too much python <-> C
	# interface overheads

	max_block_size = 1024 # particles

	# describe the file structure as list of (num_parts, [list_of_properties]) 
	file_structure = ((ng,family.gas,["mass","x","y","z","vx","vy","vz","rho","temp","eps","metals","phi"]),
			  (nd,family.dm,["mass","x","y","z","vx","vy","vz","eps","phi"]),
			  (ns,family.star,["mass","x","y","z","vx","vy","vz","metals","tform","eps","phi"]))
	

	decorate.decorate_top_snap(self)
	

	for n_left, type, st in file_structure :
	    n_done = 0
	    self_type = self[type]
	    while n_left>0 :
		n_block = max(n_left,max_block_size)

		# Read in the block
		g = np.fromstring(f.read(len(st)*n_block*4),'f').byteswap().reshape((n_block,len(st)))

		# Copy into the correct arrays
		for i, name in enumerate(st) :
		    self_type[name][n_done:n_done+n_block] = g[:,i]

		# Increment total ptcls read in, decrement ptcls left of this type
		n_left-=n_block
		n_done+=n_block


	

		
	

    def _read_array(self, array_name) :
	filename = self._filename+"."+array_name  # could do some cleverer mapping here
	
	f = util.open_(filename)
	try :
	    l = int(f.readline())
	except ValueError :
	    raise IOError, "Incorrect file format"
	if l!=len(self) :
	    raise IOError, "Incorrect file format"
	
	# Inspect the first line to see whether it's float or int
	l = f.readline()
	if "." in l :
	    tp = float
	else :
	    tp = int

	# Restart at head of file
	del f
	f = util.open_(filename)

	# N.B. np.loadtxt seems to be quite slow, and we might be
	# forced to implement something faster in C? Or does an
	# alternative already exist?
	data = np.loadtxt(f, skiprows=1, dtype=tp)
	ndim = len(data)/len(self)

	if ndim*len(self)!=len(data) :
	    raise IOError, "Incorrect file format"
	
	if ndim>1 :
	    dims = (len(data),ndim)
	else :
	    dims = len(data)
	    
	self._arrays[array_name] = data.reshape(dims).view(array.SimArray)

	return self[array_name]
	
	
    @staticmethod
    def _can_load(f) :
	# to implement!
	return True
