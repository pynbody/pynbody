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


	

    def loadable_keys(self) :
	"""Produce and return a list of loadable arrays for this TIPSY file."""

	def is_readable_array(x) :
	    try:
		f = util.open_(x)
		return int(f.readline()) == len(self)
	    except (IOError, ValueError) :
		return False
	    
	import glob
	fs = glob.glob(self._filename+".*")
	return map(lambda q: q[len(self._filename)+1:], filter(is_readable_array, fs))
	
    def _read_array(self, array_name, fam=None) :
	"""Read a TIPSY-ASCII file with the specified name. If fam is not None,
	read only the particles of the specified family."""
	
	# N.B. this code is a bit inefficient for loading
	# family-specific arrays, because it reads in the whole array
	# then slices it.  But of course the memory is only wasted
	# while still inside this routine, so it's only really the
	# wasted time that's a problem.
	
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

	if fam is None :
	    self._arrays[array_name] = data.reshape(dims).view(array.SimArray)
	else :
	    self._create_family_array(array_name, fam, ndim, data.dtype)
	    self._get_family_array(array_name, fam)[:] = data[self._get_slice_for_family(fam)]
	
	
    @staticmethod
    def _can_load(f) :
	# to implement!
	return True


@decorate.sim_decorator
def param2units(sim) :
    import sys, math, os, glob

    x = os.path.abspath(sim.filename)

    filename=None
    for i in xrange(2) :
	l = glob.glob(os.path.join(x,"*.param"))
	if len(l)==1 :
	    filename = l[0]
	    break

	x = os.path.dirname(x)
	
    print "Using .param file ",filename

    if filename==None :
	return
    

    if filename!=None  :
	f = file(filename)
	munit = dunit = hub = None
	for line in f :
	    try :
		s = line.split()
		if s[0]=="dMsolUnit" :
		    munit_st = s[2]+" Msol"
		    munit = float(s[2])
		elif s[0]=="dKpcUnit" :
		    dunit_st = s[2]+" kpc a"
		    dunit = float(s[2])
		elif s[0]=="dHubble0" :
		    hub = float(s[2])
		elif s[0]=="dOmega0" :
		    om_m0 = s[2]
		elif s[0]=="dLambda" :
		    om_lam0 = s[2]

	    except IndexError, ValueError :
		pass

	if munit==None or dunit==None or hub==None :
	    raise RuntimeError("Can't find all parameters required in .param file")

	denunit = munit/dunit**3
	denunit_st = str(denunit)+" Msol kpc^-3 a^-3"

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
	velunit_st = ("%.5g"%velunit)+" km s^-1 a"

	enunit_st = "%.5g km^2 s^-2"%(velunit**2)


	hubunit = 10. * velunit / dunit
	hubunit_st = ("%.3f"%(hubunit*hub))

	sim["vel"].units = velunit_st
	sim["eps"].units = dunit_st
	sim["pos"].units = dunit_st
	sim.gas["rho"].units = denunit_st
	sim["mass"].units = munit_st
	
	
	sim.properties['h'] = hubunit
	sim.properties['omegaM0'] = om_m0
	sim.properties['omegaL0'] = om_lam0
	
