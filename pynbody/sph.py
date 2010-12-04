"""
pynbody SPH rendering module.

This module encompasses Kernel objects, which return C fragments from which
a final C code to perform the rendering is derived. 

For most users, the function of interest will be render_image."""

import numpy as np
import scipy, scipy.weave
from scipy.weave import inline
import math

class Kernel(object) :
    def __init__(self) :
	self.h_power = 3
	# Return the power of the smoothing length which appears in
	# the denominator of the expression for the general kernel.
	# Will be 3 for 3D kernels, 2 for 2D kernels.

	self.max_d = 2
	# The maximum value of the displacement over the smoothing for
	# which the kernel is non-zero

    def get_c_code(self) :
	# placeholder square kernel!
	code =""
	
	sample_pts = np.arange(0,2.01,0.01)
	samples = [self.get_value(x) for x in sample_pts]
	samples_s = str(samples)
	samples_s = "{"+samples_s[1:-1]+"};"
	h_str="*h"*(self.h_power-1)
	if self.h_power==2 :
	    code+="#define Z_CONDITION(dz,h) true\n"
	else :
	    code+="#define Z_CONDITION(dz,h) abs(dz)<MAX_D_OVER_H*(h)\n"

	if self.h_power==2 :
	    code+="#define DISTANCE(dx,dy,dz) sqrt((dx)*(dx)+(dy)*(dy))"
	else :
	    code+="#define DISTANCE(dx,dy,dz) sqrt((dx)*(dx)+(dy)*(dy)+(dz)*(dz))"

	    
	code+= """
	const float KERNEL_VALS[] = %s
	#define KERNEL1(d,h) (d<2*h)?KERNEL_VALS[int(d/(0.01*h))]/(h%s):0
	#define KERNEL(dx,dy,dz,h) KERNEL1(DISTANCE(dx,dy,dz),h)
	#define MAX_D_OVER_H %d
	"""%(samples_s,h_str,self.max_d)
	return code
    
    def get_value(self, d, h=1) :
	"""Get the value of the kernel for a given smoothing length."""
	# Default : spline kernel
	if d<1 :
	    f = 1.-(3./2)*d**2 + (3./4.)*d**3
	elif d<2 :
	    f = 0.25*(2.-d)**3
	else :
	    f = 0
	
	return f/(math.pi*h**3)

class Kernel2D(Kernel) :
    def __init__(self, k_orig=Kernel()) :
	self.h_power = 2
	self.max_d = k_orig.max_d
	self.k_orig = k_orig

    def get_value(self, d, h=1) :
	import scipy.integrate as integrate
	import numpy as np
	return 2*integrate.quad(lambda z : self.k_orig.get_value(np.sqrt(z**2+d**2), h), 0, h)[0]


class TopHatKernel(object) :
    def __init__(self) :
	self.h_power = 3
	self.max_d = 2
	
    def get_c_code(self) :
	code = """#define KERNEL1(d,h) (d<2*h)?%.5e/(h*h*h):0
	#define KERNEL(dx,dy,dz,h) KERNEL1(sqrt((dx)*(dx)+(dy)*(dy)+(dz)*(dz)),h)
	#define Z_CONDITION(dz,h) abs(dz)<(2*h)
	#define MAX_D_OVER_H %d"""%(3./(math.pi*4),self.max_d)
	return code
	

    
def render_image(snap, qty='rho', x2=100, nx=500, y2=None, ny=None, x1=None, y1=None,
		 z_plane = 0.0, out_units=None, kernel=Kernel()) :

    """Render an SPH image using a typical (mass/rho)-weighted 'scatter'
    scheme.

    Keyword arguments:
    qty -- The name of the array within the simulation to render
    x2 -- The x-coordinate of the right edge of the image (default 100.0)
    nx -- The number of pixels wide to make the image (default 500)
    y2 -- The y-coordinate of the upper edge of the image (default x2)
    ny -- The number of pixels tall to make the image (default nx)
    x1 -- The x-coordinate of the left edge of the image (default -x2)
    y1 -- The y-coordinate of the lower edge of the image (default -y2)
    z_plane -- The z-coordinate of the plane of the image (default 0.0)
    out_units -- The units to convert the output image into (default no conversion)
    kernel -- The Kernel object to use (default Kernel(), a 3D spline kernel)
    """
    
    import os, os.path

    if y2 is None :
	y2 = x2
    if ny is None :
	ny = nx
    if x1 is None :
	x1 = -x2
    if y1 is None :
	y1 = -y2
    	
    x1, x2, y1, y2, z1 = [float(q) for q in x1,x2,y1,y2,z_plane]
    
    result = np.zeros((nx,ny))

    n_part = len(snap)
    x = snap['x']
    y = snap['y']
    z = snap['z']
    try:
	sm = snap['smooth']
    except KeyError :
	sm = ((snap['mass']/snap['rho'])**(1,3))
	sm.convert_units(x.units)

    qty = snap[qty]
    mass = snap['mass']
    rho = snap['rho']
    
    if out_units is not None :
	# Calculate the ratio now so we don't waste time calculating
	# the image only to throw a UnitsException later
	conv_ratio = (qty.units*mass.units/(rho.units*x.units**kernel.h_power)).ratio(out_units,
										      **x.conversion_context())
    
    code = kernel.get_c_code()
  
    
    code+=file(os.path.join(
	os.path.dirname(__file__),
	'sph_image.c')).read()


    # before inlining, the views on the arrays must be standard np.ndarray
    # otherwise the normal numpy macros are not generated
    x,y,z,sm,qty, mass, rho = [q.view(np.ndarray) for q in x,y,z,sm,qty, mass, rho]

    
    inline(code, ['result', 'nx', 'ny', 'x', 'y', 'z', 'sm',
    		  'x1', 'x2', 'y1', 'y2', 'z1',  'qty', 'mass', 'rho'])

    if out_units is not None :
	result*=conv_ratio

 
    return result
