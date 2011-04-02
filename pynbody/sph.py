"""
pynbody SPH rendering module.

This module encompasses Kernel objects, which return C fragments from which
a final C code to perform the rendering is derived.

For most users, the function of interest will be render_image."""

import numpy as np
import scipy, scipy.weave
from scipy.weave import inline
import snapshot, array
import math
from . import snapshot, array

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
        #define KERNEL1(d,h) (d<2*h)?KERNEL_VALS[(int)(d/(0.01*h))]/(h%s):0
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
        code = """#define KERNEL1(d,h) (d<%d *h)?%.5e/(h*h*h):0
        #define KERNEL(dx,dy,dz,h) KERNEL1(sqrt((dx)*(dx)+(dy)*(dy)+(dz)*(dz)),h)
        #define Z_CONDITION(dz,h) abs(dz)<(%d*h)
        #define MAX_D_OVER_H %d"""%(self.max_d,3./(math.pi*4*self.max_d**self.h_power),self.max_d, self.max_d)
        return code



def render_spherical_image(snap, qty='rho', nside = 8, distance = 10.0, kernel=Kernel()) :
    """Render an SPH image on a spherical surface. Note this is written in pure python and
    could be optimized into C, but would then need linking with the healpix libraries.
    Also currently uses a top-hat 3D kernel only."""

    import healpy as hp

    try :
        import healpy_f as hpf
        query_disc = hpf.query_disc
    except ImportError :
        query_disc = lambda a, b, c : hp.query_disc(a,b,c,deg=False)

    ind = np.where(np.abs(snap["r"]-distance)<snap["smooth"]*kernel.max_d)

    print "Spherical image from",len(ind[0]),"particles"
    D = snap["r"]
    h = snap["smooth"]

    im = np.zeros(hp.nside2npix(nside))

    for i in ind[0] :
        # angular radius subtended by the intersection of the boundary
        # of the SPH particle with the boundary surface of the calculation
        rad = np.arctan(math.sqrt((h[i]*kernel.max_d)**2 - (D[i]-distance)**2)/distance)
        try :
            i2 = query_disc(nside, snap["pos"][i], rad)
        except UnboundLocalError :
            i2 = []

        im[i2]+=((snap[qty][i]*snap["mass"][i]/snap["rho"][i]) / (math.pi*4*((kernel.max_d*h[i])**3)/3))

    im = im.view(array.SimArray)
    im.units = snap[qty].units*snap["mass"].units/snap["rho"].units/snap["smooth"].units**3
    im.sim = snap

    return im

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

    result = np.zeros((nx,ny),dtype=np.float32)

    n_part = len(snap)
    x = snap['x']
    y = snap['y']
    z = snap['z']

    sm = snap['smooth']

    if sm.units!=x.units :
        sm.convert_units(x.units)

    qty_s = qty
    qty = snap[qty]
    mass = snap['mass']
    rho = snap['rho']

    if out_units is not None :
        # Calculate the ratio now so we don't waste time calculating
        # the image only to throw a UnitsException later
        conv_ratio = (qty.units*mass.units/(rho.units*sm.units**kernel.h_power)).ratio(out_units,
                                                                                      **x.conversion_context())



    code = kernel.get_c_code()


    try:
        import pyopencl as cl
        import struct
        import squadron
        
        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(ctx)

        mf = cl.mem_flags

        pos = snap['pos']
        par = struct.pack('ffffi', (x2-x1)/nx, (y2-y1)/ny, x1, y1, len(pos))
    
        par_buf = cl.Buffer(ctx, mf.READ_ONLY, len(par))
        cl.enqueue_write_buffer(queue, par_buf, par)
        
        
        
        sm_buf, qty_buf, pos_buf = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x.astype(np.float32))
                                              for x in sm, (qty*mass/rho), pos]

        par_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=par)
        
        dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

        local_pos = cl.LocalMemory(500*3*4)
        local_sm = cl.LocalMemory(500*4)
        local_qty = cl.LocalMemory(500*4)
        
        code+=file(os.path.join(os.path.dirname(__file__),
                                'sph_image.cl')).read()

        print code
        
        prg = cl.Program(ctx, code).build(devices=[cl.get_platforms()[0].get_devices()[0]])

        prg.render(queue, result.shape, None, sm_buf, qty_buf, pos_buf, par_buf,
                   dest_buf,
                   local_pos, local_sm, local_qty )
        
        cl.enqueue_read_buffer(queue, dest_buf, result).wait()

    except ImportError :
        
        code+=file(os.path.join(
            os.path.dirname(__file__),
            'sph_image.c')).read()


        # before inlining, the views on the arrays must be standard np.ndarray
        # otherwise the normal numpy macros are not generated
        x,y,z,sm,qty, mass, rho = [q.view(np.ndarray) for q in x,y,z,sm,qty, mass, rho]



        inline(code, ['result', 'nx', 'ny', 'x', 'y', 'z', 'sm',
                      'x1', 'x2', 'y1', 'y2', 'z1',  'qty', 'mass', 'rho'])


    result = result.view(array.SimArray)

    # The weighting works such that there is a factor of (M_u/rho_u)h_u^3
    # where M-u, rho_u and h_u are mass, density and smoothing units
    # respectively. This is dimensionless, but may not be 1 if the units
    # have been changed since load-time.
    if out_units is None :
        result*= (snap['mass'].units / (snap['rho'].units)).ratio(snap['x'].units**3, **snap['x'].conversion_context())

        # The following will be the units of outputs after the above conversion is applied
        result.units = snap[qty_s].units/snap['x'].units**kernel.h_power
    else:
        result*=conv_ratio
        result.units = out_units

    result.sim = snap
    return result


def calculate_smoothing(snap, nleaf=10, nn=16, timing=False):
    """
    Construct a KDTree using the scipy.spatial.KDTree class and determine
    the smoothing lenghts for all particles in the sim snapshot.
    The smoothing length is defined as smooth = 0.5*d, where d is the distance
    of the most distant nearest neighbor.

    The kd tree is saved in the snapshot and the smoothing values are
    saved as a new array.
    """

    import scipy.spatial.ckdtree as kdtree
    from time import time

    t1 = time()
    snap.kdt = kdtree.cKDTree(snap['pos'], leafsize=nleaf)
    t2 = time()
    if timing:
        print 'tree built in ' + str(t2-t1) + ' seconds'

    t3 = time()
    nd,ni = snap.kdt.query(snap['pos'],k=nn)
    t4 = time()
    if timing:
        print 'nn search in ' + str(t4-t3) + ' seconds'

    # add the smoothing length as an array
    # nd is sorted, so only need the last item
    # also set the units to be the same as the current distance
    snap['smooth'] = array.SimArray(0.5*nd[:,nn-1], snap['pos'].units)

    # keep the list of nearest-neighbor indices
    del snap['nn_index']
    snap['nn_index'] = ni


#@snapshot.SimSnap.derived_quantity
#def smooth(sim):
#    calculate_smoothing(sim)
