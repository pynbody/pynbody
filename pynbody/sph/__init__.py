"""

sph
===

pynbody SPH rendering module.

This module encompasses Kernel objects, which return C fragments from which
a final C code to perform the rendering is derived.

For most users, the function of interest will be :func:`~pynbody.sph.render_image`.

"""

import numpy as np
import scipy, scipy.ndimage
import math
import time
import sys
import threading
import copy

from .. import snapshot, array, config, units, util, config_parser

try:
    from . import kdtree
except ImportError :
    raise ImportError, "Pynbody cannot import the kdtree subpackage. This can be caused when you try to import pynbody directly from the installation folder. Try changing to another folder before launching python"
import os

def _get_threaded_smooth() : 
    return config_parser.getboolean('sph','threaded-smooth') and config['number_of_threads']

def _get_threaded_image() : 
    return config_parser.getboolean('sph','threaded-image') and config['number_of_threads']

_threaded_smooth = _get_threaded_smooth()
_threaded_image = _get_threaded_image()
_approximate_image = config_parser.getboolean('sph','approximate-fast-images')


def _thread_map(func, *args) :
    threads = []
    for arg_this in zip(*args) :
        threads.append(threading.Thread(target=func, args=arg_this))
        threads[-1].start()
    for t in threads:
        while t.is_alive() :
            # just calling t.join() with no timeout can make it harder to
            # debug deadlocks!
            t.join(1.0)

def build_tree(sim) :
    if hasattr(sim,'kdtree') is False :
        # n.b. getting the following arrays through the full framework is
        # not possible because it can cause a deadlock if the build_tree
        # has been triggered by getting an array in the calling thread.
        pos, vel, mass = [np.asanyarray(sim._get_array(x),dtype=np.float64) for x in 'pos','vel','mass']
        sim.kdtree = kdtree.KDTree(pos, vel, mass, leafsize=config['sph']['tree-leafsize'])
    
def _tree_decomposition(obj) :
    return [obj[i::_get_threaded_smooth()] for i in range(_get_threaded_smooth())]

def _get_tree_objects(sim) :
    return map(getattr,_tree_decomposition(sim),['kdtree']*_get_threaded_smooth())

def build_tree_or_trees(sim) :
    global _threaded_smooth
    _threaded_smooth = _get_threaded_smooth()
    
    if _threaded_smooth :
        bits = _tree_decomposition(sim)
        if all(map(hasattr, bits, ['kdtree']*len(bits))) : return
    else :
        if hasattr(sim,'kdtree') : return
    
    if config['verbose'] :
        if _threaded_smooth :
            print>>sys.stderr, 'Building %d trees with leafsize=%d'%(_threaded_smooth,config['sph']['tree-leafsize'])
        else :
            print>>sys.stderr, 'Building tree with leafsize=%d'%config['sph']['tree-leafsize']

    if config['tracktime'] :
        import time
        start = time.clock()

    if _threaded_smooth :
        # trigger any necessary 'lazy' activity from this thread,
        # it won't be available to the individual worker threads
        sim['pos'], sim['vel'], sim['mass']
        _thread_map(build_tree, bits)
    else :
        build_tree(sim)

    if config['tracktime'] :
        end = time.clock()
        print>>sys.stderr, 'Tree build done in %5.3g s'%(end-start)
    elif config['verbose'] and verbose: print>>sys.stderr, 'Tree build done.'

        
@snapshot.SimSnap.stable_derived_quantity
def smooth(self):
    build_tree_or_trees(self)
    
    if config['verbose']: print>>sys.stderr, 'Smoothing with %d nearest neighbours'%config['sph']['smooth-particles']
    sm = array.SimArray(np.empty(len(self['pos'])), self['pos'].units)
    if config['tracktime']:
        import time
        start = time.time()

    _threaded_smooth = _get_threaded_smooth()

    if _threaded_smooth :
        _thread_map(kdtree.KDTree.populate,
                    _get_tree_objects(self),
                    _tree_decomposition(sm),
                    ['hsm']*_threaded_smooth,
                    [config['sph']['smooth-particles']]*_threaded_smooth)

        sm/=_threaded_smooth**0.3333
    
            
    else :
        self.kdtree.populate(sm, 'hsm', nn=config['sph']['smooth-particles']) 
    
    if config['tracktime'] : 
        end = time.time()
        print>>sys.stderr, 'Smoothing done in %5.3gs'%(end-start)
    elif config['verbose']: print>>sys.stderr, 'Smoothing done.'

    return sm 

@snapshot.SimSnap.stable_derived_quantity
def rho(self):
    build_tree_or_trees(self)
    if config['verbose']: print>>sys.stderr, 'Calculating SPH density'
    rho = array.SimArray(np.empty(len(self['pos'])), self['mass'].units/self['pos'].units**3)

    smooth = self['smooth']
    
    if config['tracktime']:
        import time
        start = time.time()

    _threaded_smooth = _get_threaded_smooth()

    if _threaded_smooth :
        _thread_map(kdtree.KDTree.populate,
                    _get_tree_objects(self),
                    _tree_decomposition(rho),
                    ['rho']*_threaded_smooth,
                    [config['sph']['smooth-particles']]*_threaded_smooth,
                    _tree_decomposition(smooth))
        rho*=_threaded_smooth
    else :
        self.kdtree.populate(rho, 'rho', nn=config['sph']['smooth-particles'], smooth=smooth)
    
    if config['tracktime'] : 
        end = time.time()
        print>>sys.stderr, 'Density calculation done in %5.3g s'%(end-start)
    elif config['verbose']: print>>sys.stderr, 'Density done.'
    
    return rho

class Kernel(object) :
    def __init__(self) :
        self.h_power = 3
        # Return the power of the smoothing length which appears in
        # the denominator of the expression for the general kernel.
        # Will be 3 for 3D kernels, 2 for 2D kernels.

        self.max_d = 2
        # The maximum value of the displacement over the smoothing for
        # which the kernel is non-zero

        self.safe = threading.Lock()

    def get_c_code(self) :
        
        if not hasattr(self, "_code") :
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
            self._code = code
        
        return self._code

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
        self.safe = threading.Lock()
        
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

def render_spherical_image(snap, qty='rho', nside=8, distance=10.0, kernel=Kernel(),
                           kstep=0.5,denoise=False,out_units=None,threaded=None) :
    """Render an SPH image on a spherical surface. Requires healpy libraries.

    **Keyword arguments:**

    *qty* ('rho'): The name of the simulation array to render
    
    *nside* (8): The healpix nside resolution to use (must be power of 2)

    *distance* (10.0): The distance of the shell (for 3D kernels) or maximum distance
        of the skewers (2D kernels)

    *kernel*: The Kernel object to use (defaults to 3D spline kernel)

    *kstep* (0.5): The sampling distance when projecting onto the spherical surface in units of the
        smoothing length

    *denoise* (False): if True, divide through by an estimate of the discreteness noise.
      The returned image is then not strictly an SPH estimate, but this option can be
      useful to reduce noise.

    *threaded*: if False, render on a single core. Otherwise, the number of threads to use.
      Defaults to a value specified in your configuration files. *Currently multi-threaded
      rendering is slower than single-threaded because healpy does not release the gil*.
    """
    renderer = _render_spherical_image
    
    if threaded is None : threaded = _get_threaded_image()

    if threaded :
        im= _threaded_render_image(renderer,snap,qty,nside,distance,kernel,kstep,denoise,out_units, num_threads=threaded)
    else :
        im= renderer(snap, qty, nside,distance,kernel,kstep,denoise,out_units)
    return im
    
def _render_spherical_image(snap, qty='rho', nside = 8, distance = 10.0, kernel=Kernel(),
                           kstep=0.5, denoise=False, out_units=None,__threaded=False,snap_slice=None) :

    import healpy as hp
    from . import _spherical

    if out_units is not None :
        conv_ratio = (snap[qty].units*snap['mass'].units/(snap['rho'].units*snap['smooth'].units**kernel.h_power)).ratio(out_units,
                                                                                                                         **snap.conversion_context())

    if snap_slice is None : snap_slice = slice(len(snap))
    with snap.immediate_mode :
        D,h,pos,mass,rho,qtyar = [snap[x].view(np.ndarray)[snap_slice] for x in 'r','smooth','pos','mass','rho',qty]

    
    
    ds = np.arange(kstep,kernel.max_d+kstep/2,kstep)
    weights = np.zeros_like(ds)

    for i,d1 in enumerate(ds) :
        d0 = d1-kstep
        # work out int_d0^d1 x K(x), then set our discretized kernel to
        # match that
        dvals = np.arange(d0,d1,0.05)
        ivals = map(kernel.get_value,dvals)
        ivals*=dvals
        integ = ivals.sum()*0.05
        weights[i] = 2*integ / (d1**2-d0**2)
  
    weights[:-1]-=weights[1:]

    if kernel.h_power==3 :
        ind = np.where(np.abs(D-distance)<h*kernel.max_d)[0]
        
        # angular radius subtended by the intersection of the boundary
        # of the SPH particle with the boundary surface of the calculation:
        rad = np.arctan(np.sqrt(h[ind,np.newaxis]**2-(D[ind,np.newaxis]-distance)**2)/distance)
        
    elif kernel.h_power==2 :
        ind = np.where(D<distance)[0]
        
        # angular radius taken at distance of particle:
        rad = np.arctan(h[ind,np.newaxis]*ds[np.newaxis,:]/D[ind,np.newaxis])
    else :
        raise ValueError, "render_spherical_image doesn't know how to handle this kernel"

    im, im2 = _spherical.render_spherical_image_core(rho,mass,qtyar,pos,D,h,ind,ds,weights,nside)
    
    im = im.view(array.SimArray)
    if denoise :
        im/=im2
    im.units = snap[qty].units*snap["mass"].units/snap["rho"].units/snap["smooth"].units**(kernel.h_power)
    im.sim = snap

    if out_units is not None :
        im.convert_units(out_units)
    
    return im

def _threaded_render_image(fn, s,*args, **kwargs) :
    """
    Render an SPH image using multiple threads.

    The arguments are exactly the same as those to render_image, but
    additionally you can specify the number of threads using the
    keyword argument *num_threads*. The default is given by your configuration
    file, probably 4. It should probably match the number of cores on your
    machine. """
    
    with s.immediate_mode:
        num_threads = kwargs['num_threads']
        del kwargs['num_threads']

        verbose = kwargs.get('verbose', config['verbose'])

        kwargs['__threaded']=True # will pass into render_image

        ts = []
        outputs = []

        if verbose : print "Rendering image on %d threads..."%num_threads

        for i in xrange(num_threads) :
            kwargs_local = copy.copy(kwargs)
            kwargs_local['snap_slice'] = slice(i,None,num_threads)
            args_local = [outputs, s]+list(args)
            ts.append(threading.Thread(target = _render_image_bridge(fn), args=args_local, kwargs=kwargs_local))
            ts[-1].start()

        for t in ts : 
            t.join()

    return sum(outputs)

def _interpolated_renderer(fn, levels) :
    """
    Render an SPH image using interpolation to speed up rendering where smoothing
    lengths are large.
    """
    if levels==1 :
        return fn
    
    def render_fn(*args, **kwargs) :
        kwargs['smooth_range']=(0,2)
        kwargs['res_downgrade']=1
        sub=1
        base = fn(*args, **kwargs)
        kwargs['smooth_range']=(1,2)
        for i in xrange(1,levels) :
            sub*=2
            if i==levels-1:
                kwargs['smooth_range']=(1,100000)
            kwargs['res_downgrade']=sub
            new_im=fn(*args,**kwargs)
            base+=scipy.ndimage.interpolation.zoom(new_im, float(base.shape[0])/new_im.shape[0], order=1)
        return base
    return render_fn
    
def _render_image_bridge(fn) :
    """Helper function for threaded_render_image; do not call directly"""
    def bridge(*args, **kwargs) :
        output_list = args[0]
        X = fn(*args[1:],**kwargs)
        output_list.append(X)
    return bridge

def render_image(snap, qty='rho', x2=100, nx=500, y2=None, ny=None, x1=None, 
                 y1=None, z_plane = 0.0, out_units=None, xy_units=None,
                 kernel=Kernel(),
                 z_camera=None,
                 smooth='smooth',
                 smooth_in_pixels = False,
                 force_quiet=False,
                 approximate_fast=_approximate_image,
                 threaded=None,
                 denoise=False) :
    """
    Render an SPH image using a typical (mass/rho)-weighted 'scatter'
    scheme.

    **Keyword arguments:**

    *qty* ('rho'): The name of the array within the simulation to render

    *x2* (100.0): The x-coordinate of the right edge of the image

    *nx* (500): The number of pixels wide to make the image

    *y2*: The y-coordinate of the upper edge of the image (default x2,
     or if ny is specified, x2*ny/nx)

    *ny* (nx): The number of pixels tall to make the image
    
    *x1* (-x2): The x-coordinate of the left edge of the image

    *y1* (-y2): The y-coordinate of the lower edge of the image 

    *z_plane* (0.0): The z-coordinate of the plane of the image

    *out_units* (no conversion): The units to convert the output image into

    *xy_units*: The units for the x and y axes

    *kernel*: The Kernel object to use (default Kernel(), a 3D spline kernel)

    *z_camera*: If this is set, a perspective image is rendered,
     assuming the kernel is suitable (i.e. is a projecting
     kernel). The camera is at the specified z coordinate looking
     towards -ve z, and each pixel represents a line-of-sight radially
     outwards from the camera. The width then specifies the width of
     the image in the z=0 plane. Particles too close to the camera are
     also excluded.

     *smooth*: The name of the array which contains the smoothing lengths
      (default 'smooth')

     *smooth_in_pixels*: If True, the smoothing array contains the smoothing
       length in image pixels, rather than in real distance units (default False)

     *approximate_fast*: if True, render high smoothing length particles at
       progressively lower resolution, resample and sum

     *denoise*: if True, divide through by an estimate of the discreteness noise.
       The returned image is then not strictly an SPH estimate, but this option
       can be useful to reduce noise especially when rendering AMR grids which
       often introduce problematic edge effects.

     *verbose*: if True, all text output suppressed

     *threaded*: if False (or None), render on a single core. Otherwise,
      the number of threads to use (defaults to a value specified in your
      configuration files).
    """

    if approximate_fast :
        base_renderer = _interpolated_renderer(_render_image, int(np.floor(np.log2(nx/20))))
    else :
        base_renderer = _render_image

    if threaded is None: threaded = _get_threaded_image()

    if threaded :
        im =  _threaded_render_image(base_renderer,snap, qty, x2, nx, y2, ny, x1, y1, z_plane,
                                      out_units, xy_units, kernel, z_camera, smooth,
                                      smooth_in_pixels, True, 
                                      num_threads=threaded)
    else :
        im =  base_renderer(snap, qty, x2, nx, y2, ny, x1, y1, z_plane,
                               out_units, xy_units, kernel, z_camera, smooth,
                               smooth_in_pixels, False)
        
    if denoise :
        # call self to render a 'flat field'
        snap['__one']=1
        im2 = render_image(snap, '__one', x2, nx, y2, ny, x1, y1, z_plane, None,
                           xy_units, kernel, z_camera, smooth, smooth_in_pixels,
                           force_quiet, approximate_fast, threaded, False)
        del snap.ancestor['__one']
        im2 = im/im2
        im2.units = im.units
        return im2

    else :
        return im


            
def _render_image(snap, qty, x2, nx, y2, ny, x1, 
                 y1, z_plane, out_units, xy_units, kernel, z_camera,
                 smooth, smooth_in_pixels,  force_quiet,
                 smooth_range=None, res_downgrade=None, snap_slice=None,
                 __threaded=False) :

    """The single-threaded image rendering core function. External calls
    should be made to the render_image function."""

    
    import os, os.path
    global config


    track_time = config["tracktime"] and not force_quiet
    verbose = config["verbose"] and not force_quiet

    snap_proxy = {}
    
    # cache the arrays and take a slice of them if we've been asked to
    for arname in 'x','y','z', 'pos',smooth,qty,'rho','mass' :
        snap_proxy[arname] = snap[arname]
        if snap_slice is not None :
            snap_proxy[arname] = snap_proxy[arname][snap_slice]
            
    
    if track_time :
        import time
        in_time = time.time()
    

    if y2 is None :
        if ny is not None :
            y2 = x2*float(ny)/nx
        else :
            y2 = x2
            
    if ny is None :
        ny = nx
    if x1 is None :
        x1 = -x2
    if y1 is None :
        y1 = -y2

    if res_downgrade is not None :
        # calculate original resolution
        dx = float(x2-x1)/nx
        dy = float(y2-y1)/ny

        # degrade resolution
        nx/=res_downgrade
        ny/=res_downgrade
        
        # shift boundaries (since x1, x2 etc refer to centres of pixels,
        # not edges, but we want the *edges* to remain invariant)
        sx = dx*float(res_downgrade-1)/2
        sy = dy*float(res_downgrade-1)/2
        x1-=sx
        y1-=sy
        x2+=sx
        y2+=sy
    
    x1, x2, y1, y2, z1 = [float(q) for q in x1,x2,y1,y2,z_plane]

    if smooth_range is not None :
        smooth_lo = int(smooth_range[0])
        smooth_hi = int(smooth_range[1])
    else :
        smooth_lo = 0
        smooth_hi = 0

    result = np.zeros((ny,nx),dtype=np.float32)

    n_part = len(snap)

    if xy_units is None :
        xy_units = snap_proxy['x'].units

    x = snap_proxy['x'].in_units(xy_units)
    y = snap_proxy['y'].in_units(xy_units)
    z = snap_proxy['z'].in_units(xy_units)

    sm = snap_proxy[smooth]

    if sm.units!=x.units and not smooth_in_pixels:
        sm = sm.in_units(x.units)
    
    
    qty_s = qty
    qty = snap_proxy[qty]
    mass = snap_proxy['mass']
    rho = snap_proxy['rho']
    

    if out_units is not None :
        # Calculate the ratio now so we don't waste time calculating
        # the image only to throw a UnitsException later
        conv_ratio = (qty.units*mass.units/(rho.units*sm.units**kernel.h_power)).ratio(out_units,
                                                                                      **x.conversion_context())

    try :
        kernel.safe.acquire(True)
        code = kernel.get_c_code()
    finally :
        kernel.safe.release()

    perspective = z_camera is not None

    
    
    if perspective :
        z_camera = float(z_camera)
        code+="#define PERSPECTIVE 1\n"
    if smooth_range is not None :
        code+="#define SMOOTH_RANGE 1\n"
        
    if __threaded :
        code+="#define THREAD 1\n"

    if smooth_in_pixels :
        code+="#define SMOOTH_IN_PIXELS 1\n"
    
    try:
        #import pyopencl as cl
        #import struct
        #import squadron
        raise ImportError
    
        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(ctx)

        mf = cl.mem_flags

        pos = snap_proxy['pos']
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

      
        prg = cl.Program(ctx, code).build(devices=[cl.get_platforms()[0].get_devices()[0]])

        prg.render(queue, result.shape, None, sm_buf, qty_buf, pos_buf, par_buf,
                   dest_buf,
                   local_pos, local_sm, local_qty )
        
        cl.enqueue_read_buffer(queue, dest_buf, result).wait()

    except ImportError :

        
        code+=file(os.path.join(os.path.dirname(__file__),'sph_image.c')).read()


        # before inlining, the views on the arrays must be standard np.ndarray
        # otherwise the normal numpy macros are not generated
        x,y,z,sm,qty, mass, rho = [np.asarray(q, dtype=float) for q in x,y,z,sm,qty, mass, rho]
        #qty[np.where(qty < 1e-15)] = 1e-15

        if verbose: print>>sys.stderr, "Rendering SPH image"

        if track_time:
            print>>sys.stderr, "Beginning SPH render at %.2f s"%(time.time()-in_time)
        util.threadsafe_inline( code, ['result', 'nx', 'ny', 'x', 'y', 'z', 'sm',
                      'x1', 'x2', 'y1', 'y2', 'z_camera', 'z1',   
                      'qty', 'mass', 'rho', 'smooth_lo' ,'smooth_hi'],verbose=2)
        
        if track_time:
            print>>sys.stderr, "Render done at %.2f s"%(time.time()-in_time)


    result = result.view(array.SimArray)

    # The weighting works such that there is a factor of (M_u/rho_u)h_u^3
    # where M-u, rho_u and h_u are mass, density and smoothing units
    # respectively. This is dimensionless, but may not be 1 if the units
    # have been changed since load-time.
    if out_units is None :
        result*= (snap_proxy['mass'].units / (snap_proxy['rho'].units)).ratio(snap_proxy['x'].units**3, **snap_proxy['x'].conversion_context())

        # The following will be the units of outputs after the above conversion is applied
        result.units = snap_proxy[qty_s].units*snap_proxy['x'].units**(3-kernel.h_power)
    else:
        result*=conv_ratio
        result.units = out_units

    result.sim = snap
    return result


def to_3d_grid(snap, qty='rho', nx=None, ny=None, nz=None, x2=None, out_units=None,
               xy_units=None, kernel=Kernel(), smooth='smooth', approximate_fast=_approximate_image,
               threaded=None,snap_slice=None, denoise=False) :
    """

    Project SPH onto a grid using a typical (mass/rho)-weighted 'scatter'
    scheme.

    **Keyword arguments:**

    *qty* ('rho'): The name of the array within the simulation to render

    *nx* (x2-x1 / soft): The number of pixels wide to make the grid

    *ny* (nx): The number of pixels tall to make the grid

    *nz* (nx): The number of pixels deep to make the grid
    
    *out_units* (no conversion): The units to convert the output grid into

    *xy_units*: The units for the x and y axes

    *kernel*: The Kernel object to use (default Kernel(), a 3D spline kernel)

    *smooth*: The name of the array which contains the smoothing lengths
      (default 'smooth')

    *denoise*: if True, divide through by an estimate of the discreteness noise.
      The returned image is then not strictly an SPH estimate, but this option
      can be useful to reduce noise especially when rendering AMR grids which
      often introduce problematic edge effects.

    """

    import os, os.path
    global config
 
   
    
    if config["tracktime"] :
        import time
        in_time = time.time()
    

   
    if x2 is None :
        x1 = np.min(snap['x'])
        x2 = np.max(snap['x'])
        y1 = np.min(snap['y'])
        y2 = np.max(snap['y'])
        z1 = np.min(snap['z'])
        z2 = np.max(snap['z'])
    else :
        x1 = -x2
        y1 = -x2
        z1 = -x2
        z2 = x2
        y2 = x2
            
    if nx is None :
        nx = np.ceil((x2 - x1) / np.min(snap['eps']))
    if ny is None :
        ny = nx
    if nz is None :
        nz = nx

    x1, x2, y1, y2, z1, z2 = [float(q) for q in x1,x2,y1,y2,z1,z2]
    nx, ny, nz = [int(q) for q in nx,ny,nz]

    if approximate_fast :
        renderer = _interpolated_renderer(_to_3d_grid, int(np.floor(np.log2(nx/20))))
    else :
        renderer = _to_3d_grid

    if threaded is None : threaded = _get_threaded_image()

    if threaded :
        im= _threaded_render_image(renderer,snap, qty, nx, ny, nz, x1, x2, y1, y2, z1, z2, out_units,
                                    xy_units, kernel, smooth, num_threads=threaded)
    else :
        im= renderer(snap, qty, nx, ny, nz, x1, x2, y1, y2, z1, z2, out_units,
                      xy_units, kernel, smooth, False)
        
    if config["tracktime"] :
        print>>sys.stderr, "Render done at %.2f s"%(time.time()-in_time)

        
    if denoise :
        # call self to render a 'flat field'
        snap['__one']=1
        im2 = to_3d_grid(snap, '__one',nx,ny,nz,x2,None,xy_units,kernel,smooth,
                         approximate_fast,threaded,snap_slice,False)
        del snap.ancestor['__one']
        im2 = im/im2
        im2.units = im.units
        return im2

    else :
        return im

        


def _to_3d_grid(snap, qty, nx, ny, nz, x1, x2, y1, y2, z1, z2, out_units,
               xy_units, kernel, smooth, __threaded=False,res_downgrade=None,
               snap_slice=None,
               smooth_range=None) :

    snap_proxy = {}

    # cache the arrays and take a slice of them if we've been asked to
    for arname in 'x','y','z', 'pos',smooth,qty,'rho','mass' :
        snap_proxy[arname] = snap[arname]
        
        if snap_slice is not None :
            snap_proxy[arname] = snap_proxy[arname][snap_slice]
            
    if res_downgrade is not None :
        dx = float(x2-x1)/nx
        dy = float(y2-y1)/ny
        dz = float(z2-z1)/nz
        
        nx/=res_downgrade
        ny/=res_downgrade
        nz/=res_downgrade

        # shift boundaries (see _render_image above for explanation)
        sx,sy,sz = [d_i*float(res_downgrade-1)/2 for d_i in [dx,dy,dz]]
        x1-=sx
        y1-=sy
        z1-=sz
        x2+=sx
        y2+=sy
        z2+=sz
        
        
    result = np.zeros((nx,ny,nz),dtype=np.float32)
    n_part = len(snap)

    if xy_units is None :
        xy_units = snap_proxy['x'].units

    x = snap_proxy['x'].in_units(xy_units)
    y = snap_proxy['y'].in_units(xy_units)
    z = snap_proxy['z'].in_units(xy_units)

    sm = snap_proxy[smooth]

    if sm.units!=x.units :
        sm = sm.in_units(x.units)

    qty_s = qty
    qty = snap_proxy[qty]
    mass = snap_proxy['mass']
    rho = snap_proxy['rho']

    if out_units is not None :
        # Calculate the ratio now so we don't waste time calculating
        # the image only to throw a UnitsException later
        conv_ratio = (qty.units*mass.units/(rho.units*sm.units**kernel.h_power)).ratio(out_units,
                                                                                      **x.conversion_context())

    try :
        kernel.safe.acquire(True)
        code = kernel.get_c_code()
    finally :
        kernel.safe.release()

    if __threaded :
        code+="#define THREAD 1\n"
    if smooth_range is not None :
        code+="#define SMOOTH_RANGE 1\n"
        smooth_lo=float(smooth_range[0])
        smooth_hi=float(smooth_range[1])
    else :
        smooth_lo = 0
        smooth_hi = 0

       
    code+=file(os.path.join(os.path.dirname(__file__),'sph_to_grid.c')).read()
    
    
    # before inlining, the views on the arrays must be standard np.ndarray
    # otherwise the normal numpy macros are not generated
    x,y,z,sm,qty, mass, rho = [q.view(np.ndarray) for q in x,y,z,sm,qty, mass, rho]
        #qty[np.where(qty < 1e-15)] = 1e-15
    
    if config['verbose']: print>>sys.stderr, "Gridding particles"
   
    util.threadsafe_inline( code, ['result', 'nx', 'ny', 'nz', 'x', 'y', 'z', 'sm',
                   'x1', 'x2', 'y1', 'y2', 'z1',  'z2',
                   'qty', 'mass', 'rho', 'smooth_lo','smooth_hi'],verbose=2)
   
    result = result.view(array.SimArray)

    # The weighting works such that there is a factor of (M_u/rho_u)h_u^3
    # where M_u, rho_u and h_u are mass, density and smoothing units
    # respectively. This is dimensionless, but may not be 1 if the units
    # have been changed since load-time.
    if out_units is None :
        result*= (snap_proxy['mass'].units / (snap_proxy['rho'].units)).ratio(snap_proxy['x'].units**3, **snap_proxy['x'].conversion_context())

        # The following will be the units of outputs after the above conversion is applied
        result.units = snap_proxy[qty_s].units*snap_proxy['x'].units**(3-kernel.h_power)
    else:
        result*=conv_ratio
        result.units = out_units

    result.sim = snap
    return result


def spectra(snap, qty='rho', x1=0.0, y1=0.0, v2=400, nvel=200, v1=None,
            element='H', ion='I',
            xy_units=units.Unit('kpc'), vel_units = units.Unit('km s^-1'),
            smooth='smooth', __threaded=False) :

    """

    Render an SPH spectrum using a (mass/rho)-weighted 'scatter'
    scheme of all the particles that have a smoothing length within
    2 h_sm of the position.

    **Keyword arguments:**

    *qty* ('rho'): The name of the array within the simulation to render

    *x1* (0.0): The x-coordinate of the line of sight.

    *y1* (0.0): The y-coordinate of the line of sight.

    *v1* (-400.0): The minimum velocity of the spectrum

    *v2* (400.0): The maximum velocity of the spectrum

    *nvel* (500): The number of resolution elements in spectrum

    *xy_units* ('kpc'): The units for the x and y axes

    *smooth*: The name of the array which contains the smoothing lengths
      (default 'smooth')

    """

    import os, os.path
    global config
    
    if config["tracktime"] :
        import time
        in_time = time.time()
    
    kernel=Kernel2D()
            
    if v1 is None:
        v1 = -v2
    dvel = (v2 - v1) / nvel
    v1, v2, dvel, nvel = [float(q) for q in v1,v2,dvel,nvel]
    vels = np.arange(v1+0.5*dvel, v2, dvel)

    tau = np.zeros((nvel),dtype=np.float32)
    
    n_part = len(snap)

    if xy_units is None :
        xy_units = snap['x'].units

    x = snap['x'].in_units(xy_units) - x1
    y = snap['y'].in_units(xy_units) - y1
    vz = snap['vz'].in_units(vel_units)
    temp = snap['temp'].in_units(units.Unit('K'))

    sm = snap[smooth]

    if sm.units!=x.units :
        sm = sm.in_units(x.units)

    nucleons = {'H':1, 'He':4, 'Li':6, 'C':12, 'N':14, 'O':16, 'Mg':24, 'Si':28,
                'S':32, 'Ca':40, 'Fe':56}
    nnucleons = nucleons[element]
    
    qty_s = qty
    qty = snap[qty]
    mass = snap['mass']
    rho = snap['rho']

    conv_ratio = (qty.units*mass.units/(rho.units*sm.units**kernel.h_power)).ratio(str(nnucleons)+' m_p cm^-2', **x.conversion_context())

    try :
        kernel.safe.acquire(True)
        code = kernel.get_c_code()
    finally :
        kernel.safe.release()

    if __threaded :
        code+="#define THREAD 1\n"
    
        
    code+=file(os.path.join(os.path.dirname(__file__),'sph_spectra.c')).read()

    # before inlining, the views on the arrays must be standard np.ndarray
    # otherwise the normal numpy macros are not generated
    x,y,vz,temp,sm,qty, mass, rho = [q.view(np.ndarray) for q in x,y,vz,temp,sm,qty, mass, rho]

    if config['verbose']: print>>sys.stderr, "Constructing SPH spectrum"

    if config["tracktime"] :
        print>>sys.stderr, "Beginning SPH render at %.2f s"%(time.time()-in_time)
    #import pdb; pdb.set_trace()
    util.threadsafe_inline( code, ['tau', 'nvel', 'x', 'y', 'vz', 'temp', 'sm', 'v1', 'v2',
                   'nnucleons','qty', 'mass', 'rho'],verbose=2)

    if config["tracktime"] :
        print>>sys.stderr, "Render done at %.2f s"%(time.time()-in_time)

    mass_e = 9.10938188e-28
    e = 4.803206e-10
    c = 2.99792458e10
    pi = 3.14159267
    tauconst = pi*e*e / mass_e / c / np.sqrt(pi)
    oscwav0 = 1031.9261*0.13250*1e-8
    tau = tauconst*oscwav0*tau*conv_ratio
    #tau = tau*conv_ratio
    print "tauconst: %g oscwav0: %g"%(tauconst,oscwav0)
    print "tauconst*oscwav0: %g"%(tauconst*oscwav0)
    print "conv_ratio: %g"%conv_ratio
    print "max(N): %g"%(np.max(tau))
    tau = tau.view(array.SimArray)

    tau.sim = snap
    return vels, tau
