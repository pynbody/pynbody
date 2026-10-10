import numpy as np

cimport cython
cimport libc.math as cmath
cimport numpy as np

np.import_array()
from libc.math cimport acos, atan, cos, fabs, floor, pow, sqrt
from libc.stdlib cimport free, malloc


cdef double PI = 3.14159265358979323846

# The following slightly odd repetitiveness is to force Cython to generate
# code for different permutations of the possible integer inputs.
#
# Using just one fused type requires the types to be consistent across all
# arguments.

ctypedef fused fused_input_type_1:
    np.float32_t
    np.float64_t

ctypedef fused fused_input_type_2:
    np.float32_t
    np.float64_t

ctypedef fused fused_input_type_3:
    np.float32_t
    np.float64_t

ctypedef fused fused_input_type_4:
    np.float32_t
    np.float64_t

ctypedef fused fused_input_type_5:
    np.float32_t
    np.float64_t


ctypedef np.float32_t image_output_type
np_image_output_type = np.float32

ctypedef np.float64_t fixed_input_type

cdef extern size_t query_disc_c(size_t nside, double* vec0, double radius, size_t *listpix, double *listdist) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def render_spherical_image_core(np.ndarray[fused_input_type_1, ndim=1] rho, # array of particle densities
                                np.ndarray[fused_input_type_2, ndim=1] mass, # array of particle masses
                                np.ndarray[fused_input_type_3, ndim=1] qtyar, # array of quantity to make image of
                                np.ndarray[fused_input_type_4, ndim=1] x, # arrays of positions
                                np.ndarray[fused_input_type_4, ndim=1] y,
                                np.ndarray[fused_input_type_4, ndim=1] z,
                                np.ndarray[fused_input_type_5, ndim=1] h, # particle smoothing length
                                unsigned int nside,
                                kernel,
                                fixed_input_type shell_radius=0.0) :
    """Render an SPH quantity onto a healpix sphere around the origin.

    Two modes are supported, selected by the dimension of the ``kernel``:

    * A projected (column) image, obtained by passing a 2D (projected) kernel
      (``kernel.h_power == 2``). Every particle whose smoothing sphere crosses
      the line of sight contributes its full column, giving a quantity per unit
      solid angle. ``shell_radius`` is ignored in this mode.

    * A thin-shell image, obtained by passing a 3D kernel (``kernel.h_power == 3``)
      together with a positive ``shell_radius``. The 3D kernel is evaluated on the
      sphere of radius ``shell_radius``, i.e. the field is sampled where each
      particle's kernel intersects that shell. This is the spherical analogue of
      the thin-slice mode of :func:`render_image`.
    """

    cdef np.ndarray[image_output_type,ndim=1] im

    if nside & (nside - 1) != 0:
        raise ValueError('nside value must be a power of 2')

    cdef int kernel_dim = kernel.h_power
    if kernel_dim != 2 and kernel_dim != 3:
        raise ValueError('Only kernels of dimension 2 (projected) or 3 (thin shell) '
                         'are supported for healpix maps')

    # projected -> column image; otherwise a thin shell at shell_radius
    cdef int projected = (kernel_dim == 2)

    if not projected and shell_radius <= 0.0:
        raise ValueError('A positive shell_radius is required to render a 3D (thin shell) healpix map')

    cdef fixed_input_type max_d_over_h = kernel.max_d
    cdef fixed_input_type max_d_over_h_2 = max_d_over_h * max_d_over_h

    cdef np.ndarray[image_output_type, ndim=1] samples = kernel.get_samples(dtype=np_image_output_type)
    cdef int num_samples = len(samples)
    cdef image_output_type * samples_c = <image_output_type *> samples.data

    cdef size_t npix = 12 * nside * nside

    im = np.zeros(npix, dtype=np_image_output_type)

    # these numpy arrays are being created just for temporary memory management and will be discarded
    cdef np.ndarray[size_t, ndim=1] index = np.empty(npix, dtype=np.uintp)
    cdef np.ndarray[double, ndim=1] angle = np.empty(npix, dtype=np.float64)

    cdef size_t* index_buffer = <size_t*>index.data
    cdef double* angle_buffer = <double*>angle.data
    cdef size_t num_pixels
    cdef double[3] pos_i
    cdef double angular_size
    cdef double distance, distance2
    cdef double smooth_2
    cdef double kernel_max_2
    cdef double physical_offset
    cdef double cos_alpha              # thin shell: cosine of the intersection half-angle
    cdef double shell_2 = shell_radius * shell_radius
    cdef double d2                     # thin shell: squared 3D distance particle->shell point
    cdef double two_rs_d               # thin shell: 2 * shell_radius * distance
    cdef fused_input_type_3 qty_i

    # per-particle value of h to the power of the kernel dimension, which get_kernel
    # divides the (h=1) samples by. For the projected image this additionally carries
    # a distance^2 Jacobian to express the column per unit solid angle rather than per
    # unit transverse area.
    cdef image_output_type h_to_kdim

    cdef image_output_type kern

    cdef size_t n_part = len(x)

    with nogil:
        for i in range(n_part):
            qty_i = qtyar[i]
            if qty_i != qty_i:
                continue
            pos_i[0] = x[i]
            pos_i[1] = y[i]
            pos_i[2] = z[i]
            distance2 = pos_i[0]*pos_i[0] + pos_i[1]*pos_i[1] + pos_i[2]*pos_i[2]
            distance = sqrt(distance2)
            smooth_2 = h[i]*h[i]
            kernel_max_2 = smooth_2*max_d_over_h_2

            if projected:
                angular_size = max_d_over_h * h[i] / distance
                h_to_kdim = smooth_2 / distance2
            else:
                # The kernel support (sphere of radius sqrt(kernel_max_2) about the
                # particle) meets the shell in a circle. Its angular radius about the
                # origin follows from the law of cosines for the triangle
                # origin-particle-shellpoint:
                #     cos(alpha) = (shell^2 + distance^2 - kernel_max_2) / (2 shell distance)
                two_rs_d = 2.0 * shell_radius * distance
                cos_alpha = (shell_2 + distance2 - kernel_max_2) / two_rs_d
                if cos_alpha >= 1.0:
                    # closest approach still outside the kernel: no intersection
                    continue
                elif cos_alpha <= -1.0:
                    angular_size = PI
                else:
                    angular_size = acos(cos_alpha)
                h_to_kdim = smooth_2 * h[i]

            num_pixels = query_disc_c(nside, pos_i, angular_size, index_buffer, angle_buffer)

            for j in range(num_pixels):
                if projected:
                    physical_offset = distance * angle_buffer[j]
                    d2 = physical_offset*physical_offset
                else:
                    # 3D separation between the particle and the shell point in this
                    # direction, again by the law of cosines
                    d2 = shell_2 + distance2 - two_rs_d * cos(angle_buffer[j])
                kern = get_kernel(d2, kernel_max_2, h_to_kdim, num_samples, samples_c)
                im[index_buffer[j]] += qty_i * kern * mass[i] / rho[i]

    return im



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef image_output_type get_kernel(fixed_input_type d2, fixed_input_type kernel_max_2,
                                  image_output_type h_to_the_kdim, int num_samples,
                                  image_output_type* kvals) nogil :
    cdef unsigned int index = <unsigned int>(num_samples*(d2/kernel_max_2))
    if index<num_samples :
        return kvals[index]/h_to_the_kdim
    else :
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef image_output_type get_kernel_xyz(fixed_input_type x, fixed_input_type y, fixed_input_type z, fixed_input_type kernel_max_2,
                                       image_output_type h_to_the_kdim, int num_samples,
                                 image_output_type* kvals) nogil :
     return get_kernel(x*x+y*y+z*z,kernel_max_2,h_to_the_kdim,num_samples,kvals)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def render_image(int nx, int ny,
                 np.ndarray[fused_input_type_1,ndim=1] x,
                 np.ndarray[fused_input_type_1,ndim=1] y,
                 np.ndarray[fused_input_type_1,ndim=1] z,
                 np.ndarray[fused_input_type_2,ndim=1] sm,
                 fixed_input_type x1,fixed_input_type x2,fixed_input_type y1,
                 fixed_input_type y2,fixed_input_type z_camera, fixed_input_type z0,
                 np.ndarray[fused_input_type_3,ndim=1] qty,
                 np.ndarray[fused_input_type_4,ndim=1] mass,
                 np.ndarray[fused_input_type_5,ndim=1] rho,
                 fixed_input_type smooth_lo, fixed_input_type smooth_hi,
                 fixed_input_type z_lo, fixed_input_type z_hi,
                 fixed_input_type min_smooth,
                 kernel,
                 wrap_offsets_x=[0], wrap_offsets_y=[0]) :

    cdef fixed_input_type pixel_dx = (x2-x1)/nx
    cdef fixed_input_type pixel_dy = (y2-y1)/ny
    cdef fixed_input_type x_start = x1+pixel_dx/2
    cdef fixed_input_type y_start = y1+pixel_dy/2
    cdef int n_part = len(x)
    cdef int nn=0, i=0
    cdef fixed_input_type x_i, y_i, z_i, sm_i, qty_i
    cdef fixed_input_type x_pixel, y_pixel, z_pixel
    cdef int x_pos, y_pos
    cdef int x_pix_start, x_pix_stop, y_pix_start, y_pix_stop

    # following are only used for "perspective" rendering
    cdef float per_z_dx = (x2-x1)/(2*z_camera)
    cdef float per_z_dy = (y2-y1)/(2*z_camera)
    cdef float mid_x = (x2+x1)/2
    cdef float mid_y = (y2+y1)/2
    cdef float dz_i

    cdef float wrap_offset_x, wrap_offset_y


    cdef int kernel_dim = kernel.h_power
    cdef fixed_input_type max_d_over_h = kernel.max_d


    cdef np.ndarray[image_output_type,ndim=1] samples = kernel.get_samples(dtype=np_image_output_type)
    cdef int num_samples = len(samples)
    cdef image_output_type* samples_c = <image_output_type*>samples.data
    cdef image_output_type sm_to_kdim   # minimize casting when same type as output

    cdef fixed_input_type kernel_max_2 # minimize casting when same type as input

    cdef np.ndarray[image_output_type,ndim=2] result = np.zeros((ny,nx),dtype=np_image_output_type)

    z_pixel = z0
    cdef int total_ptcls = 0

    cdef int use_z = 1 if kernel_dim>=3 else 0

    assert kernel_dim==2 or kernel_dim==3, "Only kernels of dimension 2 or 3 currently supported"
    assert len(x) == len(y) == len(z) == len(sm) == len(qty) == len(mass) == len(rho), "Inconsistent array lengths passed to render_image_core"

    for wrap_offset_x in wrap_offsets_x :
        for wrap_offset_y in wrap_offsets_y :
            with nogil:
                for i in range(n_part) :
                    # load particle details
                    x_i = x[i]+wrap_offset_x; y_i=y[i]+wrap_offset_y;
                    z_i=z[i]; sm_i = sm[i]; qty_i = qty[i]*mass[i]/rho[i]

                    if z_i<z_lo or z_i>z_hi :
                        continue

                    if qty_i!=qty_i:
                        continue

                    if z_camera!=0.0 :
                        # perspective image -
                        # update image bounds for the current z
                        if (z_i>z_camera and z_camera>0) or (z_i<z_camera and z_camera<0) :
                            # behind camera
                            continue
                        dz_i = z_camera-z_i
                        x1 = mid_x - per_z_dx*dz_i
                        x2 = mid_x + per_z_dx*dz_i
                        y1 = mid_y - per_z_dy*dz_i
                        y2 = mid_y + per_z_dy*dz_i
                        pixel_dx = (x2-x1)/nx
                        pixel_dy = (y2-y1)/ny
                        x_start = x1+pixel_dx/2
                        y_start = y1+pixel_dy/2

                    # minimum smoothing can be specified to create a smoother image (esp useful for contour plots)
                    if sm_i<min_smooth:
                        sm_i = min_smooth

                    # check particle smoothing is within specified range
                    if sm_i<pixel_dx*smooth_lo or sm_i>pixel_dx*smooth_hi : continue


                    total_ptcls+=1

                    # check particle is within bounds
                    if not ((use_z*cmath.fabs(z_i-z0)<max_d_over_h*sm_i)
                            and x_i>x1-2*sm_i and x_i<x2+2*sm_i and y_i>y1-2*sm_i and y_i<y2+2*sm_i) :
                        continue

                    # pre-cache sm^kdim and (sm*max_d_over_h)**2; tests showed massive speedups when doing this
                    if kernel_dim==2 :
                        sm_to_kdim = sm_i*sm_i
                    else :
                        sm_to_kdim = sm_i*sm_i*sm_i
                        # only 2, 3 supported

                    kernel_max_2 = (sm_i*sm_i)*(max_d_over_h*max_d_over_h)

                    # decide whether this is a single pixel or a multi-pixel particle
                    if (max_d_over_h*sm_i/pixel_dx<1 and max_d_over_h*sm_i/pixel_dy<1) :
                        # single pixel, get pixel location
                        x_pos = int((x_i-x1)/pixel_dx)
                        y_pos = int((y_i-y1)/pixel_dy)

                        # work out pixel centre
                        x_pixel = (pixel_dx*<fixed_input_type>(x_pos)+x_start)
                        y_pixel = (pixel_dy*<fixed_input_type>(y_pos)+y_start)

                        # final bounds check
                        if x_pos>=0 and x_pos<nx and y_pos>=0 and y_pos<ny :
                            result[y_pos,x_pos]+=qty_i*get_kernel_xyz(x_i-x_pixel, y_i-y_pixel, (z_i-z_pixel)*use_z, kernel_max_2 ,sm_to_kdim,num_samples,samples_c)
                    else :
                        # multi-pixel
                        x_pix_start = int((x_i-max_d_over_h*sm_i-x1)/pixel_dx)
                        x_pix_stop =  int((x_i+max_d_over_h*sm_i-x1)/pixel_dx)
                        y_pix_start = int((y_i-max_d_over_h*sm_i-y1)/pixel_dy)
                        y_pix_stop =  int((y_i+max_d_over_h*sm_i-y1)/pixel_dy)
                        if x_pix_start<0 : x_pix_start = 0
                        if x_pix_stop>nx : x_pix_stop = nx
                        if y_pix_start<0 : y_pix_start = 0
                        if y_pix_stop>ny : y_pix_stop = ny
                        for x_pos in range(x_pix_start, x_pix_stop) :
                            x_pixel = pixel_dx*<fixed_input_type>(x_pos)+x_start
                            for y_pos in range(y_pix_start, y_pix_stop) :
                                y_pixel = pixel_dy*<fixed_input_type>(y_pos)+y_start

                                # could accessing the buffer manually be
                                # faster? It seems to be FAR faster (x10!) but
                                # only when using stack-allocated memory for
                                # c_result, and when writing to memory, not
                                # also reading (i.e. = instead of +=).  The
                                # instruction that, according to Instruments,
                                # holds everything up and disappears is
                                # cvtss2sd, but it's not clear to me why this
                                # disappears from the compiled version in the
                                # instance described above.  Anyway for now, there
                                # is no advantage to the manual approach -

                                #c_result[x_pos+nx*y_pos]+=qty_i*get_kernel_xyz(x_i-x_pixel, y_i-y_pixel, (z_i-z_pixel)*use_z, kernel_max_2 ,sm_to_kdim,num_samples,samples_c)

                                result[y_pos,x_pos]+=qty_i*get_kernel_xyz(x_i-x_pixel, y_i-y_pixel, (z_i-z_pixel)*use_z, kernel_max_2 ,sm_to_kdim,num_samples,samples_c)

    return result




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def to_3d_grid(int nx, int ny, int nz,
                 np.ndarray[fused_input_type_1,ndim=1] x,
                 np.ndarray[fused_input_type_1,ndim=1] y,
                 np.ndarray[fused_input_type_1,ndim=1] z,
                 np.ndarray[fused_input_type_2,ndim=1] sm,
                 fixed_input_type x1,fixed_input_type x2,fixed_input_type y1,
                 fixed_input_type y2,fixed_input_type z1, fixed_input_type z2,
                 np.ndarray[fused_input_type_3,ndim=1] qty,
                 np.ndarray[fused_input_type_4,ndim=1] mass,
                 np.ndarray[fused_input_type_5,ndim=1] rho,
                 fixed_input_type smooth_lo, fixed_input_type smooth_hi,
                 kernel,
                 wrap_offsets_x=[0], wrap_offsets_y=[0],wrap_offsets_z=[0]) :



    cdef fixed_input_type pixel_dx = (x2-x1)/nx
    cdef fixed_input_type pixel_dy = (y2-y1)/ny
    cdef fixed_input_type pixel_dz = (z2-z1)/nz
    cdef fixed_input_type x_start = x1+pixel_dx/2
    cdef fixed_input_type y_start = y1+pixel_dy/2
    cdef fixed_input_type z_start = z1+pixel_dz/2
    cdef int n_part = len(x)
    cdef int nn=0, i=0
    cdef fixed_input_type x_i, y_i, z_i, sm_i, qty_i
    cdef fixed_input_type x_pixel, y_pixel, z_pixel
    cdef int x_pos, y_pos, z_pos
    cdef int x_pix_start, x_pix_stop, y_pix_start, y_pix_stop, z_pix_start, z_pix_stop

    cdef int kernel_dim = kernel.h_power
    cdef fixed_input_type max_d_over_h = kernel.max_d

    cdef np.ndarray[image_output_type,ndim=1] samples = kernel.get_samples(dtype=np_image_output_type)
    cdef int num_samples = len(samples)
    cdef image_output_type* samples_c = <image_output_type*>samples.data
    cdef image_output_type sm_to_kdim   # minimize casting when same type as output

    cdef fixed_input_type kernel_max_2 # minimize casting when same type as input

    cdef np.ndarray[image_output_type,ndim=3] result = np.zeros((nx,ny,nz),dtype=np_image_output_type)

    cdef int total_ptcls = 0

    cdef int use_z = 1 if kernel_dim>=3 else 0

    cdef float wrap_offset_x, wrap_offset_y, wrap_offset_z

    if kernel_dim<3:
        raise ValueError, \
          "Cannot render to 3D grid without 3-dimensional kernel or greater"

    assert len(x) == len(y) == len(z) == len(sm) == \
            len(qty) == len(mass) == len(rho), \
            "Inconsistent array lengths passed to render_image_core"

    for wrap_offset_x in wrap_offsets_x :
        for wrap_offset_y in wrap_offsets_y :
            for wrap_offset_z in wrap_offsets_z :
                with nogil:
                    for i in range(n_part) :
                        # load particle details
                        x_i = x[i]+wrap_offset_x; y_i=y[i]+wrap_offset_y; z_i=z[i]+wrap_offset_z
                        sm_i = sm[i]
                        qty_i = qty[i]*mass[i]/rho[i]

                        # check particle smoothing is within specified range
                        if sm_i<pixel_dx*smooth_lo or sm_i>pixel_dx*smooth_hi : continue

                        total_ptcls+=1

                        # check particle is within bounds
                        if not (z_i>z1-2*sm_i and z_i<z2+2*sm_i \
                                and x_i>x1-2*sm_i and x_i<x2+2*sm_i \
                                and y_i>y1-2*sm_i and y_i<y2+2*sm_i) :
                            continue

                        # pre-cache sm^kdim and (sm*max_d_over_h)**2; tests showed massive speedups when doing this
                        if kernel_dim==2 :
                            sm_to_kdim = sm_i*sm_i
                        else :
                            sm_to_kdim = sm_i*sm_i*sm_i
                            # only 2, 3 supported

                        kernel_max_2 = (sm_i*sm_i)*(max_d_over_h*max_d_over_h)

                        # decide whether this is a single pixel or a multi-pixel particle
                        if (max_d_over_h*sm_i/pixel_dx<1 and max_d_over_h*sm_i/pixel_dy<1) :
                            # single pixel, get pixel location
                            x_pos = int((x_i-x1)/pixel_dx)
                            y_pos = int((y_i-y1)/pixel_dy)
                            z_pos = int((z_i-z1)/pixel_dz)

                            # work out pixel centre
                            x_pixel = (pixel_dx*<fixed_input_type>(x_pos)+x_start)
                            y_pixel = (pixel_dy*<fixed_input_type>(y_pos)+y_start)
                            z_pixel = (pixel_dz*<fixed_input_type>(z_pos)+z_start)

                            # final bounds check
                            if x_pos>=0 and x_pos<nx and y_pos>=0 and y_pos<ny \
                               and z_pos>=0 and z_pos<nz :
                                result[x_pos,y_pos,z_pos]+=qty_i*get_kernel_xyz(x_i-x_pixel, y_i-y_pixel, (z_i-z_pixel)*use_z, kernel_max_2 ,sm_to_kdim,num_samples,samples_c)
                        else :
                            # multi-pixel
                            x_pix_start = int((x_i-max_d_over_h*sm_i-x1)/pixel_dx)
                            x_pix_stop =  int((x_i+max_d_over_h*sm_i-x1)/pixel_dx)
                            y_pix_start = int((y_i-max_d_over_h*sm_i-y1)/pixel_dy)
                            y_pix_stop =  int((y_i+max_d_over_h*sm_i-y1)/pixel_dy)
                            z_pix_start = int((z_i-max_d_over_h*sm_i-z1)/pixel_dz)
                            z_pix_stop =  int((z_i+max_d_over_h*sm_i-z1)/pixel_dz)
                            if x_pix_start<0 : x_pix_start = 0
                            if x_pix_stop>nx : x_pix_stop = nx
                            if y_pix_start<0 : y_pix_start = 0
                            if y_pix_stop>ny : y_pix_stop = ny
                            if z_pix_start<0 : z_pix_start = 0
                            if z_pix_stop>nz : z_pix_stop = nz
                            for x_pos in range(x_pix_start, x_pix_stop) :
                                x_pixel = pixel_dx*<fixed_input_type>(x_pos)+x_start
                                for y_pos in range(y_pix_start, y_pix_stop) :
                                    y_pixel = pixel_dy*<fixed_input_type>(y_pos)+y_start

                                    for z_pos in range(z_pix_start,z_pix_stop) :
                                        z_pixel = pixel_dz*<fixed_input_type>(z_pos)+z_start
                                        result[x_pos,y_pos,z_pos]+=qty_i*get_kernel_xyz(x_i-x_pixel, y_i-y_pixel, (z_i-z_pixel), kernel_max_2 ,sm_to_kdim,num_samples,samples_c)

    return result



# ---------------------------------------------------------------------------------------------------------------
# Rendering of AMR cells
#
# Here each 'particle' is a cubic cell of side sm, oriented along axes which are the columns of a rotation matrix
# (expressed in the image frame). Rather than a symmetric kernel, the cell contributes its exact (anisotropic)
# cross-section in a slice, or its exact chord length through the cell in a projection, averaged over each pixel.
# Because the cells tile space, the resulting image is then an exact rendering of the piecewise-constant field.
# ---------------------------------------------------------------------------------------------------------------

cdef enum:
    MAX_POLY = 16  # maximum number of vertices of a clipped polygon (a pixel clipped by a cube slice has <= 10)

cdef inline double _overlap(double a0, double a1, double b0, double b1) noexcept nogil:
    """Length of the overlap between intervals [a0, a1] and [b0, b1]"""
    cdef double lo = a0 if a0 > b0 else b0
    cdef double hi = a1 if a1 < b1 else b1
    return hi - lo if hi > lo else 0.0

@cython.cdivision(True)
cdef inline double _cell_chord(double dx, double dy, const double *axes, double half,
                               double t_lo, double t_hi, int *label) noexcept nogil:
    """Length of the chord along the z direction through a rotated cube, at offset (dx, dy) from its centre.

    The cube has half-width *half* and its kth axis is (axes[3k], axes[3k+1], axes[3k+2]). The chord is additionally
    restricted to the range of z offsets [t_lo, t_hi]. On exit, *label* identifies which pair of faces the ray enters
    and exits through (or -1 if the ray misses). Within a region of constant label, the chord length is a linear
    function of (dx, dy); and such regions are convex.
    """
    cdef int k, entry = 6, exit = 7  # 6 and 7 label the z-range limits
    cdef double s, ez, a, b
    for k in range(3):
        s = axes[3*k] * dx + axes[3*k+1] * dy
        ez = axes[3*k+2]
        if ez < 1e-12 and ez > -1e-12:
            # this axis lies in the image plane; it constrains (dx, dy) but not the chord
            if s > half or s < -half:
                label[0] = -1
                return 0.0
            continue
        if ez > 0:
            a = (-half - s) / ez
            b = (half - s) / ez
            if a > t_lo:
                t_lo = a
                entry = 2*k
            if b < t_hi:
                t_hi = b
                exit = 2*k + 1
        else:
            a = (half - s) / ez
            b = (-half - s) / ez
            if a > t_lo:
                t_lo = a
                entry = 2*k + 1
            if b < t_hi:
                t_hi = b
                exit = 2*k
    if t_hi <= t_lo:
        label[0] = -1
        return 0.0
    label[0] = entry * 8 + exit
    return t_hi - t_lo

@cython.cdivision(True)
cdef int _clip_polygon(const double *px, const double *py, int n, double a, double b, double c,
                       double *ox, double *oy) noexcept nogil:
    """Clip the convex polygon (px, py) of n vertices to the half-plane a x + b y <= c (Sutherland-Hodgman).

    The result is written into (ox, oy), and its number of vertices is returned."""
    cdef int i, m = 0
    cdef double d0, d1, t
    if n == 0:
        return 0
    d0 = a * px[n-1] + b * py[n-1] - c
    for i in range(n):
        d1 = a * px[i] + b * py[i] - c
        if (d0 <= 0) != (d1 <= 0):
            # edge crosses the line; add the intersection point
            t = d0 / (d0 - d1)
            if m < MAX_POLY:
                ox[m] = px[(i + n - 1) % n] + t * (px[i] - px[(i + n - 1) % n])
                oy[m] = py[(i + n - 1) % n] + t * (py[i] - py[(i + n - 1) % n])
                m += 1
        if d1 <= 0 and m < MAX_POLY:
            ox[m] = px[i]
            oy[m] = py[i]
            m += 1
        d0 = d1
    return m

cdef inline double _polygon_area(const double *px, const double *py, int n) noexcept nogil:
    cdef int i
    cdef double area = 0.0
    for i in range(n):
        area += px[i] * py[(i + 1) % n] - px[(i + 1) % n] * py[i]
    return 0.5 * area if area > 0 else -0.5 * area

cdef double _clipped_rectangle_area(double x0, double x1, double y0, double y1,
                                    const double *hp_a, const double *hp_b, const double *hp_c,
                                    int n_hp) noexcept nogil:
    """Area of the rectangle [x0, x1] x [y0, y1] intersected with n_hp half-planes a x + b y <= c"""
    cdef double buf_x[2][MAX_POLY]
    cdef double buf_y[2][MAX_POLY]
    cdef int n = 4, k, cur = 0
    buf_x[0][0] = x0; buf_y[0][0] = y0
    buf_x[0][1] = x1; buf_y[0][1] = y0
    buf_x[0][2] = x1; buf_y[0][2] = y1
    buf_x[0][3] = x0; buf_y[0][3] = y1
    for k in range(n_hp):
        n = _clip_polygon(buf_x[cur], buf_y[cur], n, hp_a[k], hp_b[k], hp_c[k], buf_x[1-cur], buf_y[1-cur])
        cur = 1 - cur
        if n < 3:
            return 0.0
    return _polygon_area(buf_x[cur], buf_y[cur], n)

cdef inline int _inside_all(double x, double y, const double *hp_a, const double *hp_b, const double *hp_c,
                            int n_hp) noexcept nogil:
    cdef int k
    for k in range(n_hp):
        if hp_a[k] * x + hp_b[k] * y > hp_c[k]:
            return 0
    return 1

cdef int _convex_hull(double *px, double *py, int n, double *hx, double *hy) noexcept nogil:
    """Convex hull (anticlockwise) of a small number of points by Andrew's monotone chain. Returns vertex count."""
    cdef int i, j, k = 0, lower_size
    cdef double tx, ty
    # insertion sort by x then y
    for i in range(1, n):
        tx = px[i]; ty = py[i]
        j = i - 1
        while j >= 0 and (px[j] > tx or (px[j] == tx and py[j] > ty)):
            px[j+1] = px[j]; py[j+1] = py[j]
            j -= 1
        px[j+1] = tx; py[j+1] = ty
    for i in range(n):
        while k >= 2 and ((hx[k-1]-hx[k-2])*(py[i]-hy[k-2]) - (hy[k-1]-hy[k-2])*(px[i]-hx[k-2])) <= 0:
            k -= 1
        hx[k] = px[i]; hy[k] = py[i]; k += 1
    lower_size = k + 1
    for i in range(n - 2, -1, -1):
        while k >= lower_size and ((hx[k-1]-hx[k-2])*(py[i]-hy[k-2]) - (hy[k-1]-hy[k-2])*(px[i]-hx[k-2])) <= 0:
            k -= 1
        hx[k] = px[i]; hy[k] = py[i]; k += 1
    return k - 1

cdef inline int _rectangle_outside_hull(double x0, double x1, double y0, double y1,
                                        const double *hx, const double *hy, int nh) noexcept nogil:
    """True if some edge of the anticlockwise convex hull separates it from the rectangle"""
    cdef int i, i1
    cdef double nx_, ny_
    for i in range(nh):
        i1 = (i + 1) % nh
        nx_ = hy[i1] - hy[i]
        ny_ = hx[i] - hx[i1]
        if (nx_ * (x0 - hx[i]) + ny_ * (y0 - hy[i]) > 0 and nx_ * (x1 - hx[i]) + ny_ * (y0 - hy[i]) > 0 and
            nx_ * (x0 - hx[i]) + ny_ * (y1 - hy[i]) > 0 and nx_ * (x1 - hx[i]) + ny_ * (y1 - hy[i]) > 0):
            return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def render_image_cells(int nx, int ny,
                       np.ndarray[fused_input_type_1,ndim=1] x,
                       np.ndarray[fused_input_type_1,ndim=1] y,
                       np.ndarray[fused_input_type_1,ndim=1] z,
                       np.ndarray[fused_input_type_2,ndim=1] sm,
                       fixed_input_type x1, fixed_input_type x2, fixed_input_type y1, fixed_input_type y2,
                       fixed_input_type z0,
                       np.ndarray[fused_input_type_3,ndim=1] qty,
                       np.ndarray[fused_input_type_4,ndim=1] mass,
                       np.ndarray[fused_input_type_5,ndim=1] rho,
                       fixed_input_type smooth_lo, fixed_input_type smooth_hi,
                       fixed_input_type z_lo, fixed_input_type z_hi,
                       fixed_input_type min_smooth,
                       bint projected,
                       orientation,
                       int subsamples = 4,
                       wrap_offsets_x=[0], wrap_offsets_y=[0]):
    """Render cubic (AMR) cells onto an image, either as an exact slice at z=z0 or an exact projection along z.

    Each cell has side length sm and the edges of all cells are parallel to the columns of the 3x3 rotation
    matrix *orientation*, expressed in the image frame. The value in each pixel is the average over the pixel
    area of the slice through (or projection of) the piecewise-constant field qty, with mass/rho giving the cell
    volume. In the projected case, the z range of the projection is restricted to [z_lo, z_hi], with cells being
    exactly truncated at those limits. A slice lying exactly along cell faces samples the cells on its +z side.

    If the cells are aligned with the image axes, the pixel averages are computed analytically. For a slice
    through rotated cells, the exact area of overlap between each pixel and each cell's (polygonal) cross-section
    is calculated. For a projection through rotated cells, the chord length is linear over most of the projected
    area of the cell, and in such regions the exact pixel average is again computed. However pixels straddling
    the projection of a cell edge are supersampled with subsamples x subsamples rays. Note that the supersampling
    grid is common to all cells, so the result remains an exact rendering of the field along each of the rays.

    Perspective (z_camera) rendering is not currently supported.
    """

    cdef double pixel_dx = (x2 - x1) / nx
    cdef double pixel_dy = (y2 - y1) / ny
    cdef double pixel_area = pixel_dx * pixel_dy
    cdef int n_part = len(x)
    cdef int i, k, ix, iy, ix0, ix1, iy0, iy1, isub, jsub, n_hp, nh, nvert
    cdef int label, label_00, label_10, label_01, label_11, uniform
    cdef double x_i, y_i, z_i, sm_i, half, weight, tz, value, chord_sum, sx, sy, Lz
    cdef double bx0, bx1, by0, by1, bz, X0, X1, Y0, Y1
    cdef double ex, ey, ez, d_cut
    cdef double z_lo_d = z_lo, z_hi_d = z_hi
    # A slice exactly along cell faces would, after round-off, be inconsistently assigned to the cells either side
    # (double-counting or missing them). Nudging the plane by much more than round-off, but much less than any
    # plausible cell size, consistently selects the cells on the +z side.
    cdef double z0_d = z0 + 1e-10 * (fabs(z0) + (x2 - x1))
    cdef double hp_a[6]
    cdef double hp_b[6]
    cdef double hp_c[6]
    cdef double vx[MAX_POLY]
    cdef double vy[MAX_POLY]
    cdef double hx[MAX_POLY]
    cdef double hy[MAX_POLY]
    cdef double axes[9]
    cdef double sub_step_x = pixel_dx / subsamples, sub_step_y = pixel_dy / subsamples
    cdef double inv_nsub2 = 1.0 / (subsamples * subsamples)
    cdef int aligned
    cdef double wrap_offset_x, wrap_offset_y

    cdef np.ndarray[image_output_type, ndim=2] result = np.zeros((ny, nx), dtype=np_image_output_type)
    cdef image_output_type *res = <image_output_type *> result.data

    # labels for a row of pixel corners; only used for projections of rotated cells
    cdef np.ndarray[np.int32_t, ndim=1] corner_labels_np = np.empty(2 * (nx + 1), dtype=np.int32)
    cdef np.int32_t *corner_labels = <np.int32_t *> corner_labels_np.data
    cdef np.int32_t *lower_row
    cdef np.int32_t *upper_row
    cdef np.int32_t *swap_row

    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.shape != (3, 3):
        raise ValueError("orientation must be a 3x3 matrix")
    if subsamples < 1:
        raise ValueError("subsamples must be at least 1")
    assert len(x) == len(y) == len(z) == len(sm) == len(qty) == len(mass) == len(rho), \
        "Inconsistent array lengths passed to render_image_cells"

    for k in range(3):
        for i in range(3):
            # axes[3k + i] is component i (in the image frame) of cell axis k
            axes[3*k + i] = orientation[i, k]

    # cells are aligned with the image if the orientation is a signed permutation matrix
    aligned = bool(np.all(np.sum(np.abs(orientation) > 1e-10, axis=0) == 1))

    for wrap_offset_x in wrap_offsets_x:
        for wrap_offset_y in wrap_offsets_y:
            with nogil:
                for i in range(n_part):
                    x_i = x[i] + wrap_offset_x
                    y_i = y[i] + wrap_offset_y
                    z_i = z[i]
                    sm_i = sm[i]
                    weight = qty[i] * mass[i] / rho[i]
                    if weight != weight:
                        continue

                    if sm_i < min_smooth:
                        sm_i = min_smooth
                    if sm_i < pixel_dx * smooth_lo or sm_i > pixel_dx * smooth_hi:
                        continue

                    half = 0.5 * sm_i
                    # normalise by the cell volume, so that the result for a slice is qty and for a projection is
                    # the integral of qty along the line of sight
                    weight /= sm_i * sm_i * sm_i

                    # extent of the cell in the image frame
                    bx0 = half * (fabs(axes[0]) + fabs(axes[3]) + fabs(axes[6]))
                    by0 = half * (fabs(axes[1]) + fabs(axes[4]) + fabs(axes[7]))
                    bz = half * (fabs(axes[2]) + fabs(axes[5]) + fabs(axes[8]))
                    bx1 = x_i + bx0
                    bx0 = x_i - bx0
                    by1 = y_i + by0
                    by0 = y_i - by0

                    if bx1 <= x1 or bx0 >= x2 or by1 <= y1 or by0 >= y2:
                        continue

                    tz = z0_d - z_i

                    if projected:
                        if z_i + bz <= z_lo_d or z_i - bz >= z_hi_d:
                            continue
                    else:
                        # half-open interval, so that a slice exactly along a cell face does not double count
                        if z_i - bz > z0_d or z_i + bz <= z0_d:
                            continue

                    if not projected and not aligned:
                        # Cross-section of the cube in the plane z=z0 is the intersection of up to six half-planes
                        n_hp = 0
                        for k in range(3):
                            ex = axes[3*k]
                            ey = axes[3*k+1]
                            ez = axes[3*k+2]
                            if fabs(ex) < 1e-12 and fabs(ey) < 1e-12:
                                # this axis is along z, giving no constraint in the plane
                                continue
                            hp_a[n_hp] = ex; hp_b[n_hp] = ey
                            hp_c[n_hp] = half - ez * tz + ex * x_i + ey * y_i
                            n_hp += 1
                            hp_a[n_hp] = -ex; hp_b[n_hp] = -ey
                            hp_c[n_hp] = half + ez * tz - ex * x_i - ey * y_i
                            n_hp += 1

                        # get a tighter bounding box from the cross-section polygon itself
                        hx[0] = bx0; hy[0] = by0
                        hx[1] = bx1; hy[1] = by0
                        hx[2] = bx1; hy[2] = by1
                        hx[3] = bx0; hy[3] = by1
                        nvert = 4
                        for k in range(n_hp):
                            nvert = _clip_polygon(hx, hy, nvert, hp_a[k], hp_b[k], hp_c[k], vx, vy)
                            for ix in range(nvert):
                                hx[ix] = vx[ix]; hy[ix] = vy[ix]
                        if nvert < 3:
                            continue
                        bx0 = bx1 = hx[0]
                        by0 = by1 = hy[0]
                        for k in range(1, nvert):
                            if hx[k] < bx0: bx0 = hx[k]
                            if hx[k] > bx1: bx1 = hx[k]
                            if hy[k] < by0: by0 = hy[k]
                            if hy[k] > by1: by1 = hy[k]

                    # range of pixels touched
                    ix0 = <int> floor((bx0 - x1) / pixel_dx)
                    ix1 = <int> floor((bx1 - x1) / pixel_dx)
                    iy0 = <int> floor((by0 - y1) / pixel_dy)
                    iy1 = <int> floor((by1 - y1) / pixel_dy)
                    if ix0 < 0: ix0 = 0
                    if iy0 < 0: iy0 = 0
                    if ix1 > nx - 1: ix1 = nx - 1
                    if iy1 > ny - 1: iy1 = ny - 1

                    if aligned:
                        # exact, separable pixel overlaps
                        if projected:
                            Lz = _overlap(z_i - half, z_i + half, z_lo_d, z_hi_d)
                        else:
                            Lz = 1.0
                        weight *= Lz / pixel_area
                        for iy in range(iy0, iy1 + 1):
                            Y0 = y1 + iy * pixel_dy
                            value = weight * _overlap(Y0, Y0 + pixel_dy, y_i - half, y_i + half)
                            for ix in range(ix0, ix1 + 1):
                                X0 = x1 + ix * pixel_dx
                                res[iy * nx + ix] += value * _overlap(X0, X0 + pixel_dx, x_i - half, x_i + half)

                    elif not projected:
                        # slice through rotated cell: exact area of overlap of pixel with cross-section polygon
                        for iy in range(iy0, iy1 + 1):
                            Y0 = y1 + iy * pixel_dy
                            Y1 = Y0 + pixel_dy
                            for ix in range(ix0, ix1 + 1):
                                X0 = x1 + ix * pixel_dx
                                X1 = X0 + pixel_dx
                                if (_inside_all(X0, Y0, hp_a, hp_b, hp_c, n_hp) and
                                    _inside_all(X1, Y0, hp_a, hp_b, hp_c, n_hp) and
                                    _inside_all(X0, Y1, hp_a, hp_b, hp_c, n_hp) and
                                    _inside_all(X1, Y1, hp_a, hp_b, hp_c, n_hp)):
                                    value = 1.0
                                else:
                                    value = _clipped_rectangle_area(X0, X1, Y0, Y1, hp_a, hp_b, hp_c, n_hp) \
                                            / pixel_area
                                res[iy * nx + ix] += weight * value

                    else:
                        # projection through rotated cell. Find the convex hull of the projected vertices, used to
                        # quickly reject pixels which do not overlap the cell
                        for k in range(8):
                            vx[k] = x_i + half * ((1 if k & 1 else -1) * axes[0] +
                                                  (1 if k & 2 else -1) * axes[3] +
                                                  (1 if k & 4 else -1) * axes[6])
                            vy[k] = y_i + half * ((1 if k & 1 else -1) * axes[1] +
                                                  (1 if k & 2 else -1) * axes[4] +
                                                  (1 if k & 4 else -1) * axes[7])
                        nh = _convex_hull(vx, vy, 8, hx, hy)

                        lower_row = corner_labels
                        upper_row = corner_labels + (nx + 1)
                        Y0 = y1 + iy0 * pixel_dy
                        for ix in range(ix0, ix1 + 2):
                            _cell_chord(x1 + ix * pixel_dx - x_i, Y0 - y_i, axes, half,
                                        z_lo_d - z_i, z_hi_d - z_i, &label)
                            lower_row[ix] = label

                        for iy in range(iy0, iy1 + 1):
                            Y0 = y1 + iy * pixel_dy
                            Y1 = Y0 + pixel_dy
                            for ix in range(ix0, ix1 + 2):
                                _cell_chord(x1 + ix * pixel_dx - x_i, Y1 - y_i, axes, half,
                                            z_lo_d - z_i, z_hi_d - z_i, &label)
                                upper_row[ix] = label

                            for ix in range(ix0, ix1 + 1):
                                X0 = x1 + ix * pixel_dx
                                X1 = X0 + pixel_dx
                                label_00 = lower_row[ix]
                                label_10 = lower_row[ix + 1]
                                label_01 = upper_row[ix]
                                label_11 = upper_row[ix + 1]
                                uniform = (label_00 == label_10 and label_00 == label_01 and label_00 == label_11)

                                if uniform and label_00 != -1:
                                    # chord is linear over the whole pixel, so its average is the central value
                                    value = _cell_chord(X0 + 0.5 * pixel_dx - x_i, Y0 + 0.5 * pixel_dy - y_i,
                                                        axes, half, z_lo_d - z_i, z_hi_d - z_i, &label)
                                else:
                                    if _rectangle_outside_hull(X0, X1, Y0, Y1, hx, hy, nh):
                                        continue
                                    # pixel straddles the edge of a linear region: supersample
                                    chord_sum = 0.0
                                    for jsub in range(subsamples):
                                        sy = Y0 + (jsub + 0.5) * sub_step_y
                                        if sy < by0 or sy > by1:
                                            continue
                                        for isub in range(subsamples):
                                            sx = X0 + (isub + 0.5) * sub_step_x
                                            if sx < bx0 or sx > bx1:
                                                continue
                                            chord_sum += _cell_chord(sx - x_i, sy - y_i, axes, half,
                                                                     z_lo_d - z_i, z_hi_d - z_i, &label)
                                    value = chord_sum * inv_nsub2

                                res[iy * nx + ix] += weight * value

                            swap_row = lower_row
                            lower_row = upper_row
                            upper_row = swap_row

    return result
