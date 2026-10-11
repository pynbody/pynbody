"""Rendering of AMR cells onto images, as exact slices through or projections of cubes.

Unlike the SPH renderer in :mod:`pynbody.imaging._render`, there is no smoothing kernel. Each 'particle' is a cubic cell
of side length given by its smoothing length (the cell width), with edges along axes which are the columns of a
rotation matrix expressed in the image frame. A slice shows the cross-section of each cell, and a projection the
chord length through it, averaged over each pixel. Because the cells tile space, the resulting image is a
rendering of the piecewise-constant field.

The Python interface is :class:`pynbody.imaging.renderers.AMRImageRenderer`.
"""

import threading

import numpy as np

cimport cython
cimport numpy as np

np.import_array()
from libc.math cimport NAN, fabs, isfinite, sqrt

# See _render.pyx for the reason for the separate fused types
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


cdef enum:
    MAX_POLY = 16  # maximum number of vertices of a clipped polygon (a pixel clipped by a cube slice has <= 10)

cdef inline int _floor_int(double v) noexcept nogil:
    """floor(v) as an int, avoiding a library call; v must be within the range of int"""
    cdef int i = <int> v
    return i - 1 if v < i else i

cdef inline double _overlap(double a0, double a1, double b0, double b1) noexcept nogil:
    """Length of the overlap between intervals [a0, a1] and [b0, b1]"""
    cdef double lo = a0 if a0 > b0 else b0
    cdef double hi = a1 if a1 < b1 else b1
    return hi - lo if hi > lo else 0.0

@cython.cdivision(True)
cdef inline double _cell_chord(double dx, double dy, const double *axes, double half,
                               double t_lo, double t_hi) noexcept nogil:
    """Length of the chord along the z direction through a rotated cube, at offset (dx, dy) from its centre.

    The cube has half-width *half* and its kth axis is (axes[3k], axes[3k+1], axes[3k+2]). The chord is additionally
    restricted to the range of z offsets [t_lo, t_hi].
    """
    cdef int k
    cdef double s, ez, a, b
    for k in range(3):
        s = axes[3*k] * dx + axes[3*k+1] * dy
        ez = axes[3*k+2]
        if ez < 1e-12 and ez > -1e-12:
            # this axis lies in the image plane; it constrains (dx, dy) but not the chord
            if s > half or s < -half:
                return 0.0
            continue
        if ez > 0:
            a = (-half - s) / ez
            b = (half - s) / ez
        else:
            a = (half - s) / ez
            b = (-half - s) / ez
        if a > t_lo:
            t_lo = a
        if b < t_hi:
            t_hi = b
    return t_hi - t_lo if t_hi > t_lo else 0.0


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

cdef inline double _dot3(const double *a, const double *b) noexcept nogil:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@cython.cdivision(True)
cdef inline double _square_area_below_line(double half, double a, double b, double c) noexcept nogil:
    """Area of the square [-half, half]^2 with a u + b v <= c.

    For u, v uniform over the square, a u + b v is the sum of two uniform variables on [-A, A] and [-B, B], whose
    cumulative distribution is piecewise quadratic."""
    cdef double A = fabs(a) * half, B = fabs(b) * half, t
    if B > A:
        t = A; A = B; B = t
    if c >= A + B:
        return 4.0 * half * half
    if c <= -(A + B):
        return 0.0
    if c <= B - A:
        return 4.0 * half * half * (c + A + B) * (c + A + B) / (8.0 * A * B)
    if c < A - B:
        return 4.0 * half * half * (c + A) / (2.0 * A)
    return 4.0 * half * half * (1.0 - (A + B - c) * (A + B - c) / (8.0 * A * B))


@cython.cdivision(True)
cdef int _cube_cross_section(const double *axes, double half, double zc, double *px, double *py) noexcept nogil:
    """Write the cross-section of the cube (centred on the origin) in the plane z = zc, as a polygon of (x, y)
    vertices, into px, py (which must have space for MAX_POLY vertices). Returns the number of vertices."""
    cdef double buf_x[MAX_POLY]
    cdef double buf_y[MAX_POLY]
    cdef double ex, ey, ez
    cdef double bound_x = half * (fabs(axes[0]) + fabs(axes[3]) + fabs(axes[6])) * 1.01
    cdef double bound_y = half * (fabs(axes[1]) + fabs(axes[4]) + fabs(axes[7])) * 1.01
    cdef int k, n = 4, j
    px[0] = -bound_x; py[0] = -bound_y
    px[1] = bound_x; py[1] = -bound_y
    px[2] = bound_x; py[2] = bound_y
    px[3] = -bound_x; py[3] = bound_y
    for k in range(3):
        ex = axes[3 * k]
        ey = axes[3 * k + 1]
        ez = axes[3 * k + 2]
        if fabs(ex) < 1e-12 and fabs(ey) < 1e-12:
            # this axis is along z, so the plane is either within its range or not
            if fabs(ez * zc) > half:
                return 0
            continue
        n = _clip_polygon(px, py, n, ex, ey, half - ez * zc, buf_x, buf_y)
        n = _clip_polygon(buf_x, buf_y, n, -ex, -ey, half + ez * zc, px, py)
    return n if n >= 3 else 0


@cython.cdivision(True)
cdef double _slab_quadrant_volume(const double *axes, double half, double dx, double dy, double z_lo, double z_hi,
                                  const double *cap_x, const double *cap_y, int n_cap) noexcept nogil:
    """Volume of the cube (centred on the origin) with x <= dx, y <= dy and z_lo <= z <= z_hi.

    (cap_x, cap_y) must be the cross-section of the cube at z_lo, from _cube_cross_section. The divergence theorem
    is applied about r0 = (dx, dy, z_hi), which lies in three of the clipping planes, so that only the cube faces
    and the cap at z_lo contribute: the volume is a third of the sum of (distance of plane from r0) x (area).
    """
    cdef double normals[12]
    cdef double offsets[4]
    cdef double hp_a[4]
    cdef double hp_b[4]
    cdef double hp_c[4]
    cdef double buf_x[MAX_POLY]
    cdef double buf_y[MAX_POLY]
    cdef double buf2_x[MAX_POLY]
    cdef double buf2_y[MAX_POLY]
    cdef double total = 0.0, sign, area, a, b, c, nk, limit
    cdef double extent_x = half * (fabs(axes[0]) + fabs(axes[3]) + fabs(axes[6]))
    cdef double extent_y = half * (fabs(axes[1]) + fabs(axes[4]) + fabs(axes[7]))
    cdef int k, k1, k2, s, i, n_hp, n
    cdef const double *n_i
    # r0 is used as the origin for the divergence theorem, so must not be far from the cube to avoid cancellation;
    # moving the planes to the edge of the cube makes no difference to the volume
    if dx > extent_x: dx = extent_x
    if dy > extent_y: dy = extent_y
    if dx < -extent_x: return 0.0
    if dy < -extent_y: return 0.0
    normals[0] = 1.0; normals[1] = 0.0; normals[2] = 0.0; offsets[0] = dx
    normals[3] = 0.0; normals[4] = 1.0; normals[5] = 0.0; offsets[1] = dy
    normals[6] = 0.0; normals[7] = 0.0; normals[8] = 1.0; offsets[2] = z_hi
    normals[9] = 0.0; normals[10] = 0.0; normals[11] = -1.0; offsets[3] = -z_lo

    for k in range(3):
        k1 = (k + 1) % 3
        k2 = (k + 2) % 3
        for s in range(2):
            sign = 1.0 if s else -1.0
            # constraints on the face, in face coordinates r = sign half e_k + u e_k1 + v e_k2
            n_hp = 0
            for i in range(4):
                n_i = normals + 3 * i
                a = _dot3(n_i, axes + 3 * k1)
                b = _dot3(n_i, axes + 3 * k2)
                nk = _dot3(n_i, axes + 3 * k)
                c = offsets[i] - sign * half * nk
                limit = half * (fabs(a) + fabs(b))
                if fabs(a) + fabs(b) < 1e-12:
                    # plane parallel to face: the face is either excluded or not. Where the plane coincides with the
                    # face and faces the same way, the face is excluded; only the z_lo plane carries a cap to take
                    # its place, but faces in the other planes contribute nothing anyway (being at zero distance).
                    if c < -1e-12 * half or (c <= 1e-12 * half and sign * nk > 0):
                        n_hp = -1
                        break
                    continue
                if c <= -limit:
                    n_hp = -1
                    break
                if c < limit:
                    hp_a[n_hp] = a; hp_b[n_hp] = b; hp_c[n_hp] = c
                    n_hp += 1
            if n_hp < 0:
                continue
            if n_hp == 0:
                area = 4.0 * half * half
            elif n_hp == 1:
                area = _square_area_below_line(half, hp_a[0], hp_b[0], hp_c[0])
            else:
                area = _clipped_rectangle_area(-half, half, -half, half, hp_a, hp_b, hp_c, n_hp)
            total += (half - sign * (axes[3 * k] * dx + axes[3 * k + 1] * dy + axes[3 * k + 2] * z_hi)) * area

    # the cap at z_lo, clipped by x <= dx and y <= dy
    if n_cap >= 3:
        n = _clip_polygon(cap_x, cap_y, n_cap, 1.0, 0.0, dx, buf_x, buf_y)
        n = _clip_polygon(buf_x, buf_y, n, 0.0, 1.0, dy, buf2_x, buf2_y)
        if n >= 3:
            total += (z_hi - z_lo) * _polygon_area(buf2_x, buf2_y, n)
    return total / 3.0


@cython.cdivision(True)
cdef double _quadrant_volume(const double *axes, double half, double dx, double dy,
                             double z_lo, double z_hi, int clip_z) noexcept nogil:
    """Volume of the cube (centred on origin) with x <= dx and y <= dy, and optionally z_lo <= z <= z_hi.

    Without z clipping, the divergence theorem is applied about the point r0 = (dx, dy, 0), which lies in both
    clipping planes so that the caps there make no contribution. The volume is then a third of the sum over the
    cube faces of (distance of face plane from r0) x (clipped face area). With z clipping, see
    _slab_quadrant_volume (to which the cross-section at z_lo can be passed directly, if already known).
    """
    cdef double cap_x[MAX_POLY]
    cdef double cap_y[MAX_POLY]
    cdef double hp_a[2]
    cdef double hp_b[2]
    cdef double hp_c[2]
    cdef double total = 0.0, sign, area, ax, bx_, cx, ay, by_, cy, ex, ey
    cdef int k, k1, k2, s, n_hp
    cdef double extent_x = half * (fabs(axes[0]) + fabs(axes[3]) + fabs(axes[6]))
    cdef double extent_y = half * (fabs(axes[1]) + fabs(axes[4]) + fabs(axes[7]))
    if clip_z:
        return _slab_quadrant_volume(axes, half, dx, dy, z_lo, z_hi, cap_x, cap_y,
                                     _cube_cross_section(axes, half, z_lo, cap_x, cap_y))
    # r0 = (dx, dy, 0) is used as the origin for the divergence theorem, so must not be far from the cube to avoid
    # cancellation; moving the planes to the edge of the cube makes no difference to the volume
    if dx > extent_x: dx = extent_x
    if dy > extent_y: dy = extent_y
    if dx < -extent_x or dy < -extent_y:
        return 0.0

    for k in range(3):
        k1 = (k + 1) % 3
        k2 = (k + 2) % 3
        ex = axes[3 * k]
        ey = axes[3 * k + 1]
        # in face coordinates (u, v), x = sign half ex + u axes[3 k1] + v axes[3 k2], and similarly y
        ax = axes[3 * k1]; bx_ = axes[3 * k2]
        ay = axes[3 * k1 + 1]; by_ = axes[3 * k2 + 1]
        for s in range(2):
            sign = 1.0 if s else -1.0
            cx = dx - sign * half * ex
            cy = dy - sign * half * ey
            # skip clipping where the face lies entirely on one side of a plane
            if cx < -half * (fabs(ax) + fabs(bx_)) or cy < -half * (fabs(ay) + fabs(by_)):
                continue
            n_hp = 0
            if cx < half * (fabs(ax) + fabs(bx_)):
                hp_a[n_hp] = ax; hp_b[n_hp] = bx_; hp_c[n_hp] = cx
                n_hp += 1
            if cy < half * (fabs(ay) + fabs(by_)):
                hp_a[n_hp] = ay; hp_b[n_hp] = by_; hp_c[n_hp] = cy
                n_hp += 1
            if n_hp == 0:
                area = 4.0 * half * half
            elif n_hp == 1:
                area = _square_area_below_line(half, hp_a[0], hp_b[0], hp_c[0])
            else:
                area = _clipped_rectangle_area(-half, half, -half, half, hp_a, hp_b, hp_c, n_hp)
            total += (half - sign * (ex * dx + ey * dy)) * area
    return total / 3.0


@cython.boundscheck(False)
@cython.wraparound(False)
def cell_quadrant_volume(orientation, dx, dy, double z_lo=-np.inf, double z_hi=np.inf):
    """Volume of a unit cube centred on the origin, with edges along the columns of orientation, for which
    x <= dx and y <= dy (and z_lo <= z <= z_hi). Vectorised over dx and dy; for testing."""
    cdef double axes[9]
    cdef int i, k, clip_z = np.isfinite(z_lo) or np.isfinite(z_hi)
    orientation = np.asarray(orientation, dtype=np.float64)
    for k in range(3):
        for i in range(3):
            axes[3 * k + i] = orientation[i, k]
    dx, dy = np.broadcast_arrays(np.asarray(dx, dtype=np.float64), np.asarray(dy, dtype=np.float64))
    cdef np.ndarray[np.float64_t, ndim=1] dx_ = np.ascontiguousarray(dx).ravel()
    cdef np.ndarray[np.float64_t, ndim=1] dy_ = np.ascontiguousarray(dy).ravel()
    cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(len(dx_))
    cdef Py_ssize_t j, n = len(dx_)
    if z_lo < -2.0: z_lo = -2.0
    if z_hi > 2.0: z_hi = 2.0
    with nogil:
        for j in range(n):
            out[j] = _quadrant_volume(axes, 0.5, dx_[j], dy_[j], z_lo, z_hi, clip_z)
    return out.reshape(dx.shape)


@cython.cdivision(True)
cdef double _split_pixel_integral(double X0, double Y0, double pixel_dx, double pixel_dy,
                                  double px, double py, double dx, double dy,
                                  double x_i, double y_i, double half, const double *axes,
                                  double zl, double zh) noexcept nogil:
    """Average chord length over a pixel which is crossed by just one line where the chord's gradient changes.

    The line passes through (px, py) with direction (dx, dy). Each part of the pixel on either side lies within
    a single region where the chord is linear, so the integral over that part is its area times the chord at its
    centroid.
    """
    cdef double sx[4]
    cdef double sy[4]
    cdef double ox[MAX_POLY]
    cdef double oy[MAX_POLY]
    cdef double area_pix = pixel_dx * pixel_dy, area_a = 0.0, cxa = 0.0, cya = 0.0, cross, area_b, cxb, cyb
    cdef double total = 0.0
    cdef int n, k, k1
    sx[0] = X0; sy[0] = Y0
    sx[1] = X0 + pixel_dx; sy[1] = Y0
    sx[2] = X0 + pixel_dx; sy[2] = Y0 + pixel_dy
    sx[3] = X0; sy[3] = Y0 + pixel_dy
    # side a: (-dy, dx) . (r - p) <= 0
    n = _clip_polygon(sx, sy, 4, -dy, dx, -dy * px + dx * py, ox, oy)
    for k in range(n):
        k1 = (k + 1) % n
        cross = ox[k] * oy[k1] - ox[k1] * oy[k]
        area_a += cross
        cxa += (ox[k] + ox[k1]) * cross
        cya += (oy[k] + oy[k1]) * cross
    if area_a != 0.0:
        cxa /= 3.0 * area_a
        cya /= 3.0 * area_a
    area_a *= 0.5
    if area_a < 0.0:
        area_a = -area_a
    area_b = area_pix - area_a
    if area_a > 1e-10 * area_pix:
        total += area_a * _cell_chord(cxa - x_i, cya - y_i, axes, half, zl, zh)
    if area_b > 1e-10 * area_pix:
        cxb = ((X0 + 0.5 * pixel_dx) * area_pix - cxa * area_a) / area_b
        cyb = ((Y0 + 0.5 * pixel_dy) * area_pix - cya * area_a) / area_b
        total += area_b * _cell_chord(cxb - x_i, cyb - y_i, axes, half, zl, zh)
    return total / area_pix


cdef inline int _segments_collinear(const double *x0, const double *y0, const double *x1, const double *y1,
                                    int a, int b) noexcept nogil:
    """True if segments a and b (with end points (x0, y0), (x1, y1)) lie along the same line, to round-off"""
    cdef double dxa = x1[a] - x0[a], dya = y1[a] - y0[a], dxb = x1[b] - x0[b], dyb = y1[b] - y0[b]
    cdef double len2a = dxa * dxa + dya * dya, len2b = dxb * dxb + dyb * dyb
    cdef double tol = 1e-9 * sqrt(len2a * len2b)
    if fabs(dxa * dyb - dya * dxb) > tol:
        return 0
    # offset of segment b's start from segment a's line
    return fabs(dxa * (y0[b] - y0[a]) - dya * (x0[b] - x0[a])) <= 1e-9 * len2a


cdef struct CornerCache:
    int ix               # pixel-corner column for which F values are cached, or -1
    double F_lower       # F at (X, Y0)
    double F_upper       # F at (X, Y1)


cdef inline double _rows_F(const double *axes, double half, double dx, double dy, double zl, double zh, int clip_z,
                           const double *cap_x, const double *cap_y, int n_cap) noexcept nogil:
    """Exact volume of the cell with x <= dx, y <= dy (and within [zl, zh] if clip_z is set)"""
    if clip_z:
        return _slab_quadrant_volume(axes, half, dx, dy, zl, zh, cap_x, cap_y, n_cap)
    else:
        return _quadrant_volume(axes, half, dx, dy, 0.0, 0.0, 0)


@cython.cdivision(True)
cdef void _project_rotated_cell_rows(double *dres, int nx, double x1, double y1, double pixel_dx, double pixel_dy,
                                     int ix0, int ix1, int iy0, int iy1, double x_i, double y_i, double half,
                                     const double *axes, double zl, double zh, int clip_z,
                                     double weight) noexcept nogil:
    """Add the exact projection of a rotated cell to an image held as second differences along x.

    Within a row of pixels, the cell's contribution I_i (weight times the pixel-averaged chord length) is linear in
    i, except in pixels crossed by a line along which the chord's gradient changes. These are the projections of
    the cube's 12 edges and, if clip_z is set, of the edges of its cross-sections at z offsets zl and zh, which
    limit the chord. The second difference D_i = I_i - 2 I_{i-1} + I_{i-2} therefore vanishes except within two
    pixels after any such crossing. Only those non-zero values are added to dres, from which the image can be
    recovered by two cumulative sums along x. The cost is therefore proportional to the cell's perimeter in pixels,
    rather than its area.

    Pixels crossed by a single such line (and containing none of its end points) are evaluated exactly by
    _split_pixel_integral. Other crossed pixels are evaluated exactly as differences of F, the volume of the cell
    with x <= X and y <= Y (see _quadrant_volume), across their corners. All remaining pixels are linear across the
    pixel, so their average is the central chord length. Pixels with i < ix0 or i > ix1 are taken to be zero;
    [iy0, iy1] must lie within the image. zl and zh must be finite.
    """
    cdef double seg_x0[24]
    cdef double seg_y0[24]
    cdef double seg_x1[24]
    cdef double seg_y1[24]
    cdef double poly_x[2][MAX_POLY]  # cross-sections at zl and zh, relative to the cell centre
    cdef double poly_y[2][MAX_POLY]
    cdef int n_poly[2]
    cdef int plane
    cdef int run_start[72]
    cdef int run_end[72]
    cdef int run_segment[72]  # the single line crossing pixels of the run, or -1
    cdef int visit_start[73]
    cdef int visit_end[73]
    cdef int piece_start[3]
    cdef int piece_end[3]
    cdef int piece_segment[3]
    cdef int n_piece, pe, q
    cdef int n_visit
    cdef double pixel_area = pixel_dx * pixel_dy
    cdef double inv_pixel_dx = 1.0 / pixel_dx
    cdef double Y0, Y1, ta, tb, xa, xb, t, I_m2, I_m1, I_0, Xc, Yc
    cdef double p0x, p0y, p1x, p1y, ox, oy
    cdef int k, k1, k2, s1, s2, n_seg = 0, n_run, m, j, i, iy, a, b, run_index
    cdef CornerCache left, right

    # projected edges: for each axis k, the four edges parallel to it
    for k in range(3):
        k1 = (k + 1) % 3
        k2 = (k + 2) % 3
        for s1 in range(2):
            for s2 in range(2):
                ox = x_i + half * ((2 * s1 - 1) * axes[3 * k1] + (2 * s2 - 1) * axes[3 * k2])
                oy = y_i + half * ((2 * s1 - 1) * axes[3 * k1 + 1] + (2 * s2 - 1) * axes[3 * k2 + 1])
                p0x = ox - half * axes[3 * k]
                p0y = oy - half * axes[3 * k + 1]
                p1x = ox + half * axes[3 * k]
                p1y = oy + half * axes[3 * k + 1]
                if p0y <= p1y:
                    seg_x0[n_seg] = p0x; seg_y0[n_seg] = p0y; seg_x1[n_seg] = p1x; seg_y1[n_seg] = p1y
                else:
                    seg_x0[n_seg] = p1x; seg_y0[n_seg] = p1y; seg_x1[n_seg] = p0x; seg_y1[n_seg] = p0y
                n_seg += 1

    n_poly[0] = 0
    n_poly[1] = 0
    if clip_z:
        # edges of the cross-sections of the cube in the planes z = zl and z = zh (relative to its centre)
        for plane in range(2):
            n_poly[plane] = _cube_cross_section(axes, half, zl if plane == 0 else zh, poly_x[plane], poly_y[plane])
            for j in range(n_poly[plane]):
                k = (j + 1) % n_poly[plane]
                p0x = x_i + poly_x[plane][j]
                p0y = y_i + poly_y[plane][j]
                p1x = x_i + poly_x[plane][k]
                p1y = y_i + poly_y[plane][k]
                if fabs(p1x - p0x) + fabs(p1y - p0y) <= 1e-12 * half:
                    continue
                if p0y <= p1y:
                    seg_x0[n_seg] = p0x; seg_y0[n_seg] = p0y; seg_x1[n_seg] = p1x; seg_y1[n_seg] = p1y
                else:
                    seg_x0[n_seg] = p1x; seg_y0[n_seg] = p1y; seg_x1[n_seg] = p0x; seg_y1[n_seg] = p0y
                n_seg += 1

    for iy in range(iy0, iy1 + 1):
        Y0 = y1 + iy * pixel_dy
        Y1 = Y0 + pixel_dy
        Yc = Y0 + 0.5 * pixel_dy

        # find the pixels crossed by projected edges within this row, as runs [a, b]
        n_run = 0
        for k in range(n_seg):
            if seg_y1[k] < Y0 or seg_y0[k] > Y1:
                continue
            if seg_y1[k] - seg_y0[k] <= 1e-12 * pixel_dy:
                xa = seg_x0[k]
                xb = seg_x1[k]
            else:
                ta = (Y0 - seg_y0[k]) / (seg_y1[k] - seg_y0[k])
                tb = (Y1 - seg_y0[k]) / (seg_y1[k] - seg_y0[k])
                if ta < 0.0: ta = 0.0
                if tb > 1.0: tb = 1.0
                xa = seg_x0[k] + ta * (seg_x1[k] - seg_x0[k])
                xb = seg_x0[k] + tb * (seg_x1[k] - seg_x0[k])
            if xa > xb:
                t = xa; xa = xb; xb = t
            a = _floor_int((xa - x1) * inv_pixel_dx)
            b = _floor_int((xb - x1) * inv_pixel_dx)
            if a < ix0: a = ix0
            if b > ix1: b = ix1
            if a > b:
                continue
            # Pixels containing an end point of the edge are crossed by more than one edge (since they contain a
            # vertex), and are marked with segment -1. The end points have the extreme values of x, so lie in
            # the first or last pixel of the run.
            piece_start[0] = a
            piece_end[0] = b
            piece_segment[0] = k
            n_piece = 1
            for q in range(2):
                if q == 0:
                    if seg_y0[k] < Y0: continue
                    pe = _floor_int((seg_x0[k] - x1) * inv_pixel_dx)
                else:
                    if seg_y1[k] > Y1: continue
                    pe = _floor_int((seg_x1[k] - x1) * inv_pixel_dx)
                if pe < a or pe > b:
                    continue
                # split the (last) simple piece at pe, which is at one of its ends
                j = n_piece - 1
                if piece_segment[j] < 0 or pe < piece_start[j] or pe > piece_end[j]:
                    j = 0
                if piece_start[j] == piece_end[j]:
                    piece_segment[j] = -1
                elif pe == piece_start[j]:
                    piece_start[n_piece] = pe; piece_end[n_piece] = pe; piece_segment[n_piece] = -1
                    piece_start[j] = pe + 1
                    n_piece += 1
                elif pe == piece_end[j]:
                    piece_start[n_piece] = pe; piece_end[n_piece] = pe; piece_segment[n_piece] = -1
                    piece_end[j] = pe - 1
                    n_piece += 1
                else:
                    # (should not occur) treat the whole piece as complex
                    piece_segment[j] = -1
            for q in range(n_piece):
                # insert into the sorted list of runs
                m = n_run
                while m > 0 and run_start[m - 1] > piece_start[q]:
                    run_start[m] = run_start[m - 1]
                    run_end[m] = run_end[m - 1]
                    run_segment[m] = run_segment[m - 1]
                    m -= 1
                run_start[m] = piece_start[q]
                run_end[m] = piece_end[q]
                run_segment[m] = piece_segment[q]
                n_run += 1

        # merge overlapping runs of crossed pixels; the result is crossed by more than one edge, unless the edges
        # are collinear (as occurs when a face of the cube is parallel to the line of sight)
        m = 0
        for j in range(1, n_run):
            if run_start[j] <= run_end[m]:
                if run_end[j] > run_end[m]:
                    run_end[m] = run_end[j]
                if run_segment[m] >= 0 and run_segment[j] >= 0:
                    if not _segments_collinear(seg_x0, seg_y0, seg_x1, seg_y1, run_segment[m], run_segment[j]):
                        run_segment[m] = -1
                else:
                    run_segment[m] = -1
            else:
                m += 1
                run_start[m] = run_start[j]
                run_end[m] = run_end[j]
                run_segment[m] = run_segment[j]
        if n_run > 0:
            n_run = m + 1

        # The second difference may be non-zero only within two pixels after a crossed run, and at the first two
        # pixels of the range (since pixels before ix0 are taken to be zero). Make a merged list of these intervals.
        visit_start[0] = ix0
        visit_end[0] = ix0 + 1
        n_visit = 1
        for j in range(n_run):
            if run_start[j] <= visit_end[n_visit - 1] + 1:
                if run_end[j] + 2 > visit_end[n_visit - 1]:
                    visit_end[n_visit - 1] = run_end[j] + 2
            else:
                visit_start[n_visit] = run_start[j]
                visit_end[n_visit] = run_end[j] + 2
                n_visit += 1

        left.ix = -1
        right.ix = -1
        run_index = 0
        for m in range(n_visit):
            a = visit_start[m]
            b = visit_end[m]
            # the two pixels before a visited interval are never crossed by an edge, so are linear
            I_m2 = _cell_chord(x1 + (a - 1.5) * pixel_dx - x_i, Yc - y_i, axes, half, zl, zh) \
                if a - 2 >= ix0 else 0.0
            I_m1 = _cell_chord(x1 + (a - 0.5) * pixel_dx - x_i, Yc - y_i, axes, half, zl, zh) \
                if a - 1 >= ix0 else 0.0
            for i in range(a, b + 1):
                if i > ix1:
                    I_0 = 0.0
                else:
                    while run_index < n_run and run_end[run_index] < i:
                        run_index += 1
                    if run_index < n_run and run_start[run_index] <= i and run_segment[run_index] >= 0:
                        # crossed by a single edge
                        k = run_segment[run_index]
                        I_0 = _split_pixel_integral(x1 + i * pixel_dx, Y0, pixel_dx, pixel_dy, seg_x0[k], seg_y0[k],
                                                    seg_x1[k] - seg_x0[k], seg_y1[k] - seg_y0[k],
                                                    x_i, y_i, half, axes, zl, zh)
                    elif run_index < n_run and run_start[run_index] <= i:
                        # crossed by several lines: integrate exactly from F at the corners
                        Xc = x1 + i * pixel_dx - x_i
                        if left.ix != i:
                            if right.ix == i:
                                left = right
                            else:
                                left.ix = i
                                left.F_lower = _rows_F(axes, half, Xc, Y0 - y_i, zl, zh, clip_z,
                                                       poly_x[0], poly_y[0], n_poly[0])
                                left.F_upper = _rows_F(axes, half, Xc, Y1 - y_i, zl, zh, clip_z,
                                                       poly_x[0], poly_y[0], n_poly[0])
                        right.ix = i + 1
                        right.F_lower = _rows_F(axes, half, Xc + pixel_dx, Y0 - y_i, zl, zh, clip_z,
                                                poly_x[0], poly_y[0], n_poly[0])
                        right.F_upper = _rows_F(axes, half, Xc + pixel_dx, Y1 - y_i, zl, zh, clip_z,
                                                poly_x[0], poly_y[0], n_poly[0])
                        I_0 = ((right.F_upper - right.F_lower) - (left.F_upper - left.F_lower)) / pixel_area
                        if I_0 < 0.0:
                            I_0 = 0.0
                    else:
                        I_0 = _cell_chord(x1 + (i + 0.5) * pixel_dx - x_i, Yc - y_i, axes, half, zl, zh)
                if i < nx:
                    dres[iy * nx + i] += weight * (I_0 - 2.0 * I_m1 + I_m2)
                I_m2 = I_m1
                I_m1 = I_0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cell_projection_cdf_table(orientation, int n):
    """Tabulate the fraction of a unit cube's volume with x <= xi and y <= eta, for a cube centred on the origin
    with edges along the columns of the rotation matrix *orientation*.

    Returns (table, bx, by) where table[i, j] corresponds to xi = -bx + 2 bx i / n, eta = -by + 2 by j / n;
    and [-bx, bx] x [-by, by] is the extent of the cube's projection. The table is generated by an exact projection
    of the cube onto an n x n grid (see _project_rotated_cell_rows), followed by cumulative sums; so it is exact to
    round-off.
    """
    cdef double axes[9]
    cdef int i, k
    cdef double bx = 0.0, by = 0.0
    orientation = np.asarray(orientation, dtype=np.float64)
    for k in range(3):
        for i in range(3):
            axes[3 * k + i] = orientation[i, k]
        bx += 0.5 * fabs(axes[3 * k])
        by += 0.5 * fabs(axes[3 * k + 1])
    cdef double hx_ = 2.0 * bx / n, hy_ = 2.0 * by / n
    # column[iy, ix] is the average chord length over each pixel, initially held as second differences along x
    cdef np.ndarray[np.float64_t, ndim=2] column = np.zeros((n, n), dtype=np.float64)
    with nogil:
        _project_rotated_cell_rows(<double *> column.data, n, -bx, -by, hx_, hy_, 0, n - 1, 0, n - 1,
                                   0.0, 0.0, 0.5, axes, -1.0, 1.0, 0, 1.0)
    np.cumsum(column, axis=1, out=column)
    np.cumsum(column, axis=1, out=column)
    np.maximum(column, 0.0, out=column)
    table = np.zeros((n + 1, n + 1), dtype=np.float64)
    table[1:, 1:] = np.cumsum(np.cumsum(column.T * (hx_ * hy_), axis=0), axis=1)
    table /= table[n, n]  # remove round-off in the total
    return table, bx, by


@cython.cdivision(True)
cdef inline double _table_lookup(const double *tab, int n, double bx, double by, double scale_x, double scale_y,
                                 double xi, double eta) noexcept nogil:
    """Bilinear interpolation in a table generated by cell_projection_cdf_table, clamped to its extent.

    Here scale_x = n / (2 bx) and scale_y = n / (2 by)."""
    cdef double fx = (xi + bx) * scale_x
    cdef double fy = (eta + by) * scale_y
    cdef int i, j
    if fx <= 0.0 or fy <= 0.0:
        return 0.0
    if fx >= n:
        fx = n
    if fy >= n:
        fy = n
    i = <int> fx
    j = <int> fy
    if i == n:
        i = n - 1
    if j == n:
        j = n - 1
    fx -= i
    fy -= j
    tab += i * (n + 1) + j
    return ((1.0 - fx) * ((1.0 - fy) * tab[0] + fy * tab[1]) +
            fx * ((1.0 - fy) * tab[n + 1] + fy * tab[n + 2]))


# Projections of rotated cells use interpolation in a table of the cell's volume as a function of position (see
# render_image_cells). The interpolation error in a pixel is of order 12 (cell size / pixel size) / n^2, for a
# table of resolution n, so larger cells use a finer table; and cells larger than the last are projected exactly.
CELL_TABLES = ((4.0, 256), (16.0, 512))  # (maximum cell size in pixels, table resolution)

_cdf_table_cache = {}
_cdf_table_lock = threading.Lock()

def get_cell_projection_cdf_table(orientation, int n):
    """Return cell_projection_cdf_table(orientation, n), caching recent results.

    A lock ensures that threads rendering parts of the same image do not each generate the same table."""
    orientation = np.ascontiguousarray(orientation, dtype=np.float64)
    key = (orientation.tobytes(), n)
    with _cdf_table_lock:
        result = _cdf_table_cache.get(key, None)
        if result is None:
            result = cell_projection_cdf_table(orientation, n)
            if len(_cdf_table_cache) >= 8:
                _cdf_table_cache.clear()
            _cdf_table_cache[key] = result
    return result


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
                       wrap_offsets_x=[0], wrap_offsets_y=[0],
                       tables=None):
    """Render cubic (AMR) cells onto an image, either as an exact slice at z=z0 or a projection along z.

    Each cell has side length sm and the edges of all cells are parallel to the columns of the 3x3 rotation
    matrix *orientation*, expressed in the image frame. The value in each pixel is the average over the pixel
    area of the slice through (or projection of) the piecewise-constant field qty, with mass/rho giving the cell
    volume. In the projected case, the z range of the projection is restricted to [z_lo, z_hi], with cells being
    exactly truncated at those limits. A slice lying exactly along cell faces samples the cells on its +z side.

    If the cells are aligned with the image axes, the pixel averages are computed analytically. For a slice
    through rotated cells, the exact area of overlap between each pixel and each cell's (polygonal) cross-section
    is calculated.

    For a projection through rotated cells, the integral over a pixel is a difference between the values at the
    pixel corners of F(X, Y), the volume of the cell with x <= X and y <= Y. Since all cells share the same
    orientation, F for every cell is a rescaled and translated copy of a single function, which is tabulated
    exactly (see cell_projection_cdf_table) and bilinearly interpolated. This conserves the mass of each cell
    exactly and never produces negative values; errors in individual pixels, relative to the peak column through
    the same cell, are typically below 1e-5 and at most of order 1e-3. The table resolution depends on the cell size relative to the pixels, according to
    CELL_TABLES.

    Rotated cells larger than covered by CELL_TABLES, or straddling z_lo or z_hi, are projected exactly. Their
    contributions are accumulated as second differences along each row of pixels, at a cost proportional to the
    cell perimeter rather than area (see _project_rotated_cell_rows).

    *tables* optionally provides a list of (max_cell_pixels, table, bx, by) to use in place of CELL_TABLES. In
    particular, tables=[] projects all rotated cells exactly (as requested by AMRImageRenderer with exact=True). Perspective (z_camera) rendering is
    not currently supported.
    """

    cdef double pixel_dx = (x2 - x1) / nx
    cdef double pixel_dy = (y2 - y1) / ny
    cdef double pixel_area = pixel_dx * pixel_dy
    cdef double inv_pixel_dx = 1.0 / pixel_dx, inv_pixel_dy = 1.0 / pixel_dy, inv_pixel_area = 1.0 / pixel_area
    cdef double pixel_min = pixel_dx if pixel_dx < pixel_dy else pixel_dy
    cdef int n_part = len(x)
    cdef int i, k, ix, iy, ix0, ix1, iy0, iy1, n_hp, nvert, clip_z, table_index
    cdef double x_i, y_i, z_i, sm_i, half, weight, tz, value, Lz, inv_sm, eta, xi, f_x, f_y, f_xy
    cdef double bx0, bx1, by0, by1, X0, X1, Y0, Y1, zl, zh
    cdef double ex, ey, ez
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
    cdef double ext_x = 0.0, ext_y = 0.0, ext_z = 0.0  # half-extents of a unit cell along image axes
    cdef int aligned
    cdef double wrap_offset_x, wrap_offset_y

    cdef np.ndarray[image_output_type, ndim=2] result = np.zeros((ny, nx), dtype=np_image_output_type)
    cdef image_output_type *res = <image_output_type *> result.data

    # rows of values of F at pixel corners, for the tabulated projection
    cdef np.ndarray[np.float64_t, ndim=1] corner_F_np = np.empty(2 * (nx + 1), dtype=np.float64)
    cdef double *lower_F = <double *> corner_F_np.data
    cdef double *upper_F = lower_F + (nx + 1)
    cdef double *swap_F

    # interpolation tables for rotated projections, in increasing order of the cell size (in pixels) they serve
    cdef int n_tables = 0
    cdef double table_max_pixels[4]
    cdef double *table_data[4]
    cdef int table_n[4]
    cdef double table_bx[4]
    cdef double table_by[4]
    cdef double table_sx[4]
    cdef double table_sy[4]
    cdef double *tab
    cdef int tn
    cdef double tbx, tby, tsx, tsy
    table_refs = []

    # second differences along x, for exactly projected rotated cells (see _project_rotated_cell_rows)
    cdef np.ndarray[np.float64_t, ndim=2] dres_np
    cdef double *dres = NULL

    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.shape != (3, 3):
        raise ValueError("orientation must be a 3x3 matrix")
    assert len(x) == len(y) == len(z) == len(sm) == len(qty) == len(mass) == len(rho), \
        "Inconsistent array lengths passed to render_image_cells"

    for k in range(3):
        for i in range(3):
            # axes[3k + i] is component i (in the image frame) of cell axis k
            axes[3*k + i] = orientation[i, k]
        ext_x += 0.5 * fabs(axes[3*k])
        ext_y += 0.5 * fabs(axes[3*k+1])
        ext_z += 0.5 * fabs(axes[3*k+2])

    # cells are aligned with the image if the orientation is a signed permutation matrix
    aligned = bool(np.all(np.sum(np.abs(orientation) > 1e-10, axis=0) == 1))

    if projected and not aligned and n_part > 0:
        cell_pixels = np.maximum(np.asarray(sm, dtype=np.float64), min_smooth) / pixel_min
        if tables is None:
            # only generate the tables that will actually be used
            tables = []
            lower = 0.0
            for max_pixels, resolution in CELL_TABLES:
                if np.any((cell_pixels > lower) & (cell_pixels <= max_pixels)):
                    tables.append((max_pixels,) + tuple(get_cell_projection_cdf_table(orientation, resolution)))
                else:
                    tables.append((max_pixels, None, 1.0, 1.0))
                lower = max_pixels
            # (a table has been generated for every range of sizes containing a cell)
            needs_exact = cell_pixels.max() > max(t[0] for t in CELL_TABLES)
        else:
            tables = list(tables)[:4]
            thresholds = np.array([t[0] for t in tables] + [np.inf])
            available = np.array([t[1] is not None for t in tables] + [False])
            needs_exact = not np.all(available[np.searchsorted(thresholds, cell_pixels, side='left')])
        # Cells are projected exactly unless the first table covering their size is available, and they lie
        # entirely within the z range. Only allocate the buffer for exact projections if it may be needed.
        if needs_exact or np.isfinite(z_lo) or np.isfinite(z_hi):
            dres_np = np.zeros((ny, nx), dtype=np.float64)
            dres = <double *> dres_np.data
        for max_pixels, table_np, bx, by in tables:
            table_max_pixels[n_tables] = max_pixels
            if table_np is None:
                table_data[n_tables] = NULL
            else:
                table_np = np.ascontiguousarray(table_np, dtype=np.float64)
                table_refs.append(table_np)
                table_data[n_tables] = <double *> (<np.ndarray> table_np).data
                table_n[n_tables] = table_np.shape[0] - 1
                table_bx[n_tables] = bx
                table_by[n_tables] = by
                table_sx[n_tables] = table_n[n_tables] / (2.0 * bx)
                table_sy[n_tables] = table_n[n_tables] / (2.0 * by)
            n_tables += 1

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
                    if not (sm_i > 0 and isfinite(sm_i) and isfinite(x_i) and isfinite(y_i) and isfinite(z_i)):
                        # cells with undefined position or size cannot be rendered
                        continue
                    if sm_i < pixel_dx * smooth_lo or sm_i > pixel_dx * smooth_hi:
                        continue

                    # extent of the cell in the image frame
                    bx1 = x_i + ext_x * sm_i
                    bx0 = x_i - ext_x * sm_i
                    if bx1 <= x1 or bx0 >= x2:
                        continue
                    by1 = y_i + ext_y * sm_i
                    by0 = y_i - ext_y * sm_i
                    if by1 <= y1 or by0 >= y2:
                        continue

                    if projected:
                        if z_i + ext_z * sm_i <= z_lo_d or z_i - ext_z * sm_i >= z_hi_d:
                            continue
                        clip_z = z_i - ext_z * sm_i < z_lo_d or z_i + ext_z * sm_i > z_hi_d
                    else:
                        # half-open interval, so that a slice exactly along a cell face does not double count
                        if z_i - ext_z * sm_i > z0_d or z_i + ext_z * sm_i <= z0_d:
                            continue

                    half = 0.5 * sm_i
                    tz = z0_d - z_i

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
                    # (the tests above guarantee the cell overlaps the image, so these are within range of an int unless
                    # the cell is enormous; clamp to avoid overflow in that case)
                    X0 = (bx0 - x1) * inv_pixel_dx
                    X1 = (bx1 - x1) * inv_pixel_dx
                    Y0 = (by0 - y1) * inv_pixel_dy
                    Y1 = (by1 - y1) * inv_pixel_dy
                    ix0 = _floor_int(X0) if X0 > -1.0 else -1
                    ix1 = _floor_int(X1) if X1 < nx else nx
                    iy0 = _floor_int(Y0) if Y0 > -1.0 else -1
                    iy1 = _floor_int(Y1) if Y1 < ny else ny
                    if ix0 >= nx or iy0 >= ny or ix1 < 0 or iy1 < 0:
                        # despite the tests above, round-off can place a cell just outside the image
                        continue

                    if ix0 == ix1 and iy0 == iy1 and (projected or aligned):
                        # the whole cell lies within a single pixel (which is within the image, given the tests
                        # above), so its full volume (or, for an aligned slice, cross-section) is deposited there.
                        # The deposit is weight times the fraction of the cell volume (or area) included
                        if not projected:
                            weight /= sm_i
                        elif clip_z:
                            if aligned:
                                weight *= _overlap(z_i - half, z_i + half, z_lo_d, z_hi_d) / sm_i
                            else:
                                weight *= (_quadrant_volume(axes, half, 1e30, 1e30, z_lo_d - z_i, z_hi_d - z_i, 1)
                                           / (sm_i * sm_i * sm_i))
                        res[iy0 * nx + ix0] += weight * inv_pixel_area
                        continue

                    # projection of a rotated cell: find the table appropriate to its size, if any
                    table_index = -1
                    if projected and not aligned and not clip_z:
                        for k in range(n_tables):
                            # (this must match exactly the arithmetic used above to decide whether to allocate dres)
                            if sm_i / pixel_min <= table_max_pixels[k]:
                                if table_data[k] != NULL:
                                    table_index = k
                                break
                        if table_index < 0 and dres == NULL:
                            # should not occur; but rather than fail, fall back on the largest available table
                            for k in range(n_tables - 1, -1, -1):
                                if table_data[k] != NULL:
                                    table_index = k
                                    break
                        if table_index >= 0:
                            tab = table_data[table_index]
                            tn = table_n[table_index]
                            tbx = table_bx[table_index]
                            tby = table_by[table_index]
                            tsx = table_sx[table_index]
                            tsy = table_sy[table_index]
                            inv_sm = 1.0 / sm_i

                    if table_index >= 0 and ix1 - ix0 <= 1 and iy1 - iy0 <= 1:
                        # A small cell, covering at most 2x2 pixels, so with at most one pixel corner (X, Y) lying
                        # within its projection. The fractions of its volume with x <= X (f_x), with y <= Y (f_y) and
                        # with both (f_xy) determine its distribution between the pixels.
                        xi = (x1 + ix1 * pixel_dx - x_i) * inv_sm
                        eta = (y1 + iy1 * pixel_dy - y_i) * inv_sm
                        f_x = _table_lookup(tab, tn, tbx, tby, tsx, tsy, xi, 1e30) if ix1 > ix0 else 1.0
                        f_y = _table_lookup(tab, tn, tbx, tby, tsx, tsy, 1e30, eta) if iy1 > iy0 else 1.0
                        if ix1 > ix0 and iy1 > iy0:
                            f_xy = _table_lookup(tab, tn, tbx, tby, tsx, tsy, xi, eta)
                        else:
                            f_xy = f_x * f_y  # (one of these is unity)
                        weight *= inv_pixel_area
                        if iy0 >= 0:
                            if ix0 >= 0:
                                res[iy0 * nx + ix0] += weight * f_xy
                            if ix1 > ix0 and ix1 < nx and f_y > f_xy:
                                res[iy0 * nx + ix1] += weight * (f_y - f_xy)
                        if iy1 > iy0 and iy1 < ny:
                            if ix0 >= 0 and f_x > f_xy:
                                res[iy1 * nx + ix0] += weight * (f_x - f_xy)
                            if ix1 > ix0 and ix1 < nx and 1.0 - f_x - f_y + f_xy > 0:
                                res[iy1 * nx + ix1] += weight * (1.0 - f_x - f_y + f_xy)
                        continue

                    if ix0 < 0: ix0 = 0
                    if iy0 < 0: iy0 = 0
                    if ix1 > nx - 1: ix1 = nx - 1
                    if iy1 > ny - 1: iy1 = ny - 1

                    # normalise by the cell volume, so that the result for a slice is qty and for a projection is
                    # the integral of qty along the line of sight
                    weight /= sm_i * sm_i * sm_i

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
                        continue

                    if not projected:
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
                        continue

                    if table_index >= 0:
                        # interpolate the tabulated volume fraction F at each pixel corner; each pixel receives the
                        # cell's volume times the difference of F across its corners
                        weight *= sm_i * sm_i * sm_i / pixel_area
                        eta = (y1 + iy0 * pixel_dy - y_i) * inv_sm
                        for ix in range(ix0, ix1 + 2):
                            lower_F[ix] = _table_lookup(tab, tn, tbx, tby, tsx, tsy,
                                                        (x1 + ix * pixel_dx - x_i) * inv_sm, eta)
                        for iy in range(iy0, iy1 + 1):
                            eta = (y1 + (iy + 1) * pixel_dy - y_i) * inv_sm
                            for ix in range(ix0, ix1 + 2):
                                upper_F[ix] = _table_lookup(tab, tn, tbx, tby, tsx, tsy,
                                                            (x1 + ix * pixel_dx - x_i) * inv_sm, eta)
                            for ix in range(ix0, ix1 + 1):
                                value = (upper_F[ix + 1] - upper_F[ix]) - (lower_F[ix + 1] - lower_F[ix])
                                if value > 0:
                                    # (the table is monotonic so value is non-negative, up to round-off)
                                    res[iy * nx + ix] += weight * value
                            swap_F = lower_F
                            lower_F = upper_F
                            upper_F = swap_F
                    elif dres != NULL:
                        # a large cell, or one truncated by the z range: project exactly, visiting only the pixels
                        # near its projected edges. Finite limits for the z range, enough to contain the cell, are
                        # needed when evaluating F.
                        zl = z_lo_d - z_i
                        zh = z_hi_d - z_i
                        if zl < -2.0 * ext_z * sm_i: zl = -2.0 * ext_z * sm_i
                        if zh > 2.0 * ext_z * sm_i: zh = 2.0 * ext_z * sm_i
                        _project_rotated_cell_rows(dres, nx, x1, y1, pixel_dx, pixel_dy, ix0, ix1, iy0, iy1,
                                                   x_i, y_i, half, axes, zl, zh, clip_z, weight)

    if dres != NULL:
        # Recover the image from its second differences; values below zero can only arise from round-off. This is
        # done in place to avoid temporary copies of the image (the buffer must be double precision, since round-off
        # errors in the second differences are amplified by the cumulative sums).
        np.cumsum(dres_np, axis=1, out=dres_np)
        np.cumsum(dres_np, axis=1, out=dres_np)
        np.maximum(dres_np, 0.0, out=dres_np)
        result += dres_np

    return result
