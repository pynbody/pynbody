"""Exact perspective projections of AMR cells.

The camera is at (mid_x, mid_y, z_camera), where (mid_x, mid_y) is the centre of the image, and looks along the z axis
towards z = 0 (or beyond). As for :func:`pynbody.imaging._render.render_image`, the image coordinates (X, Y) of a ray
are those of the point where it crosses the plane z = 0, so that the image covers [x1, x2] x [y1, y2] in that plane.
A point p is mapped to

    X = mid_x + (p_x - mid_x) / w,   Y = mid_y + (p_y - mid_y) / w,   with w = (z_camera - p_z) / z_camera,

so that w is proportional to the distance from the camera along the z axis (and is 1 at z = 0).

The value in each pixel is the average over the pixel of the line integral of qty along the ray. This is computed
exactly. A projective map takes planes to planes, so on each face of a cell, 1/w is an affine function of (X, Y). The
depth range of a ray through a cell, |z_camera| (w_far - w_near), is therefore a sum over faces of +/- |z_camera| / (affine
function), and its integral over a pixel is a sum of integrals of 1 / (affine function) over convex polygons, which
have a closed form. Since neighbouring cells share faces, which contribute with opposite signs, cells tile exactly.
The depth range along z is converted to a length along the ray by multiplying by the secant of the angle between
the ray and the z axis; within each pixel, this is approximated by its quadratic Taylor expansion about the pixel
centre, which is integrated together with the depth (with errors of third order in the size of a pixel relative to
the distance to the camera, and in the size of the polygons relative to their distance).

The Python interface is :class:`pynbody.imaging.renderers.AMRImageRenderer`.
"""

import numpy as np

cimport cython
cimport numpy as np

np.import_array()
from libc.math cimport NAN, fabs, isfinite, log, log1p, sqrt


cdef enum:
    MAX_POLY = 24  # maximum number of vertices of a clipped polygon


@cython.cdivision(True)
cdef inline double _log1p_over_u(double u) noexcept nogil:
    """log(1 + u) / u, accurate also for small u"""
    if fabs(u) < 1e-4:
        return 1.0 - u * (0.5 - u * (1.0 / 3.0 - 0.25 * u))
    return log1p(u) / u


@cython.cdivision(True)
cdef inline double _dd_xlogx(double a, double b) noexcept nogil:
    """The divided difference (b log b - a log a) / (b - a), for 0 < a <= b, accurate also for b close to a"""
    cdef double u = (b - a) / a
    return log(a) + (1.0 + u) * _log1p_over_u(u)


@cython.cdivision(True)
cdef double _triangle_mean_inverse(double l1, double l2, double l3) noexcept nogil:
    """The mean of 1/L over a triangle on which L is an affine function taking values l1, l2, l3 at the vertices.

    The values must all have the same sign. By the Hermite-Genocchi formula, the mean of g(L) over a triangle is twice
    the second divided difference of G at (l1, l2, l3), where G'' = g; here, G = L log L. For nearly equal values, a
    series expansion about the mean is used instead."""
    cdef double m = (l1 + l2 + l3) / 3.0
    cdef double d1 = l1 / m - 1.0, d2 = l2 / m - 1.0, d3 = l3 / m - 1.0
    cdef double e = fabs(d1), a, b, c, t, hk, hx, hxy, total
    cdef int k
    if fabs(d2) > e: e = fabs(d2)
    if fabs(d3) > e: e = fabs(d3)
    if e < 2e-2:
        # With d_i the relative deviations from the mean (which sum to zero), and t the barycentric coordinates, the
        # mean of 1/L is (1/m) sum_k (-1)^k <(t.d)^k>, where for t uniform over the simplex,
        # <(t.d)^k> = 2 h_k(d) / ((k + 1)(k + 2)), with h_k the complete homogeneous symmetric polynomial. The terms
        # are of order e^k, so stopping at k = 6 gives a relative error of order 1e-12.
        hx = 1.0   # h_k(d1)
        hxy = 1.0  # h_k(d1, d2)
        hk = 1.0   # h_k(d1, d2, d3)
        total = 1.0
        for k in range(1, 7):
            hx = hx * d1
            hxy = hx + d2 * hxy
            hk = hxy + d3 * hk
            if k >= 2:
                if k % 2 == 0:
                    total += 2.0 * hk / ((k + 1) * (k + 2))
                else:
                    total -= 2.0 * hk / ((k + 1) * (k + 2))
        return total / m
    a = 1.0 + d1; b = 1.0 + d2; c = 1.0 + d3
    # sort so that a <= b <= c
    if a > b: t = a; a = b; b = t
    if b > c: t = b; b = c; c = t
    if a > b: t = a; a = b; b = t
    # c - a >= e >= 2e-2, so the final division loses at most two significant figures
    return 2.0 * (_dd_xlogx(b, c) - _dd_xlogx(a, b)) / (c - a) / m


@cython.cdivision(True)
cdef double _polygon_integral(const double *px, const double *py, int n,
                              double lx, double ly, double l0, double K, double w_ref,
                              double s0, double gx, double gy, double hxx, double hxy, double hyy,
                              double Xp, double Yp) noexcept nogil:
    """The integral over a convex polygon of S(X, Y) (K / L(X, Y) - w_ref), where L(X, Y) = lx X + ly Y + l0 and S is
    the quadratic s0 + g . u + u . H . u / 2, with u = (X - Xp, Y - Yp).

    On each triangle of a fan, the integral of S/L is approximated as the product of the means of S and of 1/L, plus
    the first-order approximation to their covariance, -(grad S . M . grad L) / L_c^2, where M is the covariance
    matrix of position over the triangle and L_c the value of L at the centroid. The error is third order in the size
    of the triangle relative to the scales over which S and L vary. The mean of S is exact, and so is the term in
    w_ref."""
    cdef int i
    cdef double area, total = 0.0, cx, cy, sT, m, ux0, uy0, ux1, uy1, ux2, uy2, mxx, mxy, myy, dsx, dsy
    cdef double L0, L1, L2
    if n < 3:
        return 0.0
    L0 = lx * px[0] + ly * py[0] + l0
    L1 = lx * px[1] + ly * py[1] + l0
    for i in range(1, n - 1):
        L2 = lx * px[i + 1] + ly * py[i + 1] + l0
        area = 0.5 * fabs((px[i] - px[0]) * (py[i + 1] - py[0]) - (px[i + 1] - px[0]) * (py[i] - py[0]))
        if area > 0:
            cx = (px[0] + px[i] + px[i + 1]) / 3.0
            cy = (py[0] + py[i] + py[i + 1]) / 3.0
            m = (L0 + L1 + L2) / 3.0
            ux0 = px[0] - cx; uy0 = py[0] - cy
            ux1 = px[i] - cx; uy1 = py[i] - cy
            ux2 = px[i + 1] - cx; uy2 = py[i + 1] - cy
            mxx = (ux0 * ux0 + ux1 * ux1 + ux2 * ux2) / 12.0
            mxy = (ux0 * uy0 + ux1 * uy1 + ux2 * uy2) / 12.0
            myy = (uy0 * uy0 + uy1 * uy1 + uy2 * uy2) / 12.0
            cx -= Xp
            cy -= Yp
            # mean of S over the triangle, and its gradient at the centroid
            sT = s0 + gx * cx + gy * cy + 0.5 * (hxx * (cx * cx + mxx) + 2.0 * hxy * (cx * cy + mxy)
                                                 + hyy * (cy * cy + myy))
            dsx = gx + hxx * cx + hxy * cy
            dsy = gy + hxy * cx + hyy * cy
            total += area * (K * (sT * _triangle_mean_inverse(L0, L1, L2)
                                  - (dsx * (mxx * lx + mxy * ly) + dsy * (mxy * lx + myy * ly)) / (m * m))
                             - w_ref * sT)
        L1 = L2
    return total


@cython.cdivision(True)
cdef int _clip_polygon(const double *px, const double *py, int n, double a, double b, double c,
                       double *ox, double *oy) noexcept nogil:
    """Clip the convex polygon (px, py) of n vertices to the half-plane a x + b y <= c (Sutherland-Hodgman)"""
    cdef int i, j, m = 0
    cdef double d0, d1, t
    if n == 0:
        return 0
    d0 = a * px[n-1] + b * py[n-1] - c
    for i in range(n):
        j = i - 1 if i > 0 else n - 1
        d1 = a * px[i] + b * py[i] - c
        if (d0 <= 0) != (d1 <= 0):
            t = d0 / (d0 - d1)
            if m < MAX_POLY:
                ox[m] = px[j] + t * (px[i] - px[j])
                oy[m] = py[j] + t * (py[i] - py[j])
                m += 1
        if d1 <= 0 and m < MAX_POLY:
            ox[m] = px[i]
            oy[m] = py[i]
            m += 1
        d0 = d1
    return m


@cython.cdivision(True)
cdef int _clip_polygon_3d(const double *px, const double *py, const double *pz, int n,
                          double ga, double gb, double gc, double gd,
                          double *ox, double *oy, double *oz) noexcept nogil:
    """Clip the convex 3d polygon (px, py, pz) of n vertices to the half-space ga x + gb y + gc z + gd >= 0"""
    cdef int i, j, m = 0
    cdef double d0, d1, t
    if n == 0:
        return 0
    d0 = ga * px[n-1] + gb * py[n-1] + gc * pz[n-1] + gd
    for i in range(n):
        j = i - 1 if i > 0 else n - 1
        d1 = ga * px[i] + gb * py[i] + gc * pz[i] + gd
        if (d0 >= 0) != (d1 >= 0):
            t = d0 / (d0 - d1)
            if m < MAX_POLY:
                ox[m] = px[j] + t * (px[i] - px[j])
                oy[m] = py[j] + t * (py[i] - py[j])
                oz[m] = pz[j] + t * (pz[i] - pz[j])
                m += 1
        if d1 >= 0 and m < MAX_POLY:
            ox[m] = px[i]
            oy[m] = py[i]
            oz[m] = pz[i]
            m += 1
        d0 = d1
    return m


@cython.cdivision(True)
cdef int _cube_cross_section(const double *axes, double half, double zc, double *px, double *py) noexcept nogil:
    """The cross-section of the cube (centred on the origin) in the plane z = zc, as a polygon of (x, y) vertices
    written into px, py. Returns the number of vertices."""
    cdef double buf_x[MAX_POLY]
    cdef double buf_y[MAX_POLY]
    cdef double ex, ey, ez
    cdef double bound_x = half * (fabs(axes[0]) + fabs(axes[3]) + fabs(axes[6])) * 1.01
    cdef double bound_y = half * (fabs(axes[1]) + fabs(axes[4]) + fabs(axes[7])) * 1.01
    cdef int k, n = 4
    px[0] = -bound_x; py[0] = -bound_y
    px[1] = bound_x; py[1] = -bound_y
    px[2] = bound_x; py[2] = bound_y
    px[3] = -bound_x; py[3] = bound_y
    for k in range(3):
        ex = axes[3 * k]
        ey = axes[3 * k + 1]
        ez = axes[3 * k + 2]
        if fabs(ex) < 1e-12 and fabs(ey) < 1e-12:
            if fabs(ez * zc) > half:
                return 0
            continue
        n = _clip_polygon(px, py, n, ex, ey, half - ez * zc, buf_x, buf_y)
        n = _clip_polygon(buf_x, buf_y, n, -ex, -ey, half + ez * zc, px, py)
    return n if n >= 3 else 0


cdef inline int _floor_int(double v) noexcept nogil:
    cdef int i = <int> v
    return i - 1 if v < i else i


cdef inline int _ceil_int(double v) noexcept nogil:
    cdef int i = <int> v
    return i + 1 if v > i else i


@cython.cdivision(True)
cdef void _polygon_extent_at(const double *px, const double *py, int n, double y, double *lo, double *hi) noexcept nogil:
    """The range of x covered by the convex polygon (px, py) along the line at height y, which must lie within its
    range of y"""
    cdef int i, j
    cdef double xi
    lo[0] = 1e300
    hi[0] = -1e300
    for i in range(n):
        j = (i + 1) % n
        if (py[i] <= y <= py[j]) or (py[j] <= y <= py[i]):
            if py[i] == py[j]:
                if px[i] < lo[0]: lo[0] = px[i]
                if px[i] > hi[0]: hi[0] = px[i]
                if px[j] < lo[0]: lo[0] = px[j]
                if px[j] > hi[0]: hi[0] = px[j]
            else:
                xi = px[i] + (y - py[i]) * (px[j] - px[i]) / (py[j] - py[i])
                if xi < lo[0]: lo[0] = xi
                if xi > hi[0]: hi[0] = xi


@cython.cdivision(True)
cdef int _split_polygon(const double *px, const double *py, int n, double a, double b, double c,
                        double *ix_, double *iy_, int *n_in, double *ox, double *oy) noexcept nogil:
    """Split the convex polygon (px, py) by the line a x + b y = c, writing the part with a x + b y <= c into
    (ix_, iy_) (with its number of vertices in n_in) and the rest into (ox, oy). Returns the number of vertices of
    the latter."""
    cdef int i, j, m_in = 0, m_out = 0
    cdef double d0, d1, t, xc, yc
    if n == 0:
        n_in[0] = 0
        return 0
    d0 = a * px[n-1] + b * py[n-1] - c
    for i in range(n):
        j = i - 1 if i > 0 else n - 1
        d1 = a * px[i] + b * py[i] - c
        if (d0 <= 0) != (d1 <= 0):
            t = d0 / (d0 - d1)
            xc = px[j] + t * (px[i] - px[j])
            yc = py[j] + t * (py[i] - py[j])
            if m_in < MAX_POLY:
                ix_[m_in] = xc; iy_[m_in] = yc; m_in += 1
            if m_out < MAX_POLY:
                ox[m_out] = xc; oy[m_out] = yc; m_out += 1
        if d1 <= 0:
            if m_in < MAX_POLY:
                ix_[m_in] = px[i]; iy_[m_in] = py[i]; m_in += 1
        elif m_out < MAX_POLY:
            ox[m_out] = px[i]; oy[m_out] = py[i]; m_out += 1
        d0 = d1
    n_in[0] = m_in
    return m_out


@cython.cdivision(True)
cdef double _piece_integral(const double *px, const double *py, int n, double Xp, double Yp,
                            double lx, double ly, double l0, double K, double w_ref,
                            double s0, double gx, double gy, double hxx, double hxy, double hyy) noexcept nogil:
    """The integral over a convex polygon, lying within the pixel centred on (Xp, Yp), of S (K / L - w_ref), where
    L = lx X + ly Y + l0, and S = s0 + g.u + u.H.u / 2 with u = (X - Xp, Y - Yp).

    Where L varies by less than 1% across the polygon, 1/L is expanded to third order about the pixel centre, and
    integrated using the moments of the polygon up to third order (from Green's theorem). Terms in the products of S
    with this expansion are kept to the same overall order. The relative error is of order 1e-8. Otherwise, the
    integral is calculated by _polygon_integral."""
    cdef int i, j
    cdef double lam = lx * Xp + ly * Yp + l0
    cdef double x0, y0, x1, y1, cr, e, v, emax = 0.0
    cdef double A = 0.0, Mx = 0.0, My = 0.0, Mxx = 0.0, Mxy = 0.0, Myy = 0.0
    cdef double Mxxx = 0.0, Mxxy = 0.0, Mxyy = 0.0, Myyy = 0.0
    cdef double inv, V1, V2, V3, G1, GV, GVV, H2, HV, main
    if n < 3:
        return 0.0
    for i in range(n):
        v = fabs(lx * (px[i] - Xp) + ly * (py[i] - Yp))
        if v > emax:
            emax = v
    if emax >= 1e-2 * fabs(lam):
        return _polygon_integral(px, py, n, lx, ly, l0, K, w_ref, s0, gx, gy, hxx, hxy, hyy, Xp, Yp)
    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        x0 = px[i] - Xp; y0 = py[i] - Yp
        x1 = px[j] - Xp; y1 = py[j] - Yp
        cr = x0 * y1 - x1 * y0
        A += cr
        Mx += (x0 + x1) * cr
        My += (y0 + y1) * cr
        Mxx += (x0 * x0 + x0 * x1 + x1 * x1) * cr
        Myy += (y0 * y0 + y0 * y1 + y1 * y1) * cr
        Mxy += (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0) * cr
        Mxxx += (x0 * x0 * x0 + x0 * x0 * x1 + x0 * x1 * x1 + x1 * x1 * x1) * cr
        Myyy += (y0 * y0 * y0 + y0 * y0 * y1 + y0 * y1 * y1 + y1 * y1 * y1) * cr
        Mxxy += (x0 * x0 * (3.0 * y0 + y1) + 2.0 * x0 * x1 * (y0 + y1) + x1 * x1 * (y0 + 3.0 * y1)) * cr
        Mxyy += (y0 * y0 * (3.0 * x0 + x1) + 2.0 * y0 * y1 * (x0 + x1) + y1 * y1 * (x0 + 3.0 * x1)) * cr
    A /= 2.0; Mx /= 6.0; My /= 6.0; Mxx /= 12.0; Myy /= 12.0; Mxy /= 24.0
    Mxxx /= 20.0; Myyy /= 20.0; Mxxy /= 60.0; Mxyy /= 60.0
    if A < 0:
        A = -A; Mx = -Mx; My = -My; Mxx = -Mxx; Myy = -Myy; Mxy = -Mxy
        Mxxx = -Mxxx; Myyy = -Myyy; Mxxy = -Mxxy; Mxyy = -Mxyy
    inv = 1.0 / lam
    # moments of v = lx x + ly y, and of its products with g.u and u.H.u
    V1 = lx * Mx + ly * My
    V2 = lx * lx * Mxx + 2.0 * lx * ly * Mxy + ly * ly * Myy
    V3 = lx * lx * lx * Mxxx + 3.0 * lx * lx * ly * Mxxy + 3.0 * lx * ly * ly * Mxyy + ly * ly * ly * Myyy
    G1 = gx * Mx + gy * My
    GV = gx * (lx * Mxx + ly * Mxy) + gy * (lx * Mxy + ly * Myy)
    GVV = gx * (lx * lx * Mxxx + 2.0 * lx * ly * Mxxy + ly * ly * Mxyy) + \
          gy * (lx * lx * Mxxy + 2.0 * lx * ly * Mxyy + ly * ly * Myyy)
    H2 = hxx * Mxx + 2.0 * hxy * Mxy + hyy * Myy
    HV = hxx * (lx * Mxxx + ly * Mxxy) + 2.0 * hxy * (lx * Mxxy + ly * Mxyy) + hyy * (lx * Mxyy + ly * Myyy)
    # 1/L = (1/lam) (1 - v/lam + v^2/lam^2 - v^3/lam^3 + ...)
    main = s0 * (A - inv * (V1 - inv * (V2 - inv * V3))) + (G1 - inv * (GV - inv * GVV)) + 0.5 * (H2 - inv * HV)
    return K * inv * main - w_ref * (s0 * A + G1 + 0.5 * H2)


cdef struct View:
    # geometry of the image and camera, shared by all cells
    int nx, ny
    double x1, y1, pixel_dx, pixel_dy
    double mid_x, mid_y, z_camera
    # frustum planes, expanded by a pixel, as g(p) = a x + b y + c z + d >= 0
    double frustum[4][4]


@cython.cdivision(True)
cdef inline double _secant_model(const View *view, int ix, int iy, double *gx, double *gy, double *hxx, double *hxy,
                                 double *hyy) noexcept nogil:
    """The secant of the angle between the ray through the centre of pixel (ix, iy) and the z axis, which converts depth
    into length along the ray. Its derivatives with respect to X and Y are written into gx, gy (first) and hxx, hxy,
    hyy (second), and its mean over the pixel, to second order, is returned."""
    cdef double tx = (view.x1 + (ix + 0.5) * view.pixel_dx - view.mid_x) / view.z_camera
    cdef double ty = (view.y1 + (iy + 0.5) * view.pixel_dy - view.mid_y) / view.z_camera
    cdef double s0 = sqrt(1.0 + tx * tx + ty * ty)
    cdef double c = 1.0 / (view.z_camera * view.z_camera * s0 * s0 * s0)
    gx[0] = tx / (view.z_camera * s0)
    gy[0] = ty / (view.z_camera * s0)
    hxx[0] = (1.0 + ty * ty) * c
    hxy[0] = -tx * ty * c
    hyy[0] = (1.0 + tx * tx) * c
    return s0 + (hxx[0] * view.pixel_dx * view.pixel_dx + hyy[0] * view.pixel_dy * view.pixel_dy) / 24.0


@cython.cdivision(True)
cdef inline double _full_pixel_value(const View *view, double Xp, double Yp, double lx, double ly, double l0,
                                     double K, double w_ref, double cx2, double cy2) noexcept nogil:
    """The integral over the whole pixel centred on (Xp, Yp) of S (K / L - w_ref), divided by the mean of S over the
    pixel, where L = lx X + ly Y + l0 and S is the secant of the angle of the ray to the z axis.

    cx2 and cy2 are (lx dx)^2 and (ly dy)^2. The caller must ensure that these are less than 1e-4 L^2, so that the
    series expansion of 1/L converges quickly."""
    cdef double dx = view.pixel_dx, dy = view.pixel_dy
    cdef double tx = (Xp - view.mid_x) / view.z_camera
    cdef double ty = (Yp - view.mid_y) / view.z_camera
    # Evaluating the mean of 1/L at an offset of (dx^2 / 12, dy^2 / 12) * grad S / S from the pixel centre accounts
    # for the covariance of S with 1/L across the pixel, to first order
    cdef double shift = 1.0 / (12.0 * view.z_camera * (1.0 + tx * tx + ty * ty))
    cdef double inv = 1.0 / (lx * (Xp + dx * dx * tx * shift) + ly * (Yp + dy * dy * ty * shift) + l0)
    cdef double ex2 = cx2 * inv * inv, ey2 = cy2 * inv * inv
    # with u, v uniform over [-1/2, 1/2], the mean of 1/(1 + ex u + ey v), to fourth order
    return dx * dy * (K * inv * (1.0 + (ex2 + ey2) / 12.0 + (ex2 * ex2 + ey2 * ey2) / 80.0 + ex2 * ey2 / 24.0)
                      - w_ref)


@cython.cdivision(True)
cdef inline double _edge_pixel_value(const View *view, int ix, int iy, const double *px, const double *py, int n,
                                     double lx, double ly, double l0, double K, double w_ref) noexcept nogil:
    """The integral over the polygon (px, py), which lies within pixel (ix, iy), of S (K / L - w_ref), divided by
    the mean of S over the pixel; see _full_pixel_value"""
    cdef double gx, gy, hxx, hxy, hyy
    cdef double mean_S = _secant_model(view, ix, iy, &gx, &gy, &hxx, &hxy, &hyy)
    cdef double s0 = mean_S - (hxx * view.pixel_dx * view.pixel_dx + hyy * view.pixel_dy * view.pixel_dy) / 24.0
    return _piece_integral(px, py, n, view.x1 + (ix + 0.5) * view.pixel_dx, view.y1 + (iy + 0.5) * view.pixel_dy,
                           lx, ly, l0, K, w_ref, s0, gx, gy, hxx, hxy, hyy) / mean_S


@cython.cdivision(True)
cdef void _add_face(const View *view, double *fx, double *fy, double *fz, int n,
                    double nx_, double ny_, double nz_, double d, int clip_frustum, double w_ref, double scale,
                    double *acc) noexcept nogil:
    """Add the contribution of one face of a cell to the image accumulator.

    The face is the convex polygon (fx, fy, fz) of n vertices, lying in the plane n.p = d, where n is the outward
    normal of the cell. For each pixel, the contribution is the integral over the pixel of the secant of the angle of
    the ray to the z axis, S, times the depth (in units of |z_camera| w) at which the ray crosses the face relative to
    w_ref, divided by the mean of S over the pixel. It is multiplied by *scale*, and by +1 for faces through which rays
    leave the cell or -1 for faces through which they enter.

    S is approximated within each pixel by its quadratic Taylor expansion about the pixel centre, which is the same for
    all faces, so that the w_ref terms cancel exactly between the faces of a cell. For pixels on the edge of the
    projected face, the integral is calculated over the part of the pixel covered by the face, as for
    _polygon_integral. For pixels entirely within the face, this simplifies to the mean over the pixel of the depth,
    evaluated by its series expansion, at a point offset from the pixel centre to account for the covariance of S with
    the depth."""
    cdef double ox[MAX_POLY]
    cdef double oy[MAX_POLY]
    cdef double oz[MAX_POLY]
    cdef double PX[MAX_POLY]
    cdef double PY[MAX_POLY]
    cdef double qx[MAX_POLY]
    cdef double qy[MAX_POLY]
    cdef double rx[MAX_POLY]
    cdef double ry[MAX_POLY]
    cdef double f, K, w, lx, ly, l0, X0, X1, Y0, Y1, Xmin, Xmax, Ymin, Ymax, area2, value
    cdef double Xp, Yp, tx, ty, mean_S, gx, gy, hxx, hxy, hyy, L0, L1, R0, R1
    cdef double lam, cx2, cy2, s0, Smin, Smax
    cdef double sx[MAX_POLY]
    cdef double sy[MAX_POLY]
    cdef int jx0, jx1, mq
    cdef double dx = view.pixel_dx, dy = view.pixel_dy
    cdef int i, j, k, m, ix, iy, ix0, ix1, iy0, iy1, in0, in1
    cdef double *acc_row

    if clip_frustum:
        for k in range(4):
            m = _clip_polygon_3d(fx, fy, fz, n, view.frustum[k][0], view.frustum[k][1], view.frustum[k][2],
                                 view.frustum[k][3], ox, oy, oz)
            for i in range(m):
                fx[i] = ox[i]; fy[i] = oy[i]; fz[i] = oz[i]
            n = m
            if n < 3:
                return

    # which side of the face is the camera on?
    f = nx_ * view.mid_x + ny_ * view.mid_y + nz_ * view.z_camera - d
    if fabs(f) <= 1e-12 * (fabs(d) + fabs(view.mid_x) + fabs(view.mid_y) + fabs(view.z_camera)
                           + fabs(fx[0]) + fabs(fy[0]) + fabs(fz[0])):
        # the face is seen edge-on (to within round-off, relative to the scale of the coordinates), and so
        # contributes nothing
        return
    # rays enter the cell through faces whose outer side faces the camera
    if f > 0:
        scale = -scale

    # Along the ray through (X, Y), the plane is crossed at w = K / (lx X + ly Y + l0)
    K = -f
    lx = nx_
    ly = ny_
    l0 = -nx_ * view.mid_x - ny_ * view.mid_y - nz_ * view.z_camera
    cx2 = lx * dx * lx * dx
    cy2 = ly * dy * ly * dy

    # project the vertices
    Xmin = Ymin = 1e300
    Xmax = Ymax = -1e300
    for i in range(n):
        w = (view.z_camera - fz[i]) / view.z_camera
        PX[i] = view.mid_x + (fx[i] - view.mid_x) / w
        PY[i] = view.mid_y + (fy[i] - view.mid_y) / w
        if PX[i] < Xmin: Xmin = PX[i]
        if PX[i] > Xmax: Xmax = PX[i]
        if PY[i] < Ymin: Ymin = PY[i]
        if PY[i] > Ymax: Ymax = PY[i]

    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area2 += PX[i] * PY[j] - PX[j] * PY[i]
    if area2 == 0.0:
        return

    ix0 = <int> ((Xmin - view.x1) / dx) if Xmin > view.x1 else 0
    ix1 = <int> ((Xmax - view.x1) / dx) if Xmax > view.x1 else -1
    iy0 = <int> ((Ymin - view.y1) / dy) if Ymin > view.y1 else 0
    iy1 = <int> ((Ymax - view.y1) / dy) if Ymax > view.y1 else -1
    if ix1 >= view.nx: ix1 = view.nx - 1
    if iy1 >= view.ny: iy1 = view.ny - 1

    if ix0 == ix1 and iy0 == iy1 and Xmin >= view.x1 and Ymin >= view.y1 and \
            Xmax <= view.x1 + view.nx * dx and Ymax <= view.y1 + view.ny * dy:
        # the face lies within a single pixel
        Xp = view.x1 + (ix0 + 0.5) * dx
        Yp = view.y1 + (iy0 + 0.5) * dy
        mean_S = _secant_model(view, ix0, iy0, &gx, &gy, &hxx, &hxy, &hyy)
        s0 = mean_S - (hxx * dx * dx + hyy * dy * dy) / 24.0
        acc[<size_t> iy0 * view.nx + ix0] += scale * _piece_integral(PX, PY, n, Xp, Yp, lx, ly, l0, K, w_ref,
                                                                    s0, gx, gy, hxx, hxy, hyy) / mean_S
        return

    for iy in range(iy0, iy1 + 1):
        Y0 = view.y1 + iy * dy
        Y1 = Y0 + dy
        Yp = 0.5 * (Y0 + Y1)
        acc_row = acc + <size_t> iy * view.nx

        # the part of the polygon within this row
        m = _clip_polygon(PX, PY, n, 0.0, -1.0, -Y0, qx, qy)
        m = _clip_polygon(qx, qy, m, 0.0, 1.0, Y1, sx, sy)
        if m < 3:
            continue
        Smin = Smax = sx[0]
        for i in range(1, m):
            if sx[i] < Smin: Smin = sx[i]
            if sx[i] > Smax: Smax = sx[i]
        jx0 = _floor_int((Smin - view.x1) / dx)
        jx1 = _floor_int((Smax - view.x1) / dx)
        if jx0 < 0:
            # discard the part of the strip to the left of the image
            jx0 = 0
            m = _split_polygon(sx, sy, m, 1.0, 0.0, view.x1, qx, qy, &mq, rx, ry)
            for j in range(m):
                sx[j] = rx[j]; sy[j] = ry[j]
        if jx1 >= view.nx:
            # and to the right
            jx1 = view.nx - 1
            m = _split_polygon(sx, sy, m, 1.0, 0.0, view.x1 + view.nx * dx, qx, qy, &mq, rx, ry)
            for j in range(mq):
                sx[j] = qx[j]; sy[j] = qy[j]
            m = mq
        if jx0 > jx1 or m < 3:
            continue

        # the range of pixels in this row lying entirely within the polygon
        in0 = jx1 + 1
        in1 = jx0 - 1
        if Y0 >= Ymin and Y1 <= Ymax:
            _polygon_extent_at(PX, PY, n, Y0, &L0, &R0)
            _polygon_extent_at(PX, PY, n, Y1, &L1, &R1)
            if L1 > L0: L0 = L1
            if R1 < R0: R0 = R1
            if R0 > L0:
                in0 = _ceil_int((L0 - view.x1) / dx)
                in1 = _floor_int((R0 - view.x1) / dx) - 1
                if in0 < jx0: in0 = jx0
                if in1 > jx1: in1 = jx1
        if in0 <= in1:
            # The series expansion for the mean of 1/L over a pixel must converge quickly; since L is monotonic along
            # the row, it suffices to check the ends of the range
            for i in range(2):
                Xp = view.x1 + ((in0 if i == 0 else in1) + 0.5) * dx
                lam = lx * Xp + ly * Yp + l0
                if (cx2 + cy2) > 1e-4 * lam * lam:
                    in0 = jx1 + 1
                    in1 = jx0 - 1
                    break
        if in0 > in1:
            # no interior pixels: the whole row is peeled off from the left
            in0 = jx1 + 1
            in1 = jx1

        # pixels at the left end of the row, split successively from the strip
        for ix in range(jx0, in0):
            X1 = view.x1 + (ix + 1) * dx
            if ix < jx1:
                m = _split_polygon(sx, sy, m, 1.0, 0.0, X1, qx, qy, &mq, rx, ry)
                for j in range(m):
                    sx[j] = rx[j]; sy[j] = ry[j]
            else:
                mq = m
                for j in range(m):
                    qx[j] = sx[j]; qy[j] = sy[j]
            if mq >= 3:
                acc_row[ix] += scale * _edge_pixel_value(view, ix, iy, qx, qy, mq, lx, ly, l0, K, w_ref)

        # pixels at the right end of the row, split successively from the strip
        for ix in range(jx1, in1, -1):
            X0 = view.x1 + ix * dx
            m = _split_polygon(sx, sy, m, -1.0, 0.0, -X0, qx, qy, &mq, rx, ry)
            for j in range(m):
                sx[j] = rx[j]; sy[j] = ry[j]
            if mq >= 3:
                acc_row[ix] += scale * _edge_pixel_value(view, ix, iy, qx, qy, mq, lx, ly, l0, K, w_ref)

        # pixels entirely within the polygon
        for ix in range(in0, in1 + 1):
            Xp = view.x1 + (ix + 0.5) * dx
            acc_row[ix] += scale * _full_pixel_value(view, Xp, Yp, lx, ly, l0, K, w_ref, cx2, cy2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def render_image_cells_perspective(int nx, int ny,
                                   const double[::1] x, const double[::1] y, const double[::1] z,
                                   const double[::1] sm,
                                   double x1, double x2, double y1, double y2, double z_camera,
                                   const double[::1] qty, const double[::1] mass, const double[::1] rho,
                                   double smooth_lo, double smooth_hi, double z_lo, double z_hi,
                                   double min_smooth, orientation,
                                   wrap_offsets_x=(0.0,), wrap_offsets_y=(0.0,)):
    """Render a perspective projection of cubic (AMR) cells.

    The camera is at z = z_camera, above the centre of the image; see the module docstring for the geometry. Each cell
    has side length sm, and its edges are parallel to the columns of the 3x3 rotation matrix *orientation*, expressed
    in the image frame. The value in each pixel is the average over the pixel of the integral of the
    piecewise-constant field qty along the ray from the camera, with mass/rho giving the volume of each cell. Cells are
    exactly truncated at z_lo and z_hi. Cells which are not entirely in front of the camera are omitted.
    """
    cdef double pixel_dx = (x2 - x1) / nx
    cdef double pixel_dy = (y2 - y1) / ny
    cdef double pixel_area = pixel_dx * pixel_dy
    cdef double abs_zc = fabs(z_camera)
    cdef int n_part = len(x)
    cdef int i, k, j, a1, a2, s, n, ix, iy, bx0, bx1, by0, by1, clip_z, clip_frustum
    cdef double x_i, y_i, z_i, sm_i, half, weight, w_c, w, Xc, Yc, Xmin, Xmax, Ymin, Ymax, zmin, zmax
    cdef double nx_, ny_, nz_, d, delta, e2, e4, az, value, sec
    cdef double wrap_offset_x, wrap_offset_y
    cdef double axes[9]
    cdef double vx[8]
    cdef double vy[8]
    cdef double vz[8]
    cdef double fx[MAX_POLY]
    cdef double fy[MAX_POLY]
    cdef double fz[MAX_POLY]
    cdef double ox[MAX_POLY]
    cdef double oy[MAX_POLY]
    cdef double oz[MAX_POLY]
    cdef double cx[MAX_POLY]
    cdef double cy[MAX_POLY]
    cdef double margin_x = pixel_dx, margin_y = pixel_dy
    cdef View view
    cdef double gx, gy, hxx, hxy, hyy, scale
    # the vertices of each face, as indices into the 8 vertices (bit k of the index gives the sign along axis k)
    cdef int face_vertices[6][4]
    cdef int corner_order[4][2]

    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros((ny, nx), dtype=np.float32)
    cdef np.float32_t *res = <np.float32_t *> result.data
    # the image is accumulated in double precision, since the contributions of the faces of each cell largely cancel
    cdef np.ndarray[np.float64_t, ndim=2] acc_np = np.zeros((ny, nx), dtype=np.float64)
    cdef double *acc = <double *> acc_np.data

    if z_camera == 0:
        raise ValueError("z_camera must be non-zero for a perspective image")
    if not (len(y) == len(z) == len(sm) == len(qty) == len(mass) == len(rho) == n_part):
        raise ValueError("Inconsistent array lengths passed to render_image_cells_perspective")
    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.shape != (3, 3):
        raise ValueError("orientation must be a 3x3 matrix")
    for k in range(3):
        for j in range(3):
            # axes[3k + j] is component j (in the image frame) of cell axis k
            axes[3 * k + j] = orientation[j, k]

    view.nx = nx
    view.ny = ny
    view.x1 = x1
    view.y1 = y1
    view.pixel_dx = pixel_dx
    view.pixel_dy = pixel_dy
    view.mid_x = 0.5 * (x1 + x2)
    view.mid_y = 0.5 * (y1 + y2)
    view.z_camera = z_camera
    # X >= x1 - margin, i.e. (p_x - mid_x) z_camera - (x1 - margin - mid_x)(z_camera - p_z) >= 0 for z_camera > 0
    # (multiplied through by z_camera, so that the inequality holds whatever the sign of z_camera)
    for k in range(4):
        if k == 0:
            value = x1 - margin_x - view.mid_x
            view.frustum[k][0] = z_camera; view.frustum[k][1] = 0.0
        elif k == 1:
            value = x2 + margin_x - view.mid_x
            view.frustum[k][0] = -z_camera; view.frustum[k][1] = 0.0
        elif k == 2:
            value = y1 - margin_y - view.mid_y
            view.frustum[k][0] = 0.0; view.frustum[k][1] = z_camera
        else:
            value = y2 + margin_y - view.mid_y
            view.frustum[k][0] = 0.0; view.frustum[k][1] = -z_camera
        # the plane passes through the camera, and through (mid + value, z = 0) at w = 1
        if k == 0 or k == 2:
            view.frustum[k][2] = value
            view.frustum[k][3] = -value * z_camera - z_camera * (view.mid_x if k == 0 else view.mid_y)
        else:
            view.frustum[k][2] = -value
            view.frustum[k][3] = value * z_camera + z_camera * (view.mid_x if k == 1 else view.mid_y)
        if z_camera < 0:
            for j in range(4):
                view.frustum[k][j] = -view.frustum[k][j]

    for k in range(3):
        a1 = (k + 1) % 3
        a2 = (k + 2) % 3
        for s in range(2):
            corner_order[0][0] = 0; corner_order[0][1] = 0
            corner_order[1][0] = 1; corner_order[1][1] = 0
            corner_order[2][0] = 1; corner_order[2][1] = 1
            corner_order[3][0] = 0; corner_order[3][1] = 1
            for j in range(4):
                face_vertices[2 * k + s][j] = (s << k) | (corner_order[j][0] << a1) | (corner_order[j][1] << a2)

    if True:
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
                            continue
                        half = 0.5 * sm_i
                        weight /= sm_i * sm_i * sm_i

                        w_c = (z_camera - z_i) / z_camera
                        if w_c <= 0:
                            continue
                        if sm_i / w_c < pixel_dx * smooth_lo or sm_i / w_c > pixel_dx * smooth_hi:
                            continue

                        # vertices, and their extent in z and in the image
                        zmin = zmax = z_i
                        Xmin = Ymin = 1e300
                        Xmax = Ymax = -1e300
                        for j in range(8):
                            vx[j] = x_i
                            vy[j] = y_i
                            vz[j] = z_i
                            for k in range(3):
                                s = 1 if (j >> k) & 1 else -1
                                vx[j] += s * half * axes[3 * k]
                                vy[j] += s * half * axes[3 * k + 1]
                                vz[j] += s * half * axes[3 * k + 2]
                            if vz[j] < zmin: zmin = vz[j]
                            if vz[j] > zmax: zmax = vz[j]
                            w = (z_camera - vz[j]) / z_camera
                            if w <= 0:
                                break
                            Xc = view.mid_x + (vx[j] - view.mid_x) / w
                            Yc = view.mid_y + (vy[j] - view.mid_y) / w
                            if Xc < Xmin: Xmin = Xc
                            if Xc > Xmax: Xmax = Xc
                            if Yc < Ymin: Ymin = Yc
                            if Yc > Ymax: Ymax = Yc
                        if w <= 0:
                            # the cell is not entirely in front of the camera
                            continue
                        if zmax <= z_lo or zmin >= z_hi:
                            continue
                        if Xmax <= x1 or Xmin >= x2 or Ymax <= y1 or Ymin >= y2:
                            continue
                        clip_z = zmin < z_lo or zmax > z_hi
                        clip_frustum = Xmin < x1 - margin_x or Xmax > x2 + margin_x or \
                                       Ymin < y1 - margin_y or Ymax > y2 + margin_y

                        bx0 = <int> ((Xmin - x1) / pixel_dx) if Xmin > x1 else 0
                        bx1 = <int> ((Xmax - x1) / pixel_dx) if Xmax < x2 else nx - 1
                        by0 = <int> ((Ymin - y1) / pixel_dy) if Ymin > y1 else 0
                        by1 = <int> ((Ymax - y1) / pixel_dy) if Ymax < y2 else ny - 1
                        if bx1 >= nx: bx1 = nx - 1
                        if by1 >= ny: by1 = ny - 1
                        if bx0 > bx1 or by0 > by1:
                            continue

                        delta = z_camera - z_i
                        if bx0 == bx1 and by0 == by1 and not clip_z and half < 1e-2 * fabs(delta) and \
                                Xmin >= x1 and Xmax <= x2 and Ymin >= y1 and Ymax <= y2:
                            # The cell lies within a single pixel, so contributes the integral over the image plane
                            # of its depth range, which is the integral over its volume of 1/w^2. Expanding
                            # 1/(delta - u)^2 in the offset u along z from the centre of the cell gives
                            # sum_n (n + 1) <u^n> / delta^(n+2), with only even moments non-zero.
                            e2 = half * half / 3.0  # since the squared z components of the axes sum to one
                            e4 = 0.0
                            for k in range(3):
                                az = axes[3 * k + 2] * axes[3 * k + 2]
                                e4 += az * az * half ** 4 / 5.0
                                for j in range(k + 1, 3):
                                    e4 += 6.0 * az * axes[3 * j + 2] * axes[3 * j + 2] * (half * half / 3.0) ** 2
                            value = sm_i * sm_i * sm_i / (w_c * w_c) * \
                                    (1.0 + 3.0 * e2 / (delta * delta) + 5.0 * e4 / (delta * delta * delta * delta))
                            # convert depth to length along the ray, at the projected centre of the cell
                            Xc = (x_i - view.mid_x) / w_c
                            Yc = (y_i - view.mid_y) / w_c
                            sec = sqrt(1.0 + (Xc * Xc + Yc * Yc) / (z_camera * z_camera))
                            # (the accumulator holds values divided by the mean of the secant over each pixel)
                            acc[<size_t> by0 * nx + bx0] += weight * value * sec / pixel_area / \
                                _secant_model(&view, bx0, by0, &gx, &gy, &hxx, &hxy, &hyy)
                            continue

                        scale = weight * abs_zc / pixel_area

                        for k in range(3):
                            for s in range(2):
                                nx_ = axes[3 * k] if s else -axes[3 * k]
                                ny_ = axes[3 * k + 1] if s else -axes[3 * k + 1]
                                nz_ = axes[3 * k + 2] if s else -axes[3 * k + 2]
                                d = nx_ * x_i + ny_ * y_i + nz_ * z_i + half
                                n = 4
                                for j in range(4):
                                    fx[j] = vx[face_vertices[2 * k + s][j]]
                                    fy[j] = vy[face_vertices[2 * k + s][j]]
                                    fz[j] = vz[face_vertices[2 * k + s][j]]
                                if clip_z:
                                    n = _clip_polygon_3d(fx, fy, fz, n, 0.0, 0.0, 1.0, -z_lo, ox, oy, oz)
                                    n = _clip_polygon_3d(ox, oy, oz, n, 0.0, 0.0, -1.0, z_hi, fx, fy, fz)
                                if n >= 3:
                                    _add_face(&view, fx, fy, fz, n, nx_, ny_, nz_, d, clip_frustum, w_c, scale, acc)

                        if clip_z:
                            # the faces created by truncating the cell at z_lo and z_hi
                            for s in range(2):
                                if s == 0:
                                    if zmin >= z_lo: continue
                                    value = z_lo
                                else:
                                    if zmax <= z_hi: continue
                                    value = z_hi
                                n = _cube_cross_section(axes, half, value - z_i, cx, cy)
                                for j in range(n):
                                    fx[j] = x_i + cx[j]
                                    fy[j] = y_i + cy[j]
                                    fz[j] = value
                                if n >= 3:
                                    _add_face(&view, fx, fy, fz, n, 0.0, 0.0, 1.0 if s else -1.0,
                                              value if s else -value, clip_frustum, w_c, scale, acc)

    with nogil:
        for iy in range(ny):
            for ix in range(nx):
                res[iy * nx + ix] = acc[<size_t> iy * nx + ix] * _secant_model(&view, ix, iy, &gx, &gy, &hxx, &hxy, &hyy)

    return result


def triangle_mean_inverse(double l1, double l2, double l3):
    """The mean of 1/L over a triangle on which L is affine, with values l1, l2, l3 at the vertices (for testing)"""
    return _triangle_mean_inverse(l1, l2, l3)


def polygon_integrals(px, py, double Xp, double Yp, double lx, double ly, double l0, double K, double w_ref,
                      double s0, double gx, double gy, double hxx, double hxy, double hyy):
    """The integral of S (K/L - w_ref) over a polygon, by the series and the exact methods (for testing)"""
    cdef double cpx[MAX_POLY]
    cdef double cpy[MAX_POLY]
    cdef int i, n = len(px)
    for i in range(n):
        cpx[i] = px[i]; cpy[i] = py[i]
    return (_piece_integral(cpx, cpy, n, Xp, Yp, lx, ly, l0, K, w_ref, s0, gx, gy, hxx, hxy, hyy),
            _polygon_integral(cpx, cpy, n, lx, ly, l0, K, w_ref, s0, gx, gy, hxx, hxy, hyy, Xp, Yp))


def render_image_cells_perspective_approximate(int nx, int ny, x, y, z, sm,
                                               double x1, double x2, double y1, double y2, double z_camera,
                                               qty, mass, rho, double smooth_lo, double smooth_hi,
                                               double z_lo, double z_hi, double min_smooth, orientation,
                                               wrap_offsets_x=(0.0,), wrap_offsets_y=(0.0,),
                                               double tolerance=0.1):
    """Render a perspective projection of cubic (AMR) cells, approximating where this is accurate to *tolerance*.

    The arguments and result are as for :func:`render_image_cells_perspective`. However, a cell whose perspective
    image differs from its orthographic image, scaled to its distance from the camera and centred on the projection
    of its centre, by less than *tolerance* pixels at every corner is rendered in the latter way, using the tabulated
    orthographic projections of :func:`pynbody.imaging._render_amr.render_image_cells`. Its total is scaled to that
    of the exact perspective image (i.e. the integral of 1/w^2 over its volume, times the secant of the angle of the
    ray through its centre). Other cells are projected exactly.

    The neglected distortion arises from the obliquity of the rays to the z axis, and from the variation of the
    perspective scale across each cell. Writing the offset of a corner from the centre of the cell as u, the distance
    of the centre from the camera along z as D, and t as the tangent vector of the angle of the ray through the
    centre, the corner is displaced in the image by (u_z / w) (t + u_xy / D) / (1 - u_z / D), where w is the
    perspective scale factor of the centre.

    Since many cells in a typical image are smaller than a pixel or far from the camera, most can be approximated,
    which reduces the cost to approximately that of an orthographic image. As the distortion is shared by neighbouring
    cells, they continue to tile space to a good approximation.
    """
    from . import _render_amr

    orientation = np.asarray(orientation, dtype=np.float64)
    x, y, z, sm, qty, mass, rho = (np.ascontiguousarray(a, dtype=np.float64)
                                   for a in (x, y, z, sm, qty, mass, rho))
    sm = np.maximum(sm, min_smooth)
    pixel_dx = (x2 - x1) / nx
    pixel_dy = (y2 - y1) / ny
    pixel_min = min(pixel_dx, pixel_dy)
    mid_x = 0.5 * (x1 + x2)
    mid_y = 0.5 * (y1 + y2)

    # the half-extent along z of a unit cell, and the fourth moment of its distribution along z (see the single-pixel
    # case of render_image_cells_perspective)
    az = orientation[2, :] ** 2
    ext_z = 0.5 * np.abs(orientation[2, :]).sum()
    c4 = (az ** 2).sum() / 5.0 + 6.0 * (az[0] * az[1] + az[0] * az[2] + az[1] * az[2]) / 9.0

    result = np.zeros((ny, nx), dtype=np.float32)
    for wrap_offset_x in wrap_offsets_x:
        for wrap_offset_y in wrap_offsets_y:
            xw = x + wrap_offset_x
            yw = y + wrap_offset_y
            delta = z_camera - z                     # distance from the camera along z, with the sign of z_camera
            w = delta / z_camera                     # perspective scale factor
            half_z = ext_z * sm                      # half-extent of each cell along z
            with np.errstate(divide='ignore', invalid='ignore'):
                tx = (xw - mid_x) / delta
                ty = (yw - mid_y) / delta
                margin = 1.0 - half_z / np.abs(delta)
                distortion = (half_z / np.abs(w)) * (np.sqrt(tx * tx + ty * ty) + 0.87 * sm / np.abs(delta)) / margin
                approximate = ((w > 0) & (margin > 0.5) & (distortion < tolerance * pixel_min) &
                               (z - half_z >= z_lo) & (z + half_z <= z_hi) & (sm < 0.1 * np.abs(delta)))
            exact = ~approximate & (z + half_z > z_lo) & (z - half_z < z_hi)

            if np.any(exact):
                result += render_image_cells_perspective(
                    nx, ny, xw[exact], yw[exact], z[exact], sm[exact], x1, x2, y1, y2, z_camera,
                    qty[exact], mass[exact], rho[exact], smooth_lo, smooth_hi, z_lo, z_hi, 0.0, orientation)

            if np.any(approximate):
                wa = w[approximate]
                da = delta[approximate]
                half = 0.5 * sm[approximate]
                Xc = mid_x + (xw[approximate] - mid_x) / wa
                Yc = mid_y + (yw[approximate] - mid_y) / wa
                sec = np.sqrt(1.0 + ((Xc - mid_x) ** 2 + (Yc - mid_y) ** 2) / z_camera ** 2)
                # the integral over the image of the depth range of each cell is that of 1/w^2 over its volume
                d2 = (half / da) ** 2
                total = sec / wa ** 2 * (1.0 + d2 + 5.0 * c4 * d2 * d2)
                result += _render_amr.render_image_cells(
                    nx, ny, Xc, Yc, np.zeros_like(Xc), sm[approximate] / wa, x1, x2, y1, y2, 0.0,
                    qty[approximate] * total, mass[approximate], rho[approximate], smooth_lo, smooth_hi,
                    -np.inf, np.inf, 0.0, True, orientation)
    return result
