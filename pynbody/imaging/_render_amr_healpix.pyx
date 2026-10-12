"""Rendering of AMR cells onto healpix maps, as exact projections of (or exact thin shells through) cubes.

This is the healpix counterpart of :mod:`pynbody.imaging._render_amr`. Each 'particle' is a cubic cell of side
length given by its smoothing length, with edges along the columns of a rotation matrix. The map is centred on the
origin, in RING ordering.

The Python interface is :class:`pynbody.imaging.renderers.AMRHealpixRenderer`; see :func:`render_healpix_cells`
for the method.
"""

import numpy as np

cimport cython
cimport numpy as np

np.import_array()
from libc.math cimport INFINITY, M_PI, asin, atan2, ceil, cos, fabs, floor, isfinite, log2, sin, sqrt

ctypedef np.float32_t image_output_type
np_image_output_type = np.float32

cdef enum:
    MAX_LEVEL = 29     # finest healpix level (nside = 2**29) used for sub-pixels, so that indices fit in 64 bits
    STACK_SIZE = 512   # depth-first traversal pushes at most 4 nodes per level, so this is ample
    MAX_PLANES = 4     # planes bounding the cone of a pixel
    MAX_SUB = 64       # maximum number of segments into which each side of a pixel is divided (for large cells)
    MAX_SUB_LOCAL = 1048576  # ... for small cells, for which only the segments close to the cell are considered
    SIDE_CACHE = 255   # number of points along a side cached by _local_sliver_volume
    POLAR_REFINEMENT = 8  # maximum extra levels of sub-pixels used for thin shells close to the poles
    POLAR_RINGS = 64      # ... within this number of rings of the poles
    MAX_POLY = 16      # maximum number of vertices of a clipped polygon (at most 8 + 6 are needed)

# Tolerance for the treatment of the slivers between pixel sides and great circles; see _pixel_volume
cdef double SLIVER_TOLERANCE = 1e-4

# Safety factor on healpix's maximum pixel radius, used to bound each (sub-)pixel by a cone
cdef double PIXRAD_SAFETY = 1.02

# Bound on the gradient (per radian) of the continuous face coordinates of a healpix map at nside=1; the
# true maximum is 1.452, attained close to the poles
cdef double FACE_COORDINATE_GRADIENT = 1.5

cdef struct Cell:
    double c[3]        # centre, relative to the origin, in the map frame
    double ck[3]       # centre in the frame of the cell axes
    double axes[9]     # axes[3k + i] is component i (in the map frame) of cell axis k
    double half        # half of the side length
    double d           # distance of the centre from the origin
    double chat[3]     # unit vector towards the centre
    int far            # True if the origin is outside the bounding sphere of the cell
    double rho_c       # angular radius of the bounding sphere as seen from the origin (pi if not far)
    double cos_rho_c, sin_rho_c
    double value       # the quantity inside the cell (qty * mass / rho / side^3)
    double volume      # side^3
    double vert[24]    # vertices relative to the centre; vertex j is at +half along axis k if bit k of j is set
    # the 12 edges, as seen from the origin, for the test of whether a cone crosses any of them
    int edges_ready
    double eP[36]      # unit vector towards the first end
    double eQ[36]      # unit vector towards the second end
    double eN[36]      # unit normal to the plane containing the edge and the origin
    double eTP[36]     # tangent at the first end, pointing along the arc
    double eTQ[36]     # tangent at the second end, pointing back along the arc
    int eflag[12]      # 0 if the edge is seen as one or two points, so that only eP and eQ are meaningful


cdef inline double _dot3(const double *a, const double *b) noexcept nogil:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

cdef inline void _cross3(const double *a, const double *b, double *out) noexcept nogil:
    out[0] = a[1] * b[2] - a[2] * b[1]
    out[1] = a[2] * b[0] - a[0] * b[2]
    out[2] = a[0] * b[1] - a[1] * b[0]

cdef inline int _jrll(int face) noexcept nogil:
    return 2 + face // 4

cdef inline int _jpll(int face) noexcept nogil:
    cdef int f = face % 4
    if face // 4 == 1:
        return 2 * f
    return 2 * f + 1


@cython.cdivision(True)
cdef inline void _pixel_centre(long long nside, int face, long long ix, long long iy, double *v) noexcept nogil:
    """Unit vector towards the centre of the pixel with face coordinates (ix, iy), as for pix2vec in the NEST
    scheme. Accurate close to the poles."""
    cdef long long jr = _jrll(face) * nside - ix - iy - 1
    cdef long long nl4 = 4 * nside
    cdef long long nr, kshift, jp
    cdef double z, sin_theta, one_minus_z, phi, ratio
    if jr < nside:
        nr = jr
        ratio = <double> nr / <double> nside
        one_minus_z = ratio * ratio / 3.0
        z = 1.0 - one_minus_z
        sin_theta = sqrt(one_minus_z * (2.0 - one_minus_z))
        kshift = 0
    elif jr > 3 * nside:
        nr = nl4 - jr
        ratio = <double> nr / <double> nside
        one_minus_z = ratio * ratio / 3.0
        z = one_minus_z - 1.0
        sin_theta = sqrt(one_minus_z * (2.0 - one_minus_z))
        kshift = 0
    else:
        nr = nside
        z = (2 * nside - jr) * 2.0 / (3.0 * nside)
        sin_theta = sqrt((1.0 - z) * (1.0 + z))
        kshift = (jr - nside) & 1
    jp = (_jpll(face) * nr + ix - iy + 1 + kshift) / 2
    if jp > nl4:
        jp -= nl4
    if jp < 1:
        jp += nl4
    phi = (jp - (kshift + 1) * 0.5) * (0.5 * M_PI / nr)
    v[0] = sin_theta * cos(phi)
    v[1] = sin_theta * sin(phi)
    v[2] = z


@cython.cdivision(True)
cdef inline long long _ring_index(long long nside, int face, long long ix, long long iy) noexcept nogil:
    """Index in the RING scheme of the pixel with face coordinates (ix, iy)"""
    cdef long long jr = _jrll(face) * nside - ix - iy - 1
    cdef long long nl4 = 4 * nside
    cdef long long npix = 12 * nside * nside
    cdef long long ncap = 2 * nside * (nside - 1)
    cdef long long nr, n_before, kshift, jp
    if jr < nside:
        nr = jr
        n_before = 2 * nr * (nr - 1)
        kshift = 0
    elif jr > 3 * nside:
        nr = nl4 - jr
        n_before = npix - 2 * (nr + 1) * nr
        kshift = 0
    else:
        nr = nside
        n_before = ncap + (jr - nside) * nl4
        kshift = (jr - nside) & 1
    jp = (_jpll(face) * nr + ix - iy + 1 + kshift) / 2
    if jp > nl4:
        jp -= nl4
    elif jp < 1:
        jp += nl4
    return n_before + jp - 1


cdef inline double _distance_to_integer(double u) noexcept nogil:
    cdef double f = u - floor(u)
    return f if f < 1.0 - f else 1.0 - f


cdef struct FaceCoordinates:
    # continuous face coordinates at nside=1 of a direction, as defined in the healpix ang2pix algorithm; at any
    # other nside, they are multiplied by nside, and pixel boundaries are the lines on which either is an integer
    double u
    double v
    int polar   # True if in a polar cap (|z| > 2/3)
    int north   # True if z > 0
    int ntt     # the quarter of the sphere in longitude (for polar caps)


@cython.cdivision(True)
cdef void _face_coordinates(const double *c, double d, FaceCoordinates *fc) noexcept nogil:
    """Continuous face coordinates of the direction of c, at distance d from the origin"""
    cdef double z = c[2] / d
    cdef double tt = atan2(c[1], c[0]) / (0.5 * M_PI)
    cdef double tp, tmp
    if tt < 0.0:
        tt += 4.0
    if tt >= 4.0:
        tt -= 4.0
    fc.north = z > 0
    if fabs(z) <= 2.0 / 3.0:
        fc.polar = 0
        fc.ntt = 0
        fc.u = 0.5 + tt - 0.75 * z
        fc.v = 0.5 + tt + 0.75 * z
    else:
        fc.polar = 1
        fc.ntt = <int> tt
        if fc.ntt >= 4:
            fc.ntt = 3
        tp = tt - fc.ntt
        # sqrt(3 (1 - |z|)), computed without cancellation
        tmp = sqrt(3.0 * (c[0] * c[0] + c[1] * c[1]) / (d * (d + fabs(c[2]))))
        fc.u = tp * tmp
        fc.v = (1.0 - tp) * tmp


@cython.cdivision(True)
cdef double _locate(const FaceCoordinates *fc, long long nside, int *face, long long *ix,
                    long long *iy) noexcept nogil:
    """Find the pixel at the given nside containing a direction with the given face coordinates, as in the healpix
    ang2pix algorithm, and return how far the direction lies from the pixel boundary in units of the face
    coordinates at that nside.

    The distance to the nearest integer of the face coordinates changes by at most FACE_COORDINATE_GRADIENT * nside
    per radian, so a cone of angular radius r about the direction lies within the pixel if the returned margin
    exceeds FACE_COORDINATE_GRADIENT * nside * r."""
    cdef double u = fc.u * nside, v = fc.v * nside
    cdef double margin, tmp
    cdef long long jp = <long long> u, jm = <long long> v, ifp, ifm
    if not fc.polar:
        ifp = jp // nside
        ifm = jm // nside
        if ifp == ifm:
            face[0] = <int> (ifp | 4)
        elif ifp < ifm:
            face[0] = <int> ifp
        else:
            face[0] = <int> ifm + 8
        ix[0] = jm & (nside - 1)
        iy[0] = nside - (jp & (nside - 1)) - 1
    else:
        if jp >= nside:
            jp = nside - 1
        if jm >= nside:
            jm = nside - 1
        if fc.north:
            face[0] = fc.ntt
            ix[0] = nside - jm - 1
            iy[0] = nside - jp - 1
        else:
            face[0] = fc.ntt + 8
            ix[0] = jp
            iy[0] = jm
    margin = _distance_to_integer(u)
    tmp = _distance_to_integer(v)
    if tmp < margin:
        margin = tmp
    if face[0] < 0 or face[0] > 11:
        margin = -1.0
    return margin


cdef double max_pixel_radius(long long nside):
    """Maximum angular distance between a pixel centre and any point in the pixel (healpix's max_pixrad)"""
    cdef double za = 2.0 / 3.0, phia = M_PI / (4.0 * nside)
    cdef double t1 = 1.0 - 1.0 / nside
    cdef double zb = 1.0 - t1 * t1 / 3.0
    cdef double sa = sqrt((1 - za) * (1 + za)), sb = sqrt((1 - zb) * (1 + zb))
    cdef double a[3]
    cdef double b[3]
    cdef double cr[3]
    a[0] = sa * cos(phia); a[1] = sa * sin(phia); a[2] = za
    b[0] = sb; b[1] = 0.0; b[2] = zb
    _cross3(a, b, cr)
    return atan2(sqrt(_dot3(cr, cr)), _dot3(a, b))


@cython.cdivision(True)
cdef inline double _ray_moment(const Cell *g, const double *n) noexcept nogil:
    """(r_out^3 - r_in^3) / 3 along the ray from the origin in unit direction n, where r_in and r_out are the
    distances at which the ray enters and leaves the cell (r_in = 0 if the origin is inside the cell). This is the
    exact value of the integral of r^2 dr through the cell."""
    cdef double t_in = 0.0, t_out = INFINITY
    cdef double m, lo, hi, t1, t2
    cdef int k
    for k in range(3):
        m = g.axes[3 * k] * n[0] + g.axes[3 * k + 1] * n[1] + g.axes[3 * k + 2] * n[2]
        lo = g.ck[k] - g.half
        hi = g.ck[k] + g.half
        if m > 0:
            t1 = lo / m
            t2 = hi / m
        elif m < 0:
            t1 = hi / m
            t2 = lo / m
        else:
            if lo > 0 or hi < 0:
                return 0.0
            continue
        if t1 > t_in:
            t_in = t1
        if t2 < t_out:
            t_out = t2
    if t_out <= t_in:
        return 0.0
    return (t_out - t_in) * (t_out * t_out + t_out * t_in + t_in * t_in) / 3.0


@cython.cdivision(True)
cdef inline double _ramp_integral(double c, double m, double a) noexcept nogil:
    """Integral over v from -a to a of max(0, c - m v), for 0 <= m <= 1"""
    cdef double p, q
    if m < 1e-9:
        return 2.0 * a * (c if c > 0 else 0.0)
    p = c + m * a
    q = c - m * a
    p = p * p if p > 0 else 0.0
    q = q * q if q > 0 else 0.0
    return (p - q) / (2.0 * m)


@cython.cdivision(True)
cdef inline double _square_area_below_line(double a, double alpha, double beta, double gamma) noexcept nogil:
    """Area of the square [-a, a]^2 in which alpha u + beta v <= gamma"""
    cdef double t, m, c
    alpha = fabs(alpha)
    beta = fabs(beta)
    if beta > alpha:
        t = alpha
        alpha = beta
        beta = t
    if alpha == 0.0:
        return 4.0 * a * a if gamma >= 0 else 0.0
    # integrate, over v, the length of the u interval: clamp((gamma - beta v) / alpha + a, 0, 2a)
    m = beta / alpha
    c = gamma / alpha + a
    return _ramp_integral(c, m, a) - _ramp_integral(c - 2.0 * a, m, a)


@cython.cdivision(True)
cdef double _cube_section_area(double half, const double *normal, double h) noexcept nogil:
    """Area of the cross-section of the cube [-half, half]^3 with the plane normal . x = h, for unit normal.

    The section is projected onto the coordinate plane perpendicular to the largest component of the normal,
    where it is the intersection of a square with a strip."""
    cdef int k = 0, i, j
    cdef double nk
    if fabs(normal[1]) > fabs(normal[k]):
        k = 1
    if fabs(normal[2]) > fabs(normal[k]):
        k = 2
    i = (k + 1) % 3
    j = (k + 2) % 3
    nk = fabs(normal[k])
    return (_square_area_below_line(half, normal[i], normal[j], h + half * nk)
            - _square_area_below_line(half, normal[i], normal[j], h - half * nk)) / nk


@cython.cdivision(True)
cdef void _prepare_edges(Cell *g) noexcept nogil:
    cdef double V[24]
    cdef double E[3]
    cdef double N[3]
    cdef double *P
    cdef double *Q
    cdef double nP, nQ, nN, nE
    cdef int j, j2, k, i, e = 0
    for j in range(8):
        for i in range(3):
            V[3 * j + i] = g.c[i]
            for k in range(3):
                V[3 * j + i] += (g.half if (j >> k) & 1 else -g.half) * g.axes[3 * k + i]
    for k in range(3):
        for j in range(8):
            if (j >> k) & 1:
                continue
            j2 = j | (1 << k)
            P = &V[3 * j]
            Q = &V[3 * j2]
            for i in range(3):
                E[i] = Q[i] - P[i]
            nP = sqrt(_dot3(P, P))
            nQ = sqrt(_dot3(Q, Q))
            nE = sqrt(_dot3(E, E))
            g.eflag[e] = 0
            if nP <= 1e-14 * nE:
                # the edge starts at the origin, and is seen as a single point
                for i in range(3):
                    g.eP[3 * e + i] = Q[i] / nQ
                    g.eQ[3 * e + i] = Q[i] / nQ
            elif nQ <= 1e-14 * nE:
                for i in range(3):
                    g.eP[3 * e + i] = P[i] / nP
                    g.eQ[3 * e + i] = P[i] / nP
            else:
                for i in range(3):
                    g.eP[3 * e + i] = P[i] / nP
                    g.eQ[3 * e + i] = Q[i] / nQ
                _cross3(P, E, N)
                nN = sqrt(_dot3(N, N))
                # unless the line of the edge passes through the origin (in which case the edge is seen as one or
                # two points), it is seen as a great circle arc between the directions to its ends
                if nN > 1e-14 * nP * nE:
                    g.eflag[e] = 1
                    for i in range(3):
                        g.eN[3 * e + i] = N[i] / nN
                    _cross3(&g.eN[3 * e], &g.eP[3 * e], &g.eTP[3 * e])
                    _cross3(&g.eQ[3 * e], &g.eN[3 * e], &g.eTQ[3 * e])
            e += 1
    g.edges_ready = 1


cdef inline int _cone_meets_edges(const Cell *g, const double *n, double cos_r, double sin_r) noexcept nogil:
    """True if any edge of the cell, as seen from the origin, comes within the cone of the given half-angle about
    unit vector n.

    For a projection, the integrand is a smooth function of direction except where the ray passes through an edge,
    so a cone that meets no edge either misses the cell entirely, or sees a smooth integrand."""
    cdef int e
    cdef double s
    for e in range(12):
        if _dot3(n, &g.eP[3 * e]) >= cos_r or _dot3(n, &g.eQ[3 * e]) >= cos_r:
            return 1
        if g.eflag[e]:
            s = _dot3(n, &g.eN[3 * e])
            if fabs(s) <= sin_r and _dot3(n, &g.eTP[3 * e]) >= 0 and _dot3(n, &g.eTQ[3 * e]) >= 0:
                return 1
    return 0


@cython.cdivision(True)
cdef inline void _face_point(int face, double x, double y, double *v, double *one_minus_abs_z) noexcept nogil:
    """Unit vector towards the point with continuous coordinates (x, y) in [0, 1]^2 within the given base face (so
    that the pixel with face coordinates (ix, iy) at a given nside spans [ix, ix+1]/nside x [iy, iy+1]/nside).
    Accurate close to the poles. Also returns 1 - |z|."""
    cdef double jr = _jrll(face) - x - y
    cdef double nr, z, omz, sin_theta, tmp, phi
    if jr < 1.0:
        nr = jr
        omz = nr * nr / 3.0
        z = 1.0 - omz
        sin_theta = sqrt(omz * (2.0 - omz))
    elif jr > 3.0:
        nr = 4.0 - jr
        omz = nr * nr / 3.0
        z = omz - 1.0
        sin_theta = sqrt(omz * (2.0 - omz))
    else:
        nr = 1.0
        z = (2.0 - jr) * (2.0 / 3.0)
        omz = 1.0 - fabs(z)
        sin_theta = sqrt((1.0 - z) * (1.0 + z))
    tmp = _jpll(face) * nr + x - y
    if tmp < 0:
        tmp += 8.0
    if tmp >= 8.0:
        tmp -= 8.0
    phi = 0.0 if nr < 1e-15 else 0.25 * M_PI * tmp / nr
    v[0] = sin_theta * cos(phi)
    v[1] = sin_theta * sin(phi)
    v[2] = z
    one_minus_abs_z[0] = omz


cdef inline double _triangle_solid_angle(const double *p, const double *q, const double *r) noexcept nogil:
    """Signed solid angle of the spherical triangle with unit vector vertices p, q, r (positive if anticlockwise
    as seen from outside the sphere)"""
    cdef double qr[3]
    _cross3(q, r, qr)
    return 2.0 * atan2(_dot3(p, qr), 1.0 + _dot3(p, q) + _dot3(q, r) + _dot3(r, p))


@cython.cdivision(True)
cdef int _clip_polygon(const double *poly, int n, const double *m, double b, double *out) noexcept nogil:
    """Clip the polygon (n vertices, as consecutive triples) to the half-space m . y + b >= 0 (Sutherland-Hodgman),
    preserving its orientation. Returns the number of vertices written to out."""
    cdef int i, j, k, n_out = 0
    cdef double si, sj, t
    if n == 0:
        return 0
    sj = _dot3(m, &poly[3 * (n - 1)]) + b
    j = n - 1
    for i in range(n):
        si = _dot3(m, &poly[3 * i]) + b
        if (si >= 0) != (sj >= 0):
            t = sj / (sj - si)
            for k in range(3):
                out[3 * n_out + k] = poly[3 * j + k] + t * (poly[3 * i + k] - poly[3 * j + k])
            n_out += 1
        if si >= 0:
            for k in range(3):
                out[3 * n_out + k] = poly[3 * i + k]
            n_out += 1
        sj = si
        j = i
    return n_out


@cython.cdivision(True)
cdef double _cone_volume(const Cell *g, const double *planes, int n_planes) noexcept nogil:
    """Exact volume of the part of the cell within the convex cone {x : planes[p] . x >= 0 for all p}, whose apex
    is at the origin.

    Each face of the cube is clipped to the cone. Since the faces of the clipped polyhedron that lie on the planes
    of the cone pass through the origin, the volume is (1/3) sum_f h_f A_f, summed only over (clipped) faces of
    the cube, where h_f is the signed distance of face f from the origin and A_f its clipped area. To avoid
    round-off when the cell is small compared to its distance, positions are taken relative to the centre of the
    cell, c, so that for a polygon with vertices c + y_i the contribution of the triangle (0, i, i+1) of a fan is
    c . ((y_i - y_0) x (y_{i+1} - y_0)) + y_0 . (y_i x y_{i+1})."""
    cdef double offset[MAX_PLANES]
    cdef int active[MAX_PLANES]
    cdef int inside_mask[MAX_PLANES]  # bit j set if vertex j is inside the plane
    cdef int n_active = 0, p, j, k, k2, f, sgn, n, i, a, b, face_mask
    cdef int face_vertices[4]
    cdef double buf1[3 * MAX_POLY]
    cdef double buf2[3 * MAX_POLY]
    cdef double *poly
    cdef double *other
    cdef double *tmp
    cdef double e1[3]
    cdef double e2[3]
    cdef double cr[3]
    cdef double total = 0.0, vt
    for p in range(n_planes):
        offset[p] = _dot3(&planes[3 * p], g.c)
        inside_mask[p] = 0
        for j in range(8):
            if _dot3(&planes[3 * p], &g.vert[3 * j]) + offset[p] >= 0:
                inside_mask[p] |= 1 << j
        if inside_mask[p] == 0:
            return 0.0
        if inside_mask[p] != 255:
            active[n_active] = p
            n_active += 1
    if n_active == 0:
        return g.volume

    for f in range(6):
        k = f // 2
        sgn = f % 2  # 1 for the face at +half along axis k
        a = (k + 1) % 3
        b = (k + 2) % 3
        # vertex index has bit k set for the + face; corners in anticlockwise order seen from outside
        face_mask = 0
        for i in range(4):
            j = (sgn << k)
            if sgn:
                # order (-,-), (+,-), (+,+), (-,+) in (a, b)
                j |= ((1 if i == 1 or i == 2 else 0) << a) | ((1 if i >= 2 else 0) << b)
            else:
                j |= ((1 if i >= 2 else 0) << a) | ((1 if i == 1 or i == 2 else 0) << b)
            face_vertices[i] = j
            face_mask |= 1 << j
        poly = buf1
        other = buf2
        for i in range(4):
            for k2 in range(3):
                poly[3 * i + k2] = g.vert[3 * face_vertices[i] + k2]
        n = 4
        for i in range(n_active):
            p = active[i]
            if (inside_mask[p] & face_mask) == face_mask:
                continue  # the face is entirely inside this plane
            if (inside_mask[p] & face_mask) == 0:
                n = 0  # the face is entirely outside this plane
                break
            n = _clip_polygon(poly, n, &planes[3 * p], offset[p], other)
            tmp = poly
            poly = other
            other = tmp
            if n < 3:
                break
        if n < 3:
            continue
        for i in range(1, n - 1):
            for k2 in range(3):
                e1[k2] = poly[3 * i + k2] - poly[k2]
                e2[k2] = poly[3 * i + 3 + k2] - poly[k2]
            _cross3(e1, e2, cr)
            vt = _dot3(g.c, cr)
            _cross3(&poly[3 * i], &poly[3 * i + 3], cr)
            total += vt + _dot3(poly, cr)
    return fabs(total) / 6.0


cdef double _triangle_volume(const Cell *g, const double *a, const double *b, const double *c) noexcept nogil:
    """Signed volume of the cell within the cone of the spherical triangle a, b, c: positive if the triangle is
    anticlockwise as seen from outside the sphere, negative otherwise"""
    cdef double planes[9]
    cdef double sign
    cdef int k
    _cross3(a, b, planes)
    _cross3(b, c, &planes[3])
    _cross3(c, a, &planes[6])
    sign = 1.0 if _dot3(a, &planes[3]) >= 0 else -1.0
    if sign < 0:
        for k in range(9):
            planes[k] = -planes[k]
    return sign * _cone_volume(g, planes, 3)


@cython.cdivision(True)
cdef inline void _side_point(int face, double xa, double ya, double xb, double yb, long long j, long long n,
                             const double *a, const double *b, double *v) noexcept nogil:
    """The point a fraction j/n of the way along a side of a pixel, between corners a and b at face coordinates
    (xa, ya) and (xb, yb)"""
    cdef double t, dummy
    cdef int k
    if j == 0:
        for k in range(3):
            v[k] = a[k]
    elif j == n:
        for k in range(3):
            v[k] = b[k]
    else:
        t = <double> j / n
        _face_point(face, xa + t * (xb - xa), ya + t * (yb - ya), v, &dummy)


cdef struct SidePoints:
    # a cache of points along a side of a pixel, at parameters j/n for first <= j <= first + SIDE_CACHE
    int face
    double xa, ya, xb, yb
    const double *a
    const double *b
    long long n, first
    double point[3 * (SIDE_CACHE + 1)]
    char have[SIDE_CACHE + 1]


cdef inline void _cached_side_point(SidePoints *sp, long long j, double *v) noexcept nogil:
    cdef long long i = j - sp.first
    cdef int k
    if i < 0 or i > SIDE_CACHE:
        _side_point(sp.face, sp.xa, sp.ya, sp.xb, sp.yb, j, sp.n, sp.a, sp.b, v)
        return
    if not sp.have[i]:
        _side_point(sp.face, sp.xa, sp.ya, sp.xb, sp.yb, j, sp.n, sp.a, sp.b, &sp.point[3 * i])
        sp.have[i] = 1
    for k in range(3):
        v[k] = sp.point[3 * i + k]


@cython.cdivision(True)
cdef double _local_sliver_volume(Cell *g, int face, double xa, double ya, double xb, double yb, const double *a,
                                 const double *b, long long n, double t_lo, double t_hi) noexcept nogil:
    """Volume of a small cell within the sliver between a side of a pixel (from corner a to b) and the great circle
    through its ends, as in _pixel_volume but considering only the part of the side between parameters t_lo and
    t_hi (in [0, 1]), outside which the cell does not reach. The side is approximated by a polyline of n segments
    (n a power of 2), the polygon between it and the great circle split into triangles by recursive bisection, and
    the parabolic segments between the polyline and the side approximated as in _pixel_volume. Triangles and segments
    entirely outside the range, or not reaching the cell, are skipped. The volume within the remaining triangles is
    calculated exactly where they meet the edges of the cell, while for triangles small compared to the cell, over
    which the integrand is smooth, it is approximated using the integrand at the centroid. (Since the triangles are
    thin, they contain a small fraction of the cell, so that the resulting error is negligible.)"""
    cdef double p0[3]
    cdef double p1[3]
    cdef double p2[3]
    cdef double cr[3]
    cdef double cen[3]
    cdef double dv = 0.0, area, norm, radius
    cdef long long step, span, j, j1
    cdef int k
    cdef SidePoints sp
    if not g.edges_ready:
        _prepare_edges(g)
    if t_lo < 0:
        t_lo = 0.0
    if t_hi > 1:
        t_hi = 1.0
    sp.face = face
    sp.xa = xa
    sp.ya = ya
    sp.xb = xb
    sp.yb = yb
    sp.a = a
    sp.b = b
    sp.n = n
    sp.first = <long long> floor(t_lo * n) - 1
    for k in range(SIDE_CACHE + 1):
        sp.have[k] = 0
    step = n // 2
    while step >= 1:
        span = 2 * step
        j = (<long long> floor(t_lo * n / span)) * span
        j1 = (<long long> ceil(t_hi * n / span)) * span
        if j1 > n:
            j1 = n
        while j < j1:
            _cached_side_point(&sp, j, p0)
            _cached_side_point(&sp, j + step, p1)
            _cached_side_point(&sp, j + span, p2)
            j += span
            # bounding cone of the (thin) triangle
            for k in range(3):
                cen[k] = p0[k] + p1[k] + p2[k]
            norm = sqrt(_dot3(cen, cen))
            for k in range(3):
                cen[k] /= norm
            for k in range(3):
                cr[k] = p0[k] - p2[k]
            radius = 0.6 * sqrt(_dot3(cr, cr)) + 1e-12  # (the triangle lies close to the chord between p0 and p2)
            if g.rho_c + radius < 0.5 * M_PI and _dot3(cen, g.chat) < cos(g.rho_c + radius):
                continue  # the triangle does not reach the cell
            if radius < 0.5 * g.rho_c and not _cone_meets_edges(g, cen, cos(radius), sin(radius)):
                dv += _ray_moment(g, cen) * _triangle_solid_angle(p0, p1, p2)
            else:
                dv += _triangle_volume(g, p0, p1, p2)
        step //= 2
    j = <long long> floor(t_lo * n)
    j1 = <long long> ceil(t_hi * n)
    if j1 > n:
        j1 = n
    while j < j1:
        _cached_side_point(&sp, j, p0)
        _cached_side_point(&sp, j + 1, p2)
        _side_point(face, xa, ya, xb, yb, 2 * j + 1, 2 * n, a, b, p1)
        j += 1
        _cross3(p0, p2, cr)
        if g.rho_c < 2.0 * fabs(_dot3(cr, p1)) / sqrt(_dot3(cr, cr)):
            continue  # the cell is too small compared to the width of the parabolic segment to be sampled reliably
        area = (4.0 / 3.0) * _triangle_solid_angle(p0, p1, p2)
        dv += area * _ray_moment(g, p1)
    return dv


@cython.cdivision(True)
cdef inline int _cone_meets_arc(const double *n, double radius, const double *a, const double *b, const double *normal,
                                double normal_length) noexcept nogil:
    """True if the cone of the given angular radius about unit vector n comes within the shorter great circle arc
    between unit vectors a and b, where normal = a x b has the given length"""
    cdef double cos_r, s
    cdef double ta[3]
    cdef double tb[3]
    if radius >= 0.5 * M_PI:
        return 1
    cos_r = cos(radius)
    if _dot3(n, a) >= cos_r or _dot3(n, b) >= cos_r:
        return 1
    s = _dot3(n, normal) / normal_length
    if fabs(s) > sin(radius):
        return 0
    # is the closest point of the great circle within the arc?
    _cross3(normal, a, ta)
    _cross3(b, normal, tb)
    return _dot3(n, ta) >= 0 and _dot3(n, tb) >= 0


@cython.cdivision(True)
cdef double _pixel_volume(Cell *g, long long nside, int face, long long ix, long long iy, const double *corner,
                          const double *omz) noexcept nogil:
    """Volume of the cell within the cone of a pixel of the map.

    The cone through the pixel's four corners (bounded by great circles) is treated exactly by _cone_volume. The
    sides of a healpix pixel are not great circles, so the thin slivers between each side and the corresponding
    great circle are then added or subtracted. Each side is approximated by a polyline (a single segment, except
    near the poles where the sides are most curved), with a parabolic segment between each part of the polyline and
    the side. The volume within the polygon between the polyline and the great circle is calculated exactly; within
    the parabolic segments, the integrand is approximated as constant across the segment, and integrated along it.
    Approximating the sides in this way gives pixel areas that are correct to 3e-5 of a pixel.

    For a cell smaller than the side, the polyline is refined until the parabolic segments are narrower than
    SLIVER_TOLERANCE times the cell's angular size (or a tenth of the side, if larger), so that the fractional error
    in its volume is at most of that order; only the part of the side near the cell is then considered (see
    _local_sliver_volume). (Cells smaller than the width of the parabolic segments, if any remain, are not sampled reliably across
    them, so are instead divided according to the polyline.) For larger cells, the number of points used to
    integrate along each parabolic segment is increased until the error relative to the pixel is at most of order
    SLIVER_TOLERANCE. Sides whose slivers the cell does not reach are skipped.

    The pixel cones and slivers are computed identically from either side of each pixel boundary, so the volume of
    each cell is shared exactly between pixels.

    The corners of the pixel, at face coordinates (ix, iy), (ix+1, iy), (ix+1, iy+1), (ix, iy+1), are passed in
    *corner*, with 1 - |z| for each in *omz* (see _face_point)."""
    cdef double planes[12]
    cdef double norms[4]
    cdef int outside
    cdef double xs[5]
    cdef double ys[5]
    cdef double q[3 * MAX_SUB + 3]   # points along a side
    cdef double mid[3]
    cdef double cr[3]
    cdef double e1[3]
    cdef double dummy, sigma, volume, dv, t, r, sag, norm, side_length, area, u0, u1
    cdef int i, j, k, n_sub, n_quad, step
    xs[0] = xs[3] = xs[4] = <double> ix / nside
    xs[1] = xs[2] = <double> (ix + 1) / nside
    ys[0] = ys[1] = ys[4] = <double> iy / nside
    ys[2] = ys[3] = <double> (iy + 1) / nside
    for i in range(4):
        _cross3(&corner[3 * i], &corner[3 * ((i + 1) % 4)], &planes[3 * i])
    sigma = 1.0 if _dot3(&planes[0], &corner[6]) > 0 else -1.0
    if sigma < 0:
        for i in range(12):
            planes[i] = -planes[i]
    outside = 0
    for i in range(4):
        norms[i] = sqrt(_dot3(&planes[3 * i], &planes[3 * i]))
        if g.far:
            t = _dot3(&planes[3 * i], g.chat) / norms[i]
            if t < -g.sin_rho_c:
                outside = 1
                # away from the poles, the sliver along a side is narrower than 0.25 of its length (and norms[i] is
                # the sine of the length); if the cell lies beyond that, it does not overlap the pixel at all
                r = nside * sqrt(3.0 * (omz[i] if omz[i] < omz[(i + 1) % 4] else omz[(i + 1) % 4]))
                if r > 10.5 and g.rho_c + 0.4 * norms[i] < 0.5 * M_PI and t < -sin(g.rho_c + 0.4 * norms[i]):
                    return 0.0

    volume = 0.0 if outside else _cone_volume(g, planes, 4)

    for i in range(4):
        norm = norms[i]
        side_length = atan2(norm, _dot3(&corner[3 * i], &corner[3 * ((i + 1) % 4)]))
        r = nside * sqrt(3.0 * (omz[i] if omz[i] < omz[(i + 1) % 4] else omz[(i + 1) % 4]))
        if g.far and r > 10.5 and not _cone_meets_arc(g.chat, g.rho_c + 0.4 * side_length, &corner[3 * i],
                                                      &corner[3 * ((i + 1) % 4)], &planes[3 * i], norm):
            continue  # the cell does not reach the sliver (whose width away from the poles is < 0.25 side_length)
        t = 0.5
        _face_point(face, xs[i] + t * (xs[i + 1] - xs[i]), ys[i] + t * (ys[i + 1] - ys[i]), cr, &dummy)
        # sagitta (angular): distance of the middle of the side from the great circle through its ends
        sag = fabs(_dot3(&planes[3 * i], cr)) / norm
        n_sub = 8 if r < 1.5 else (4 if r < 3.5 else (2 if r < 10.5 else 1))
        n_quad = 1
        if g.far and not _cone_meets_arc(g.chat, g.rho_c + 1.5 * sag, &corner[3 * i], &corner[3 * ((i + 1) % 4)],
                                         &planes[3 * i], norm):
            continue  # the cell does not reach the sliver
        if g.far and g.rho_c < side_length:
            # For a cell smaller than the side, refine the polyline until the parabolic segments are much narrower
            # than the cell, so that the error from approximating the integrand across them is small. Only the part
            # of the side near the cell needs to be considered: find the range of the parameter along the side (with
            # a generous margin, since the parameter is not exactly proportional to the distance along it).
            # (For cells much smaller than the side, the tolerance is relative to a tenth of the side instead, since
            # such cells contribute little to the pixel.)
            t = g.rho_c if g.rho_c > 0.1 * side_length else 0.1 * side_length
            while n_sub < MAX_SUB_LOCAL and sag > SLIVER_TOLERANCE * t * n_sub * n_sub:
                n_sub *= 2
            _cross3(&corner[3 * i], &corner[3 * ((i + 1) % 4)], cr)  # (unlike planes[3 * i], not reoriented)
            t = _dot3(g.chat, cr) / (norm * norm)
            for k in range(3):
                mid[k] = g.chat[k] - t * cr[k]  # the direction of the cell projected onto the great circle
            _cross3(&corner[3 * i], mid, e1)
            t = atan2(_dot3(e1, cr) / norm, _dot3(&corner[3 * i], mid)) / side_length
            u0 = 1.5 * g.rho_c / side_length + 2.0 / n_sub
            dv = _local_sliver_volume(g, face, xs[i], ys[i], xs[i + 1], ys[i + 1], &corner[3 * i],
                                      &corner[3 * ((i + 1) % 4)], n_sub, t - u0, t + u0)
            volume += sigma * dv
            continue
        # For a larger cell, use enough points across each parabolic segment to integrate the integrand along it
        while n_quad < MAX_SUB and sag > SLIVER_TOLERANCE * side_length * n_sub * n_sub * n_quad * n_quad:
            n_quad *= 2
        for k in range(3):
            q[k] = corner[3 * i + k]
            q[3 * n_sub + k] = corner[3 * ((i + 1) % 4) + k]
        for j in range(1, n_sub):
            t = <double> j / n_sub
            _face_point(face, xs[i] + t * (xs[i + 1] - xs[i]), ys[i] + t * (ys[i + 1] - ys[i]), &q[3 * j], &dummy)
        # The polygon between the great circle and the polyline is split into triangles by recursive bisection,
        # which (unlike a fan) gives the same triangles when the side is traversed from the neighbouring pixel. The
        # volume within each triangle is calculated exactly.
        dv = 0.0
        step = n_sub // 2
        while step >= 1:
            j = 0
            while j < n_sub:
                dv += _triangle_volume(g, &q[3 * j], &q[3 * (j + step)], &q[3 * (j + 2 * step)])
                j += 2 * step
            step //= 2
        # Within the thin parabolic segments between the polyline and the side, the integrand is approximated as
        # constant across the segment, and integrated along it with n_quad points
        for j in range(n_sub):
            t = (j + 0.5) / n_sub
            _face_point(face, xs[i] + t * (xs[i + 1] - xs[i]), ys[i] + t * (ys[i + 1] - ys[i]), mid, &dummy)
            area = (4.0 / 3.0) * _triangle_solid_angle(&q[3 * j], mid, &q[3 * j + 3])
            if g.far:
                # a cell that is small compared to the width of the segment (its sagitta) is not sampled reliably,
                # so is instead assigned according to the polyline
                _cross3(&q[3 * j], &q[3 * j + 3], cr)
                if g.rho_c < 2.0 * fabs(_dot3(cr, mid)) / sqrt(_dot3(cr, cr)):
                    continue
            for k in range(n_quad):
                u0 = <double> k / n_quad
                u1 = <double> (k + 1) / n_quad
                if n_quad > 1:
                    t = (j + 0.5 * (u0 + u1)) / n_sub
                    _face_point(face, xs[i] + t * (xs[i + 1] - xs[i]), ys[i] + t * (ys[i + 1] - ys[i]), mid,
                                &dummy)
                # fraction of the area of a parabolic segment between u0 and u1 along its length
                dv += area * (3.0 * (u1 * u1 - u0 * u0) - 2.0 * (u1 * u1 * u1 - u0 * u0 * u0)) * _ray_moment(g, mid)
        volume += sigma * dv
    return volume


cdef inline double _polygon_area(const double *poly, int n) noexcept nogil:
    """Area of a planar polygon in 3D"""
    cdef double s[3]
    cdef double cr[3]
    cdef int i, k
    s[0] = s[1] = s[2] = 0.0
    for i in range(n):
        _cross3(&poly[3 * i], &poly[3 * ((i + 1) % n)], cr)
        for k in range(3):
            s[k] += cr[k]
    return 0.5 * sqrt(_dot3(s, s))


@cython.cdivision(True)
cdef double _shell_fraction(const Cell *g, int face, long long nside, long long ix, long long iy,
                            const double *n, double radius) noexcept nogil:
    """Fraction of a (sub-)pixel, with centre n, over which the shell of the given radius lies inside the cell.

    The shell is approximated by its tangent plane at the centre of the pixel, and the pixel by the polygon through
    its corners and the midpoints of its sides; the fraction is then the area of the polygon in which the cone of
    that polygon meets the plane, clipped to the cell, divided by the area of the polygon. (The polygon may be
    slightly concave, which does not affect the area of the clipped polygon.)"""
    cdef double poly[3 * MAX_POLY]
    cdef double other[3 * MAX_POLY]
    cdef double *a = poly
    cdef double *b = other
    cdef double *tmp
    cdef double v[3]
    cdef double m[3]
    cdef double dummy, scale, area0, area1
    cdef int i, k, n_vert = 8
    # positions around the boundary of the pixel in units of half a pixel, starting at (ix, iy)
    cdef int dx[8]
    cdef int dy[8]
    dx[:] = [0, 1, 2, 2, 2, 1, 0, 0]
    dy[:] = [0, 0, 0, 1, 2, 2, 2, 1]
    for i in range(8):
        _face_point(face, (2 * ix + dx[i]) / (2.0 * nside), (2 * iy + dy[i]) / (2.0 * nside), v, &dummy)
        scale = radius / _dot3(n, v)
        for k in range(3):
            a[3 * i + k] = v[k] * scale - g.c[k]
    area0 = _polygon_area(a, 8)
    if area0 <= 0:
        return 0.0
    for k in range(3):
        for i in range(2):
            # half-spaces +/- axis_k . y + half >= 0
            m[0] = g.axes[3 * k] * (1 - 2 * i)
            m[1] = g.axes[3 * k + 1] * (1 - 2 * i)
            m[2] = g.axes[3 * k + 2] * (1 - 2 * i)
            n_vert = _clip_polygon(a, n_vert, m, g.half, b)
            tmp = a
            a = b
            b = tmp
            if n_vert < 3:
                return 0.0
    area1 = _polygon_area(a, n_vert)
    return area1 / area0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def render_healpix_cells(int nside,
                         const double[::1] x, const double[::1] y, const double[::1] z,
                         const double[::1] sm, const double[::1] qty, const double[::1] mass,
                         const double[::1] rho,
                         double min_smooth, orientation, double shell_radius=0.0,
                         int edge_level=3, double cells_across=8.0, bint deposit=True):
    """Render cubic (AMR) cells onto a healpix map (RING ordering) centred on the origin.

    Each cell has side length sm (or min_smooth if that is larger), and the edges of all cells are parallel to the
    columns of the 3x3 rotation matrix *orientation*. The field inside each cell is qty * mass / rho / sm^3, so that
    the cell's volume is effectively mass/rho. Cells with non-finite position or size are skipped.

    If *shell_radius* is zero, a projection is rendered: each pixel contains the average over its solid angle of
    the integral of the field times r^2 dr along the line of sight, i.e. qty * mass / rho times the fraction of
    the cell's volume that lies within the cone of the pixel, divided by the pixel's solid angle. The sum over
    pixels of the pixel value times the pixel solid angle is therefore the sum of qty * mass / rho. If
    *shell_radius* is positive, a thin shell is rendered instead: each pixel contains the average over its solid
    angle of the field on the sphere of that radius.

    Method
    ------

    The pixels overlapping each cell are found by descending the healpix quadtree, starting from the smallest
    (sub-)pixel that contains the cell's bounding sphere as seen from the origin, and discarding (sub-)pixels whose
    bounding cones miss the cell.

    For a projection, the volume of the cell within the cone of each pixel is then computed exactly (see
    _cone_volume), with corrections for the sides of healpix pixels not being great circles (see _pixel_volume),
    which make pixel areas correct to 3e-5 and the volume of each cell within each pixel correct to around 1e-4 of
    the cell's volume (or better, for cells much smaller than a pixel). Volumes are shared exactly between
    neighbouring pixels, so that the sum of qty * mass / rho is conserved to round-off, and a set of cells that tile a
    region give the same map as a single cell occupying that region, to the same precision.

    For a thin shell, the cell is sampled on healpix sub-pixels at nside * 2^k, which have equal areas and nest
    exactly within the pixels. A sub-pixel lying wholly within the cell (or, the shell being a sphere, wholly
    within it on the shell) is accepted as a whole; one wholly outside is discarded; and otherwise it is refined, up
    to a level k_cell, at which the fraction of the sub-pixel inside the cell is calculated by approximating the
    shell by its tangent plane (see _shell_fraction). For cells larger than about a pixel, k_cell = *edge_level*;
    for smaller cells, k_cell is increased so that each cell is at least *cells_across* sub-pixels across. Close to
    the poles, where the sides of sub-pixels are most curved, sub-pixels are refined further.

    If *deposit* is True, a cell whose bounding sphere lies entirely within one pixel (as seen from the origin) of a
    projection deposits qty * mass/rho into that pixel without further calculation. This makes no difference to the
    result, other than to speed it up. In shells, cells too small to be sampled at the finest level supported
    (nside * 2^k <= 2^29) are deposited into the pixel containing their centre, using the area of the planar
    cross-section of the cell.

    The Python interface is :class:`pynbody.imaging.renderers.AMRHealpixRenderer`.
    """

    if nside < 1 or nside & (nside - 1) != 0:
        raise ValueError("nside must be a power of 2")
    if shell_radius < 0 or not np.isfinite(shell_radius):
        raise ValueError("shell_radius must be zero (for a projection) or positive and finite")
    if edge_level < 0 or not cells_across > 0:
        raise ValueError("edge_level must be non-negative, and cells_across positive")

    cdef long long n_part = len(x)
    assert len(y) == len(z) == len(sm) == len(qty) == len(mass) == len(rho) == n_part, \
        "Inconsistent array lengths passed to render_healpix_cells"

    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.shape != (3, 3):
        raise ValueError("orientation must be a 3x3 matrix")

    cdef int base_level = 0
    while (1 << base_level) < nside:
        base_level += 1
    if base_level > MAX_LEVEL:
        raise ValueError("nside is too large")

    cdef long long nside_l = nside
    cdef long long npix = 12 * nside_l * nside_l
    cdef double pixel_solid_angle = 4.0 * M_PI / npix
    cdef double inv_pixel_solid_angle = 1.0 / pixel_solid_angle
    cdef double pixel_size = sqrt(pixel_solid_angle)
    cdef int projected = shell_radius == 0.0
    cdef double R = shell_radius

    cdef np.ndarray[np.float64_t, ndim=1] acc_np = np.zeros(npix, dtype=np.float64)
    cdef double *acc = <double *> acc_np.data

    # bounding cones of (sub-)pixels at each level
    cdef double level_rho[MAX_LEVEL + 1]
    cdef double level_cos[MAX_LEVEL + 1]
    cdef double level_sin[MAX_LEVEL + 1]
    cdef double level_chord[MAX_LEVEL + 1]   # chord length subtended by the cone's half-angle
    cdef double level_weight[MAX_LEVEL + 1]  # area of a sub-pixel relative to a pixel of the map
    cdef int lev
    for lev in range(MAX_LEVEL + 1):
        level_rho[lev] = max_pixel_radius(1 << lev) * PIXRAD_SAFETY + 1e-12
        if level_rho[lev] > M_PI:
            level_rho[lev] = M_PI
        level_cos[lev] = cos(level_rho[lev])
        level_sin[lev] = sin(level_rho[lev]) if level_rho[lev] < 0.5 * M_PI else 1.0
        level_chord[lev] = 2.0 * sin(0.5 * level_rho[lev])
        level_weight[lev] = 0.25 ** (lev - base_level) if lev >= base_level else 0.0

    cdef Cell g
    cdef int k, i, j
    for k in range(3):
        for i in range(3):
            g.axes[3 * k + i] = orientation[i, k]

    # traversal stack
    cdef int stack_level[STACK_SIZE]
    cdef int stack_face[STACK_SIZE]
    cdef long long stack_ix[STACK_SIZE]
    cdef long long stack_iy[STACK_SIZE]
    cdef int stack_inside[STACK_SIZE]
    cdef int sp, face, inside, child, shift
    cdef long long ix, iy, index
    cdef long long p
    cdef double n[3]
    cdef double e[3]
    cdef double sm_i, weight, r_bound, theta_cell, margin, area, value, cos_angle, delta, ae
    cdef double cos_prune
    cdef int k_cell, finest_level, test, outside, all_inside, tiny
    cdef long long ring
    cdef FaceCoordinates fc
    cdef double grid[27]
    cdef double grid_omz[9]
    cdef double corners[12]
    cdef double corner_omz[4]

    with nogil:
        for p in range(n_part):
            weight = qty[p] * mass[p] / rho[p]
            if weight != weight or weight == 0.0:
                continue
            sm_i = sm[p]
            if sm_i < min_smooth:
                sm_i = min_smooth
            if not (sm_i > 0 and isfinite(sm_i) and isfinite(x[p]) and isfinite(y[p]) and isfinite(z[p])):
                # cells with undefined position or size cannot be rendered
                continue

            g.c[0] = x[p]
            g.c[1] = y[p]
            g.c[2] = z[p]
            g.half = 0.5 * sm_i
            g.volume = sm_i * sm_i * sm_i
            g.value = weight / g.volume
            g.d = sqrt(_dot3(g.c, g.c))
            for k in range(3):
                g.ck[k] = _dot3(&g.axes[3 * k], g.c)
            for j in range(8):
                for i in range(3):
                    g.vert[3 * j + i] = 0.0
                    for k in range(3):
                        g.vert[3 * j + i] += (g.half if (j >> k) & 1 else -g.half) * g.axes[3 * k + i]
            r_bound = g.half * sqrt(3.0)
            g.far = g.d > r_bound * (1.0 + 1e-12)
            g.edges_ready = 0

            if not projected and fabs(g.d - R) > r_bound:
                continue  # the cell does not reach the shell

            if g.far:
                for k in range(3):
                    g.chat[k] = g.c[k] / g.d
                g.rho_c = asin(r_bound / g.d)
                g.cos_rho_c = cos(g.rho_c)
                g.sin_rho_c = r_bound / g.d
            else:
                g.rho_c = M_PI

            tiny = 0
            if projected:
                finest_level = base_level
            else:
                k_cell = edge_level
                if g.far:
                    theta_cell = sm_i / g.d
                    k_cell = <int> ceil(log2(cells_across * pixel_size / theta_cell))
                    if k_cell < edge_level:
                        k_cell = edge_level
                if base_level + k_cell > MAX_LEVEL:
                    k_cell = MAX_LEVEL - base_level
                    # the cell is too small to be sampled reliably, so deposit it in the pixel containing its centre
                    tiny = g.far
                finest_level = base_level + k_cell

            sp = 0
            if g.far:
                _face_coordinates(g.c, g.d, &fc)
                # Find the deepest (sub-)pixel containing the whole bounding cone, from which to start the traversal.
                # If this is a pixel of the map, or smaller, the cell can deposit its contribution directly.
                lev = base_level if projected else finest_level - 1
                if lev < 0:
                    lev = 0
                while lev >= 0:
                    margin = _locate(&fc, 1LL << lev, &face, &ix, &iy)
                    if margin > FACE_COORDINATE_GRADIENT * (1LL << lev) * g.rho_c * (1.0 + 1e-6) + 1e-6:
                        break
                    lev -= 1
                if (deposit and projected and lev >= base_level) or tiny:
                    margin = _locate(&fc, nside_l, &face, &ix, &iy)
                    if margin >= 0:
                        index = (face * nside_l + iy) * nside_l + ix
                        if projected:
                            acc[index] += weight * inv_pixel_solid_angle
                        else:
                            for k in range(3):
                                e[k] = _dot3(&g.axes[3 * k], g.chat)
                            area = _cube_section_area(g.half, e, R - g.d)
                            acc[index] += g.value * area / (R * R) * inv_pixel_solid_angle
                        continue
                if lev >= 0:
                    stack_level[0] = lev
                    stack_face[0] = face
                    stack_ix[0] = ix
                    stack_iy[0] = iy
                    stack_inside[0] = 0
                    sp = 1

            if sp == 0:
                for face in range(12):
                    stack_level[sp] = 0
                    stack_face[sp] = face
                    stack_ix[sp] = 0
                    stack_iy[sp] = 0
                    stack_inside[sp] = 0
                    sp += 1

            while sp > 0:
                sp -= 1
                lev = stack_level[sp]
                face = stack_face[sp]
                ix = stack_ix[sp]
                iy = stack_iy[sp]
                inside = stack_inside[sp]
                value = -1.0  # the contribution to the pixel; -1 means that the node must be refined

                _pixel_centre(1LL << lev, face, ix, iy, n)

                if projected and g.far and not inside:
                    cos_angle = _dot3(n, g.chat)
                    if g.rho_c + level_rho[lev] < M_PI:
                        cos_prune = g.cos_rho_c * level_cos[lev] - g.sin_rho_c * level_sin[lev]
                        if cos_angle < cos_prune:
                            continue  # the bounding cones do not overlap

                if projected and lev == base_level:
                    # the exact volume of the cell within this pixel (only reached if the traversal starts here)
                    for i in range(4):
                        _face_point(face, <double> (ix + (1 if i == 1 or i == 2 else 0)) / nside_l,
                                    <double> (iy + (1 if i >= 2 else 0)) / nside_l, &corners[3 * i], &corner_omz[i])
                    value = g.value * _pixel_volume(&g, nside_l, face, ix, iy, corners, corner_omz) \
                            * inv_pixel_solid_angle
                    if value != 0:
                        acc[(face * nside_l + iy) * nside_l + ix] += value
                    continue

                if inside:
                    # the whole node lies within the cell (for a shell), or within its outline (for a projection)
                    if lev >= base_level:
                        value = g.value * level_weight[lev]  # (a shell; projections have been dealt with above)
                elif projected:
                    # if the node is small compared to the cell, check whether it meets the outline of the cell
                    test = not g.far or level_rho[lev] <= 0.5 * g.rho_c
                    if test:
                        if not g.edges_ready:
                            _prepare_edges(&g)
                        if not _cone_meets_edges(&g, n, level_cos[lev], level_sin[lev]):
                            if _ray_moment(&g, n) <= 0:
                                continue  # entirely outside the cell
                            inside = 1
                else:
                    delta = R * level_chord[lev]
                    outside = 0
                    all_inside = 1
                    for k in range(3):
                        ae = fabs(R * _dot3(&g.axes[3 * k], n) - g.ck[k])
                        if ae > g.half + delta:
                            outside = 1
                        if ae >= g.half - delta:
                            all_inside = 0
                    if outside:
                        continue
                    if all_inside:
                        if lev >= base_level:
                            value = g.value * level_weight[lev]
                        else:
                            inside = 1
                    elif lev >= finest_level:
                        # Close to the poles, the sides of sub-pixels are curved (with sagitta ~1/8r of their length
                        # at the rth ring from the pole), so sub-pixels there are refined further
                        ring = _jrll(face) * (1LL << lev) - ix - iy - 1
                        if (lev >= finest_level + POLAR_REFINEMENT or lev == MAX_LEVEL
                                or (ring >= POLAR_RINGS and ring <= 4 * (1LL << lev) - POLAR_RINGS)):
                            value = g.value * level_weight[lev] * _shell_fraction(&g, face, 1LL << lev, ix, iy, n,
                                                                                  R)

                if value >= 0:
                    if value > 0:
                        shift = lev - base_level
                        acc[(face * nside_l + (iy >> shift)) * nside_l + (ix >> shift)] += value
                    continue

                if projected and lev == base_level - 1:
                    # Find the exact volume of the cell within each of the four pixels of the map in this node. Their
                    # corners lie on a 3x3 grid.
                    for j in range(3):
                        for i in range(3):
                            _face_point(face, <double> (2 * ix + i) / nside_l, <double> (2 * iy + j) / nside_l,
                                        &grid[3 * (3 * j + i)], &grid_omz[3 * j + i])
                    for child in range(4):
                        for i in range(4):
                            j = 3 * ((child >> 1) + (1 if i >= 2 else 0)) + (child & 1) + (1 if i == 1 or i == 2 else 0)
                            for k in range(3):
                                corners[3 * i + k] = grid[3 * j + k]
                            corner_omz[i] = grid_omz[j]
                        value = g.value * inv_pixel_solid_angle * _pixel_volume(
                            &g, nside_l, face, 2 * ix + (child & 1), 2 * iy + (child >> 1), corners, corner_omz)
                        if value != 0:
                            acc[(face * nside_l + 2 * iy + (child >> 1)) * nside_l + 2 * ix + (child & 1)] += value
                    continue

                for child in range(4):
                    stack_level[sp] = lev + 1
                    stack_face[sp] = face
                    stack_ix[sp] = 2 * ix + (child & 1)
                    stack_iy[sp] = 2 * iy + (child >> 1)
                    stack_inside[sp] = inside
                    sp += 1

    # convert to the RING scheme, removing any negative values left by round-off where there is nothing to render
    cdef np.ndarray[image_output_type, ndim=1] result = np.zeros(npix, dtype=np_image_output_type)
    cdef image_output_type *res = <image_output_type *> result.data
    with nogil:
        for face in range(12):
            for iy in range(nside_l):
                for ix in range(nside_l):
                    value = acc[(face * nside_l + iy) * nside_l + ix]
                    res[_ring_index(nside_l, face, ix, iy)] = value if value > 0 else 0.0
    return result


def _pixel_centre_vectors(int nside, int face, ix, iy):
    """Pixel centres (for testing): unit vectors towards the centres of the pixels with face coordinates ix, iy"""
    ix = np.atleast_1d(np.asarray(ix, dtype=np.int64))
    iy = np.atleast_1d(np.asarray(iy, dtype=np.int64))
    out = np.empty((len(ix), 3))
    cdef double v[3]
    cdef Py_ssize_t j
    for j in range(len(ix)):
        _pixel_centre(nside, face, ix[j], iy[j], v)
        out[j] = (v[0], v[1], v[2])
    return out


def _ring_indices(int nside, int face, ix, iy):
    """RING indices (for testing) of the pixels with face coordinates ix, iy"""
    ix = np.atleast_1d(np.asarray(ix, dtype=np.int64))
    iy = np.atleast_1d(np.asarray(iy, dtype=np.int64))
    return np.array([_ring_index(nside, face, ix[j], iy[j]) for j in range(len(ix))], dtype=np.int64)


def _locate_pixels(int nside, vectors):
    """For testing: (ring index, margin) of the pixel containing each direction"""
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    pix = np.empty(len(vectors), dtype=np.int64)
    margins = np.empty(len(vectors))
    cdef double c[3]
    cdef int face
    cdef long long ix, iy
    cdef FaceCoordinates fc
    cdef Py_ssize_t j
    for j in range(len(vectors)):
        c[0], c[1], c[2] = vectors[j]
        _face_coordinates(c, sqrt(_dot3(c, c)), &fc)
        margins[j] = _locate(&fc, nside, &face, &ix, &iy)
        pix[j] = _ring_index(nside, face, ix, iy)
    return pix, margins


def _pixel_corners(int nside, int face, long long ix, long long iy):
    """For testing: unit vectors towards the corners of the pixel with face coordinates (ix, iy), in the order used
    by _pixel_volume"""
    out = np.empty((4, 3))
    cdef double v[3]
    cdef double omz
    cdef int i
    for i in range(4):
        _face_point(face, <double> (ix + (1 if i == 1 or i == 2 else 0)) / nside,
                    <double> (iy + (1 if i >= 2 else 0)) / nside, v, &omz)
        out[i] = (v[0], v[1], v[2])
    return out


def cube_section_area(double half, normal, double h):
    """Area of the cross-section of the cube [-half, half]^3 with the plane normal . x = h, for unit normal"""
    cdef double nv[3]
    nv[0], nv[1], nv[2] = np.asarray(normal, dtype=np.float64)
    return _cube_section_area(half, nv, h)
