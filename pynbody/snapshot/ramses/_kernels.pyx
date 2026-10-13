# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Decoding the Fortran records of RAMSES output files, without holding the GIL.

Each decoding function takes a buffer read from a file and a table of *rows*, one for each stretch of data in the
buffer to decode, saying where it lies and where its values go; puts the values of the wanted cells (or, for
particles, of each family) in place in the destination arrays; and raises ValueError if a row points outside the
buffer. Values that show the buffer holds what was meant to be read (the lengths that start and end each Fortran
record, and the records giving each level and its number of octs) are checked in a separate step,
:func:`check_int32`, with checks planned with the reads (see :mod:`._plan`); except that :func:`cell_values`, whose
records are many, checks the lengths of those it decodes itself.

Destinations may be strided (such as a column of a ``pos`` array), so are passed as numpy arrays, whose data
pointers and strides are collected before the GIL is released. The GIL is released only where there is enough to
do for it to be worth handing to another thread.

Which cells are wanted is given by a mask of one byte per oct, bit ``ind`` (0 to ``2**ndim - 1``) set if cell
``ind`` is wanted. Wanted cells are put in the order pynbody has always used: within each stretch of octs, cell
``ind`` of every oct in turn, then the next ``ind``, and so on, each row saying where the first wanted cell of its
``ind`` goes.

:func:`leaf_mask`, :func:`records` and :func:`own_layout` help to find the layout of files when a snapshot is
opened (see :mod:`._layout`), and :func:`chunk_counts` where particles go (see :mod:`._selection`).
"""

cimport numpy as np

import numpy as np

from libc.stdint cimport int32_t, int64_t, uint8_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

np.import_array()

# Below this many bytes of work, a function keeps the GIL: handing it to another thread would cost more
cdef Py_ssize_t _NOGIL_NBYTES = 65536


cdef inline bint _record_ok(const uint8_t *buf, Py_ssize_t buflen, Py_ssize_t pos, Py_ssize_t nbytes) noexcept nogil:
    """True if a Fortran record of nbytes bytes of data starts at pos in buf"""
    cdef int32_t m1, m2
    if pos < 0 or nbytes < 0 or pos + nbytes + 8 > buflen:
        return False
    memcpy(&m1, buf + pos, 4)
    memcpy(&m2, buf + pos + 4 + nbytes, 4)
    return m1 == nbytes and m2 == nbytes


cdef class _Destinations:
    """Data pointers, strides and lengths of a list of 1D arrays (or Nones), for use without the GIL"""
    cdef char **data
    cdef Py_ssize_t *stride
    cdef Py_ssize_t *length
    cdef Py_ssize_t n
    cdef list _keep

    def __cinit__(self, list arrays, Py_ssize_t itemsize):
        cdef Py_ssize_t i
        cdef np.ndarray a
        self.n = len(arrays)
        self._keep = arrays
        self.data = <char **> malloc(max(self.n, 1) * sizeof(char *))
        self.stride = <Py_ssize_t *> malloc(max(self.n, 1) * sizeof(Py_ssize_t))
        self.length = <Py_ssize_t *> malloc(max(self.n, 1) * sizeof(Py_ssize_t))
        if self.data == NULL or self.stride == NULL or self.length == NULL:
            raise MemoryError()
        for i in range(self.n):
            if arrays[i] is None:
                self.data[i] = NULL
                self.stride[i] = self.length[i] = 0
                continue
            a = arrays[i]
            if a.ndim != 1 or a.itemsize != itemsize or not a.flags.writeable:
                raise ValueError("Destinations must be writeable one-dimensional arrays with items of %d bytes"
                                 % itemsize)
            self.data[i] = <char *> np.PyArray_DATA(a)
            self.stride[i] = a.strides[0]
            self.length[i] = a.shape[0]

    def __dealloc__(self):
        free(self.data)
        free(self.stride)
        free(self.length)


cdef int _leaf_mask(const uint8_t *b, Py_ssize_t buflen, const int64_t[:, ::1] segments, int twotondim,
                    int last_level, uint8_t[::1] mask, int64_t[:, ::1] nleaf) noexcept nogil:
    cdef Py_ssize_t s, i, ind, pos, n, level, m0, nbytes
    cdef int32_t v
    cdef int64_t count
    for s in range(segments.shape[0]):
        pos, level, n, m0 = segments[s, 0], segments[s, 1], segments[s, 2], segments[s, 3]
        nbytes = 4 * n
        for i in range(n):
            mask[m0 + i] = 0
        for ind in range(twotondim):
            if not _record_ok(b, buflen, pos, nbytes):
                return 1
            count = 0
            for i in range(n):
                memcpy(&v, b + pos + 4 + 4 * i, 4)
                if level == last_level or v == 0:
                    mask[m0 + i] |= <uint8_t> (1 << ind)
                    count += 1
            nleaf[level, ind] += count
            pos += nbytes + 8
    return 0


def leaf_mask(const uint8_t[::1] buf, const int64_t[:, ::1] segments, int twotondim, int last_level,
              uint8_t[::1] mask, int64_t[:, ::1] nleaf):
    """Find the leaf cells of octs from their refinement maps.

    Each row of segments, (buffer offset, level, number of octs n, offset in mask), locates twotondim records of n
    int32s, the refinement map of a level's octs. Bit ind of the mask byte of each oct is set if cell ind is a leaf
    (unrefined, or any cell at level last_level), and nleaf[level, ind] counts the leaves."""
    cdef Py_ssize_t s, total = 0, buflen = buf.shape[0]
    cdef int failed
    cdef const uint8_t *b = &buf[0] if buflen > 0 else NULL
    if twotondim > 8 or nleaf.shape[1] < twotondim:
        raise ValueError("Unexpected number of cells per oct")
    for s in range(segments.shape[0]):
        if segments[s, 3] + segments[s, 2] > mask.shape[0] or not 0 <= segments[s, 1] < nleaf.shape[0]:
            raise ValueError("Segment lies outside the mask")
        total += segments[s, 2] * twotondim * 4
    if total >= _NOGIL_NBYTES:
        with nogil:
            failed = _leaf_mask(b, buflen, segments, twotondim, last_level, mask, nleaf)
    else:
        failed = _leaf_mask(b, buflen, segments, twotondim, last_level, mask, nleaf)
    if failed:
        raise ValueError("Unexpected record length in the refinement map of an AMR file")


cdef int _cell_positions(const uint8_t *b, Py_ssize_t buflen, const int64_t[:, ::1] rows, int ndim,
                         const uint8_t[::1] mask, double boxlen, const double[::1] offset, double[:, :] pos,
                         double[:] smooth) noexcept nogil:
    cdef Py_ssize_t r, i, j, d, ind, n, m0, level
    cdef Py_ssize_t p[3]
    cdef int twotondim = 1 << ndim
    cdef double c, dx
    cdef double sub[3]
    for r in range(rows.shape[0]):
        n, m0, level = rows[r, 3], rows[r, 4], rows[r, 5]
        for d in range(ndim):
            p[d] = rows[r, d]
            if p[d] < 0 or p[d] + 8 * n > buflen:
                return 1
        dx = boxlen * (0.5 ** (level + 1))
        for ind in range(twotondim):
            sub[0] = (ind & 1) - 0.5
            sub[1] = ((ind >> 1) & 1) - 0.5
            sub[2] = (ind >> 2) - 0.5
            j = rows[r, 6 + ind]
            for i in range(n):
                if mask[m0 + i] & (1 << ind):
                    if j < 0 or j >= pos.shape[0] or j >= smooth.shape[0]:
                        return 2
                    for d in range(ndim):
                        memcpy(&c, b + p[d] + 8 * i, 8)  # (records are not aligned)
                        c = c - offset[d]
                        pos[j, d] = boxlen * c + dx * sub[d]
                    smooth[j] = dx
                    j += 1
    return 0


def cell_positions(const uint8_t[::1] buf, const int64_t[:, ::1] rows, int ndim, const uint8_t[::1] mask,
                   double boxlen, const double[::1] offset, double[:, :] pos, double[:] smooth):
    """Find the positions and sizes of wanted cells from the centres of their octs.

    Each row, (offset of the x, y and z centres, number of octs n, offset in mask, level, then where the first
    wanted cell of each ind goes), locates the centres of n consecutive octs of a level: ndim runs of n float64s
    in buf (unused offsets are ignored). The position of each cell whose bit is set in the mask (see leaf_mask) is
    put in pos, and its size in smooth."""
    cdef Py_ssize_t r, total = 0, buflen = buf.shape[0]
    cdef int twotondim = 1 << ndim
    cdef int failed
    cdef const uint8_t *b = &buf[0] if buflen > 0 else NULL
    if ndim > 3 or pos.shape[1] < ndim or offset.shape[0] < ndim or rows.shape[1] < 6 + twotondim:
        raise ValueError("Unexpected number of dimensions")
    for r in range(rows.shape[0]):
        if rows[r, 3] < 0 or rows[r, 4] < 0 or rows[r, 4] + rows[r, 3] > mask.shape[0]:
            raise ValueError("Row lies outside the mask")
        total += rows[r, 3] * ndim * 8
    if total >= _NOGIL_NBYTES:
        with nogil:
            failed = _cell_positions(b, buflen, rows, ndim, mask, boxlen, offset, pos, smooth)
    else:
        failed = _cell_positions(b, buflen, rows, ndim, mask, boxlen, offset, pos, smooth)
    if failed == 1:
        raise ValueError("Oct centres lie outside the data read")
    if failed == 2:
        raise ValueError("Cells lie outside the destination arrays")


cdef int _cell_values(const uint8_t *b, Py_ssize_t buflen, const int64_t[:, ::1] rows, int twotondim,
                      const uint8_t[::1] mask, _Destinations dest) noexcept nogil:
    cdef Py_ssize_t r, k, i, p, n, m0, ind, d, j, nvar, var_stride, stride, length
    cdef int64_t inds
    cdef bint check
    cdef char *out
    for r in range(rows.shape[0]):
        n, m0 = rows[r, twotondim], rows[r, twotondim + 1]
        nvar, var_stride, check = rows[r, twotondim + 3], rows[r, twotondim + 4], rows[r, twotondim + 5]
        inds = rows[r, twotondim + 6]
        for ind in range(twotondim):
            if not inds & (1 << ind):
                continue
            for k in range(nvar):
                p = rows[r, ind] + k * var_stride
                if p < 0 or p + 8 * n > buflen:
                    return 1
                if check and not _record_ok(b, buflen, p - 4, 8 * n):
                    return 2
                d = rows[r, twotondim + 2] + k
                if d >= dest.n or dest.data[d] == NULL:
                    continue
                stride, length = dest.stride[d], dest.length[d]
                j = rows[r, twotondim + 7 + ind]
                out = dest.data[d] + j * stride
                for i in range(n):
                    if mask[m0 + i] & (1 << ind):
                        if j < 0 or j >= length:
                            return 3
                        memcpy(out, b + p + 8 * i, 8)
                        out += stride
                        j += 1
    return 0


def cell_values(const uint8_t[::1] buf, const int64_t[:, ::1] rows, int twotondim, const uint8_t[::1] mask,
                list destinations):
    """Find the values of variables in wanted cells, from a hydro-like file.

    Each row locates the values of consecutive variables in every cell of n consecutive octs of a level: (with T =
    twotondim) T columns of where in buf the values of cell ind of the first variable start, then n, the offset in
    mask, the first destination, the number of variables, the stride in buf from one variable to the next, whether
    the values are whole Fortran records (whose lengths are then checked), a mask of which inds to decode (bit ind
    set for each), and T columns of where the first wanted cell of each ind goes. Each variable's values are n float64s. The value of each cell whose bit ind is set in the
    mask is put in the destination of its variable (counting from the first given), an array (or None, to skip it)
    in destinations."""
    cdef Py_ssize_t r, total = 0, buflen = buf.shape[0]
    cdef _Destinations dest = _Destinations(destinations, 8)
    cdef int failed
    cdef const uint8_t *b = &buf[0] if buflen > 0 else NULL
    if not 0 < twotondim <= 8 or (rows.shape[0] > 0 and rows.shape[1] < 2 * twotondim + 7):
        raise ValueError("Rows must have 2 * twotondim + 7 columns")
    for r in range(rows.shape[0]):
        if rows[r, twotondim] < 0 or rows[r, twotondim + 1] < 0 or \
                rows[r, twotondim + 1] + rows[r, twotondim] > mask.shape[0] or rows[r, twotondim + 2] < 0 or \
                rows[r, twotondim + 3] < 0 or rows[r, twotondim + 4] < 0:
            raise ValueError("Row lies outside the mask")
        total += rows[r, twotondim] * rows[r, twotondim + 3] * twotondim * 8
    if total >= _NOGIL_NBYTES:
        with nogil:
            failed = _cell_values(b, buflen, rows, twotondim, mask, dest)
    else:
        failed = _cell_values(b, buflen, rows, twotondim, mask, dest)
    if failed == 1:
        raise ValueError("Values lie outside the data read")
    if failed == 2:
        raise ValueError("Unexpected record length")
    if failed == 3:
        raise ValueError("Cells lie outside the destination arrays")


def check_int32(const uint8_t[::1] buf, const int64_t[:, ::1] checks):
    """Return the index of the first check, a row (offset in buf, expected value, ...), whose int32 at that offset
    does not have the expected value; or -1 if all pass."""
    cdef Py_ssize_t k, p, buflen = buf.shape[0]
    cdef int32_t v
    for k in range(checks.shape[0]):
        p = checks[k, 0]
        if p < 0 or p + 4 > buflen:
            return k
        memcpy(&v, &buf[p], 4)
        if v != checks[k, 1]:
            return k
    return -1


cdef int _scatter_rows(const uint8_t *b, const int64_t[:, ::1] rows, const uint8_t *family,
                       const Py_ssize_t *itemsize, char ***data, Py_ssize_t **stride, Py_ssize_t **length,
                       const Py_ssize_t *ndest, int64_t *starts) noexcept nogil:
    cdef Py_ssize_t r, i, k, field, size, j
    cdef uint8_t f
    cdef const uint8_t *src
    cdef char *out
    for r in range(rows.shape[0]):
        field = rows[r, 3]
        size = itemsize[field]
        src = b + rows[r, 0]
        for k in range(ndest[field]):
            starts[k] = rows[r, 4 + k]
        for i in range(rows[r, 2] - rows[r, 1]):
            f = family[rows[r, 1] + i]
            if f < ndest[field] and data[field][f] != NULL:
                j = starts[f]
                if j < 0 or j >= length[field][f]:
                    return 1
                out = data[field][f] + j * stride[field][f]
                # (fixed sizes, so that each copy compiles to a single move)
                if size == 8:
                    memcpy(out, src + 8 * i, 8)
                elif size == 4:
                    memcpy(out, src + 4 * i, 4)
                elif size == 1:
                    out[0] = <char> src[i]
                else:
                    memcpy(out, src + size * i, size)
                starts[f] = j + 1
    return 0


def scatter_rows(const uint8_t[::1] buf, const int64_t[:, ::1] rows, const uint8_t[::1] family, list fields):
    """Put particle values from buf in the destination of each particle's family.

    Each row, (offset in buf, first particle, last particle + 1, field, then where the first of those particles of
    each family goes), locates the values of a field for a run of particles, whose family ids are given in family
    (ids without a destination are skipped). fields lists, for each field, (item size, destination array of each
    family or None)."""
    cdef Py_ssize_t r, k, nfields = len(fields), maxdest = 0, total = 0, buflen = buf.shape[0]
    cdef list keep = [_Destinations(list(d), size) for size, d in fields]
    cdef _Destinations dd
    cdef int failed = 0
    cdef Py_ssize_t *itemsize = <Py_ssize_t *> malloc(max(nfields, 1) * sizeof(Py_ssize_t))
    cdef Py_ssize_t *ndest = <Py_ssize_t *> malloc(max(nfields, 1) * sizeof(Py_ssize_t))
    cdef char ***data = <char ***> malloc(max(nfields, 1) * sizeof(char **))
    cdef Py_ssize_t **stride = <Py_ssize_t **> malloc(max(nfields, 1) * sizeof(Py_ssize_t *))
    cdef Py_ssize_t **length = <Py_ssize_t **> malloc(max(nfields, 1) * sizeof(Py_ssize_t *))
    cdef int64_t *starts = NULL
    try:
        if itemsize == NULL or ndest == NULL or data == NULL or stride == NULL or length == NULL:
            raise MemoryError()
        for k in range(nfields):
            dd = keep[k]
            itemsize[k], ndest[k], data[k], stride[k], length[k] = fields[k][0], dd.n, dd.data, dd.stride, dd.length
            maxdest = max(maxdest, dd.n)
        starts = <int64_t *> malloc(max(maxdest, 1) * sizeof(int64_t))
        if starts == NULL:
            raise MemoryError()
        for r in range(rows.shape[0]):
            if not 0 <= rows[r, 3] < nfields or rows.shape[1] < 4 + ndest[rows[r, 3]]:
                raise ValueError("Row names an unknown field, or too few start indices")
            if not 0 <= rows[r, 1] <= rows[r, 2] <= family.shape[0]:
                raise ValueError("Row names particles outside the family ids given")
            if rows[r, 0] < 0 or rows[r, 0] + (rows[r, 2] - rows[r, 1]) * itemsize[rows[r, 3]] > buflen:
                raise ValueError("Values lie outside the data read")
            total += (rows[r, 2] - rows[r, 1]) * itemsize[rows[r, 3]]
        if rows.shape[0] > 0:
            if total >= _NOGIL_NBYTES:
                with nogil:
                    failed = _scatter_rows(&buf[0], rows, &family[0], itemsize, data, stride, length, ndest, starts)
            else:
                failed = _scatter_rows(&buf[0], rows, &family[0], itemsize, data, stride, length, ndest, starts)
        if failed:
            raise ValueError("Particles lie outside the destination arrays")
    finally:
        free(itemsize)
        free(ndest)
        free(data)
        free(stride)
        free(length)
        free(starts)


def chunk_counts(const uint8_t[::1] family, Py_ssize_t chunk, Py_ssize_t nfamilies):
    """Count the particles of each family (ignoring ids of nfamilies or more) in each chunk of chunk particles,
    returning an array (chunks, families)"""
    cdef Py_ssize_t c, i, stop, n = family.shape[0]
    if chunk <= 0:
        raise ValueError("Chunks must hold particles")
    out = np.zeros(((n + chunk - 1) // chunk, nfamilies), dtype=np.int64)
    cdef int64_t[:, ::1] o = out
    with nogil:
        for c in range(o.shape[0]):
            stop = min(n, (c + 1) * chunk)
            for i in range(c * chunk, stop):
                if family[i] < nfamilies:
                    o[c, family[i]] += 1
    return out


def records(const uint8_t[::1] buf, Py_ssize_t pos, Py_ssize_t count):
    """Locate up to count consecutive Fortran records starting at pos in buf. Returns an array of (where its data start,
    number of bytes) for each record wholly within buf, which may be fewer than count; raises ValueError if the lengths
    at the start and end of a record disagree."""
    cdef Py_ssize_t k = 0, buflen = buf.shape[0]
    cdef int32_t m1, m2
    out = np.empty((count, 2), dtype=np.int64)
    cdef int64_t[:, ::1] o = out
    while k < count and pos + 4 <= buflen:
        memcpy(&m1, &buf[pos], 4)
        if m1 < 0:
            raise ValueError("Invalid record length")
        if pos + 8 + m1 > buflen:
            break
        memcpy(&m2, &buf[pos + 4 + m1], 4)
        if m1 != m2:
            raise ValueError("Inconsistent record lengths")
        o[k, 0] = pos + 4
        o[k, 1] = m1
        pos += 8 + m1
        k += 1
    return out[:k]


def own_layout(const int32_t[:, ::1] counts, Py_ssize_t cpu):
    """From the numbers of octs held for each level (row) and CPU or boundary (column) in an AMR file, return, for
    each level, the number of the given CPU's octs (counting CPUs from 1), and the number of octs and of nonempty
    blocks of octs that precede them in the file (which is written level by level, each CPU then each boundary)."""
    cdef Py_ssize_t level, j, nlevels = counts.shape[0], ncols = counts.shape[1]
    cdef int64_t octs = 0, blocks = 0
    if not 1 <= cpu <= ncols:
        raise ValueError("No such CPU in the AMR file")
    own, octs_before, blocks_before = np.empty((3, nlevels), dtype=np.int64)
    cdef int64_t[::1] o = own, ob = octs_before, bb = blocks_before
    for level in range(nlevels):
        for j in range(ncols):
            if j == cpu - 1:
                o[level] = counts[level, j]
                ob[level] = octs
                bb[level] = blocks
            octs += counts[level, j]
            blocks += counts[level, j] > 0
    return own, octs_before, blocks_before
