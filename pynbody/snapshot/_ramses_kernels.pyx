# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Decoding the Fortran records of RAMSES output files, without holding the GIL.

Each function takes a buffer read from a file, and a table of the *segments* of it to decode (one row each, giving
where in the buffer the segment starts and what it holds); checks that the segments hold Fortran records (each a
4-byte length, the data, and the length again) of the lengths expected; and puts the values belonging to leaf cells
(or, for particles, to each family) in place in the destination arrays. Destinations may be strided (such as a
column of a ``pos`` array), so are passed as numpy arrays, whose data pointers and strides are collected before the
GIL is released. The GIL is released only where there is enough to do for it to be worth handing to another
thread.

Cells are put in the order pynbody has always used: within each level of each file, cell ``ind`` (0 to
``2**ndim - 1``) of every oct in turn, then the next ``ind``, and so on. ``starts[level, ind]`` gives where in the
destination the first leaf cell of each level and ``ind`` goes.

See :mod:`pynbody.snapshot._ramses_reader` for how buffers are read and work shared between threads.
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


cdef inline bint _int_record_is(const uint8_t *buf, Py_ssize_t buflen, Py_ssize_t pos, int32_t value) noexcept nogil:
    cdef int32_t v
    if not _record_ok(buf, buflen, pos, 4):
        return False
    memcpy(&v, buf + pos + 4, 4)
    return v == value


cdef class _Destinations:
    """Data pointers and strides of a list of 1D arrays (or Nones), for use without the GIL"""
    cdef char **data
    cdef Py_ssize_t *stride
    cdef Py_ssize_t n
    cdef list _keep

    def __cinit__(self, list arrays, Py_ssize_t itemsize):
        cdef Py_ssize_t i
        cdef np.ndarray a
        self.n = len(arrays)
        self._keep = arrays
        self.data = <char **> malloc(max(self.n, 1) * sizeof(char *))
        self.stride = <Py_ssize_t *> malloc(max(self.n, 1) * sizeof(Py_ssize_t))
        if self.data == NULL or self.stride == NULL:
            raise MemoryError()
        for i in range(self.n):
            if arrays[i] is None:
                self.data[i] = NULL
                self.stride[i] = 0
                continue
            a = arrays[i]
            if a.ndim != 1 or a.itemsize != itemsize or not a.flags.writeable:
                raise ValueError("Destinations must be writeable one-dimensional arrays with items of %d bytes"
                                 % itemsize)
            self.data[i] = <char *> np.PyArray_DATA(a)
            self.stride[i] = a.strides[0]

    def __dealloc__(self):
        free(self.data)
        free(self.stride)


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


cdef int _leaf_positions(const uint8_t *b, Py_ssize_t buflen, const int64_t[:, ::1] segments, int ndim,
                         const uint8_t[::1] mask, double boxlen, const double[::1] offset, double[:, :] pos,
                         double[:] smooth, const int64_t[:, ::1] starts) noexcept nogil:
    cdef Py_ssize_t s, i, j, d, ind, p, n, level, m0, nbytes
    cdef int twotondim = 1 << ndim
    cdef double c, dx
    cdef double sub[3]
    for s in range(segments.shape[0]):
        p, level, n, m0 = segments[s, 0], segments[s, 1], segments[s, 2], segments[s, 3]
        nbytes = 8 * n
        for d in range(ndim):
            if not _record_ok(b, buflen, p + d * (nbytes + 8), nbytes):
                return 1
        dx = boxlen * (0.5 ** (level + 1))
        for ind in range(twotondim):
            sub[0] = (ind & 1) - 0.5
            sub[1] = ((ind >> 1) & 1) - 0.5
            sub[2] = (ind >> 2) - 0.5
            j = starts[level, ind]
            for i in range(n):
                if mask[m0 + i] & (1 << ind):
                    for d in range(ndim):
                        memcpy(&c, b + p + d * (nbytes + 8) + 4 + 8 * i, 8)  # (records are not aligned)
                        c = c - offset[d]
                        pos[j, d] = boxlen * c + dx * sub[d]
                    smooth[j] = dx
                    j += 1
    return 0


def leaf_positions(const uint8_t[::1] buf, const int64_t[:, ::1] segments, int ndim, const uint8_t[::1] mask,
                   double boxlen, const double[::1] offset, double[:, :] pos, double[:] smooth,
                   const int64_t[:, ::1] starts):
    """Find the positions and sizes of leaf cells from the centres of their octs.

    Each row of segments, (buffer offset, level, number of octs n, offset in mask), locates ndim records of n
    float64s, the centres of a level's octs. The position of each leaf cell (see leaf_mask) is put in pos, and its
    size in smooth."""
    cdef Py_ssize_t s, total = 0, buflen = buf.shape[0]
    cdef int twotondim = 1 << ndim
    cdef int failed
    cdef const uint8_t *b = &buf[0] if buflen > 0 else NULL
    if ndim > 3 or pos.shape[1] < ndim or offset.shape[0] < ndim or starts.shape[1] < twotondim:
        raise ValueError("Unexpected number of dimensions")
    for s in range(segments.shape[0]):
        if segments[s, 3] + segments[s, 2] > mask.shape[0] or not 0 <= segments[s, 1] < starts.shape[0]:
            raise ValueError("Segment lies outside the mask")
        total += segments[s, 2] * ndim * 8
    if total >= _NOGIL_NBYTES:
        with nogil:
            failed = _leaf_positions(b, buflen, segments, ndim, mask, boxlen, offset, pos, smooth, starts)
    else:
        failed = _leaf_positions(b, buflen, segments, ndim, mask, boxlen, offset, pos, smooth, starts)
    if failed:
        raise ValueError("Unexpected record length among the grid centres of an AMR file")


cdef int _leaf_values(const uint8_t *b, Py_ssize_t buflen, const int64_t[:, ::1] segments, int nvar_file,
                      const uint8_t[::1] mask, _Destinations dest, const int64_t[:, ::1] starts) noexcept nogil:
    cdef Py_ssize_t s, i, ind, ivar, p, n, level, m0, ind0, ind1, nbytes, stride
    cdef char *out
    for s in range(segments.shape[0]):
        p, level, n, m0, ind0, ind1 = segments[s, 0], segments[s, 1], segments[s, 2], segments[s, 3], \
            segments[s, 4], segments[s, 5]
        nbytes = 8 * n
        if ind0 == 0:
            if not (_int_record_is(b, buflen, p, level + 1) and _int_record_is(b, buflen, p + 12, n)):
                return 1
            p += 24
        for ind in range(ind0, ind1):
            for ivar in range(nvar_file):
                if not _record_ok(b, buflen, p, nbytes):
                    return 2
                if ivar < dest.n and dest.data[ivar] != NULL:
                    stride = dest.stride[ivar]
                    out = dest.data[ivar] + starts[level, ind] * stride
                    for i in range(n):
                        if mask[m0 + i] & (1 << ind):
                            memcpy(out, b + p + 4 + 8 * i, 8)
                            out += stride
                p += nbytes + 8
    return 0


def leaf_values(const uint8_t[::1] buf, const int64_t[:, ::1] segments, int nvar_file, const uint8_t[::1] mask,
                list destinations, const int64_t[:, ::1] starts):
    """Find the values of variables in leaf cells, from a hydro-like file.

    Each row of segments, (buffer offset, level, number of octs n, offset in mask, ind0, ind1), locates, for each ind
    from ind0 to ind1 - 1, nvar_file records of n float64s, the values of each variable in cell ind of a level's own
    octs. If ind0 is 0, they are preceded by the records giving the level (counting from 1) and n, which are
    checked. The value of each leaf cell (see leaf_mask) is put in the destination of its variable: destinations
    holds an array (or None, to skip that variable) for each of the first len(destinations) variables in the file."""
    cdef Py_ssize_t s, total = 0, buflen = buf.shape[0]
    cdef _Destinations dest = _Destinations(destinations, 8)
    cdef int failed
    cdef const uint8_t *b = &buf[0] if buflen > 0 else NULL
    for s in range(segments.shape[0]):
        if segments[s, 3] + segments[s, 2] > mask.shape[0] or not 0 <= segments[s, 1] < starts.shape[0] \
                or not 0 <= segments[s, 4] <= segments[s, 5] <= starts.shape[1]:
            raise ValueError("Segment lies outside the mask")
        total += segments[s, 2] * (segments[s, 5] - segments[s, 4]) * nvar_file * 8
    if total >= _NOGIL_NBYTES:
        with nogil:
            failed = _leaf_values(b, buflen, segments, nvar_file, mask, dest, starts)
    else:
        failed = _leaf_values(b, buflen, segments, nvar_file, mask, dest, starts)
    if failed == 1:
        raise ValueError("A level of a hydro-like file does not start where expected: the file may not match its AMR "
                         "file, or be in a format pynbody does not recognise")
    elif failed:
        raise ValueError("Unexpected record length in a hydro-like file")


cdef inline void _scatter(const uint8_t *src, Py_ssize_t n, Py_ssize_t itemsize, const uint8_t *family,
                          _Destinations dest, int64_t *starts) noexcept nogil:
    cdef Py_ssize_t i
    cdef uint8_t f
    for i in range(n):
        f = family[i]
        if f < dest.n and dest.data[f] != NULL:
            # (fixed sizes, so that each copy compiles to a single move)
            if itemsize == 8:
                memcpy(dest.data[f] + starts[f] * dest.stride[f], src + 8 * i, 8)
            elif itemsize == 4:
                memcpy(dest.data[f] + starts[f] * dest.stride[f], src + 4 * i, 4)
            elif itemsize == 1:
                dest.data[f][starts[f] * dest.stride[f]] = <char> src[i]
            else:
                memcpy(dest.data[f] + starts[f] * dest.stride[f], src + itemsize * i, itemsize)
            starts[f] += 1


def scatter_by_family(const uint8_t[::1] buf, Py_ssize_t offset, Py_ssize_t itemsize, const uint8_t[::1] family,
                      list destinations, int64_t[::1] starts):
    """Put each item of buf, starting at offset (len(family) items of itemsize bytes), in the destination of its
    family, starting at starts[family] (which is advanced past what is put there). Items of families without a
    destination are skipped."""
    cdef Py_ssize_t n = family.shape[0]
    cdef _Destinations dest = _Destinations(destinations, itemsize)
    if offset < 0 or buf.shape[0] < offset + n * itemsize:
        raise ValueError("Too few data for the family ids given")
    if starts.shape[0] < dest.n:
        raise ValueError("Too few start indices for the destinations given")
    if n == 0:
        return
    if n * itemsize >= _NOGIL_NBYTES:
        with nogil:
            _scatter(&buf[offset], n, itemsize, &family[0], dest, &starts[0])
    else:
        _scatter(&buf[offset], n, itemsize, &family[0], dest, &starts[0])


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
