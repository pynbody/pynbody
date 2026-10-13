"""Reading the binary files of a RAMSES output, with threads.

A RAMSES output has, for each CPU (domain) of the run, an AMR file describing the octs that CPU holds; hydro,
gravity and RT files holding the values of variables in the cells of those octs; and a particle file. Each file
also holds the "virtual" octs of neighbouring CPUs, which pynbody skips.

Every read is described as a list of *pieces*, in file order, each something to read from one file and a way to
decode what was read: to pick out the leaf cells (or each family's particles) and put them in place, which the
Cython functions of :mod:`._kernels` do without holding the GIL. :func:`_run` performs the pieces through the
threaded pipeline of :mod:`pynbody.util.hdf_bulk_read.execute`: consecutive pieces are gathered into jobs of a few
megabytes (so that the cost of handing a job between threads is small beside the work in it, even where each file
holds little), and consecutive jobs into tasks for the input threads, each of which reads its files one after
another, sequentially; decoding threads decode. :func:`choose_strategy` decides how many threads of each kind to use.

Most of a RAMSES file's layout follows from the numbers of octs each CPU holds at each level, which the AMR file
records. So when a snapshot is opened, :func:`scan_amr` reads the header of each AMR file, works out where its own
octs lie in it and in the corresponding hydro-like files, and reads their refinement maps to find which cells are
leaves (keeping one byte per oct). Later reads then go straight to the records wanted, rather than through every
record of every file (whose number grows as the square of the number of CPUs), and the AMR files need not be read
again to load hydro, gravity or RT variables. Each piece read is checked to hold the records expected there.

Particle files are walked record by record when the snapshot is opened (see :func:`count_particles`), so that each
field can later be read straight from where it lies.

A file whose contents are not as expected (for example, one truncated, or not matching its AMR file) raises
ValueError, rather than OSError, which pynbody's lazy loading takes to mean that an array is not on disk at all.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import typing

import numpy as np

from ... import config_parser
from ...util.hdf_bulk_read import execute, strategy
from ...util.hdf_bulk_read.common import Job
from . import _kernels as kernels

logger = logging.getLogger('pynbody.snapshot.ramses')

# The least a job reads, where there is enough to read: big enough that the work of handing jobs between threads is
# small beside it, and small enough that a few per thread take little memory
_job_nbytes = 4 * 1024 * 1024

# Particles are read in chunks of this many
_particle_chunk = _job_nbytes // 8

# What reading the header of a file, and what is found through it, is guessed to take before it has been read
_header_nbytes_guess = 256 * 1024

# Byte ranges of a file closer together than this are read in one go, with what lies between them
_max_gap = 64 * 1024


def _read_config():
    def option(name, default):
        value = config_parser.get('ramses', name, fallback=str(default)).strip().lower()
        if value == 'auto':
            return value
        try:
            return int(value)
        except ValueError:
            logger.warning("Ignoring the value %r of %s in the [ramses] configuration", value, name)
            return default

    return {'parallel-read': option('parallel-read', 8), 'io-threads': option('io-threads', 'auto')}


config = _read_config()
"""How many threads read RAMSES files: initially from the ``[ramses]`` section of the configuration, but may be changed
at run time. ``parallel-read`` is the most threads decoding (``auto`` for as many as there are CPUs); if 1 or less,
everything is done in the calling thread. ``io-threads`` is the number of threads reading files, or ``auto`` for 16
on a parallel filesystem and one otherwise (the rule used for HDF5 files: see
:mod:`pynbody.util.hdf_bulk_read.strategy`)."""


def choose_strategy(paths: list[str], num_jobs: int, max_job_nbytes: int) -> strategy.ReadStrategy:
    """Decide how many threads to read *paths* with, in *num_jobs* jobs of at most *max_job_nbytes*"""
    threads = config['parallel-read']
    if threads == 'auto':
        threads = strategy.available_cpus()
    if threads <= 1 or num_jobs <= 1:
        return strategy.ReadStrategy(io_threads=1, decode_threads=0,
                                     reason="reading in the calling thread (parallel-read is %s; %d jobs)"
                                            % (config['parallel-read'], num_jobs))
    filesystem = strategy.filesystem_type(paths[0]) if paths else None
    if config['io-threads'] != 'auto':
        io_threads = max(int(config['io-threads']), 1)
    elif filesystem in strategy.parallel_filesystem_types:
        io_threads = 16
    else:
        io_threads = 1
    decode_threads = min(threads, strategy.available_cpus(), num_jobs)
    return strategy.ReadStrategy(io_threads=io_threads, decode_threads=decode_threads,
                                 reason=f"{io_threads} input and {decode_threads} decoding thread(s) for {num_jobs} "
                                        f"jobs on a {filesystem or 'local'} filesystem",
                                 inflight_nbytes=2 * decode_threads * max(max_job_nbytes, 1))


class _Piece(typing.NamedTuple):
    """Something to read from one file, and what to do with it"""
    path: str
    nbytes: int  # roughly what it takes in memory once read
    read: typing.Callable  # given the open file, returns what was read
    decode: typing.Callable  # given what was read, puts it in place


class _Task:
    """Jobs, each reading a list of pieces, which are fetched one after another by one input thread. Consecutive
    pieces from the same file share one opening of it."""

    def __init__(self):
        self.jobs = []
        self._path = None
        self._file = None

    def file(self, path):
        if path != self._path:
            self._close()
            self._file = open(path, 'rb')
            self._path = path
        return self._file

    def _close(self):
        if self._file is not None:
            self._file.close()
            self._file = self._path = None

    def __iter__(self):
        try:
            self.jobs.reverse()
            while self.jobs:
                yield self.jobs.pop()  # (so that each job is let go of once done)
        finally:
            self._close()


def _run(pieces: list[_Piece]):
    """Read and decode the pieces, gathered into jobs and tasks (see the module docstring)"""
    tasks, job, job_nbytes, task_nbytes = [], [], 0, 0

    def finish_job():
        nonlocal job, job_nbytes
        if job:
            tasks[-1].jobs.append(_job(tasks[-1], job, job_nbytes))
            job, job_nbytes = [], 0

    for i, piece in enumerate(pieces):
        # a task (which an input thread reads straight through) ends only at the end of a file, and only once it has
        # enough to read; a job ends once it has enough
        if i == 0 or (piece.path != pieces[i - 1].path and task_nbytes >= _job_nbytes):
            finish_job()
            tasks.append(_Task())
            task_nbytes = 0
        job.append(piece)
        job_nbytes += piece.nbytes
        task_nbytes += piece.nbytes
        if job_nbytes >= _job_nbytes:
            finish_job()
    finish_job()

    plan = choose_strategy(list(dict.fromkeys(p.path for p in pieces)), sum(len(t.jobs) for t in tasks),
                           max((j.nbytes for t in tasks for j in t.jobs), default=0))
    logger.debug("Reading RAMSES files: %s", plan.reason)
    execute.perform_tasks(tasks, plan, iter)


def _job(task: _Task, pieces: list[_Piece], nbytes: int) -> Job:
    def fetch():
        return [piece.read(task.file(piece.path)) for piece in pieces]

    def process(fetched):
        for piece, data in zip(pieces, fetched):
            piece.decode(data)

    return Job(fetch, process, max(nbytes, 1))


class _Records:
    """Fortran records of an open file, read in turn"""

    def __init__(self, f):
        self._f = f

    def length(self) -> int:
        """Read the length that starts (or ends) a record"""
        b = self._f.read(4)
        if len(b) != 4:
            raise ValueError(f"Unexpected end of {self._f.name}")
        return int.from_bytes(b, sys.byteorder, signed=True)

    def read(self, dtype) -> np.ndarray:
        length = self.length()
        data = self._f.read(length)
        if len(data) != length or self.length() != length:
            raise ValueError(f"Inconsistent record lengths in {self._f.name}")
        return np.frombuffer(data, dtype=dtype)

    def tell(self) -> int:
        return self._f.tell()


def _read_spans(f, offsets, nbytes) -> tuple[np.ndarray, list]:
    """Read byte ranges of an open file (in increasing order, not overlapping) into one buffer, reading ranges that lie
    close together in one go, with what lies between them; return the buffer, and where in it each range starts"""
    spans, at, span_at = [], [], 0  # spans are [first byte, last byte + 1, where in the buffer]
    for offset, n in zip(offsets, nbytes):
        if not spans or offset - spans[-1][1] > _max_gap:
            if spans:
                span_at += spans[-1][1] - spans[-1][0]
            spans.append([offset, offset + n, span_at])
        else:
            spans[-1][1] = offset + n
        at.append(spans[-1][2] + offset - spans[-1][0])
    buf = np.empty(spans[-1][2] + spans[-1][1] - spans[-1][0] if spans else 0, dtype=np.uint8)
    view = memoryview(buf)
    for start, end, span_at in spans:
        f.seek(start)
        if f.readinto(view[span_at:span_at + end - start]) != end - start:
            raise ValueError(f"{f.name} is shorter than expected")
    return buf, at


def _records_at(f, pos: int, count: int, nbytes_guess: int) -> tuple[np.ndarray, np.ndarray]:
    """Read *count* consecutive Fortran records starting at *pos*, returning a buffer and (where in it each record's
    data start, number of bytes)"""
    nbytes = nbytes_guess
    while True:
        f.seek(pos)
        buf = np.frombuffer(f.read(nbytes), dtype=np.uint8)
        table = kernels.records(buf, 0, count)
        if len(table) == count:
            return buf, table
        if len(buf) < nbytes:
            raise ValueError(f"Unexpected end of {f.name}")
        nbytes *= 4


def _record_value(buf, table, k, dtype='i'):
    dtype = np.dtype(dtype)
    return np.frombuffer(buf, dtype, int(table[k, 1]) // dtype.itemsize, int(table[k, 0]))


def _pieces(path: str, segments: np.ndarray, decode) -> list[_Piece]:
    """Pieces reading the *segments* of a file, an array of rows (file offset, number of bytes, ...) in increasing
    order of offset, a few megabytes' worth at a time. Each piece calls decode(buffer, rows) with the rows of its
    segments, their offsets replaced by where in the buffer read they lie."""
    pieces, i, n = [], 0, len(segments)
    cumulative = np.cumsum(segments[:, 1])
    while i < n:
        # (the segments up to and including the first to take the piece to _job_nbytes)
        j = int(np.searchsorted(cumulative, (cumulative[i - 1] if i else 0) + _job_nbytes, side='left')) + 1
        j = min(max(j, i + 1), n)
        rows = segments[i:j]

        def read(f, rows=rows):
            return _read_spans(f, rows[:, 0].tolist(), rows[:, 1].tolist())

        def decode_piece(fetched, rows=rows):
            buf, at = fetched
            rows = rows.copy()
            rows[:, 0] = at
            decode(buf, rows)

        pieces.append(_Piece(path, int(rows[:, 1].sum()), read, decode_piece))
        i = j
    return pieces


def _destinations(arrays, dtype):
    """Plain ndarray views of *arrays* to write *dtype* items into directly, and a list of (array, temporary) for
    any of a different dtype, which are written into a temporary and copied over afterwards"""
    out, copies = [], []
    for a in arrays:
        if a is None:
            out.append(None)
        elif a.dtype == dtype:
            out.append(a.view(np.ndarray))
        else:
            temporary = np.zeros(a.shape, dtype=dtype)  # (zeros: not every element need be written, e.g. pos in 2D)
            copies.append((a, temporary))
            out.append(temporary)
    return out, copies


# ---------------------------------------------------------------------------------------------------------------
# AMR structure

@dataclasses.dataclass
class AmrGeometry:
    """What all the AMR files of an output share"""
    ndim: int
    nlevelmax: int
    offset: np.ndarray  # subtracted from oct centres (as stored) to give positions in units of the box

    @property
    def twotondim(self) -> int:
        return 2 ** self.ndim


@dataclasses.dataclass
class AmrLayout:
    """Where one CPU's own octs lie in its files, and which of their cells are leaves"""
    ncache: np.ndarray  # number of own octs at each level read
    amr_offset: np.ndarray  # where the own octs of each level start in the AMR file
    entries_before: np.ndarray  # (level, CPU) entries preceding the own octs of each level, in a hydro-like file
    blocks_before: np.ndarray  # of which, entries with octs
    octs_before: np.ndarray  # and the number of octs they hold
    mask: np.ndarray = None  # one byte per own oct, bit ind set if cell ind is a leaf
    nleaf: np.ndarray = None  # number of leaf cells of each ind at each level

    @property
    def nlevels(self) -> int:
        return len(self.ncache)

    @property
    def ncells(self) -> int:
        return int(self.nleaf.sum())

    @property
    def mask_offsets(self) -> np.ndarray:
        """Where in mask the octs of each level start"""
        return np.concatenate([[0], np.cumsum(self.ncache)[:-1]]).astype(np.int64)

    def leaf_starts(self, first: int) -> np.ndarray:
        """Where in the gas arrays the first leaf cell of each (level, ind) goes, if this file's cells start at
        *first*"""
        starts = np.zeros(self.nleaf.size, dtype=np.int64)
        np.cumsum(self.nleaf.ravel()[:-1], out=starts[1:])
        return (starts + first).reshape(self.nleaf.shape)


def _read_amr_header(f, bisection: bool):
    """Read the header of an AMR file, returning (header dict, octs of each CPU and boundary at each level,
    where the octs start)"""
    # The first 19 records are small; then come three of nlevelmax x ncpu integers (the heads and tails of linked
    # lists of octs, which are skipped, and the number of octs held), and a few more
    buf, table = _records_at(f, 0, 19, 8 * 1024)
    ncpu, ndim, nlevelmax = (int(_record_value(buf, table, k)[0]) for k in (0, 1, 3))
    header = dict(ncpu=ncpu, ndim=ndim, ng=_record_value(buf, table, 2), nlevelmax=nlevelmax,
                  nboundary=int(_record_value(buf, table, 5)[0]))
    list_nbytes = 4 * nlevelmax * ncpu
    pos = int(table[18, 0] + table[18, 1]) + 4 + 2 * (8 + list_nbytes)
    count = 2 + (3 if header['nboundary'] > 0 else 0) + 2 + (5 if bisection else 1) + 3
    buf, table = _records_at(f, pos, count, list_nbytes + 40 * nlevelmax + 8 * ncpu + 8 * 1024)
    if table[0, 1] != list_nbytes:
        raise ValueError(f"Unexpected layout of the header of {f.name}")
    counts = _record_value(buf, table, 0).reshape((nlevelmax, ncpu))
    if header['nboundary'] > 0:
        boundary = _record_value(buf, table, 4).reshape((nlevelmax, header['nboundary']))
        counts = np.concatenate([counts, boundary], axis=1)
    return header, np.ascontiguousarray(counts), pos + int(table[-1, 0] + table[-1, 1]) + 4


def _layout(cpu, ndim, counts, start, nlevels) -> AmrLayout:
    """Work out where the own octs of each level lie, from the numbers of octs in the blocks of the file"""
    twotondim = 2 ** ndim
    ncache, octs_before, blocks_before = kernels.own_layout(counts[:nlevels], cpu)
    # a block of n > 0 octs has nint records of n integers and ndim of n reals, taking nint (8 + 4n) + ndim (8 + 8n)
    # bytes
    nint = 3 + 1 + 2 * ndim + 3 * twotondim
    return AmrLayout(ncache=ncache,
                     amr_offset=start + (4 * nint + 8 * ndim) * octs_before + (8 * nint + 8 * ndim) * blocks_before,
                     entries_before=np.arange(nlevels, dtype=np.int64) * counts.shape[1] + (cpu - 1),
                     blocks_before=blocks_before, octs_before=octs_before)


def _amr_segments(layout: AmrLayout, ndim: int, which: str) -> np.ndarray:
    """Rows (file offset, number of bytes, level, number of octs, offset in mask) locating the oct centres ('centres')
    or refinement maps ('refinement') of each level's own octs"""
    twotondim = 2 ** ndim
    n = layout.ncache
    if which == 'centres':
        offset, nbytes = layout.amr_offset + 3 * (8 + 4 * n), ndim * (8 + 8 * n)
    else:
        offset = layout.amr_offset + (4 + 2 * ndim + 2 * twotondim) * (8 + 4 * n) + ndim * (8 + 8 * n)
        nbytes = twotondim * (8 + 4 * n)
    rows = np.empty((layout.nlevels, 5), dtype=np.int64)
    rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4] = offset, nbytes, np.arange(layout.nlevels), n, \
        layout.mask_offsets
    return rows[n > 0]


def scan_amr(paths: list[str], cpus: list[int], bisection: bool, maxlevel: int | None) \
        -> tuple[AmrGeometry, list[AmrLayout]]:
    """Find the layout and leaf cells of each CPU's own octs (see the module docstring)"""
    layouts = [None] * len(paths)
    geometry = []

    def piece(i):
        def read(f):
            header, counts, start = _read_amr_header(f, bisection)
            nlevels = min(maxlevel or header['nlevelmax'], header['nlevelmax'])
            layout = _layout(cpus[i], header['ndim'], counts, start, nlevels)
            rows = _amr_segments(layout, header['ndim'], 'refinement')
            buf, rows[:, 0] = _read_spans(f, rows[:, 0].tolist(), rows[:, 1].tolist())
            return header, layout, rows, buf

        def decode(fetched):
            header, layout, rows, buf = fetched
            twotondim = 2 ** header['ndim']
            layout.mask = np.empty(int(layout.ncache.sum()), dtype=np.uint8)
            layout.nleaf = np.zeros((layout.nlevels, twotondim), dtype=np.int64)
            kernels.leaf_mask(buf, np.ascontiguousarray(rows[:, [0, 2, 3, 4]]), twotondim, layout.nlevels - 1,
                              layout.mask, layout.nleaf)
            layouts[i] = layout
            if i == 0:
                ng = np.asarray(header['ng'], dtype=np.float64)
                offset = np.zeros(3)
                offset[:len(ng)] = ng / 2 - 0.5
                geometry.append(AmrGeometry(ndim=header['ndim'], nlevelmax=header['nlevelmax'], offset=offset))

        return _Piece(paths[i], _header_nbytes_guess, read, decode)

    _run([piece(i) for i in range(len(paths))])
    return geometry[0], layouts


def load_cell_positions(geometry: AmrGeometry, layouts: list[AmrLayout], paths: list[str], first: np.ndarray,
                        boxlen: float, pos: np.ndarray, smooth: np.ndarray):
    """Put the position and size of every leaf cell into *pos* and *smooth*, from the AMR files *paths*, the cells of
    each file starting at the corresponding element of *first*"""
    ndim = geometry.ndim
    (pos_out, smooth_out), copies = _destinations([pos, smooth], np.float64)
    pieces = []
    for layout, first_cell, path in zip(layouts, first, paths):
        def decode(buf, rows, layout=layout, starts=layout.leaf_starts(first_cell)):
            kernels.leaf_positions(buf, np.ascontiguousarray(rows[:, [0, 2, 3, 4]]), ndim, layout.mask, boxlen,
                                   geometry.offset, pos_out, smooth_out, starts)

        pieces += _pieces(path, _amr_segments(layout, ndim, 'centres'), decode)
    _run(pieces)
    for a, temporary in copies:
        a[...] = temporary


def read_header(path: str, attrs) -> tuple[dict, int]:
    """Read a header described as for FortranFile.read_attrs, returning it and where it ends"""
    with open(path, 'rb') as f:
        r = _Records(f)
        header = {}
        for name, count, dtype in attrs:
            value = r.read(dtype)
            if count == 1:
                header[name] = value[0]
        return header, r.tell()


def load_cell_values(geometry: AmrGeometry, layouts: list[AmrLayout], paths: list[str], first: np.ndarray,
                     header_nbytes: int, nvar_file: int, arrays: list):
    """Put the values of the first len(arrays) variables in the hydro-like files *paths* into *arrays* (or skip a
    variable whose array is None), the cells of each file starting at the corresponding element of *first*"""
    twotondim = geometry.twotondim
    destinations, copies = _destinations(arrays, np.float64)
    pieces = []
    for layout, first_cell, path in zip(layouts, first, paths):
        segments = []  # (file offset, number of bytes, level, number of octs, offset in mask, ind0, ind1)
        for level, n, mask_offset in zip(range(layout.nlevels), layout.ncache.tolist(), layout.mask_offsets.tolist()):
            if n == 0:
                continue
            # each (level, CPU) entry is a record of the level, one of the number of octs n, and if n > 0,
            # nvar_file records of n reals for each ind
            block = header_nbytes + 24 * int(layout.entries_before[level]) + \
                twotondim * nvar_file * 8 * int(layout.blocks_before[level] + layout.octs_before[level])
            per_ind = nvar_file * (8 + 8 * n)
            # a level is read in one segment (with the records preceding it, to be checked), or one per ind if large
            if twotondim * per_ind <= _job_nbytes:
                segments.append((block, 24 + twotondim * per_ind, level, n, mask_offset, 0, twotondim))
            else:
                segments.append((block, 24 + per_ind, level, n, mask_offset, 0, 1))
                segments += [(block + 24 + ind * per_ind, per_ind, level, n, mask_offset, ind, ind + 1)
                             for ind in range(1, twotondim)]
        if not segments:
            continue

        def decode(buf, rows, layout=layout, starts=layout.leaf_starts(first_cell)):
            kernels.leaf_values(buf, np.ascontiguousarray(rows[:, [0, 2, 3, 4, 5, 6]]), nvar_file, layout.mask,
                                destinations, starts)

        pieces += _pieces(path, np.array(segments, dtype=np.int64), decode)
    _run(pieces)
    for a, temporary in copies:
        a[...] = temporary


# ---------------------------------------------------------------------------------------------------------------
# Particles

_particle_header_records = 8  # ncpu, ndim, npart, random seed, nstar, mstar, mstar_lost, nsink


@dataclasses.dataclass
class ParticleFile:
    """Where the fields of a particle file lie"""
    path: str
    npart: int
    offsets: list  # where the data of each record after the header start
    nbytes: list  # and how many bytes each holds
    family: np.ndarray = None  # the (internal) family id of each particle
    chunk_starts: np.ndarray = None  # (chunks, families): where each chunk's particles of each family go


def count_particles(paths: list[str], field: int, dtype: str) -> list[tuple[ParticleFile, np.ndarray | None]]:
    """Find where the fields of each particle file lie, and read field number *field*, which distinguishes
    families; returns for each file its ParticleFile and that field (None if the file does not have it)"""
    results = [None] * len(paths)

    def piece(i):
        def read(f):
            buf, table = _records_at(f, 0, _particle_header_records, 4096)
            npart = int(_record_value(buf, table, 2)[0])
            # walk through the records, reading the length that ends each with the one that starts the next
            offsets, nbytes = [], []
            pos = int(table[-1, 0] + table[-1, 1]) + 4
            f.seek(pos)
            start = f.read(4)
            while len(start) == 4:
                length = int.from_bytes(start, sys.byteorder, signed=True)
                if length < 0:
                    raise ValueError(f"Negative record length in {paths[i]}: the file is damaged, or has a record "
                                     "over 2 GB, which is not supported")
                offsets.append(pos + 4)
                nbytes.append(length)
                pos += 8 + length
                f.seek(pos - 4)
                lengths = f.read(8)
                if len(lengths) < 4 or int.from_bytes(lengths[:4], sys.byteorder, signed=True) != length:
                    raise ValueError(f"Inconsistent record lengths in {paths[i]}")
                start = lengths[4:]
            data = None
            if field < len(offsets):
                data = _read_spans(f, [offsets[field]], [nbytes[field]])[0].view(dtype)
            return ParticleFile(paths[i], npart, offsets, nbytes), data

        def decode(fetched):
            results[i] = fetched

        return _Piece(paths[i], _header_nbytes_guess, read, decode)

    _run([piece(i) for i in range(len(paths))])
    return results


def set_families(files: list[ParticleFile], families: list[np.ndarray], first: list, nfamilies: int):
    """Record the family of each particle, and where each chunk of particles of each family goes"""
    for pf, fam, first_index in zip(files, families, first):
        pf.family = np.ascontiguousarray(fam).view(np.uint8)
        chunk_counts = [np.bincount(pf.family[p:p + _particle_chunk], minlength=nfamilies)[:nfamilies]
                        for p in range(0, pf.npart, _particle_chunk)]
        cumulative = np.cumsum([np.zeros(nfamilies, dtype=np.int64)] + chunk_counts, axis=0)
        pf.chunk_starts = cumulative[:-1] + np.asarray(first_index, dtype=np.int64)[:nfamilies]


def load_particle_fields(files: list[ParticleFile], fields: list[tuple[int, str, list]]):
    """Read particle fields, each given as (record number, dtype, destination array of each family)"""
    prepared, copies = [], []
    for record, dtype, arrays in fields:
        destinations, field_copies = _destinations(arrays, np.dtype(dtype))
        prepared.append((record, np.dtype(dtype), destinations))
        copies += field_copies
    for pf in files:
        for record, dtype, _ in prepared:
            if record >= len(pf.offsets) or pf.nbytes[record] != pf.npart * dtype.itemsize:
                raise ValueError(f"{pf.path} has no field {record} of {pf.npart} items of type {dtype}")
    pieces = []
    for pf in files:
        segments = []  # (file offset, number of bytes, field, chunk, first particle, last particle + 1)
        for field, (record, dtype, _) in enumerate(prepared):
            for chunk, p0 in enumerate(range(0, pf.npart, _particle_chunk)):
                p1 = min(p0 + _particle_chunk, pf.npart)
                segments.append((pf.offsets[record] + p0 * dtype.itemsize, (p1 - p0) * dtype.itemsize, field, chunk,
                                 p0, p1))
        if not segments:
            continue

        def decode(buf, rows, pf=pf):
            for at, _, field, chunk, p0, p1 in rows.tolist():
                _, dtype, destinations = prepared[field]
                kernels.scatter_by_family(buf, at, dtype.itemsize, pf.family[p0:p1], destinations,
                                          pf.chunk_starts[chunk].copy())

        pieces += _pieces(pf.path, np.array(segments, dtype=np.int64), decode)
    _run(pieces)
    for a, temporary in copies:
        a[...] = temporary
