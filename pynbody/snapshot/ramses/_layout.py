"""What is on disk in a RAMSES output: the layout of its binary files.

A RAMSES output has, for each CPU (domain) of the run, an AMR file describing the octs that CPU holds; hydro,
gravity and RT ("hydro-like") files holding the values of variables in the cells of those octs; and a particle file.
Each AMR and hydro-like file also holds the "virtual" octs of neighbouring CPUs, which pynbody skips.

Most of a file's layout follows from the numbers of octs each CPU holds at each level, which the AMR file records.
So when a snapshot is opened, :func:`scan_amr` reads the header of each AMR file and works out where the CPU's own
octs lie in it and in the corresponding hydro-like files (an :class:`AmrLayout`). It also reads their refinement
maps, to find which cells are leaves. Particle files are walked record by record (:func:`scan_particle_files`),
giving a :class:`ParticleLayout`. Nothing else is read until arrays are loaded (see :mod:`._plan`).

This module knows only what is on disk. Which octs and particles are wanted is the business of :mod:`._selection`,
and how they are read that of :mod:`._plan` and :mod:`._execute`.
"""

from __future__ import annotations

import dataclasses
import sys
import typing

import numpy as np

from . import _execute, _kernels as kernels

# What reading the header of a file, and what is found through it, is guessed to take before any file has been read;
# and the least it is guessed to take once one has (so that a job reads at most job_nbytes / this files)
_header_nbytes_guess = 256 * 1024

# Records in the header of a particle file: ncpu, ndim, npart, random seed, nstar, mstar, mstar_lost, nsink
_particle_header_records = 8


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


def _scan(piece, n: int, description: str):
    """Read and decode ``piece(i, nbytes)`` for each of *n* files, *nbytes* being what the piece is guessed to hold
    once read, which decides how many files a job reads.

    What a file's piece holds is known only once its header has been read, and finding each file's size in advance
    would take a request for each, which with many files on a parallel filesystem takes about as long as the scan.
    So the first file is read alone, and what its piece holds is the guess for the others (as the files of a
    load-balanced run are of similar sizes); and each piece is charged to the budget of memory in flight what it
    actually holds, once read (see :attr:`._execute.Piece.held`)."""
    if n == 0:
        return
    first, measured = piece(0, _header_nbytes_guess), []

    def held(fetched):
        measured.append(first.held(fetched))
        return measured[-1]

    _execute.run_pieces([first._replace(held=held)], f"{description} (the first)")
    if n > 1:
        guess = max(measured[0], _header_nbytes_guess)
        _execute.run_pieces([piece(i, guess) for i in range(1, n)],
                            f"{description} (the rest, guessed at {_execute.describe_nbytes(guess)} each)")


def _read_ranges(f, offsets, nbytes, gap: int) -> tuple[np.ndarray, list]:
    """Read byte ranges of an open file (in increasing order, not overlapping) into one buffer, reading ranges less
    than *gap* apart in one go, with what lies between them, so long as no more lies between them in all than the
    ranges hold (which bounds the buffer at twice what is wanted, as its size is not known in advance); return the
    buffer, and where in it each range starts"""
    spans, at, span_at = [], [], 0  # spans are [first byte, last byte + 1, where in the buffer]
    wanted = between = 0
    for offset, n in zip(offsets, nbytes):
        wanted += n
        space = offset - spans[-1][1] if spans else 0
        if not spans or space > gap or between + space > wanted:
            if spans:
                span_at += spans[-1][1] - spans[-1][0]
            spans.append([offset, offset + n, span_at])
        else:
            spans[-1][1] = offset + n
            between += space
        at.append(spans[-1][2] + offset - spans[-1][0])
    reads = [(start, end - start, span_at) for start, end, span_at in spans]
    nbytes_total = spans[-1][2] + spans[-1][1] - spans[-1][0] if spans else 0
    return _execute.read_into(f, reads, nbytes_total), at


# ---------------------------------------------------------------------------------------------------------------
# AMR and hydro-like files

@dataclasses.dataclass
class AmrGeometry:
    """What all the AMR files of an output share"""
    ndim: int
    nlevelmax: int
    offset: np.ndarray  # subtracted from oct centres (as stored) to give positions in units of the box

    @property
    def twotondim(self) -> int:
        return 2 ** self.ndim


@dataclasses.dataclass(frozen=True)
class HydroLikeFile:
    """What distinguishes the layout of one kind of hydro-like file (hydro, gravity or RT)"""
    header_nbytes: int  # the size of its header
    nvar: int  # the number of variables it holds


@dataclasses.dataclass
class AmrLayout:
    """Where one CPU's own octs lie in its AMR file and in its hydro-like files.

    The layouts of several CPUs can be stacked into one (see :meth:`stack`), whose arrays have a first axis for the
    CPU; its methods then take indices (CPU, level) where they take a level."""
    ndim: int
    ncache: np.ndarray  # number of own octs at each level read
    amr_offset: np.ndarray  # where the own octs of each level start in the AMR file
    entries_before: np.ndarray  # (level, CPU) entries preceding the own octs of each level, in a hydro-like file
    blocks_before: np.ndarray  # of which, entries with octs
    octs_before: np.ndarray  # and the number of octs they hold

    @classmethod
    def stack(cls, layouts: list[AmrLayout]) -> AmrLayout:
        """Stack the layouts of CPUs (which share their number of levels) into one"""
        return cls(ndim=layouts[0].ndim,
                   **{f.name: np.stack([getattr(layout, f.name) for layout in layouts])
                      for f in dataclasses.fields(cls) if f.name != 'ndim'})

    @property
    def nlevels(self) -> int:
        return self.ncache.shape[-1]

    @property
    def twotondim(self) -> int:
        return 2 ** self.ndim

    def centres_offset(self, level) -> np.ndarray:
        """Where the record of the x centres of the own octs of *level* (an int or array) starts in the AMR file; the
        records of the other coordinates follow, each 8 + 8n bytes long"""
        n = self.ncache[level]
        return self.amr_offset[level] + 3 * (8 + 4 * n)

    def refinement_offset(self, level) -> np.ndarray:
        """Where the refinement maps (twotondim records of n int32s) of the own octs of *level* start in the AMR
        file"""
        n = self.ncache[level]
        return self.amr_offset[level] + (4 + 2 * self.ndim + 2 * self.twotondim) * (8 + 4 * n) + \
            self.ndim * (8 + 8 * n)

    def hydro_block_offset(self, kind: HydroLikeFile, level) -> np.ndarray:
        """Where the own octs of *level* start in a hydro-like file of the given kind.

        Each (level, CPU) entry of such a file is a record of the level, one of the number of octs n, and if n > 0,
        nvar records of n float64s for each ind. So the own octs' block is 24 bytes (the level and n), then record
        (ind, var) of 8 + 8n bytes at 24 + (ind * nvar + var) * (8 + 8n)."""
        return kind.header_nbytes + 24 * self.entries_before[level] + \
            self.twotondim * kind.nvar * 8 * (self.blocks_before[level] + self.octs_before[level])


def _read_amr_header(f, bisection: bool, gap: int):
    """Read the header of an AMR file, returning (header dict, octs of each CPU and boundary at each level,
    where the octs start); *gap* is the read gap (see :func:`._execute.read_gap`)"""
    # The first 19 records are small; then come three of nlevelmax x ncpu integers (the heads and tails of linked
    # lists of octs, which are skipped, and the number of octs held), the total octs at each level, three more of
    # nlevelmax x nboundary integers if there are boundaries, the free memory, the ordering of domains and its
    # bounds, and lastly three records of an integer for each coarse cell
    buf, table = _records_at(f, 0, 19, 8 * 1024)
    ncpu, ndim, nlevelmax = (int(_record_value(buf, table, k)[0]) for k in (0, 1, 3))
    header = dict(ncpu=ncpu, ndim=ndim, ng=_record_value(buf, table, 2), nlevelmax=nlevelmax,
                  nboundary=int(_record_value(buf, table, 5)[0]))
    list_nbytes = 4 * nlevelmax * ncpu
    pos = int(table[18, 0] + table[18, 1]) + 4 + 2 * (8 + list_nbytes)
    count = 2 + (3 if header['nboundary'] > 0 else 0) + 2 + (5 if bisection else 1)
    # (the bounds of the domains are ncpu + 1 keys of up to 16 bytes, or for bisection ordering a tree of about 2 ncpu
    # nodes and the box of each CPU)
    nbytes = 8 * count + list_nbytes + 40 * nlevelmax + 12 * nlevelmax * header['nboundary'] + \
        (100 * ncpu if bisection else 16 * (ncpu + 1))
    coarse_nbytes = 4 * int(np.prod(header['ng']))
    # coarse cells' records are read with the rest unless reading them takes more than skipping them
    read_coarse = 3 * (8 + coarse_nbytes) <= gap
    if read_coarse:
        count += 3
        nbytes += 3 * (8 + coarse_nbytes)
    buf, table = _records_at(f, pos, count, nbytes + 8 * 1024)
    if table[0, 1] != list_nbytes:
        raise ValueError(f"Unexpected layout of the header of {f.name}")
    counts = _record_value(buf, table, 0).reshape((nlevelmax, ncpu))
    if header['nboundary'] > 0:
        boundary = _record_value(buf, table, 4).reshape((nlevelmax, header['nboundary']))
        counts = np.concatenate([counts, boundary], axis=1)
    end = int(table[-1, 0] + table[-1, 1]) + 4  # (in buf)
    if not read_coarse:
        # check the length that starts the first coarse record (usually already read), and skip the three
        if len(buf) >= end + 4:
            length = int(np.frombuffer(buf, np.int32, 1, end)[0])
        else:
            f.seek(pos + end)
            length = int.from_bytes(f.read(4), sys.byteorder, signed=True)
        if length != coarse_nbytes:
            raise ValueError(f"Unexpected layout of the header of {f.name}")
        end += 3 * (8 + coarse_nbytes)
    return header, np.ascontiguousarray(counts), pos + end


def _amr_layout(cpu, ndim, counts, start, nlevels) -> AmrLayout:
    """Work out where the own octs of each level lie, from the numbers of octs in the blocks of the file"""
    twotondim = 2 ** ndim
    ncache, octs_before, blocks_before = kernels.own_layout(counts[:nlevels], cpu)
    # a block of n > 0 octs has nint records of n integers and ndim of n reals, taking nint (8 + 4n) + ndim (8 + 8n)
    # bytes
    nint = 3 + 1 + 2 * ndim + 3 * twotondim
    return AmrLayout(ndim=ndim, ncache=ncache,
                     amr_offset=start + (4 * nint + 8 * ndim) * octs_before + (8 * nint + 8 * ndim) * blocks_before,
                     entries_before=np.arange(nlevels, dtype=np.int64) * counts.shape[1] + (cpu - 1),
                     blocks_before=blocks_before, octs_before=octs_before)


@dataclasses.dataclass
class LeafCells:
    """Which of a CPU's own cells are leaves"""
    mask: np.ndarray  # one byte per own oct (levels one after another), bit ind set if cell ind is a leaf
    nleaf: np.ndarray  # (levels, twotondim): the number of leaf cells of each ind at each level


def scan_amr(paths: list[str], cpus: list[int], bisection: bool, maxlevel: int | None) \
        -> tuple[AmrGeometry, list[AmrLayout], list[LeafCells]]:
    """Find the layout of each CPU's own octs, and which of their cells are leaves (see the module docstring)"""
    layouts, leaves = [None] * len(paths), [None] * len(paths)
    geometry = []
    gap = _execute.read_gap(paths)

    def piece(i, nbytes_guess):
        def read(f):
            header, counts, start = _read_amr_header(f, bisection, gap)
            nlevels = min(maxlevel or header['nlevelmax'], header['nlevelmax'])
            layout = _amr_layout(cpus[i], header['ndim'], counts, start, nlevels)
            levels = np.flatnonzero(layout.ncache > 0)
            offsets = layout.refinement_offset(levels)
            nbytes = layout.twotondim * (8 + 4 * layout.ncache[levels])
            buf, at = _read_ranges(f, offsets.tolist(), nbytes.tolist(), gap)
            return header, layout, levels, np.asarray(at, dtype=np.int64), buf

        def decode(fetched):
            header, layout, levels, at, buf = fetched
            mask_offsets = np.concatenate([[0], np.cumsum(layout.ncache)[:-1]]).astype(np.int64)
            rows = np.stack([at, levels, layout.ncache[levels], mask_offsets[levels]], axis=1).astype(np.int64)
            cells = LeafCells(mask=np.empty(int(layout.ncache.sum()), dtype=np.uint8),
                              nleaf=np.zeros((layout.nlevels, layout.twotondim), dtype=np.int64))
            kernels.leaf_mask(buf, np.ascontiguousarray(rows), layout.twotondim, layout.nlevels - 1, cells.mask,
                              cells.nleaf)
            layouts[i], leaves[i] = layout, cells
            if i == 0:
                ng = np.asarray(header['ng'], dtype=np.float64)
                offset = np.zeros(3)
                offset[:len(ng)] = ng / 2 - 0.5
                geometry.append(AmrGeometry(ndim=header['ndim'], nlevelmax=header['nlevelmax'], offset=offset))

        def held(fetched):
            return fetched[-1].nbytes

        return _execute.Piece(paths[i], nbytes_guess, read, decode, held)

    _scan(piece, len(paths), f"AMR headers and refinement maps of {len(paths)} files (read gap "
                             f"{_execute.describe_nbytes(gap)})")
    return geometry[0], layouts, leaves


# ---------------------------------------------------------------------------------------------------------------
# Particle files

@dataclasses.dataclass
class ParticleLayout:
    """Where the fields of a particle file lie"""
    path: str
    npart: int
    offsets: list  # where the data of each record after the header start
    nbytes: list  # and how many bytes each holds


def scan_particle_files(paths: list[str], field: int, dtype: str, convert: typing.Callable | None = None) \
        -> list[tuple[ParticleLayout, np.ndarray | None]]:
    """Find where the fields of each particle file lie, and read field number *field*, which distinguishes
    families; returns for each file its ParticleLayout and that field (None if the file does not have it), passed
    through *convert*, if given, as each file is decoded (so that only what it returns is kept: a family id for each
    particle takes less than, say, its age)"""
    results = [None] * len(paths)

    def piece(i, nbytes_guess):
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
                data = _read_ranges(f, [offsets[field]], [nbytes[field]], 0)[0].view(dtype)
            return ParticleLayout(paths[i], npart, offsets, nbytes), data

        def decode(fetched):
            layout, data = fetched
            if convert is not None and data is not None:
                data = convert(data)
            results[i] = layout, data

        def held(fetched):
            return 0 if fetched[1] is None else fetched[1].nbytes

        return _execute.Piece(paths[i], nbytes_guess, read, decode, held)

    _scan(piece, len(paths), f"headers of {len(paths)} particle files")
    return results
