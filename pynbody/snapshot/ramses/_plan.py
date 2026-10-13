"""Planning reads of RAMSES files: turning what is wanted into a ReadPlan.

A load starts from a selection of each CPU's cells or particles (see :mod:`._selection`), the layout of its files
(see :mod:`._layout`) and the records wanted: hydro-like variables, oct centres, or particle fields. A builder
(:func:`cell_values`, :func:`cell_positions`, :func:`particle_fields`) expands these into byte *ranges*, each paired
with *rows* saying how to decode it and *checks* of what the file must hold there (the lengths that start and end
each record, and the records giving each level and its number of octs). :func:`_assemble` then

1. cuts the ranges of each file into pieces of a few megabytes, keeping together the ranges of one unit (those
   that must be decoded together, such as the coordinates of the same octs);
2. merges the ranges of each piece that lie closer together than the read gap into single reads, and works out
   where each range lands in the piece's buffer;
3. points the rows and checks at those places.

The result, a :class:`ReadPlan`, says exactly what will be read, and refers to neither threads nor destination
arrays, so it can be inspected, tested and kept for reuse; the ``load_*`` functions perform one (through
:func:`._execute.run_plan`), binding the arrays the values go into.
"""

from __future__ import annotations

import bisect
import dataclasses

import numpy as np

from . import _execute, _kernels as kernels
from ._layout import AmrGeometry, AmrLayout, HydroLikeFile, ParticleLayout
from ._selection import CellSelection, ParticleSelection

# Particles are read in chunks of this many
particle_chunk = _execute.job_nbytes // 8

# Kinds of check (see _execute): a record length, or a record giving a level or its number of octs
_RECORD_LENGTH, _LEVEL_RECORD = 0, 1


@dataclasses.dataclass(frozen=True)
class PlanPiece:
    """Reads of one file into one buffer, and how to check and decode what is read"""
    path: str
    context: int  # which CPU's selection the piece decodes with
    reads: np.ndarray  # (reads, 3): file offset, number of bytes, where in the buffer
    nbytes: int  # the size of the buffer
    requested: int  # the bytes of the ranges read (excluding what lies between them)
    rows: np.ndarray  # how to decode the buffer, as the decoder for the plan's kind expects
    checks: np.ndarray  # (checks, 3): where in the buffer, the int32 expected there, kind of check


@dataclasses.dataclass
class ReadPlan:
    """What a load reads, and how its pieces are decoded"""
    kind: str
    pieces: list

    @property
    def nreads(self) -> int:
        return sum(len(p.reads) for p in self.pieces)

    @property
    def nbytes(self) -> int:
        return sum(p.nbytes for p in self.pieces)

    @property
    def requested(self) -> int:
        return sum(p.requested for p in self.pieces)

    def describe(self) -> str:
        files = len({p.path for p in self.pieces})
        return (f"{self.kind}: {files} files, {len(self.pieces)} pieces, {self.nreads} reads of "
                f"{self.nbytes / 1e6:.1f} MB for {self.requested / 1e6:.1f} MB of records")


def _assemble(paths: list[str], ranges: np.ndarray, rows: np.ndarray, row_ranges: np.ndarray, offset_columns: list[int],
              checks: np.ndarray, gap: int) -> list[PlanPiece]:
    """Cut ranges into pieces, merge each piece's nearby ranges into reads, and point rows and checks at the buffer
    (see the module docstring).

    *ranges* are rows (file, offset, number of bytes, unit), where file indexes *paths* (and the selections the
    pieces decode with); the ranges of a file must not overlap, and those of a unit, which must be read in one
    piece, must lie together in one file. Each row's *offset_columns* hold offsets relative to the start of the
    range given by the same column of *row_ranges*. *checks* are rows (range, offset relative to its start, expected
    int32, kind)."""
    if len(ranges) == 0:
        return []
    order = np.lexsort((ranges[:, 1], ranges[:, 0]))
    renumber = np.empty_like(order)
    renumber[order] = np.arange(len(order))
    file, offset, nbytes, units = ranges[order].T
    n = len(order)
    new_unit = np.ones(n, dtype=bool)
    new_unit[1:] = (units[1:] != units[:-1]) | (file[1:] != file[:-1])
    units = np.cumsum(new_unit)  # (numbered in order)

    # 1. cut into pieces: each takes ranges up to and including the first that brings it to job_nbytes, and the rest
    # of that range's unit, but ends at the end of a file
    cumulative = np.cumsum(nbytes)
    unit_end = np.searchsorted(units, units, side='right')
    file_end = np.searchsorted(file, file, side='right')
    cumulative_l, before_l = cumulative.tolist(), (cumulative - nbytes).tolist()
    unit_end_l, file_end_l = unit_end.tolist(), file_end.tolist()
    starts, i = [], 0
    while i < n:
        starts.append(i)
        j = bisect.bisect_left(cumulative_l, before_l[i] + _execute.job_nbytes) + 1
        i = unit_end_l[min(max(j, i + 1), file_end_l[i]) - 1]
    starts = np.array(starts, dtype=np.int64)
    npieces = len(starts)
    is_start = np.zeros(n, dtype=bool)
    is_start[starts] = True
    piece_of = np.cumsum(is_start) - 1

    # 2. merge ranges of a piece closer together than the gap into one read, and find where each lands in the buffer
    ends = offset + nbytes
    new_read = is_start.copy()
    new_read[1:] |= offset[1:] - ends[:-1] > gap
    read_of = np.cumsum(new_read) - 1
    first = np.flatnonzero(new_read)
    read_offset = offset[first]
    read_nbytes = ends[np.append(first[1:], n) - 1] - read_offset
    read_piece = piece_of[first]
    piece_first_read = np.searchsorted(read_piece, np.arange(npieces + 1))
    read_at = np.cumsum(read_nbytes) - read_nbytes
    read_at -= read_at[piece_first_read[:-1]][read_piece]
    at = read_at[read_of] + offset - read_offset[read_of]
    reads = np.stack([read_offset, read_nbytes, read_at], axis=1)
    piece_nbytes = np.add.reduceat(read_nbytes, piece_first_read[:-1]).tolist()
    piece_requested = np.add.reduceat(nbytes, starts).tolist()

    # 3. point rows and checks at the buffer, and share them between pieces
    rows = rows.copy()
    for k, column in enumerate(offset_columns):
        rows[:, column] += at[renumber[row_ranges[:, k]]]
    rows, row_bounds = _by_piece(rows, piece_of[renumber[row_ranges[:, 0]]], npieces)
    check_range = renumber[checks[:, 0]]
    checks = np.stack([at[check_range] + checks[:, 1], checks[:, 2], checks[:, 3]], axis=1)
    checks, check_bounds = _by_piece(checks, piece_of[check_range], npieces)

    for table in (reads, rows, checks):
        table.setflags(write=False)  # (plans are kept, and their pieces share these)
    pieces = []
    for k, f in enumerate(file[starts].tolist()):
        pieces.append(PlanPiece(path=paths[f], context=f, reads=reads[piece_first_read[k]:piece_first_read[k + 1]],
                                nbytes=piece_nbytes[k], requested=piece_requested[k],
                                rows=rows[row_bounds[k]:row_bounds[k + 1]],
                                checks=checks[check_bounds[k]:check_bounds[k + 1]]))
    return pieces


def _by_piece(table, piece, npieces):
    """Sort the rows of a table by the piece each belongs to, returning it and where each piece's rows start"""
    order = np.argsort(piece, kind='stable')
    return np.ascontiguousarray(table[order], dtype=np.int64), \
        np.searchsorted(piece[order], np.arange(npieces + 1)).tolist()


def _record_checks(range_index, data_offset, nbytes):
    """Checks of the lengths that start and end records of *nbytes*, whose data start *data_offset* into ranges"""
    k = len(range_index)
    return np.concatenate([
        np.stack([range_index, data_offset - 4, nbytes, np.full(k, _RECORD_LENGTH)], axis=1),
        np.stack([range_index, data_offset + nbytes, nbytes, np.full(k, _RECORD_LENGTH)], axis=1)]).astype(np.int64)


def _ranges(file, offset, nbytes, units=None):
    """A table of ranges for _assemble; each range is a unit of its own unless *units* are given"""
    return np.stack([file, offset, nbytes, np.arange(len(file)) if units is None else units], axis=1).astype(np.int64)


# ---------------------------------------------------------------------------------------------------------------
# Cells

class _Runs:
    """The runs of octs of the selections of several CPUs, one after another"""

    def __init__(self, layouts: list[AmrLayout], selections: list[CellSelection], first_cells):
        self.layout = AmrLayout.stack(layouts)
        self.cpu = np.repeat(np.arange(len(selections)), [sel.nruns for sel in selections])
        for name in ('level', 'first', 'count', 'mask_offset'):
            setattr(self, name, np.concatenate([getattr(sel, name) for sel in selections]))
        self.starts = np.concatenate([sel.starts + first_cell for sel, first_cell in zip(selections, first_cells)])
        self.index = (self.cpu, self.level)  # (to index the stacked layout)
        self.n = self.layout.ncache[self.index]  # the octs of each run's level
        self.whole = (self.first == 0) & (self.count == self.n)  # whether a run holds every oct of its level

    def __len__(self):
        return len(self.cpu)


def _variable_groups(wanted):
    """Split (variable, destination) pairs into groups of consecutive variables going to consecutive destinations,
    returning (first variable, first destination, number of variables) for each"""
    groups = []
    for var, dest in sorted(wanted):
        if groups and var == groups[-1][0] + groups[-1][2] and dest == groups[-1][1] + groups[-1][2]:
            groups[-1][2] += 1
        else:
            groups.append([var, dest, 1])
    return np.array(groups, dtype=np.int64).reshape(-1, 3)


def cell_values(layouts: list[AmrLayout], selections: list[CellSelection], first_cells, paths: list[str],
                kind: HydroLikeFile, wanted: list[tuple[int, int]]) -> ReadPlan:
    """Plan reading variables of selected cells from hydro-like files of the given kind. *wanted* lists (variable
    in the file, destination index) pairs. Rows are as :func:`._kernels.cell_values` takes them: (data offset,
    octs, mask offset, ind, first destination, where the first wanted cell goes, variables, record stride)."""
    if not paths or not wanted:
        return ReadPlan('cell values', [])
    runs = _Runs(layouts, selections, first_cells)
    twotondim = runs.layout.twotondim
    block = runs.layout.hydro_block_offset(kind, runs.index)
    record_nbytes = 8 + 8 * runs.n

    def values(which, var_groups, whole):
        """Ranges and rows for the given runs, ind by ind and group by group. The values of a run holding every oct
        of its level are whole records, and a group of them is read in one range; otherwise each variable's values
        are a range of their own."""
        shape = (len(which), twotondim, len(var_groups))
        run = np.broadcast_to(which[:, None, None], shape).ravel()
        ind = np.broadcast_to(np.arange(twotondim)[None, :, None], shape).ravel()
        var0, dest0, nvar = (np.broadcast_to(var_groups[None, None, :, k], shape).ravel() for k in range(3))
        record_start = block[run] + 24 + (ind * kind.nvar + var0) * record_nbytes[run]
        if whole:
            offset, nbytes = record_start, nvar * record_nbytes[run]
            data_offset, record_stride = np.full(len(run), 4), record_nbytes[run]
        else:
            offset, nbytes = record_start + 4 + 8 * runs.first[run], 8 * runs.count[run]
            data_offset = record_stride = np.zeros(len(run), dtype=np.int64)
        rows = np.stack([data_offset, runs.count[run], runs.mask_offset[run], ind, dest0, runs.starts[run, ind], nvar,
                         record_stride], axis=1)
        return (runs.cpu[run], offset, nbytes), rows

    whole_ranges, whole_rows = values(np.flatnonzero(runs.whole), _variable_groups(wanted), True)
    part_ranges, part_rows = values(np.flatnonzero(~runs.whole),
                                    np.array([[var, dest, 1] for var, dest in wanted]).reshape(-1, 3), False)
    rows = np.concatenate([whole_rows, part_rows]).astype(np.int64)
    nvalue = len(rows)

    # and a range for the records giving each level and its number of octs, which are checked
    _, first_run = np.unique(runs.cpu * runs.layout.nlevels + runs.level, return_index=True)
    nstubs = len(first_run)
    stub_ranges = (runs.cpu[first_run], block[first_run], np.full(nstubs, 24))
    four = np.full(nstubs, 4)
    expected = np.stack([four, runs.level[first_run] + 1, four, four, runs.n[first_run], four], axis=1)
    checks = np.stack([np.repeat(nvalue + np.arange(nstubs), 6), np.tile(4 * np.arange(6), nstubs), expected.ravel(),
                       np.full(6 * nstubs, _LEVEL_RECORD)], axis=1)

    ranges = _ranges(*(np.concatenate(columns) for columns in zip(whole_ranges, part_ranges, stub_ranges)))
    return ReadPlan('cell values', _assemble(paths, ranges, rows, np.arange(nvalue)[:, None], [0], checks,
                                             _execute.read_gap(paths)))


def cell_positions(layouts: list[AmrLayout], selections: list[CellSelection], first_cells, paths: list[str]) \
        -> ReadPlan:
    """Plan reading the oct centres of selected cells from the AMR files. Rows are as
    :func:`._kernels.cell_positions` takes them: (offsets of the x, y and z centres, octs, mask offset, level, then
    where the first wanted cell of each ind goes)."""
    if not paths:
        return ReadPlan('cell positions', [])
    runs = _Runs(layouts, selections, first_cells)
    ndim, nruns = runs.layout.ndim, len(runs)
    record_nbytes = 8 + 8 * runs.n
    # a range for each (run, coordinate), the coordinates of a run being one unit (as they are decoded together)
    record_start = runs.layout.centres_offset(runs.index)[:, None] + np.arange(ndim) * record_nbytes[:, None]
    whole = runs.whole[:, None]
    offset = np.where(whole, record_start, record_start + 4 + 8 * runs.first[:, None])
    nbytes = np.broadcast_to(np.where(whole, record_nbytes[:, None], 8 * runs.count[:, None]), offset.shape)
    ranges = _ranges(np.repeat(runs.cpu, ndim), offset.ravel(), nbytes.ravel(), np.repeat(np.arange(nruns), ndim))

    range_index = np.arange(nruns * ndim).reshape(nruns, ndim)
    row_ranges = np.concatenate([range_index, np.repeat(range_index[:, :1], 3 - ndim, axis=1)], axis=1)
    data_offset = np.where(runs.whole, 4, 0)
    rows = np.concatenate([np.repeat(data_offset[:, None], 3, axis=1),
                           np.stack([runs.count, runs.mask_offset, runs.level], axis=1), runs.starts], axis=1)
    w = np.flatnonzero(runs.whole)
    checks = _record_checks(range_index[w].ravel(), np.full(len(w) * ndim, 4), np.repeat(8 * runs.n[w], ndim))
    return ReadPlan('cell positions', _assemble(paths, ranges, rows.astype(np.int64), row_ranges, [0, 1, 2], checks,
                                                _execute.read_gap(paths)))


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
            temporary = np.empty(a.shape, dtype=dtype)
            copies.append((a, temporary))
            out.append(temporary)
    return out, copies


def load_cell_values(plan: ReadPlan, selections: list[CellSelection], arrays: list):
    """Perform a cell_values plan, putting values into *arrays* (indexed by the plan's destinations; None to skip)"""
    destinations, copies = _destinations(arrays, np.float64)
    _execute.run_plan(plan, lambda piece, buf: kernels.cell_values(buf, piece.rows, selections[piece.context].mask,
                                                                    destinations))
    for a, temporary in copies:
        a[...] = temporary


def load_cell_positions(plan: ReadPlan, selections: list[CellSelection], geometry: AmrGeometry, boxlen: float,
                        pos: np.ndarray, smooth: np.ndarray):
    """Perform a cell_positions plan, putting each cell's position and size into *pos* and *smooth*"""
    (pos_out, smooth_out), copies = _destinations([pos, smooth], np.float64)
    _execute.run_plan(plan, lambda piece, buf: kernels.cell_positions(
        buf, piece.rows, geometry.ndim, selections[piece.context].mask, boxlen, geometry.offset, pos_out, smooth_out))
    for a, temporary in copies:
        a[...] = temporary


# ---------------------------------------------------------------------------------------------------------------
# Particles

def particle_fields(layouts: list[ParticleLayout], selections: list[ParticleSelection],
                    fields: list[tuple[int, int]]) -> ReadPlan:
    """Plan reading particle fields, given as (record number, item size), in chunks of particle_chunk particles.
    Rows are (data offset, first particle, last particle + 1, field, then where the chunk's first particle of each
    family goes)."""
    for layout in layouts:
        for record, itemsize in fields:
            if record >= len(layout.offsets) or layout.nbytes[record] != layout.npart * itemsize:
                raise ValueError(f"{layout.path} has no field {record} of {layout.npart} items of {itemsize} bytes")
    if not layouts or not fields:
        return ReadPlan('particle fields', [])
    npart = np.array([layout.npart for layout in layouts], dtype=np.int64)
    nchunks = -(-npart // particle_chunk)
    file = np.repeat(np.arange(len(layouts)), nchunks)
    chunk = np.arange(len(file)) - np.repeat(np.cumsum(nchunks) - nchunks, nchunks)
    p0 = chunk * particle_chunk
    p1 = np.minimum(p0 + particle_chunk, npart[file])
    # (the first chunk of a record includes the length that starts it, and the last the length that ends it)
    first, last = chunk == 0, chunk == nchunks[file] - 1
    data_offset = 4 * first
    family_starts = np.concatenate([sel.chunk_starts(particle_chunk) for sel in selections])
    nfirst, nlast = np.flatnonzero(first), np.flatnonzero(last)

    ranges, rows, checks = [], [], []
    for field, (record, itemsize) in enumerate(fields):
        record_offset = np.array([layout.offsets[record] for layout in layouts], dtype=np.int64)
        record_nbytes = npart * itemsize
        ranges.append((file, record_offset[file] + p0 * itemsize - data_offset,
                       (p1 - p0) * itemsize + data_offset + 4 * last))
        rows.append(np.concatenate([np.stack([data_offset, p0, p1, np.full(len(file), field)], axis=1),
                                    family_starts], axis=1))
        base = field * len(file)
        checks.append(np.stack([base + nfirst, np.zeros(len(nfirst)), record_nbytes[file[nfirst]],
                                np.full(len(nfirst), _RECORD_LENGTH)], axis=1))
        checks.append(np.stack([base + nlast, data_offset[nlast] + (p1 - p0)[nlast] * itemsize,
                                record_nbytes[file[nlast]], np.full(len(nlast), _RECORD_LENGTH)], axis=1))
    ranges = _ranges(*(np.concatenate(columns) for columns in zip(*ranges)))
    paths = [layout.path for layout in layouts]
    return ReadPlan('particle fields', _assemble(paths, ranges, np.concatenate(rows).astype(np.int64),
                                                 np.arange(len(ranges))[:, None], [0],
                                                 np.concatenate(checks).astype(np.int64), _execute.read_gap(paths)))


def load_particle_fields(plan: ReadPlan, selections: list[ParticleSelection], fields: list[tuple[np.dtype, list]]):
    """Perform a particle_fields plan, putting each field (given in the plan's order as (dtype, destination array of
    each family)) in place"""
    prepared, copies = [], []
    for dtype, arrays in fields:
        destinations, field_copies = _destinations(arrays, np.dtype(dtype))
        prepared.append((np.dtype(dtype).itemsize, destinations))
        copies += field_copies

    def decode(piece, buf):
        family = selections[piece.context].family
        for row in piece.rows:
            at, p0, p1, field = row[:4].tolist()
            itemsize, destinations = prepared[field]
            kernels.scatter_by_family(buf, at, itemsize, family[p0:p1], destinations, row[4:].copy())

    _execute.run_plan(plan, decode)
    for a, temporary in copies:
        a[...] = temporary
