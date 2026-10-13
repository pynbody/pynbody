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

import collections
import collections.abc
import dataclasses
import typing

import numpy as np

from . import _execute, _kernels as kernels
from ._layout import AmrGeometry, AmrLayout, HydroLikeFile, ParticleLayout
from ._selection import CellSelection, ParticleSelection

# Particles are read in chunks of this many
particle_chunk = _execute.job_nbytes // 8

# The most memory a snapshot's kept plans may take (see PlanCache)
plan_cache_nbytes = 64 * 1024 * 1024

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
    gap: int = 0  # the read gap the ranges of each piece were merged with

    @property
    def nreads(self) -> int:
        return sum(len(p.reads) for p in self.pieces)

    @property
    def nbytes(self) -> int:
        return sum(p.nbytes for p in self.pieces)

    @property
    def requested(self) -> int:
        return sum(p.requested for p in self.pieces)

    @property
    def memory_nbytes(self) -> int:
        """The memory the plan's tables take (which its pieces share)"""
        tables = {}
        for p in self.pieces:
            for a in (p.reads, p.rows, p.checks):
                base = a if a.base is None else a.base
                tables[id(base)] = base.nbytes
        return sum(tables.values())

    def describe(self) -> str:
        files = len({p.path for p in self.pieces})
        return (f"{self.kind}: {files} files, {len(self.pieces)} pieces, {self.nreads} reads of "
                f"{self.nbytes / 1e6:.1f} MB for {self.requested / 1e6:.1f} MB of records (read gap "
                f"{_execute.describe_nbytes(self.gap)})")


class PlanCache(collections.abc.Mapping):
    """Plans kept for reuse, by key, up to *max_nbytes* of them in all: once there are more, those least recently used
    are let go of (and a plan larger than that is never kept)"""

    def __init__(self, max_nbytes: int | None = None):
        self.max_nbytes = plan_cache_nbytes if max_nbytes is None else max_nbytes
        self._plans = collections.OrderedDict()  # key: (plan, memory_nbytes), least recently used first
        self.nbytes = 0

    def get_or_build(self, key, build: typing.Callable[[], ReadPlan]) -> ReadPlan:
        """The plan kept for *key*, or else the plan *build* returns, which is kept"""
        if key in self._plans:
            self._plans.move_to_end(key)
            return self._plans[key][0]
        plan = build()
        nbytes = plan.memory_nbytes
        if nbytes > self.max_nbytes:
            return plan
        self._plans[key] = (plan, nbytes)
        self.nbytes += nbytes
        while self.nbytes > self.max_nbytes:
            _, (_, dropped) = self._plans.popitem(last=False)
            self.nbytes -= dropped
        return plan

    def __getitem__(self, key) -> ReadPlan:
        return self._plans[key][0]

    def __iter__(self):
        return iter(self._plans)

    def __len__(self):
        return len(self._plans)


def _assemble(paths: list[str], ranges: np.ndarray, rows: np.ndarray, row_ranges: np.ndarray, offset_columns: list[int],
              checks: np.ndarray, gap: int) -> list[PlanPiece]:
    """Cut ranges into pieces, merge each piece's nearby ranges into reads, and point rows and checks at the buffer
    (see the module docstring).

    *ranges* are rows (file, offset, number of bytes, unit), where file indexes *paths* (and the selections the
    pieces decode with); the ranges of a file must not overlap, and those of a unit, which must be read in one
    piece, must lie in one file (but may be interleaved with those of other units). Each row's *offset_columns* hold
    offsets relative to the start of the range given by the same column of *row_ranges*. *checks* are rows (range,
    offset relative to its start, expected int32, kind)."""
    if len(ranges) == 0:
        return []
    order = np.lexsort((ranges[:, 1], ranges[:, 0]))
    renumber = np.empty_like(order)
    renumber[order] = np.arange(len(order))
    file, offset, nbytes, units = ranges[order].T
    n = len(order)

    # 1. cut into pieces: each takes the ranges it can without reading (the ranges, and the bytes between those that
    # will be merged) more than job_nbytes, and at least one; then on to the first place where no unit is left
    # unfinished; and ends at the end of a file
    _, units = np.unique(units, return_inverse=True)
    last = np.zeros(units.max() + 1, dtype=np.int64)
    np.maximum.at(last, units, np.arange(n))
    may_end = np.flatnonzero(np.maximum.accumulate(last[units]) == np.arange(n))  # where a piece may end
    file_start = np.flatnonzero(np.diff(file, prepend=-1))
    file_end = np.append(file_start[1:], n) - 1
    space = np.zeros(n, dtype=np.int64)  # the bytes between each range and the one before, if read with it
    space[1:] = offset[1:] - (offset + nbytes)[:-1]
    space[(space > gap) | (space < 0)] = 0
    space[file_start] = 0
    cumulative = np.cumsum(nbytes + space)  # (a piece starting at range i reads cumulative[j] - before[i] to j)
    before = cumulative - nbytes
    starts = [file_start[cumulative[file_end] - before[file_start] <= _execute.job_nbytes]]  # (files of one piece)
    for f0, f1 in zip(file_start.tolist(), file_end.tolist()):
        if cumulative[f1] - before[f0] <= _execute.job_nbytes:
            continue
        i, piece_starts = f0, []
        while i <= f1:
            piece_starts.append(i)
            j = min(max(int(np.searchsorted(cumulative, before[i] + _execute.job_nbytes, side='right')) - 1, i), f1)
            i = int(may_end[np.searchsorted(may_end, j)]) + 1  # (a file's last range is where a piece may end)
        starts.append(np.array(piece_starts, dtype=np.int64))
    starts = np.sort(np.concatenate(starts))
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
    read_nbytes = np.maximum.reduceat(ends, first) - read_offset
    read_piece = piece_of[first]
    piece_first_read = np.searchsorted(read_piece, np.arange(npieces + 1))
    read_at = np.cumsum(read_nbytes) - read_nbytes
    read_at -= read_at[piece_first_read[:-1]][read_piece]
    at = read_at[read_of] + offset - read_offset[read_of]
    reads = np.stack([read_offset, read_nbytes, read_at], axis=1)
    piece_nbytes = np.add.reduceat(read_nbytes, piece_first_read[:-1]).tolist()
    piece_requested = np.add.reduceat(nbytes, starts).tolist()
    del ends, new_read, read_of, first, read_piece, read_at

    # 3. point rows and checks at the buffer, and share them between pieces
    row_order, row_bounds = _by_piece(piece_of[renumber[row_ranges[:, 0]]], npieces)
    rows = rows[row_order]
    for k, column in enumerate(offset_columns):
        rows[:, column] += at[renumber[row_ranges[row_order, k]]]
    check_range = renumber[checks[:, 0]]
    check_order, check_bounds = _by_piece(piece_of[check_range], npieces)
    checks = np.stack([at[check_range] + checks[:, 1], checks[:, 2], checks[:, 3]], axis=1)[check_order]

    for table in (reads, rows, checks):
        table.setflags(write=False)  # (plans are kept, and their pieces share these)
    pieces = []
    for k, f in enumerate(file[starts].tolist()):
        pieces.append(PlanPiece(path=paths[f], context=f, reads=reads[piece_first_read[k]:piece_first_read[k + 1]],
                                nbytes=piece_nbytes[k], requested=piece_requested[k],
                                rows=rows[row_bounds[k]:row_bounds[k + 1]],
                                checks=checks[check_bounds[k]:check_bounds[k + 1]]))
    return pieces


def _by_piece(piece, npieces):
    """The order that sorts the rows of a table by the piece each belongs to, and where each piece's rows start"""
    order = np.argsort(piece, kind='stable')
    return order, np.searchsorted(piece[order], np.arange(npieces + 1)).tolist()


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
    in the file, destination index) pairs. Rows are as :func:`._kernels.cell_values` takes them.

    A run holding every oct of its level has whole records, and needs one row for each group of consecutive
    variables (all the variables in one pass when all are wanted), read in one range if the group is every variable
    the file holds (its records of every ind then lie together) or else in one range per ind. Any other run needs
    a row for each variable, and a range for each ind of it."""
    variables = [var for var, _ in wanted]
    if len(set(variables)) != len(variables) or len({dest for _, dest in wanted}) != len(wanted) or \
            not all(0 <= var < kind.nvar for var in variables):
        raise ValueError(f"Variables must be distinct and fewer than the {kind.nvar} the files hold, and go to "
                         f"distinct destinations")
    if not paths or not wanted:
        return ReadPlan('cell values', [])
    runs = _Runs(layouts, selections, first_cells)
    gap = _execute.read_gap(paths)
    twotondim = runs.layout.twotondim
    ind = np.arange(twotondim)
    block = runs.layout.hydro_block_offset(kind, runs.index)
    record_nbytes = 8 + 8 * runs.n
    ranges, rows, row_ranges = [], [], []
    nranges = 0

    def add(which, file_offset, nbytes, data_offset, dest0, nvar, var_stride, check):
        """Ranges and rows for the runs *which*. *file_offset* and *nbytes* are (runs, 1) for one range per run, or
        (runs, twotondim) for one per ind; the values of each ind start *data_offset* (runs, twotondim) into its
        range. A row's ranges are one unit, read into one piece; but a run whose ranges of each ind take more than
        job_nbytes to read is given a row (and a unit) for each ind."""
        nonlocal nranges
        k, c = file_offset.shape
        range_id = nranges + np.arange(k * c).reshape(k, c)
        range_of_ind = np.broadcast_to(range_id, (k, twotondim))
        nranges += k * c
        table = np.concatenate([data_offset, np.stack([runs.count[which], runs.mask_offset[which], np.full(k, dest0),
                                                        np.full(k, nvar), var_stride, np.full(k, int(check)),
                                                        np.full(k, (1 << twotondim) - 1)], axis=1),
                                runs.starts[which]], axis=1)
        unit = np.broadcast_to(range_id[:, :1], (k, c))
        # (by what its ranges take to read, with the bytes between them that the read gap merges)
        space = file_offset[:, 1:] - (file_offset + nbytes)[:, :-1]
        reads = nbytes.sum(axis=1) + np.where((space >= 0) & (space <= gap), space, 0).sum(axis=1)
        split = np.flatnonzero((reads > _execute.job_nbytes) & (c == twotondim))
        if len(split):
            each = np.repeat(table[split], twotondim, axis=0)  # (a row for each (run, ind))
            ind_of = np.tile(ind, len(split))
            each[:, :twotondim] = each[np.arange(len(each)), ind_of][:, None]
            each[:, twotondim + 6] = 1 << ind_of
            table = np.concatenate([np.delete(table, split, axis=0), each])
            range_of_ind = np.concatenate([np.delete(range_of_ind, split, axis=0),
                                           np.repeat(range_id[split].ravel()[:, None], twotondim, axis=1)])
            unit = unit.copy()
            unit[split] = range_id[split]
        ranges.append((np.repeat(runs.cpu[which], c), file_offset.ravel(), nbytes.ravel(), unit.ravel()))
        rows.append(table)
        row_ranges.append(range_of_ind)

    whole, part = np.flatnonzero(runs.whole), np.flatnonzero(~runs.whole)
    if len(whole):
        for var0, dest0, nvar in _variable_groups(wanted).tolist():
            # the records of a group of every variable lie together across the inds, and are read in one range,
            # unless that is more than job_nbytes
            together = (nvar == kind.nvar) & (twotondim * nvar * record_nbytes[whole] <= _execute.job_nbytes)
            for which, one_range in ((whole[together], True), (whole[~together], False)):
                if len(which) == 0:
                    continue
                rec = record_nbytes[which][:, None]
                first_record = block[which][:, None] + 24 + var0 * rec
                if one_range:
                    add(which, first_record, twotondim * nvar * rec, 4 + ind * nvar * rec, dest0, nvar, rec[:, 0],
                        True)
                else:
                    add(which, first_record + ind * kind.nvar * rec, np.broadcast_to(nvar * rec, (len(which),
                        twotondim)), np.full((len(which), twotondim), 4), dest0, nvar, rec[:, 0], True)
    if len(part):
        rec = record_nbytes[part][:, None]
        for var, dest in wanted:
            add(part, block[part][:, None] + 28 + var * rec + 8 * runs.first[part][:, None] + ind * kind.nvar * rec,
                np.broadcast_to(8 * runs.count[part][:, None], (len(part), twotondim)),
                np.zeros((len(part), twotondim), dtype=np.int64), dest, 1, np.zeros(len(part), dtype=np.int64),
                False)
    if not rows:
        return ReadPlan('cell values', [])

    # and a range for the records giving each level and its number of octs, which are checked
    _, first_run = np.unique(runs.cpu * runs.layout.nlevels + runs.level, return_index=True)
    nstubs = len(first_run)
    ranges.append((runs.cpu[first_run], block[first_run], np.full(nstubs, 24), nranges + np.arange(nstubs)))
    # (each record's starting length and value: the lengths that end them are then where expected)
    four = np.full(nstubs, 4)
    expected = np.stack([four, runs.level[first_run] + 1, four, runs.n[first_run]], axis=1)
    checks = np.stack([np.repeat(nranges + np.arange(nstubs), 4), np.tile([0, 4, 12, 16], nstubs), expected.ravel(),
                       np.full(4 * nstubs, _LEVEL_RECORD)], axis=1)

    ranges = _ranges(*(np.concatenate(columns) for columns in zip(*ranges)))
    return ReadPlan('cell values', _assemble(paths, ranges, np.concatenate(rows).astype(np.int64, copy=False),
                                             np.concatenate(row_ranges), list(range(twotondim)), checks, gap), gap)


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
    gap = _execute.read_gap(paths)
    return ReadPlan('cell positions', _assemble(paths, ranges, rows.astype(np.int64, copy=False), row_ranges, [0, 1, 2],
                                                checks, gap), gap)


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


def load_cell_values(plan: ReadPlan, selections: list[CellSelection], arrays: list):
    """Perform a cell_values plan, putting values into *arrays* (indexed by the plan's destinations; None to skip)"""
    destinations, copies = _destinations(arrays, np.float64)
    _execute.run_plan(plan, lambda piece, buf: kernels.cell_values(
        buf, piece.rows, (piece.rows.shape[1] - 7) // 2, selections[piece.context].mask, destinations))
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
    """Plan reading particle fields, given as (record number, item size), in chunks of particle_chunk particles
    (skipping chunks holding no particle the selections want). Rows are (data offset, first particle, last particle + 1, field, then where the chunk's first particle of each
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
    family_starts, wanted = (np.concatenate(x) for x in zip(*[sel.chunk_starts(particle_chunk) for sel in selections]))
    # (chunks holding no wanted particle are not read)
    file, chunk, p0, p1, first, last, family_starts = (x[wanted] for x in (file, chunk, p0, p1, first, last,
                                                                           family_starts))
    if len(file) == 0:
        return ReadPlan('particle fields', [])
    data_offset = 4 * first
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
    gap = _execute.read_gap(paths)
    return ReadPlan('particle fields', _assemble(paths, ranges, np.concatenate(rows).astype(np.int64, copy=False),
                                                 np.arange(len(ranges))[:, None], [0],
                                                 np.concatenate(checks).astype(np.int64), gap), gap)


def load_particle_fields(plan: ReadPlan, selections: list[ParticleSelection], fields: list[tuple[np.dtype, list]]):
    """Perform a particle_fields plan, putting each field (given in the plan's order as (dtype, destination array of
    each family)) in place"""
    prepared, copies = [], []
    for dtype, arrays in fields:
        destinations, field_copies = _destinations(arrays, np.dtype(dtype))
        prepared.append((np.dtype(dtype).itemsize, destinations))
        copies += field_copies
    _execute.run_plan(plan, lambda piece, buf: kernels.scatter_rows(buf, piece.rows, selections[piece.context].family,
                                                                     prepared))
    for a, temporary in copies:
        a[...] = temporary
