"""Which cells and particles of a RAMSES output are wanted.

A selection says, for one CPU, which of its octs and particles a snapshot holds and where each goes in the
snapshot's arrays. It is built once, when the snapshot is opened, and every later load of an array reads through it
(see :mod:`._plan`), so that hydro, gravity and RT variables, and every particle field, are put in the same places.

* A :class:`CellSelection` holds runs of consecutive octs at each level, one byte per oct in those runs saying which
  of its cells are wanted, and where the first wanted cell of each run and ``ind`` goes.
* A :class:`ParticleSelection` holds the family of each particle (:data:`SKIP` for a particle that is not wanted)
  and where each family's particles go.

A selection holds the whole of the CPU's leaf cells and particles, or (when a region is given) those inside the
region, which :func:`in_region` decides with the region's filter.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from . import _kernels as kernels
from ._layout import AmrLayout, LeafCells

SKIP = 255
"""The family id of a particle that is not wanted"""

# Runs of wanted octs no further apart than this many octs are read as one run (reading those between costs little,
# and fewer runs make smaller plans)
run_gap_octs = 256


# Positions are tested against a filter this many at a time (bounding the memory of the snapshots made to test them)
region_chunk = 2 ** 20

# Properties derived from positions alone, which band-pass filters on them can test
_position_properties = {'x', 'y', 'z', 'r', 'rxy'}


def depends_only_on_position(region) -> bool:
    """Whether the filter *region* selects by position alone (so can be applied to positions before anything else
    is read): spheres, cuboids, discs, band-pass filters on x, y, z, r or rxy, and their combinations"""
    from ... import filt
    if isinstance(region, (filt.And, filt.Or)):
        return depends_only_on_position(region.f1) and depends_only_on_position(region.f2)
    if isinstance(region, filt.Not):
        return depends_only_on_position(region.f)
    if isinstance(region, (filt.BandPass, filt.HighPass, filt.LowPass)):
        return region._prop in _position_properties
    return type(region) in (filt.Sphere, filt.Cuboid, filt.Disc)


def in_region(region, positions: np.ndarray, length_unit, properties: dict) -> np.ndarray:
    """Which of *positions* (in *length_unit*) the filter *region*, which must select by position alone, selects, as
    a boolean array. The positions are put in minimal snapshots with the output's units and cosmological properties,
    so that the filter applies exactly as it would to the loaded snapshot (including unit strings and periodic
    wrapping)."""
    from ... import new  # (here, as pynbody.new imports the snapshot package)
    keep = np.empty(len(positions), dtype=bool)
    for start in range(0, len(positions), region_chunk):
        chunk = positions[start:start + region_chunk]
        snap = new(dm=len(chunk))
        snap['pos'] = chunk
        snap['pos'].units = length_unit
        snap.properties.update(properties)
        keep[start:start + len(chunk)] = np.asarray(region(snap), dtype=bool)
    return keep


@dataclasses.dataclass
class CellSelection:
    """The wanted octs of one CPU, as runs of consecutive octs at each level, in file order"""
    level: np.ndarray  # the level of each run
    first: np.ndarray  # the first oct of each run, counting from the first own oct of its level
    count: np.ndarray  # the number of octs in each run
    mask_offset: np.ndarray  # where in mask each run's octs start
    starts: np.ndarray  # (runs, twotondim): where the run's first wanted cell of each ind goes, counting from the
    # first cell of this CPU
    mask: np.ndarray  # one byte per oct in the runs, bit ind set if cell ind is wanted
    ncells: int  # the number of wanted cells

    @property
    def nruns(self) -> int:
        return len(self.level)

    def restricted(self, keep: np.ndarray) -> CellSelection:
        """The selection of those of this selection's cells that *keep* (a boolean for each place a cell is put)
        says to keep: wanted octs form runs, those less than run_gap_octs apart joined. The cells kept go in the
        same order as in this selection."""
        if len(keep) != self.ncells:
            raise ValueError(f"Expected {self.ncells} cells to keep or not, but {len(keep)} were given")
        twotondim = self.starts.shape[1]
        runs = np.stack([self.mask_offset, self.count], axis=1).astype(np.int64)
        mask = kernels.restrict_mask(self.mask, runs, np.ascontiguousarray(self.starts, dtype=np.int64),
                                     np.ascontiguousarray(keep).view(np.uint8))
        level, first, count, mask_offset, origin = [], [], [], [], []
        for r, (run_level, run_first, run_count, run_offset) in enumerate(zip(
                self.level.tolist(), self.first.tolist(), self.count.tolist(), self.mask_offset.tolist())):
            wanted = np.flatnonzero(mask[run_offset:run_offset + run_count])
            if len(wanted) == 0:
                continue
            breaks = np.flatnonzero(np.diff(wanted) > run_gap_octs + 1)
            starts, ends = wanted[np.append(0, breaks + 1)], wanted[np.append(breaks, len(wanted) - 1)] + 1
            level += [run_level] * len(starts)
            first += (run_first + starts).tolist()
            count += (ends - starts).tolist()
            mask_offset += (run_offset + starts).tolist()
            origin += [r] * len(starts)
        level, first, count, mask_offset, origin = (np.array(x, dtype=np.int64)
                                                    for x in (level, first, count, mask_offset, origin))
        counts = kernels.mask_counts(mask, np.stack([mask_offset, count], axis=1).reshape(-1, 2), twotondim)
        # cells go in this selection's order: each new run's cells of each ind go together, so those blocks go in
        # the order of where this selection put the first cell (kept or not) of each
        # (this selection's cells before each new run, within the run it came from: those between it and the
        # previous new run, or the start, then those of the previous new run, and so on)
        new_run = np.flatnonzero(np.diff(origin, prepend=-1) != 0)
        previous_end = np.where(np.isin(np.arange(len(origin)), new_run), self.mask_offset[origin],
                                np.roll(mask_offset + count, 1))
        segments = np.stack([np.stack([previous_end, mask_offset - previous_end], axis=1),
                             np.stack([mask_offset, count], axis=1)], axis=1).reshape(-1, 2)
        old = kernels.mask_counts(self.mask, segments, twotondim).reshape(len(origin), 2, twotondim)
        between, within = old[:, 0], old[:, 1]
        before = np.cumsum(between + within, axis=0) - within
        before -= np.repeat((before - between)[new_run], np.diff(np.append(new_run, len(origin))), axis=0)
        order = np.argsort((self.starts[origin] + before).ravel(), kind='stable')
        starts = np.empty(counts.size, dtype=np.int64)
        starts[order] = np.cumsum(counts.ravel()[order]) - counts.ravel()[order]
        return CellSelection(level=level, first=first, count=count, mask_offset=mask_offset,
                             starts=starts.reshape(counts.shape), mask=mask, ncells=int(counts.sum()))

    @classmethod
    def all_leaves(cls, layout: AmrLayout, leaves: LeafCells) -> CellSelection:
        """Select every leaf cell of the CPU: one run for each level holding own octs"""
        levels = np.flatnonzero(layout.ncache > 0)
        mask_offsets = np.concatenate([[0], np.cumsum(layout.ncache)[:-1]]).astype(np.int64)
        # cells go level by level, and within a level, ind by ind
        starts = np.zeros(leaves.nleaf.size, dtype=np.int64)
        np.cumsum(leaves.nleaf.ravel()[:-1], out=starts[1:])
        starts = starts.reshape(leaves.nleaf.shape)
        return cls(level=levels.astype(np.int64), first=np.zeros(len(levels), dtype=np.int64),
                   count=layout.ncache[levels].astype(np.int64), mask_offset=mask_offsets[levels],
                   starts=np.ascontiguousarray(starts[levels]), mask=leaves.mask, ncells=int(leaves.nleaf.sum()))


@dataclasses.dataclass
class ParticleSelection:
    """The wanted particles of one CPU, and where those of each family go"""
    family: np.ndarray  # the (internal) family id of each particle in the file, SKIP if it is not wanted
    first: np.ndarray  # where the CPU's first particle of each family goes
    _chunk_starts: dict = dataclasses.field(default_factory=dict, repr=False)

    @property
    def nfamilies(self) -> int:
        return len(self.first)

    def chunk_starts(self, chunk: int) -> tuple[np.ndarray, np.ndarray]:
        """(chunks, families): where the first particle of each family in each chunk of *chunk* particles goes; and
        (chunks,) whether each chunk holds any wanted particle"""
        if chunk not in self._chunk_starts:
            counts = kernels.chunk_counts(self.family, chunk, self.nfamilies)
            self._chunk_starts[chunk] = (np.cumsum(counts, axis=0) - counts + self.first, counts.any(axis=1))
        return self._chunk_starts[chunk]

    @classmethod
    def all_particles(cls, family: np.ndarray, first, nfamilies: int) -> ParticleSelection:
        """Select every particle of the CPU, given the family of each and where each family's first goes"""
        return cls(family=np.ascontiguousarray(family).view(np.uint8),
                   first=np.asarray(first, dtype=np.int64)[:nfamilies])
