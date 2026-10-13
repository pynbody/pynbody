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


def in_region(region, positions: np.ndarray, length_unit, properties: dict, family: str) -> np.ndarray:
    """Which of *positions* (in *length_unit*) the filter *region* selects, as a boolean array. The positions are put
    in a minimal snapshot of the given family, with the output's units and cosmological properties, so that the
    filter applies exactly as it would to the loaded snapshot (including unit strings and periodic wrapping)."""
    from ... import new  # (here, as pynbody.new imports the snapshot package)
    snap = new(**{family: len(positions)})
    snap['pos'] = positions
    snap['pos'].units = length_unit
    snap.properties.update(properties)
    return np.asarray(region(snap), dtype=bool)


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
        """The selection of those of this selection's cells that *keep* (a boolean for each, in the order they are
        put in place) says to keep: wanted octs form runs, those less than run_gap_octs apart joined. Cells are put
        in place run by run, so where a run is split, they go in a different order from this selection's."""
        twotondim = self.starts.shape[1]
        mask = kernels.restrict_mask(self.mask, np.stack([self.mask_offset, self.count], axis=1).astype(np.int64),
                                     twotondim, np.ascontiguousarray(keep).view(np.uint8))
        level, first, count, mask_offset = [], [], [], []
        for run_level, run_first, run_count, run_offset in zip(self.level.tolist(), self.first.tolist(),
                                                               self.count.tolist(), self.mask_offset.tolist()):
            wanted = np.flatnonzero(mask[run_offset:run_offset + run_count])
            if len(wanted) == 0:
                continue
            breaks = np.flatnonzero(np.diff(wanted) > run_gap_octs + 1)
            starts, ends = wanted[np.append(0, breaks + 1)], wanted[np.append(breaks, len(wanted) - 1)] + 1
            level += [run_level] * len(starts)
            first += (run_first + starts).tolist()
            count += (ends - starts).tolist()
            mask_offset += (run_offset + starts).tolist()
        level, first, count, mask_offset = (np.array(x, dtype=np.int64) for x in (level, first, count, mask_offset))
        # where each run's first cell of each ind goes: cells go run by run, and within a run, ind by ind
        bits = np.unpackbits(mask[:, None], axis=1, bitorder='little')[:, :twotondim].astype(np.int64)
        cumulative = np.concatenate([np.zeros((1, twotondim), dtype=np.int64), np.cumsum(bits, axis=0)])
        counts = cumulative[mask_offset + count] - cumulative[mask_offset]
        starts = (np.cumsum(counts.ravel()) - counts.ravel()).reshape(counts.shape)
        return CellSelection(level=level, first=first, count=count, mask_offset=mask_offset, starts=starts,
                             mask=mask, ncells=int(counts.sum()))

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

    def chunk_starts(self, chunk: int) -> np.ndarray:
        """(chunks, families): where the first particle of each family in each chunk of *chunk* particles goes"""
        if chunk not in self._chunk_starts:
            counts = kernels.chunk_counts(self.family, chunk, self.nfamilies)
            self._chunk_starts[chunk] = np.cumsum(counts, axis=0) - counts + self.first
        return self._chunk_starts[chunk]

    @classmethod
    def all_particles(cls, family: np.ndarray, first, nfamilies: int) -> ParticleSelection:
        """Select every particle of the CPU, given the family of each and where each family's first goes"""
        return cls(family=np.ascontiguousarray(family).view(np.uint8),
                   first=np.asarray(first, dtype=np.int64)[:nfamilies])
