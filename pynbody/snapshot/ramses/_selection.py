"""Which cells and particles of a RAMSES output are wanted.

A selection says, for one CPU, which of its octs and particles a snapshot holds and where each goes in the
snapshot's arrays. It is built once, when the snapshot is opened, and every later load of an array reads through it
(see :mod:`._plan`), so that hydro, gravity and RT variables, and every particle field, are put in the same places.

* A :class:`CellSelection` holds runs of consecutive octs at each level, one byte per oct in those runs saying which
  of its cells are wanted, and where the first wanted cell of each run and ``ind`` goes.
* A :class:`ParticleSelection` holds the family of each particle (:data:`SKIP` for a particle that is not wanted)
  and where each family's particles go.

For now a selection always holds the whole of the CPU's leaf cells and particles.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from . import _kernels as kernels
from ._layout import AmrLayout, LeafCells

SKIP = 255
"""The family id of a particle that is not wanted"""


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
