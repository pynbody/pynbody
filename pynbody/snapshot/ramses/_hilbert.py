"""Which CPU files of a RAMSES output hold a region of space.

RAMSES divides the box among its CPUs (domains) along a Hilbert curve: each CPU holds the octs whose centres have
Hilbert keys (at order ``levelmax + 1``) in a range of its own, listed in the info file (:func:`domain_keys`). An
oct's cells lie within its father cell, so a cell can lie up to one father cell (at most a cell of ``levelmin``)
outside its CPU's domain; particles likewise drift up to a cell of their level beyond the domain they were assigned
to (by a fraction of a cell, measured). :func:`cpus_in_region` therefore finds the CPUs whose domains come within
1.25 ``levelmin`` cells of a region (or cells of the coarsest level loaded, if ``maxlevel`` is less than
``levelmin``).

It does so coarse to fine. Starting from the 8 cells of a 2x2x2 grid, it keeps the cells that the region (grown by
the padding) may intersect, as the filter's ``cubic_cell_intersection`` says. A cell whose range of keys lies within
one CPU's domain selects that CPU; one straddling domains is divided into its 8 children, and so on, until the cells
that straddle become too many, at which point every CPU any of them touches is selected. The result can include
CPUs holding nothing in the region (the test is conservative), but never omits one that does.
"""

from __future__ import annotations

import numpy as np

from . import _kernels as kernels

# Refining stops once this many cells straddle domains (keeping the cost of choosing CPUs to milliseconds)
max_cells = 2 ** 17

# Hilbert keys in the info file are written to 15 significant figures, so key ranges are compared with this
# relative tolerance (counting a cell near a domain boundary as straddling it)
_key_tolerance = 1e-13

# How far beyond its domain a cell or particle may lie, in cells of the coarsest level loaded: up to one for the cells
# of an oct (which lie within its father cell, whose centre's key decides its CPU), and a fraction of a cell more for
# particles drifting from the cells their CPU holds (measured at under 0.25)
padding_cells = 1.25


def domain_keys(info_path: str) -> np.ndarray:
    """The Hilbert keys bounding the CPUs' domains, from the info file: CPU *i* (counting from 1) holds keys from
    element *i - 1* up to element *i*"""
    with open(info_path) as f:
        lines = f.read().splitlines()
    for start, line in enumerate(lines):
        if line.split()[:3] == ['DOMAIN', 'ind_min', 'ind_max']:
            break
    else:
        raise ValueError(f"{info_path} has no table of Hilbert domains")
    rows = [line.split() for line in lines[start + 1:] if line.strip()]
    keys = np.array([[float(x) for x in row[1:3]] for row in rows], dtype=np.float64)
    if len(keys) == 0 or not np.array_equal(keys[1:, 0], keys[:-1, 1]):
        raise ValueError(f"The Hilbert domains in {info_path} are not contiguous")
    return np.append(keys[:, 0], keys[-1, 1])


def cpus_in_region(region, bound_keys: np.ndarray, levelmin: int, levelmax: int, boxlen: float,
                   ncpu: int | None = None) -> list[int]:
    """The CPUs (counting from 1) whose domains come within ``padding_cells`` cells of level *levelmin* (the coarsest
    level of leaf cells) of the region of *region*, a filter (see the module docstring). Positions are in the
    snapshot's units, in which the box is *boxlen* across. If there are more domains than *ncpu* (as RAMSES writes
    with ``overload`` above one), domain *i* is held by CPU ``(i - 1) % ncpu + 1``."""
    ndomains = len(bound_keys) - 1
    bits = levelmax + 1  # (RAMSES's order of Hilbert keys)
    if not np.isclose(bound_keys[-1], 8.0 ** bits, rtol=1e-12):
        raise NotImplementedError("The Hilbert domains do not end at 8**(levelmax+1), as expected")
    pad = boxlen * padding_cells * 0.5 ** levelmin
    selected = np.zeros(ndomains + 1, dtype=np.int64)  # (+1 and -1 bounding runs of selected domains)
    cells = np.stack(np.meshgrid(*[np.arange(2)] * 3, indexing='ij'), axis=-1).reshape(-1, 3).astype(np.int64)
    for level in range(1, min(bits, 17) + 1):
        n = 2 ** level
        centres = (cells + 0.5) * (boxlen / n)
        hit = np.asarray(region.cubic_cell_intersection(centres, deltax=boxlen / n + 2 * pad, boxsize=boxlen),
                         dtype=bool)
        cells = cells[hit]
        span = 8.0 ** (bits - level)
        first_key = kernels.hilbert_keys(cells[:, 0].copy(), cells[:, 1].copy(), cells[:, 2].copy(),
                                         level).astype(np.float64) * span
        last_key = first_key + span
        first = np.searchsorted(bound_keys, first_key * (1 - _key_tolerance), side='right') - 1
        last = np.searchsorted(bound_keys, last_key * (1 + _key_tolerance), side='left') - 1
        first, last = np.clip(first, 0, ndomains - 1), np.clip(last, 0, ndomains - 1)
        within = first == last
        straddle = ~within
        if level == min(bits, 17) or 8 * straddle.sum() > max_cells:
            within[:] = True  # (select every domain each cell touches)
        np.add.at(selected, first[within], 1)
        np.add.at(selected, last[within] + 1, -1)
        if within.all():
            break
        cells = (2 * cells[straddle, None, :] + _children).reshape(-1, 3)
    domains = np.flatnonzero(np.cumsum(selected)[:ndomains] > 0)
    if ncpu is not None:
        domains = np.unique(domains % ncpu)
    return (domains + 1).tolist()


_children = np.stack(np.meshgrid(*[np.arange(2)] * 3, indexing='ij'), axis=-1).reshape(1, 8, 3).astype(np.int64)
