"""Turning requests to read data into units of work, and grouping units of work into tasks for threads.

A :class:`ReadRequest` says which rows of a dataset to read, and where to put them. :func:`plan` turns requests into
units of work (:class:`_Work`), each reading from one file: most requests are one unit, but a request of a virtual
dataset is split where one source file gives way to the next. The work is then described to the rules in
:mod:`.strategy` (:func:`summarise`), and grouped into tasks for threads (:func:`group`).
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

from .common import ReadProperties
from .datasets import is_direct_reader
from .strategy import ReadSummary
from .virtual import _VirtualReader

# An index selection is read a block of rows at a time, each block into a buffer of at most this many bytes, from
# which the rows selected are then copied out
_gather_block_nbytes = 16 * 1024 * 1024


@dataclasses.dataclass(eq=False)
class ReadRequest:
    """A request to read some rows of an HDF5 dataset into an array.

    Pass requests to :meth:`pynbody.util.hdf_bulk_read.BulkReader.read`, which reads them in whatever way it judges
    best. Requests are read into separate places, so their destinations must not overlap.
    """

    dataset: typing.Any
    """The dataset: an h5py dataset, or anything with h5py's ``shape``, ``dtype`` and ``read_direct``."""

    rows: slice | np.ndarray
    """The rows to read: a slice (with a step of 1), or a sorted array of distinct row numbers."""

    destination: np.ndarray
    """Where to put them: an array of shape ``(number of rows,) + dataset.shape[1:]``, into whose dtype they are
    converted as they are read."""


class _Work:
    """A unit of work: reading rows of a dataset, all from one file, into a destination."""

    __slots__ = ('source', 'rows', 'destination', 'filename')

    def __init__(self, source, rows: slice | np.ndarray, destination: np.ndarray, filename: str):
        self.source = source  # a direct reader (see .datasets and .virtual), or an h5py dataset
        self.rows = rows  # a slice with explicit start and stop, or a sorted array of rows (not all consecutive)
        self.destination = destination  # a plain ndarray (see plan)
        self.filename = filename  # the file the data come from, for sharing work between threads

    @property
    def span(self) -> tuple[int, int]:
        """The range of rows that reading touches"""
        if isinstance(self.rows, slice):
            return self.rows.start, self.rows.stop
        return int(self.rows[0]), int(self.rows[-1]) + 1

    def properties(self) -> ReadProperties:
        """What matters about this work for deciding how to perform it"""
        if is_direct_reader(self.source):
            return self.source.properties(*self.span)
        return ReadProperties(direct=False)

    def shares_chunk_with(self, following: _Work) -> bool:
        """True if this work and the *following* both need the chunk at the boundary between them"""
        if following.source is not self.source or not is_direct_reader(self.source):
            return False
        chunk = self.source.chunk_containing(self.span[1] - 1)
        return chunk is not None and chunk == self.source.chunk_containing(following.span[0])

    def prepare(self):
        """Do now every HDF5 lookup that performing this work will need (see _DirectReader.prepare)"""
        if hasattr(self.source, 'prepare'):
            self.source.prepare(*self.span)

    def perform(self):
        if isinstance(self.rows, slice):
            self.source.read_direct(self.destination, source_sel=self.rows)
        else:
            _gather(self.source, self.rows, self.destination)


def _gather(source, rows: np.ndarray, destination: np.ndarray):
    """Read the given (sorted) rows of *source* into *destination*, a block of rows at a time"""
    row_nbytes = max(destination.dtype.itemsize * int(np.prod(destination.shape[1:], dtype=np.int64)), 1)
    block_rows = max(_gather_block_nbytes // row_nbytes, 1)
    i = 0
    while i < len(rows):
        start = int(rows[i])
        j = int(np.searchsorted(rows, start + block_rows, side='left'))
        stop = int(rows[j - 1]) + 1
        if stop - start == j - i:
            # consecutive rows, which can be read straight into place
            source.read_direct(destination[i:j], source_sel=slice(start, stop))
        else:
            block = np.empty((stop - start,) + destination.shape[1:], dtype=destination.dtype)
            source.read_direct(block, source_sel=slice(start, stop))
            destination[i:j] = block[rows[i:j] - start]
            del block
        i = j


def plan(requests: typing.Iterable[ReadRequest], open_dataset: typing.Callable) -> list[_Work]:
    """Turn requests into units of work, in the order given.

    *open_dataset* opens a dataset for reading (see BulkReader.open); each dataset is opened once, however many
    requests there are of it. Requests are not kept, so that if *requests* is an iterator, nothing more than the
    units of work holds on to the datasets (see .execute)."""
    works = []
    opened = {}  # id(dataset) -> (dataset, source, filename); the dataset is kept so that its id stays unique
    for request in requests:
        entry = opened.get(id(request.dataset))
        if entry is None:
            source = open_dataset(request.dataset)
            entry = opened[id(request.dataset)] = (request.dataset, source, _filename(source))
        _, source, filename = entry

        rows = _normalise_rows(request.rows, len(request.destination))
        if rows is None:
            continue
        # Write through a plain ndarray view: slicing an ndarray subclass (such as pynbody's SimArray) runs its
        # __array_finalize__, which can be far more expensive than copying the data
        destination = request.destination.view(np.ndarray)
        if isinstance(source, _VirtualReader):
            works.extend(_split_at_sources(source, rows, destination))
        else:
            works.append(_Work(source, rows, destination, filename))
    return works


def _filename(source) -> str:
    if is_direct_reader(source):
        return source.filename
    try:
        return source.file.filename
    except AttributeError:
        return ''


def _normalise_rows(rows, num_rows: int) -> slice | np.ndarray | None:
    """Rows as a slice with explicit start and stop, or as an array of rows that are not all consecutive; None if
    there are none"""
    if isinstance(rows, slice):
        if rows.step not in (None, 1) or rows.start is None or rows.stop is None:
            raise ValueError("Rows to read must be given as a slice with a start, a stop and a step of 1, or as an "
                             "array")
        start, stop = int(rows.start), int(rows.stop)
        if stop - start != num_rows:
            raise ValueError(f"The destination has room for {num_rows} rows, but {stop - start} are to be read")
        return slice(start, stop) if stop > start else None
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) != num_rows:
        raise ValueError(f"The destination has room for {num_rows} rows, but {len(rows)} are to be read")
    if len(rows) == 0:
        return None
    if rows[-1] - rows[0] == len(rows) - 1:
        return slice(int(rows[0]), int(rows[-1]) + 1)
    return rows


def _split_at_sources(source: _VirtualReader, rows: slice | np.ndarray, destination: np.ndarray) -> list[_Work]:
    """Split reading rows of a virtual dataset where one source file gives way to the next.

    Each part becomes a unit of work attributed to its source file, so that the work is shared between threads just
    as if the source files had been read one by one."""
    if isinstance(rows, slice):
        start, stop = rows.start, rows.stop
    else:
        start, stop = int(rows[0]), int(rows[-1]) + 1
    segments = source.source_segments(start, stop)
    edges = [row for row, _ in segments[1:]]
    if isinstance(rows, slice):
        bounds = [start] + edges + [stop]
        parts = [(a - start, b - start, slice(a, b)) for a, b in zip(bounds[:-1], bounds[1:])]
    else:
        cuts = [0] + [int(k) for k in np.searchsorted(rows, edges)] + [len(rows)]
        parts = [(a, b, rows[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
    works = []
    for (lo, hi, part_rows), (_, filename) in zip(parts, segments):
        if hi > lo:
            part_rows = _normalise_rows(part_rows, hi - lo)
            works.append(_Work(source, part_rows, destination[lo:hi], filename))
    return works


def summarise(works: list[_Work]) -> ReadSummary:
    """Describe the work, for the rules deciding how to perform it"""
    properties = [work.properties() for work in works]
    return ReadSummary(paths=tuple(dict.fromkeys(work.filename for work in works)),
                       num_reads=len(works),
                       num_files=len({work.filename for work in works}),
                       all_direct=all(p.direct for p in properties),
                       compressed=any(p.compressed for p in properties),
                       max_chunk_nbytes=max((p.chunk_nbytes for p in properties), default=0))


def group(works: list[_Work], per_file: bool) -> list[list[_Work]]:
    """Group units of work into tasks for threads, each task being performed in order by one thread.

    If *per_file*, there is one task per file, in order of first appearance. Otherwise there is one task per unit of
    work, except that consecutive units needing the same chunk are kept together, so that the chunk is decoded once,
    by one thread, rather than by several at once."""
    if not per_file:
        tasks = []
        for work in works:
            if tasks and tasks[-1][-1].shares_chunk_with(work):
                tasks[-1].append(work)
            else:
                tasks.append([work])
        return tasks
    tasks = {}
    for work in works:
        tasks.setdefault(work.filename, []).append(work)
    return list(tasks.values())
