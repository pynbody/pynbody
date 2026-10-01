"""Turning requests to read data into units of work, and grouping units of work into tasks for threads.

A :class:`ReadRequest` says which rows of a dataset to read, and where to put them. :func:`plan` turns requests into
units of work (:class:`_Work`), each reading from one place in one file: most requests are one unit, but a reader
may divide a request into parts (a virtual dataset divides one into a part for each source dataset it draws on).
The work is then described to the rules in :mod:`.strategy` (:func:`summarise`), and grouped by file into tasks for
input threads (:func:`group_by_file`). Finally each task is turned into jobs (see :class:`.common.Job`), as it is
performed (:func:`jobs`): for a chunked dataset, one per chunk, however many units of work want rows of it.
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from .common import Job, ReadProperties
from .strategy import ReadSummary

# An index selection is read a block of rows at a time: from an h5py dataset, each block into a buffer of at most
# this many bytes, from which the rows selected are then copied out; from anything else, as many rows as fill this
# many bytes, by indexing it with them
_gather_block_nbytes = 16 * 1024 * 1024


@dataclasses.dataclass(eq=False)
class ReadRequest:
    """A request to read some rows of an HDF5 dataset into an array.

    Pass requests to :meth:`pynbody.util.hdf_bulk_read.BulkReader.read`, which reads them in whatever way it judges
    best. Requests are read into separate places, so their destinations must not overlap. Requests of the same
    dataset are best given in increasing order of rows (as reading through a file sequentially is), since each
    chunk of a chunked dataset is then decoded only once.
    """

    dataset: typing.Any
    """The dataset: an h5py dataset, or anything else with h5py's ``shape``, ``dtype`` and ``read_direct``, and which
    can be indexed with an array of rows (such as a remote dataset served through ``hdfstream``). Such a dataset is
    read through ``read_direct`` for a slice of rows, and by indexing it with them for an array of rows, so that it
    can fetch just those rows."""

    rows: slice | np.ndarray
    """The rows to read: a slice with a step of 1, or an array of row numbers in increasing order (counted from
    *offset*). An array is used as it is, without being copied, so it must not change until it has been read."""

    destination: np.ndarray
    """Where to put them: a writeable, C-contiguous array of shape ``(number of rows,) + dataset.shape[1:]``, into
    whose dtype they are converted as they are read."""

    offset: int = 0
    """Added to every one of *rows*: for example, where in the dataset lies the piece of it that *rows* index. This
    lets *rows* be a view of an existing array of indices, where adding the offset to them would make a copy."""


class _Work:
    """A unit of work: reading rows of a dataset, all from one file, into a destination."""

    __slots__ = ('source', 'rows', 'offset', 'destination', 'filename')

    def __init__(self, source, rows: slice | np.ndarray, offset: int, destination: np.ndarray, filename: str):
        self.source = source  # a direct reader (see .datasets and .virtual), or an h5py dataset
        # the rows: a slice with explicit start and stop (and *offset* 0), or a sorted array of rows (not all
        # consecutive) to which *offset* is still to be added
        self.rows = rows
        self.offset = offset
        self.destination = destination  # a plain ndarray (see plan)
        self.filename = filename  # the file the data come from, for sharing work between threads

    @property
    def span(self) -> tuple[int, int]:
        """The range of rows that reading touches"""
        if isinstance(self.rows, slice):
            return self.rows.start, self.rows.stop
        return int(self.rows[0]) + self.offset, int(self.rows[-1]) + 1 + self.offset

    def properties(self) -> ReadProperties:
        """What matters about this work for deciding how to perform it"""
        if hasattr(self.source, 'properties'):
            return self.source.properties(*self.span, destination_dtype=self.destination.dtype,
                                          gathered=not isinstance(self.rows, slice))
        return ReadProperties(direct=False)  # (an h5py dataset)

    def prepare(self):
        """Do now every HDF5 lookup that performing this work will need (see _DirectReader.prepare)"""
        if hasattr(self.source, 'prepare'):
            self.source.prepare(*self.span)

    def perform(self):
        if isinstance(self.rows, slice):
            self.source.read_direct(self.destination, source_sel=self.rows)
        elif h5py is not None and isinstance(self.source, h5py.Dataset):
            _gather(self.source, self.rows, self.offset, self.destination)
        else:
            _index(self.source, self.rows, self.offset, self.destination)


def _gather(source, rows: np.ndarray, offset: int, destination: np.ndarray):
    """Read the given (sorted) rows of *source*, each plus *offset*, into *destination*, a block of rows at a time.

    Anything the size of *rows* is only ever made a block at a time, so that the memory needed beyond *rows* itself
    is bounded by the size of a block."""
    row_nbytes = max(destination.dtype.itemsize * int(np.prod(destination.shape[1:], dtype=np.int64)), 1)
    block_rows = max(_gather_block_nbytes // row_nbytes, 1)
    i = 0
    while i < len(rows):
        first = int(rows[i])
        j = int(np.searchsorted(rows, first + block_rows, side='left'))
        last = int(rows[j - 1])
        start, stop = first + offset, last + 1 + offset
        if last - first == j - i - 1:
            # consecutive rows, which can be read straight into place
            source.read_direct(destination[i:j], source_sel=slice(start, stop))
        else:
            block = np.empty((stop - start,) + destination.shape[1:], dtype=destination.dtype)
            source.read_direct(block, source_sel=slice(start, stop))
            destination[i:j] = block[rows[i:j] - first]
            del block
        i = j


def _index(source, rows: np.ndarray, offset: int, destination: np.ndarray):
    """Read the given (sorted) rows of *source*, a dataset other than an h5py one, each plus *offset*, into
    *destination*, by indexing *source* with them a block of rows at a time.

    Unlike h5py, which reads an index selection slowly, other datasets may read one well: in particular, a remote
    dataset then fetches only the rows selected, rather than every row between them."""
    row_nbytes = max(destination.dtype.itemsize * int(np.prod(destination.shape[1:], dtype=np.int64)), 1)
    block_rows = max(_gather_block_nbytes // row_nbytes, 1)
    for i in range(0, len(rows), block_rows):
        destination[i:i + block_rows] = source[rows[i:i + block_rows] + offset, ...]


def plan(requests: typing.Iterable[ReadRequest], open_dataset: typing.Callable) -> list[_Work]:
    """Turn requests into units of work, in the order given.

    *open_dataset* opens a dataset for reading (see BulkReader.open); each dataset is opened once, however many
    requests there are of it (even through different h5py objects). Requests are not kept, so that if *requests* is
    an iterator, nothing more than the units of work holds on to the datasets (see .execute)."""
    works = []
    opened = {}  # dataset identity -> (dataset, source, filename); the dataset is kept, so that its identity is too
    for request in requests:
        dataset = request.dataset
        key = _identity(dataset)
        entry = opened.get(key)
        if entry is None:
            source = open_dataset(dataset)
            entry = opened[key] = (dataset, source, _filename(dataset, source))
        _, source, filename = entry

        destination = _check_destination(request.destination, dataset)
        rows, offset = _normalise_rows(request.rows, int(request.offset), len(destination), dataset.shape[0])
        if rows is None:
            continue
        # Write through a plain ndarray view: slicing an ndarray subclass (such as pynbody's SimArray) runs its
        # __array_finalize__, which can be far more expensive than copying the data
        destination = destination.view(np.ndarray)
        if hasattr(source, 'divide'):
            for part in source.divide(rows, offset, destination):
                part_source, part_rows, part_offset, part_destination, part_filename = part
                part_rows, part_offset = _normalise_rows(part_rows, part_offset, len(part_destination))
                if part_rows is not None:
                    works.append(_Work(part_source, part_rows, part_offset, part_destination, part_filename))
        else:
            works.append(_Work(source, rows, offset, destination, filename))
    return works


def _identity(dataset):
    """Something equal for every object representing the same dataset"""
    try:
        return hash(dataset.id), dataset.id  # (h5py's identifiers compare equal for the same object in the file)
    except (AttributeError, TypeError):
        return id(dataset)


def _filename(dataset, source) -> str:
    filename = getattr(source, 'filename', None)
    if filename is not None:
        return filename
    try:
        return dataset.file.filename
    except AttributeError:
        return ''


def _check_destination(destination, dataset) -> np.ndarray:
    if not isinstance(destination, np.ndarray):
        raise TypeError("The destination of a read must be a numpy array")
    if destination.ndim < 1 or destination.shape[1:] != tuple(dataset.shape[1:]):
        raise ValueError(f"A destination of shape {destination.shape} cannot hold rows of a dataset of shape "
                         f"{tuple(dataset.shape)}")
    if not destination.flags.c_contiguous or not destination.flags.writeable:
        raise ValueError("The destination of a read must be writeable and C-contiguous")
    return destination


def _normalise_rows(rows, offset: int, num_rows: int, num_dataset_rows: int | None = None) \
        -> tuple[slice | np.ndarray | None, int]:
    """Return rows as (a slice with explicit start and stop, 0), or as (an array of rows that are not all consecutive,
    the offset to add to them); or (None, 0) if there are none. An array is returned as it is (a view, not a copy)
    wherever possible. If *num_dataset_rows* is given, the rows are checked to be ones the dataset has."""
    if isinstance(rows, slice):
        start, stop, step = rows.start, rows.stop, rows.step
        if step not in (None, 1):
            raise ValueError("Rows to read must be given as a slice with a step of 1, or as an array")
        if num_dataset_rows is not None and offset == 0:
            start, stop, _ = rows.indices(num_dataset_rows)  # (so that e.g. slice(None, 10) is understood)
        if start is None or stop is None:
            raise ValueError("A slice of rows to read, with an offset, must give its start and stop")
        start, stop = int(start) + offset, max(int(stop), int(start)) + offset
        if stop - start != num_rows:
            raise ValueError(f"The destination has room for {num_rows} rows, but {stop - start} are to be read")
        if num_dataset_rows is not None and stop > start and (start < 0 or stop > num_dataset_rows):
            raise ValueError(f"Rows to read must lie between 0 and {num_dataset_rows - 1}")
        return (slice(start, stop), 0) if stop > start else (None, 0)
    rows = np.asarray(rows)
    if rows.ndim != 1 or (len(rows) > 0 and rows.dtype.kind not in 'iu'):
        raise ValueError("Rows to read must be a slice, or a one-dimensional array of integers")
    if rows.dtype != np.int64:
        rows = rows.astype(np.int64)
    if len(rows) != num_rows:
        raise ValueError(f"The destination has room for {num_rows} rows, but {len(rows)} are to be read")
    if len(rows) == 0:
        return None, 0
    first, last = int(rows[0]) + offset, int(rows[-1]) + offset
    if num_dataset_rows is not None:
        if first < 0 or last >= num_dataset_rows:
            raise ValueError(f"Rows to read must lie between 0 and {num_dataset_rows - 1}")
        if len(rows) > 1 and not _strictly_increasing(rows):
            raise ValueError("Rows to read must be in increasing order, each given once")
    if last - first == len(rows) - 1:
        return slice(first, last + 1), 0
    return rows, offset


def _strictly_increasing(rows: np.ndarray, block: int = 1 << 20) -> bool:
    """True if *rows* increase strictly, found a block at a time so as to need little memory beyond *rows*"""
    for a in range(0, len(rows) - 1, block):
        piece = rows[a:a + block + 1]
        if not np.all(piece[1:] > piece[:-1]):
            return False
    return True


def summarise(works: list[_Work]) -> ReadSummary:
    """Describe the work, for the rules deciding how to perform it"""
    properties = [work.properties() for work in works]
    return ReadSummary(paths=tuple(dict.fromkeys(work.filename for work in works)),
                       num_reads=len(works),
                       num_files=len({work.filename for work in works}),
                       all_direct=all(p.direct for p in properties),
                       compressed=any(p.compressed for p in properties),
                       max_chunk_nbytes=max((p.chunk_nbytes for p in properties), default=0),
                       max_job_nbytes=max((p.job_nbytes for p in properties), default=0),
                       num_chunks=sum(p.num_chunks for p in properties))


def group_by_file(works: list[_Work]) -> list[list[_Work]]:
    """Group units of work into tasks for input threads: one per file, in order of first appearance, each keeping
    its units of work in order (so that a file is read sequentially)"""
    tasks = {}
    for work in works:
        tasks.setdefault(work.filename, []).append(work)
    return list(tasks.values())


def jobs(task: list[_Work]) -> typing.Iterator[Job]:
    """Yield the jobs that perform a task's units of work, in order, emptying *task* as it goes (so that each
    dataset is let go of once its jobs are done).

    Consecutive units of work reading the same source are given to it together, so that it can make one job of
    each chunk they need, rather than a job of each chunk for each unit of work."""
    task.reverse()
    while task:
        work = task.pop()
        source = work.source
        targets = [(work.rows, work.offset, work.destination)]
        del work
        while task and task[-1].source is source:
            work = task.pop()
            targets.append((work.rows, work.offset, work.destination))
            del work
        if hasattr(source, 'jobs'):
            yield from source.jobs(targets)
        else:
            # (an h5py dataset: performing the work is all input)
            for rows, offset, destination in targets:
                yield Job(_Work(source, rows, offset, destination, '').perform)
        del source, targets
