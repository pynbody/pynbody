"""Running reads of RAMSES files on threads.

Work arrives as *pieces*, each something to read from one file and a way to decode what was read. :func:`run_pieces`
gathers consecutive pieces into jobs of a few megabytes (so that handing a job between threads costs little beside
the work in it, even where each file holds little) and consecutive jobs into tasks for the input threads, each of
which reads its files one after another, sequentially; decoding threads decode. It runs them through the threaded
pipeline of :mod:`pynbody.util.hdf_bulk_read.execute`, with as many threads of each kind as :func:`choose_strategy`
decides.

Most reads are planned in advance (see :mod:`._plan`), and :func:`run_plan` runs a :class:`._plan.ReadPlan`: each of
its pieces reads byte ranges into one buffer, checks the values the plan expects to find there, and is decoded by a
function given the piece and its buffer. Reads whose extent is known only once a file's header has been read (when
a snapshot is opened; see :mod:`._layout`) are pieces given to :func:`run_pieces` directly.

A file whose contents are not as expected raises ValueError, rather than OSError, which pynbody's lazy loading takes
to mean that an array is not on disk at all.
"""

from __future__ import annotations

import logging
import typing

import numpy as np

from ... import config_parser
from ...util.hdf_bulk_read import execute, strategy
from ...util.hdf_bulk_read.common import Job
from . import _kernels as kernels

logger = logging.getLogger('pynbody.snapshot.ramses')

# The least a job reads, where there is enough to read: big enough that the work of handing jobs between threads is
# small beside it, and small enough that a few per thread take little memory
job_nbytes = 4 * 1024 * 1024

# Byte ranges of a file closer together than this are read in one go, with what lies between them (see read_gap)
_read_gap = 64 * 1024


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


def read_gap(paths: list[str]) -> int:
    """How far apart byte ranges of the files *paths* may be and still be read in one go, with what lies between
    them: reading what lies between costs less than making another request"""
    return _read_gap


class Piece(typing.NamedTuple):
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


def run_pieces(pieces: list[Piece]):
    """Read and decode the pieces, gathered into jobs and tasks (see the module docstring)"""
    tasks, job, nbytes, task_nbytes = [], [], 0, 0

    def finish_job():
        nonlocal job, nbytes
        if job:
            tasks[-1].jobs.append(_job(tasks[-1], job, nbytes))
            job, nbytes = [], 0

    for i, piece in enumerate(pieces):
        # a task (which an input thread reads straight through) ends only at the end of a file, and only once it has
        # enough to read; a job ends once it has enough
        if i == 0 or (piece.path != pieces[i - 1].path and task_nbytes >= job_nbytes):
            finish_job()
            tasks.append(_Task())
            task_nbytes = 0
        job.append(piece)
        nbytes += piece.nbytes
        task_nbytes += piece.nbytes
        if nbytes >= job_nbytes:
            finish_job()
    finish_job()

    plan = choose_strategy(list(dict.fromkeys(p.path for p in pieces)), sum(len(t.jobs) for t in tasks),
                           max((j.nbytes for t in tasks for j in t.jobs), default=0))
    logger.debug("Reading RAMSES files: %s", plan.reason)
    execute.perform_tasks(tasks, plan, iter)


def _job(task: _Task, pieces: list[Piece], nbytes: int) -> Job:
    def fetch():
        return [piece.read(task.file(piece.path)) for piece in pieces]

    def process(fetched):
        for piece, data in zip(pieces, fetched):
            piece.decode(data)

    return Job(fetch, process, max(nbytes, 1))


def read_into(f, reads, nbytes: int) -> np.ndarray:
    """Read the byte ranges *reads*, rows (file offset, number of bytes, where in the buffer), of an open file into a
    new buffer of *nbytes*"""
    buf = np.empty(nbytes, dtype=np.uint8)
    view = memoryview(buf)
    for offset, n, at in reads:
        f.seek(offset)
        if f.readinto(view[at:at + n]) != n:
            raise ValueError(f"{f.name} is shorter than expected")
    return buf


# What a failed check (see ._plan) of each kind means
_check_failures = {0: "Unexpected record length in {path}",
                   1: "A level of {path} does not start where expected: the file may not match its AMR file, or be in "
                      "a format pynbody does not recognise"}


def run_plan(plan, decode: typing.Callable):
    """Perform a ReadPlan: read each piece, check it, and call decode(piece, buffer)"""
    logger.debug("Reading RAMSES files: %s", plan.describe())

    def make(piece):
        def read(f):
            return read_into(f, piece.reads.tolist(), piece.nbytes)

        def decode_piece(buf):
            failed = kernels.check_int32(buf, piece.checks)
            if failed >= 0:
                raise ValueError(_check_failures[int(piece.checks[failed, 2])].format(path=piece.path))
            try:
                decode(piece, buf)
            except ValueError as e:
                raise ValueError(f"{e} in {piece.path}") from e

        return Piece(piece.path, piece.nbytes, read, decode_piece)

    run_pieces([make(piece) for piece in plan.pieces])
