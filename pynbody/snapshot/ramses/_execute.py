"""Running reads of RAMSES files on threads.

Work arrives as *pieces*, each something to read from one file and a way to decode what was read. :func:`run_pieces`
gathers consecutive pieces into jobs of a few megabytes (so that handing a job between threads costs little beside
the work in it, even where each file holds little) and consecutive jobs into tasks for the input threads, each of
which reads its files one after another, sequentially; decoding threads decode. It runs them through the pipeline of
input and decode threads shared with the HDF5 reader (:mod:`pynbody.util.bulk_read.pipeline`), with as many threads of
each kind as :func:`choose_strategy` decides, from the shared rules for each kind of filesystem
(:mod:`pynbody.util.bulk_read.filesystem`).

Most reads are planned in advance (see :mod:`._plan`), and :func:`run_plan` runs a :class:`._plan.ReadPlan`: each of
its pieces reads byte ranges into one buffer, checks the values the plan expects to find there, and is decoded by a
function given the piece and its buffer. Reads whose extent is known only once a file's header has been read (when
a snapshot is opened; see :mod:`._layout`) are pieces given to :func:`run_pieces` directly, with a guess at what they
will hold, and each job of them is charged what it does hold once read.

A file whose contents are not as expected raises ValueError, rather than OSError, which pynbody's lazy loading takes
to mean that an array is not on disk at all.
"""

from __future__ import annotations

import logging
import typing

import numpy as np

from ... import config_parser
from ...util.bulk_read import filesystem
from ...util.bulk_read.pipeline import Job, ReadStrategy, perform_tasks
from . import _kernels as kernels

logger = logging.getLogger('pynbody.snapshot.ramses')

# The least a job reads, where there is enough to read: big enough that the work of handing jobs between threads is
# small beside it, and small enough that a few per thread take little memory
job_nbytes = 4 * 1024 * 1024


def _read_config():
    def option(name, default, minimum=None):
        value = config_parser.get('ramses', name, fallback=str(default)).strip().lower()
        if value == 'auto':
            return value
        try:
            if minimum is None or int(value) >= minimum:
                return int(value)
        except ValueError:
            pass
        logger.warning("Ignoring the value %r of %s in the [ramses] configuration", value, name)
        return default

    def choice(name, default, choices):
        value = config_parser.get('ramses', name, fallback=default).strip().lower()
        if value in choices:
            return value
        logger.warning("Ignoring the value %r of %s in the [ramses] configuration", value, name)
        return default

    return {'parallel-read': option('parallel-read', 8), 'io-threads': option('io-threads', 'auto'),
            'read-gap': option('read-gap', 'auto', 0),
            'hydro-loading': choice('hydro-loading', 'requested', ('requested', 'all'))}


config = _read_config()
"""How RAMSES files are read: initially from the ``[ramses]`` section of the configuration, but may be changed at run
time. ``parallel-read`` is the most threads decoding (``auto`` for as many as there are CPUs); if 1 or less,
everything is done in the calling thread. ``io-threads`` is the number of threads reading files, or ``auto`` for 16
on a parallel filesystem and one otherwise (the rule used for HDF5 files: see
:mod:`pynbody.util.bulk_read.filesystem`). ``read-gap`` is the read gap in bytes, or ``auto`` (see
:func:`read_gap`). ``hydro-loading`` is ``requested`` to read only the hydro, gravity or RT variable asked for (with
the other components of a vector), or ``all`` to read every variable of the file at once."""


def choose_strategy(paths: list[str], num_jobs: int, max_job_nbytes: int) -> ReadStrategy:
    """Decide how many threads to read *paths* with, in *num_jobs* jobs of at most *max_job_nbytes*"""
    threads = config['parallel-read']
    if threads == 'auto':
        threads = filesystem.available_cpus()
    if threads <= 1 or num_jobs <= 1:
        return ReadStrategy(io_threads=1, decode_threads=0,
                                     reason="reading in the calling thread (parallel-read is %s; %d jobs)"
                                            % (config['parallel-read'], num_jobs))
    fs_type = filesystem.filesystem_type(paths[0]) if paths else None
    if config['io-threads'] != 'auto':
        io_threads = max(int(config['io-threads']), 1)
    else:
        io_threads = filesystem.default_io_threads(fs_type)
    decode_threads = min(threads, filesystem.available_cpus(), num_jobs)
    return ReadStrategy(io_threads=io_threads, decode_threads=decode_threads,
                                 reason=f"{io_threads} input and {decode_threads} decoding thread(s) for {num_jobs} "
                                        f"jobs on a {fs_type or 'local'} filesystem",
                                 inflight_nbytes=2 * decode_threads * max(max_job_nbytes, 1))


def read_gap(paths: list[str]) -> int:
    """How far apart byte ranges of the files *paths* may be and still be read in one go, with what lies between
    them, because reading what lies between costs less than making another request.

    The ``read-gap`` option gives it in bytes, or ``auto`` chooses about what could be read in the time a request
    takes: 64 KB on a local filesystem, 256 KB on a network filesystem served by one server (such as NFS), and 1 MB
    on a parallel filesystem (such as Lustre). It affects speed, not memory: the bytes read between ranges count
    towards the size of jobs, which bounds memory, and the gap is never more than a quarter of a job."""
    value = config['read-gap']
    if value != 'auto':
        try:
            value = int(value)
            if value < 0:
                raise ValueError
        except (TypeError, ValueError):
            logger.warning("Ignoring the read gap %r, which is neither a number of bytes nor 'auto'", value)
            value = 'auto'
    if value == 'auto':
        value = filesystem.request_gap(filesystem.filesystem_type(paths[0]) if paths else None)
    return min(value, job_nbytes // 4)


def describe_nbytes(n: int) -> str:
    """A number of bytes, for the log"""
    return f"{n // 1024 ** 2} MB" if n >= 1024 ** 2 else f"{n // 1024} KB" if n >= 1024 else f"{n} bytes"


class Piece(typing.NamedTuple):
    """Something to read from one file, and what to do with it"""
    path: str
    nbytes: int  # roughly what it takes in memory once read (or a guess, if *held* is given)
    read: typing.Callable  # given the open file, returns what was read
    decode: typing.Callable  # given what was read, puts it in place
    held: typing.Callable | None = None  # given what was read, what it takes in memory, where nbytes is a guess


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


def run_pieces(pieces: list[Piece], description: str = "pieces"):
    """Read and decode the pieces, gathered into jobs and tasks (see the module docstring); *description* says what
    is read, for the log"""
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
    logger.debug("Reading RAMSES %s: %s", description, plan.reason)
    perform_tasks(tasks, plan, iter)


def _job(task: _Task, pieces: list[Piece], nbytes: int) -> Job:
    def fetch():
        fetched = [piece.read(task.file(piece.path)) for piece in pieces]
        if any(piece.held is not None for piece in pieces):
            # (so that the job is charged what it holds, rather than what it was guessed to)
            job.nbytes = max(sum(piece.nbytes if piece.held is None else piece.held(data)
                                 for piece, data in zip(pieces, fetched)), 1)
        return fetched

    def process(fetched):
        for piece, data in zip(pieces, fetched):
            piece.decode(data)

    job = Job(fetch, process, max(nbytes, 1))
    return job


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
    """Perform a ReadPlan: read each piece, check it, and call decode(piece, buffer). The pieces are run by
    :func:`run_pieces`, which hands them to input and decoding threads (or does everything in this thread, if
    :func:`choose_strategy` says so); *decode* is called on a decoding thread, so must release the GIL to run in
    parallel, as the kernels do."""

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

    run_pieces([make(piece) for piece in plan.pieces], plan.describe())
