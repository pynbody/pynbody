"""A pipeline of input and decode threads, performing jobs of reading as a :class:`ReadStrategy` says.

Work is done as jobs (see :class:`Job`), each an input step (*fetch*) and optionally a computation on what was fetched
(*process*), grouped into *tasks*, each the jobs reading one file (or a run of files) in order. There are two stages,
with their own threads:

* input threads each take a task and fetch its jobs in order, so that each file is read sequentially;
* decode threads process what has been fetched, in whatever order it arrives.

The memory taken by what has been fetched but not yet processed is kept within the strategy's ``inflight_nbytes``:
an input thread waits before fetching more until enough has been processed. (A job that finds, once fetched, that it
holds more than it was thought to is charged what it holds, and so may take that over the limit.) With no decode
threads, input threads process what they fetch themselves, and the same limit applies to how many do so at once; with
one input thread and no decode threads, everything is done in the calling thread.

Each task, and each job, is let go of once done, so that what a file's jobs hold is freed once all the work on that
file is done, rather than when everything has been read.

The HDF5 reader (:mod:`pynbody.util.hdf_bulk_read.execute`) and the RAMSES reader
(:mod:`pynbody.snapshot.ramses._execute`) each plan their jobs and choose a strategy, then call
:func:`perform_tasks`.
"""

from __future__ import annotations

import concurrent.futures
import concurrent.futures.thread  # (imported lazily by concurrent.futures otherwise, which could fail mid-read)
import dataclasses
import threading
import traceback


class Job:
    """One step of reading: input (*fetch*), then, optionally, computation on what was fetched (*process*).

    The pipeline runs the fetches of each task in order on an input thread, which is all that thread does, and hands
    what they fetch to *process* on a thread of a decoding pool, which is all that pool does; or, when there is no
    pool, runs both one after the other. *nbytes* is roughly the memory that what is fetched takes until processed,
    which the pipeline keeps within a budget. Where that is known only once fetched, *fetch* may set *nbytes* to it,
    and the job is charged that instead.

    *fetch* returns what *process* is to be given, or None if nothing is left to do (for example, because the fetch
    itself put the data in place)."""

    __slots__ = ('fetch', 'process', 'nbytes')

    def __init__(self, fetch, process=None, nbytes: int = 0):
        if process is not None and nbytes <= 0:
            raise ValueError("A job that processes what it fetches must say how much memory that takes")
        self.fetch = fetch
        self.process = process
        self.nbytes = nbytes

    def run(self):
        """Fetch and process, one after the other, in this thread"""
        fetched = self.fetch()
        if fetched is not None and self.process is not None:
            self.process(fetched)


@dataclasses.dataclass(frozen=True)
class ReadStrategy:
    """How many threads of each kind perform a read, and how much memory may wait between them."""

    io_threads: int
    """The number of threads reading files; 1 means the calling thread reads them, one after another"""

    decode_threads: int
    """The number of threads decoding what has been read; 0 means that the reading threads decode it themselves"""

    reason: str
    """Why, in words"""

    inflight_nbytes: int = 0
    """The most memory to be taken by data read and waiting to be decoded"""

    @property
    def serial(self) -> bool:
        """True if everything is done in the calling thread"""
        return self.io_threads <= 1 and self.decode_threads == 0


def perform_tasks(tasks: list, strategy: ReadStrategy, jobs=iter):
    """Perform *tasks*, whose jobs (in order) are given by ``jobs(task)``, on the threads *strategy* says, emptying
    *tasks* as it goes."""
    if strategy.serial:
        _perform_serially(tasks, jobs)
    else:
        _Pipeline(strategy, jobs).run(tasks)


def _perform_serially(tasks: list, jobs):
    tasks.reverse()
    while tasks:
        for job in jobs(tasks.pop()):
            job.run()


class _Budget:
    """A limit on the memory taken by data fetched and not yet processed.

    A job whose data would take the total over the limit waits, unless nothing else is waiting to be processed (so
    that a job larger than the whole limit can still go ahead, alone)."""

    def __init__(self, limit: int):
        self._limit = limit
        self._used = 0
        self._condition = threading.Condition()

    def acquire(self, nbytes: int, stop: threading.Event) -> bool:
        """Wait until *nbytes* fit within the limit, and take them; return False, taking nothing, if *stop* is set"""
        with self._condition:
            while self._used > 0 and self._used + nbytes > self._limit:
                if stop.is_set():
                    return False
                self._condition.wait(timeout=0.1)
            if stop.is_set():
                return False
            self._used += nbytes
            return True

    def release(self, nbytes: int):
        with self._condition:
            self._used -= nbytes
            self._condition.notify_all()

    def charge(self, nbytes: int):
        """Take *nbytes* (or give them back, if negative) without waiting, for memory already taken"""
        if nbytes < 0:
            self.release(-nbytes)
        else:
            with self._condition:
                self._used += nbytes


class _Pipeline:
    """Input threads fetching jobs, and a pool of decode threads processing what they fetch."""

    def __init__(self, strategy: ReadStrategy, jobs=iter):
        self._strategy = strategy
        self._jobs = jobs
        self._budget = _Budget(strategy.inflight_nbytes)
        self._stop = threading.Event()  # set once anything has gone wrong, or the read is interrupted
        self._error = None  # the first exception raised while processing
        self._outstanding = 0  # jobs handed to decode threads and not yet processed
        self._condition = threading.Condition()
        self._decoders = None

    def run(self, tasks: list):
        io_threads = min(self._strategy.io_threads, len(tasks))
        if self._strategy.decode_threads > 0:
            self._decoders = concurrent.futures.ThreadPoolExecutor(max_workers=self._strategy.decode_threads,
                                                                   thread_name_prefix="pynbody-decode")
        readers = None
        try:
            if io_threads <= 1:
                self._read(tasks)
            else:
                readers = concurrent.futures.ThreadPoolExecutor(max_workers=io_threads,
                                                                thread_name_prefix="pynbody-input")
                # list() waits for every task, and raises the first exception any of them raised
                list(readers.map(self._read_task, self._take_each(tasks)))
            self._wait_for_decoding()
        except BaseException:
            # (before waiting for any thread, so that an interruption, such as KeyboardInterrupt in this thread,
            # stops the others after the job each is doing)
            self._stop.set()
            raise
        finally:
            if readers is not None:
                readers.shutdown(wait=True, cancel_futures=True)
            if self._decoders is not None:
                self._decoders.shutdown(wait=True, cancel_futures=True)
        if self._error is not None:
            raise self._error

    @staticmethod
    def _take_each(tasks):
        """Yield the tasks one by one, so that *tasks* no longer holds each once it is handed out"""
        tasks.reverse()
        while tasks:
            yield tasks.pop()

    def _read(self, tasks):
        for task in self._take_each(tasks):
            self._read_task(task)

    def _read_task(self, task):
        """Fetch the jobs of a task in order, handing each to the decode threads (or processing it here)"""
        try:
            self._fetch_jobs(task)
        except BaseException:
            self._stop.set()  # (now, so that other threads stop at once, rather than once this reaches run)
            raise

    def _fetch_jobs(self, task):
        for job in self._jobs(task):
            if self._stop.is_set():
                return
            if job.process is None:
                job.run()  # (nothing is held once fetched)
                continue
            nbytes = job.nbytes
            if not self._budget.acquire(nbytes, self._stop):
                return
            if self._decoders is None:
                try:
                    job.run()
                finally:
                    self._budget.release(nbytes)
                continue
            try:
                fetched = job.fetch()
            except BaseException:
                self._budget.release(nbytes)
                raise
            # (a job that has found, once fetched, that it holds more or less than was thought is charged what it
            # holds, so that no more is fetched until enough has been processed)
            if job.nbytes != nbytes:
                self._budget.charge(job.nbytes - nbytes)
            if fetched is None:
                self._budget.release(job.nbytes)
                continue
            with self._condition:
                self._outstanding += 1
            # (in a list, which the decode thread empties, so that what was fetched is freed once processed, rather
            # than when the pool lets go of the call)
            self._decoders.submit(self._process, job, [fetched])
            del job, fetched

    def _process(self, job, holder):
        try:
            fetched = holder.pop()
            if not self._stop.is_set():
                job.process(fetched)
        except BaseException as e:
            with self._condition:
                if self._error is None:
                    # (keeping the traceback, but not what its frames held, such as the data being decoded)
                    traceback.clear_frames(e.__traceback__)
                    self._error = e
            self._stop.set()
        finally:
            fetched = None
            self._budget.release(job.nbytes)
            with self._condition:
                self._outstanding -= 1
                self._condition.notify_all()

    def _wait_for_decoding(self):
        with self._condition:
            while self._outstanding > 0:
                self._condition.wait(timeout=0.1)
