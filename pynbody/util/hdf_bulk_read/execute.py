"""Performing units of work (see :mod:`.plan`) as a :class:`.strategy.ReadStrategy` says.

Work is done as jobs (see :class:`.common.Job`), each an input step (*fetch*) and optionally a computation on what
was fetched (*process*), in two stages with their own threads:

* input threads each take a file's work (a task; see :func:`.plan.group_by_file`) and fetch its jobs in order, so
  that each file is read sequentially;
* decode threads process what has been fetched, in whatever order it arrives.

The memory taken by what has been fetched but not yet processed is kept within the strategy's ``inflight_nbytes``:
an input thread waits before fetching more until enough has been processed. With no decode threads, input threads
process what they fetch themselves, and the same limit applies to how many do so at once; with one input thread and
no decode threads, everything is done in the calling thread.

Each unit of work, and each job, is let go of once done, so that a file's datasets are freed once all the work on
that file is done, rather than when every request has been read.
"""

from __future__ import annotations

import concurrent.futures
import concurrent.futures.thread  # (imported lazily by concurrent.futures otherwise, which could fail mid-read)
import threading

from . import plan
from .strategy import ReadStrategy


def perform(works: list[plan._Work], strategy: ReadStrategy):
    """Perform the work, emptying *works* as it goes."""
    # Make every HDF5 lookup the work will need now, in this thread: from several threads at once they would only
    # queue for h5py's lock, and the handing over of that lock is itself costly
    for work in works:
        work.prepare()
    tasks = plan.group_by_file(works)
    works.clear()
    if strategy.serial:
        _perform_serially(tasks)
    else:
        _Pipeline(strategy).run(tasks)


def _perform_serially(tasks: list[list[plan._Work]]):
    tasks.reverse()
    while tasks:
        for job in plan.jobs(tasks.pop()):
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


class _Pipeline:
    """Input threads fetching jobs, and a pool of decode threads processing what they fetch."""

    def __init__(self, strategy: ReadStrategy):
        self._strategy = strategy
        self._budget = _Budget(strategy.inflight_nbytes)
        self._stop = threading.Event()  # set once anything has gone wrong, or the read is interrupted
        self._error = None  # the first exception raised while processing
        self._outstanding = 0  # jobs handed to decode threads and not yet processed
        self._condition = threading.Condition()
        self._decoders = None

    def run(self, tasks: list[list[plan._Work]]):
        io_threads = min(self._strategy.io_threads, len(tasks))
        if self._strategy.decode_threads > 0:
            self._decoders = concurrent.futures.ThreadPoolExecutor(max_workers=self._strategy.decode_threads,
                                                                   thread_name_prefix="pynbody-hdf-decode")
        try:
            if io_threads <= 1:
                self._read(tasks)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=io_threads,
                                                           thread_name_prefix="pynbody-hdf-input") as readers:
                    # list() waits for every task, and raises the first exception any of them raised; tasks not yet
                    # started are then cancelled (by map), as they are if waiting is interrupted
                    list(readers.map(self._read_task, self._take_each(tasks)))
            self._wait_for_decoding()
        except BaseException:
            self._stop.set()
            raise
        finally:
            if self._decoders is not None:
                self._decoders.shutdown(wait=True, cancel_futures=True)
        if self._error is not None:
            raise self._error

    @staticmethod
    def _take_each(tasks):
        """Yield the tasks one by one, letting go of each as it is handed out"""
        tasks.reverse()
        while tasks:
            yield tasks.pop()

    def _read(self, tasks):
        for task in self._take_each(tasks):
            self._read_task(task)

    def _read_task(self, task: list[plan._Work]):
        """Fetch the jobs of a task in order, handing each to the decode threads (or processing it here)"""
        try:
            self._fetch_jobs(task)
        except BaseException:
            self._stop.set()  # (now, so that other threads stop at once, rather than once this reaches run)
            raise

    def _fetch_jobs(self, task: list[plan._Work]):
        for job in plan.jobs(task):
            if self._stop.is_set():
                return
            if job.process is None:
                job.run()  # (nothing is held once fetched)
                continue
            if not self._budget.acquire(job.nbytes, self._stop):
                return
            if self._decoders is None:
                try:
                    job.run()
                finally:
                    self._budget.release(job.nbytes)
                continue
            try:
                fetched = job.fetch()
            except BaseException:
                self._budget.release(job.nbytes)
                raise
            if fetched is None:
                self._budget.release(job.nbytes)
                continue
            with self._condition:
                self._outstanding += 1
            self._decoders.submit(self._process, job, fetched)
            del job, fetched

    def _process(self, job, fetched):
        try:
            if not self._stop.is_set():
                job.process(fetched)
        except BaseException as e:
            with self._condition:
                if self._error is None:
                    self._error = e
            self._stop.set()
        finally:
            del fetched
            self._budget.release(job.nbytes)
            with self._condition:
                self._outstanding -= 1
                self._condition.notify_all()

    def _wait_for_decoding(self):
        with self._condition:
            while self._outstanding > 0:
                self._condition.wait(timeout=0.1)
