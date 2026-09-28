"""Performing units of work (see :mod:`.plan`), serially or in threads, as a :class:`.strategy.ReadStrategy` says.

Each unit of work is let go of once it has been performed, so that a file's datasets, and the chunks decoded from
them, are freed once all the work on that file is done, rather than when every request has been read.

For now, each unit of work is performed whole by one thread, which both reads and decodes. Separating the two (so
that, for example, one thread reads a file sequentially while several decode what it has read) would change this
module, and how readers divide work into steps, but not the planning or anything outside this package: chunked
readers already get each chunk in separate steps of locating, fetching and decoding (see
:meth:`.datasets._ChunkedReader._get_chunk`).
"""

from __future__ import annotations

import concurrent.futures
import concurrent.futures.thread  # (imported lazily by concurrent.futures otherwise, which could fail mid-read)

from . import plan
from .strategy import ReadStrategy


def perform(works: list[plan._Work], strategy: ReadStrategy):
    """Perform the work, emptying *works* as it goes."""
    if strategy.threads > 1:
        # Make every HDF5 lookup the work will need now, serially: from several threads at once they would only
        # queue for h5py's lock, and the handing over of that lock is itself costly
        for work in works:
            work.prepare()
        tasks = plan.group(works, strategy.per_file)
        works.clear()
        _perform_in_threads(tasks, strategy.threads)
    else:
        _perform_task(works)


def _perform_task(task: list[plan._Work]):
    """Perform the units of work of a task in order, letting go of each once performed"""
    task.reverse()
    while task:
        task.pop().perform()


def _perform_in_threads(tasks: list[list[plan._Work]], num_threads: int):
    """Perform tasks concurrently, in a pool of *num_threads* threads, each task in order by one thread."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads,
                                               thread_name_prefix="pynbody-hdf-read") as executor:
        # list() waits for every task, and raises the first exception any of them raised; tasks not yet started
        # are then cancelled (by map), as they are if waiting is interrupted (e.g. by KeyboardInterrupt)
        list(executor.map(_perform_task, tasks))
