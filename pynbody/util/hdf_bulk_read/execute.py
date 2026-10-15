"""Performing units of work (see :mod:`.plan`) on the threads a :class:`.strategy.ReadStrategy` says.

The work on each file becomes a task of jobs (see :func:`.plan.group_by_file`), performed by the pipeline of input
and decode threads shared with pynbody's other bulk readers (:mod:`pynbody.util.bulk_read.pipeline`).
"""

from __future__ import annotations

from ..bulk_read.pipeline import perform_tasks
from . import plan
from .strategy import ReadStrategy


def prepare(works: list[plan._Work]):
    """Make every HDF5 lookup the work will need now, in this thread: from several threads at once they would only
    queue for h5py's lock, and the handing over of that lock is itself costly"""
    for work in works:
        work.prepare()


def perform(works: list[plan._Work], strategy: ReadStrategy):
    """Perform the work (prepared by :func:`prepare`), emptying *works* as it goes."""
    tasks = plan.group_by_file(works)
    works.clear()
    perform_tasks(tasks, strategy, plan.jobs)
