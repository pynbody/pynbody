"""Reading large amounts of data from files on threads, shared by pynbody's bulk readers.

The readers of HDF5 data (:mod:`pynbody.util.hdf_bulk_read`) and of RAMSES outputs (:mod:`pynbody.snapshot.ramses`)
work the same way: they plan their reads as jobs, each an input step and a computation on what it read, and run them
through one pipeline of input and decode threads (:mod:`.pipeline`), with as many threads of each kind as suits the
filesystem and machine (:mod:`.filesystem`).
"""

from . import filesystem, pipeline
from .pipeline import Job, ReadStrategy, perform_tasks
