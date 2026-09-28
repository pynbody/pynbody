"""Rules deciding how many threads read a snapshot's bulk data, and how the work is shared between them.

pynbody reads the bulk data of gadget-like HDF5 snapshots (including SWIFT) in pieces, and can read pieces in
several threads at once when it reads the data directly rather than through h5py (see
:mod:`pynbody.util.hdf_bulk_read`). Whether that helps depends on where the data are and how they are stored, so
before loading an array pynbody calls :func:`choose_read_strategy`, which applies these rules, in order:

1. **Reads through h5py are serial.** If any part of the array has to be read through h5py (because direct reading
   is switched off, or the data are stored in a way pynbody does not read itself), everything is read in one
   thread: h5py's lock serialises its reads anyway, and threads contending for it cost more than they save.

2. **A fixed number of threads, if one is configured.** If ``bulk-read-threads`` is a number rather than ``auto``,
   that many threads are used (1 meaning serial), sharing the work as rule 3 or 4 would.

3. **Parallel filesystems: one thread per file, up to** ``parallel-filesystem-threads`` **(default 16).** On Lustre
   and similar systems a spanned snapshot's files usually live on different servers, while each file lives on one.
   Each thread therefore reads whole files, in order: several servers are busy at once, and each file is still read
   sequentially, which keeps the filesystem's readahead effective. Threads sharing one file would compete for one
   server and defeat readahead, so a single-file snapshot is read serially here. (The pieces of a virtual dataset
   count as separate files, since their data come from different source files.)

4. **Other filesystems: threads only for compressed data, up to** ``compressed-data-threads`` **(default 4).** On a
   local disk, reading is rarely what limits speed; decompression is, and it can proceed in parallel. So compressed
   data are read piece by piece in several threads, even from a single file, while uncompressed data are read
   serially.

In every case, the number of threads is capped by the number of pieces of work and by the number of CPUs this
process may use. The options all live in the ``[gadgethdf]`` section of the configuration; they can also be changed
for a session through :data:`config`, for example ``pynbody.util.hdf_read_strategy.config['bulk-read-threads'] = 8``.
The strategy chosen for the most recent load is logged at debug level, together with the reason for it.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os

from .. import config_parser

logger = logging.getLogger('pynbody.util.hdf_read_strategy')

# Filesystem types on which files are spread over several servers (so that reading several files at once pays off)
parallel_filesystem_types = {'lustre', 'gpfs', 'beegfs', 'wekafs', 'ceph', 'fuse.ceph', 'pvfs2', 'orangefs', 'panfs',
                             'nfs', 'nfs4'}


def _read_config():
    def option(name, default):
        return config_parser.get('gadgethdf', name, fallback=default).strip()
    return {'bulk-read-threads': option('bulk-read-threads', 'auto'),
            'parallel-filesystem-threads': int(option('parallel-filesystem-threads', '16')),
            'compressed-data-threads': int(option('compressed-data-threads', '4'))}


config = _read_config()
"""The options governing threaded reading, initially from the configuration files; may be changed at run time."""


@dataclasses.dataclass(frozen=True)
class ReadSummary:
    """What the planner knows about the reads needed to load one array."""

    paths: tuple[str, ...]
    """The files the data come from (for a virtual dataset, the file holding it)"""

    num_reads: int
    """The number of pieces to be read"""

    num_files: int
    """The number of distinct files among the pieces, counting each piece of a virtual dataset as its own file"""

    all_direct: bool
    """True if every piece can be read directly, rather than through h5py"""

    compressed: bool
    """True if any of the data are compressed"""


@dataclasses.dataclass(frozen=True)
class ReadStrategy:
    """How to perform the reads needed to load one array."""

    threads: int
    """Number of threads to read with; 1 means serially, in the calling thread"""

    per_file: bool
    """True if each thread reads whole files; False if pieces of the same file may be read by different threads"""

    reason: str
    """Why, in words"""


def choose_read_strategy(summary: ReadSummary) -> ReadStrategy:
    """Decide how to read an array, applying the rules in the module docstring."""
    if summary.num_reads <= 1:
        return ReadStrategy(1, True, "there is only one piece to read")
    if not summary.all_direct:
        return ReadStrategy(1, True, "some of the data must be read through h5py, which serialises reads")

    filesystem = filesystem_type(summary.paths[0]) if summary.paths else None
    on_parallel_filesystem = filesystem in parallel_filesystem_types
    per_file = on_parallel_filesystem
    tasks = summary.num_files if per_file else summary.num_reads

    requested = config['bulk-read-threads']
    if requested.lower() != 'auto':
        threads = int(requested)
        reason = f"bulk-read-threads is set to {threads}"
    elif on_parallel_filesystem:
        threads = config['parallel-filesystem-threads']
        reason = f"the files are on a parallel filesystem ({filesystem}), so each thread reads whole files"
    elif summary.compressed:
        threads = config['compressed-data-threads']
        reason = (f"the data are compressed, and on a {filesystem or 'local'} filesystem, so threads share the work "
                  f"of decompressing them")
    else:
        return ReadStrategy(1, per_file, f"the data are uncompressed, on a {filesystem or 'local'} filesystem, where "
                                         f"one thread reads as fast as several")

    limit = min(tasks, available_cpus())
    if threads > limit:
        threads = limit
        reason += f" (limited to {limit} by the number of {'files' if per_file else 'pieces'} and of CPUs)"
    if threads <= 1:
        return ReadStrategy(1, per_file, reason + ", which leaves only one thread's worth of work")
    return ReadStrategy(threads, per_file, reason)


def available_cpus() -> int:
    """The number of CPUs this process may run on"""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def filesystem_type(path: str) -> str | None:
    """The type of the filesystem holding *path* (e.g. 'lustre', 'ext4'), or None if it cannot be found out.

    Found from the mount table on Linux; elsewhere, returns None, which the rules treat as a local filesystem."""
    mounts = _mount_table()
    if not mounts:
        return None
    path = os.path.realpath(path)
    best, best_type = "", None
    for mount_point, fs_type in mounts:
        if (path == mount_point or path.startswith(mount_point.rstrip('/') + '/')) and len(mount_point) > len(best):
            best, best_type = mount_point, fs_type
    return best_type


@functools.lru_cache(maxsize=1)
def _mount_table() -> tuple[tuple[str, str], ...]:
    """(mount point, filesystem type) for each mounted filesystem, or () if unavailable"""
    try:
        with open('/proc/self/mounts') as f:
            lines = f.readlines()
    except OSError:
        return ()
    table = []
    for line in lines:
        fields = line.split()
        if len(fields) >= 3:
            # the mount table escapes spaces and some other characters in octal
            mount_point = fields[1].encode().decode('unicode_escape')
            table.append((mount_point, fields[2]))
    return tuple(table)
