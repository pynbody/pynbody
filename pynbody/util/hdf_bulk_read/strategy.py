"""Rules deciding how many threads read, and how the work is shared between them.

:meth:`pynbody.util.hdf_bulk_read.BulkReader.read` can read in several threads at once when it reads the data
directly rather than through h5py. Whether that helps depends on where the data are and how they are stored, so
before reading it calls :func:`choose_read_strategy`, with a description of the work (a :class:`ReadSummary`),
which applies these rules, in order:

1. **Reads through h5py are serial.** If any of the data have to be read through h5py (because direct reading is
   switched off, or the data are stored in a way pynbody does not read itself), everything is read in one thread:
   h5py's lock serialises its reads anyway, and threads contending for it cost more than they save.

2. **A fixed number of threads, if one is configured.** If the ``threads`` option is a number rather than
   ``auto``, that many threads are used (1 meaning serial), sharing the work as rule 3 or 4 would.

3. **Parallel filesystems: one thread per file, up to** ``parallel-filesystem-threads`` **(default 16).** On Lustre
   and similar systems a spanned snapshot's files usually live on different servers, while each file lives on one.
   Each thread therefore reads whole files, in order: several servers are busy at once, and each file is still read
   sequentially, which keeps the filesystem's readahead effective. Threads sharing one file would compete for one
   server and defeat readahead, so a single file is read serially here. (A virtual dataset's work is attributed to
   its source files, so it is shared out just as the source files would be.)

4. **Other filesystems: threads only for compressed data, up to** ``compressed-data-threads`` **(default 4).** On a
   local disk, reading is rarely what limits speed; decompression is, and it can proceed in parallel. So compressed
   data are read piece by piece in several threads, even from a single file, while uncompressed data are read
   serially.

In every case, the number of threads is capped by the number of pieces of work, by the number of CPUs this
process may use, and by ``decode-memory`` (default 2 GiB): each thread needs about three times the size of the
chunks it decodes, which matters only for data stored in very large chunks.

The options live in the ``[hdf-bulk-read]`` section of the configuration; they can also be changed for a session
through :data:`config`, for example ``pynbody.util.hdf_bulk_read.strategy.config['threads'] = 8``. The strategy
chosen for each read is logged at debug level, together with the reason for it.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os

from ... import config_parser

logger = logging.getLogger('pynbody.util.hdf_bulk_read.strategy')

# Filesystem types on which files are spread over several servers (so that reading several files at once pays off)
# (NFS is not among them: it is usually served by a single server)
parallel_filesystem_types = {'lustre', 'gpfs', 'beegfs', 'wekafs', 'ceph', 'fuse.ceph', 'pvfs2', 'orangefs', 'panfs'}


def _read_config():
    def option(name, default):
        return config_parser.get('hdf-bulk-read', name, fallback=default).strip()

    def whole_number(name, default, allowed=()):
        value = option(name, str(default))
        if value.lower() in allowed:
            return value.lower()
        try:
            number = int(value)
            if number < 1:
                raise ValueError
            return number
        except ValueError:
            logger.warning("Ignoring the value %r of %s in the [hdf-bulk-read] configuration: expected a whole "
                           "number of at least 1%s; using %s", value, name,
                           "".join(f" or '{a}'" for a in allowed), default)
            return default

    # (options that lived in [gadgethdf] while this package was being developed)
    for old_name in ('bulk-read-threads', 'parallel-filesystem-threads', 'compressed-data-threads',
                     'bulk-read-memory'):
        if config_parser.has_option('gadgethdf', old_name):
            logger.warning("Ignoring %s in the [gadgethdf] configuration: options for reading HDF5 data now live in "
                           "[hdf-bulk-read] (see pynbody.util.hdf_bulk_read.strategy)", old_name)

    return _Options({'threads': str(whole_number('threads', 'auto', allowed=('auto',))),
                     'parallel-filesystem-threads': whole_number('parallel-filesystem-threads', 16),
                     'compressed-data-threads': whole_number('compressed-data-threads', 4),
                     'decode-memory': whole_number('decode-memory', 2 * 1024 ** 3)})


class _Options(dict):
    """The options, which can be changed but not added to, so that a misspelt name is not silently ignored"""

    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f"{key!r} is not an option for reading HDF5 data; the options are {', '.join(self)}")
        super().__setitem__(key, value)


config = _read_config()
"""The options governing threaded reading, initially from the configuration files; may be changed at run time."""


@dataclasses.dataclass(frozen=True)
class ReadSummary:
    """What the planner knows about the reads needed to load one array."""

    paths: tuple[str, ...]
    """The files the data come from (for a virtual dataset, its source files)"""

    num_reads: int
    """The number of pieces to be read"""

    num_files: int
    """The number of distinct files among the pieces"""

    all_direct: bool
    """True if every piece can be read directly, rather than through h5py"""

    compressed: bool
    """True if any of the data are compressed"""

    max_chunk_nbytes: int = 0
    """The size of the largest chunk to be decoded (0 if the data are not chunked)"""


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
    if summary.num_reads == 0:
        return ReadStrategy(1, True, "there is nothing to read")
    if summary.num_reads == 1:
        return ReadStrategy(1, True, "there is only one piece to read")
    if not summary.all_direct:
        return ReadStrategy(1, True, "some of the data must be read through h5py, which serialises reads")

    filesystem = filesystem_type(summary.paths[0]) if summary.paths else None
    on_parallel_filesystem = filesystem in parallel_filesystem_types
    per_file = on_parallel_filesystem
    tasks = summary.num_files if per_file else summary.num_reads

    threads = _requested_threads()
    if threads is not None:
        reason = f"the threads option is set to {threads}"
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
    # each thread holds a chunk compressed and decompressed while decoding it, and may keep one for its next read
    memory_limit = max(1, config['decode-memory'] // max(3 * summary.max_chunk_nbytes, 1))
    if threads > memory_limit:
        threads = memory_limit
        reason += (f" (limited to {memory_limit} by decode-memory, since chunks decode to as much as "
                   f"{summary.max_chunk_nbytes / 2 ** 20:.0f} MB)")
    if threads <= 1:
        return ReadStrategy(1, per_file, reason + ", which leaves only one thread's worth of work")
    return ReadStrategy(threads, per_file, reason)


def _requested_threads() -> int | None:
    """The number of threads the threads option asks for, or None for 'auto' (or a value that makes no sense)"""
    requested = str(config['threads']).strip().lower()
    if requested == 'auto':
        return None
    try:
        return max(int(requested), 1)
    except ValueError:
        logger.warning("Ignoring the threads option, %r, which is neither a number nor 'auto'", requested)
        return None


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
