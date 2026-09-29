"""Rules deciding how many threads read, and how many decode.

:meth:`pynbody.util.hdf_bulk_read.BulkReader.read` reads in two stages, each with its own threads (see
:mod:`.execute`): *input* threads each read a file at a time, sequentially, and *decode* threads decompress what they
have read and put it in place. How many of each help depends on different things: input on where the files are
(several files on a parallel filesystem can be read at once, but a local disk is best read by one reader at a time),
and decoding on the CPUs available. So before reading, :meth:`~.BulkReader.read` calls :func:`choose_read_strategy`
with a description of the work (a :class:`ReadSummary`), which applies these rules:

1. **Reads through h5py are serial.** If any of the data have to be read through h5py (because direct reading is
   switched off, or the data are stored in a way pynbody does not read itself), everything is read in one thread:
   h5py's lock serialises its reads anyway, and threads contending for it cost more than they save.

2. **Input threads: one per file on a parallel filesystem, up to** ``parallel-filesystem-io-threads`` **(default
   16); otherwise one.** On Lustre and similar systems a spanned snapshot's files usually live on different
   servers, so several can be read at once; each is still read sequentially by one thread, which keeps the
   filesystem's readahead effective. A local disk is read fastest by one reader. The ``io-threads`` option, if a
   number rather than ``auto``, fixes the number instead.

3. **Decode threads: for compressed data, up to** ``max-decode-threads`` **(default 8).** Decompression is what
   limits the speed of reading compressed data, wherever they are, and it proceeds in parallel, even for data read
   from a single file by a single input thread. Beyond about 8 threads it gains little: disks and filesystems
   rarely deliver data faster than 8 threads decode them, and even data already in memory are decoded only
   modestly faster by more, which contend with each other, at a cost in CPU time and memory. For uncompressed data there is nothing to decode, and the input
   threads put the data in place themselves. The ``decode-threads`` option, if a number rather than ``auto``, fixes
   the number instead (0 meaning that input threads decode what they read themselves).

Neither number exceeds what there is work for (files, for input; chunks, for decoding), and under ``auto`` decode
threads do not exceed the CPUs this process may use, and are not used at all for chunks smaller than 128 kB: the
Python work around decompressing each chunk, which threads cannot share, then takes as long as the decompression. Memory limits decode threads too: each decode thread, with the
data waiting for it, needs about five times the size of the chunks it decodes, and ``decode-memory`` (default 2 GiB)
bounds the total, which matters only for data stored in very large chunks. (With no decode threads, it limits
instead how many input threads may hold a chunk at once.)

The options live in the ``[hdf-bulk-read]`` section of the configuration; they can also be changed for a session
through :data:`config`, for example ``pynbody.util.hdf_bulk_read.strategy.config['decode-threads'] = 8``. The
strategy chosen for each read is logged at debug level, together with the reason for it.
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

    def whole_number(name, default, minimum=1):
        value = option(name, str(default))
        if value.lower() == 'auto' and default == 'auto':
            return 'auto'
        try:
            number = int(value)
            if number < minimum:
                raise ValueError
            return number
        except ValueError:
            logger.warning("Ignoring the value %r of %s in the [hdf-bulk-read] configuration: expected a whole "
                           "number of at least %d%s; using %s", value, name, minimum,
                           " or 'auto'" if default == 'auto' else "", default)
            return default

    # options that were renamed while this package was being developed
    old_names = {'gadgethdf': ('bulk-read-threads', 'parallel-filesystem-threads', 'compressed-data-threads',
                               'bulk-read-memory'),
                 'hdf-bulk-read': ('threads', 'parallel-filesystem-threads', 'compressed-data-threads')}
    for section, names in old_names.items():
        for old_name in names:
            if config_parser.has_option(section, old_name):
                logger.warning("Ignoring %s in the [%s] configuration, which is no longer an option: see "
                               "pynbody.util.hdf_bulk_read.strategy for the options in [hdf-bulk-read]",
                               old_name, section)

    return _Options({'io-threads': whole_number('io-threads', 'auto'),
                     'parallel-filesystem-io-threads': whole_number('parallel-filesystem-io-threads', 16),
                     'decode-threads': whole_number('decode-threads', 'auto', minimum=0),
                     'max-decode-threads': whole_number('max-decode-threads', 8),
                     'decode-memory': whole_number('decode-memory', 2 * 1024 ** 3)})


class _Options(dict):
    """The options, which can be changed but not added to, so that a misspelt name is not silently ignored"""

    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f"{key!r} is not an option for reading HDF5 data; the options are {', '.join(self)}")
        super().__setitem__(key, value)


config = _read_config()
"""The options governing threaded reading, initially from the configuration files; may be changed at run time."""


# Under 'auto', compressed data in chunks smaller than this are decoded by the input threads, not decode threads
_min_chunk_nbytes_for_decode_threads = 128 * 1024


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

    max_job_nbytes: int = 0
    """The most data any job holds between fetching and processing it (see .common.ReadProperties); if 0, taken to
    be max_chunk_nbytes"""

    num_chunks: int = 0
    """Roughly how many chunks are to be decoded"""


@dataclasses.dataclass(frozen=True)
class ReadStrategy:
    """How to perform the reads needed to load one array."""

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


_SERIAL = dict(io_threads=1, decode_threads=0)


def choose_read_strategy(summary: ReadSummary) -> ReadStrategy:
    """Decide how to read an array, applying the rules in the module docstring."""
    if summary.num_reads == 0:
        return ReadStrategy(**_SERIAL, reason="there is nothing to read")
    if not summary.all_direct:
        return ReadStrategy(**_SERIAL, reason="some of the data must be read through h5py, which serialises reads")

    filesystem = filesystem_type(summary.paths[0]) if summary.paths else None
    where = f"on a parallel filesystem ({filesystem})" if filesystem in parallel_filesystem_types else \
        f"on a {filesystem or 'local'} filesystem"

    requested = _requested('io-threads')
    if requested is not None:
        io_threads, io_reason = requested, f"the io-threads option is set to {requested}"
    elif filesystem in parallel_filesystem_types:
        io_threads = config['parallel-filesystem-io-threads']
        io_reason = f"the files are {where}, where several can be read at once"
    else:
        io_threads, io_reason = 1, f"the files are {where}, which one reader reads fastest"
    if io_threads > summary.num_files:
        io_threads = max(summary.num_files, 1)
        io_reason += f" (but there are only {summary.num_files} files)"

    # (what a job holds; the budget and memory limits are counted in these)
    chunk_nbytes = max(summary.max_job_nbytes or summary.max_chunk_nbytes, 1)
    # each decode thread holds a chunk compressed and decompressed while decoding it, perhaps a buffer the size of the
    # rows it copies out of it, and up to two chunks read and waiting for it
    memory_limit = config['decode-memory'] // (5 * chunk_nbytes)
    if not summary.compressed:
        decode_threads, decode_reason = 0, "the data are not compressed, so there is nothing to decode"
    else:
        requested = _requested('decode-threads')
        if requested is not None:
            decode_threads, decode_reason = requested, f"the decode-threads option is set to {requested}"
        elif summary.max_chunk_nbytes < _min_chunk_nbytes_for_decode_threads:
            decode_threads = 0
            decode_reason = (f"the data are compressed, but in chunks of at most {summary.max_chunk_nbytes / 1024:.0f} "
                             f"kB, too small for decompressing them to be worth sharing between threads")
        else:
            decode_threads = min(config['max-decode-threads'], available_cpus())
            decode_reason = f"the data are compressed, and {available_cpus()} CPUs are available"
        if decode_threads > summary.num_chunks:
            decode_threads = summary.num_chunks
            decode_reason += f" (but there are only {summary.num_chunks} chunks to decode)"
        if decode_threads > memory_limit:
            decode_threads = memory_limit
            decode_reason += (f" (limited to {memory_limit} by decode-memory, since chunks decode to as much as "
                              f"{chunk_nbytes / 2 ** 20:.0f} MB)")
        if decode_threads <= 1 and io_threads <= 1 and summary.num_chunks <= 1:
            decode_threads = 0

    io_threads, decode_threads = max(io_threads, 1), max(decode_threads, 0)
    reason = f"{io_threads} input thread(s) because {io_reason}; {decode_threads} decode thread(s) because " \
             f"{decode_reason}"
    if decode_threads > 0:
        # Two waiting for each decode thread. (That can leave input threads waiting, when there are more of them than
        # twice the decode threads; but then decoding is what limits the speed of reading, and more data waiting for
        # it would only take more memory.)
        inflight_chunks = 2 * decode_threads
    else:
        # Input threads process what they fetch themselves, each holding a chunk and whatever decoding it takes, so
        # decode-memory limits how many do so at once
        inflight_chunks = max(min(io_threads, memory_limit), 1)
        if io_threads > 1 and inflight_chunks < io_threads and summary.num_chunks > 0:
            reason += f" (and only {inflight_chunks} at a time may hold a chunk, limited by decode-memory)"
    return ReadStrategy(io_threads=io_threads, decode_threads=decode_threads, reason=reason,
                        inflight_nbytes=inflight_chunks * chunk_nbytes)


def _requested(option: str) -> int | None:
    """The number of threads an option asks for, or None for 'auto' (or a value that makes no sense)"""
    requested = str(config[option]).strip().lower()
    if requested == 'auto':
        return None
    try:
        return max(int(requested), 0 if option == 'decode-threads' else 1)
    except ValueError:
        logger.warning("Ignoring the %s option, %r, which is neither a number nor 'auto'", option, requested)
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
