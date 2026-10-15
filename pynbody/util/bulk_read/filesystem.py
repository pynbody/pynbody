"""What pynbody knows about where files are read from, shared by its bulk readers.

How best to read depends on the filesystem: several files on a parallel filesystem (Lustre, GPFS and similar) can be
read at once, but a local disk is best read by one reader at a time; and every request to a network or parallel
filesystem costs a round trip to a server, in which a good deal could have been read instead.
"""

from __future__ import annotations

import functools
import os

# Filesystem types on which files are spread over several servers (so that reading several files at once pays off)
# (NFS is not among them: it is usually served by a single server)
parallel_filesystem_types = {'lustre', 'gpfs', 'beegfs', 'wekafs', 'ceph', 'fuse.ceph', 'pvfs2', 'orangefs', 'panfs'}

# Filesystem types served by a single server over a network
network_filesystem_types = {'nfs', 'nfs4', 'cifs', 'smb3', 'smbfs', 'fuse.sshfs', 'afs', '9p'}

# Filesystem types of high latency and bandwidth (object stores and the like), beside the parallel filesystems
high_latency_filesystem_types = {'glusterfs', 'fuse.glusterfs', 'fuse.dfuse', 'fuse.s3fs', 'fuse.gcsfuse',
                                 'fuse.rclone', 'fuse.juicefs'}

# How many input threads read files at once on a parallel filesystem (one otherwise)
parallel_filesystem_io_threads = 16

# About the bytes that could be read in the time a request takes, on each kind of filesystem (see request_gap)
local_request_gap = 64 * 1024  # local disks
network_request_gap = 256 * 1024  # a single file server (a round trip of ~0.3 ms at ~1 GB/s, or more at less)
parallel_request_gap = 1024 * 1024  # parallel filesystems and object stores (~1 ms at several GB/s)


def available_cpus() -> int:
    """The number of CPUs this process may run on"""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def filesystem_type(path: str) -> str | None:
    """The type of the filesystem holding *path* (e.g. 'lustre', 'ext4'), or None if it cannot be found out.

    Found from the mount table on Linux; elsewhere, returns None, which is treated as a local filesystem."""
    mounts = _mount_table()
    if not mounts:
        return None
    path = os.path.realpath(path)
    best, best_type = "", None
    for mount_point, fs_type in mounts:
        if (path == mount_point or path.startswith(mount_point.rstrip('/') + '/')) and len(mount_point) > len(best):
            best, best_type = mount_point, fs_type
    return best_type


def is_parallel(filesystem: str | None) -> bool:
    """Whether files on a filesystem of this type are best read several at once"""
    return filesystem in parallel_filesystem_types


def default_io_threads(filesystem: str | None) -> int:
    """How many input threads should read files on a filesystem of this type: several on a parallel filesystem
    (each reading a file sequentially), one otherwise"""
    return parallel_filesystem_io_threads if is_parallel(filesystem) else 1


def request_gap(filesystem: str | None) -> int:
    """About the bytes that could be read from a filesystem of this type in the time a request takes, so that two
    stretches of a file closer than this are better read in one request, with what lies between them"""
    if is_parallel(filesystem) or filesystem in high_latency_filesystem_types:
        return parallel_request_gap
    if filesystem in network_filesystem_types:
        return network_request_gap
    return local_request_gap


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
