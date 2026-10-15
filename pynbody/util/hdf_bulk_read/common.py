"""Definitions shared by the modules of :mod:`pynbody.util.hdf_bulk_read`."""

from __future__ import annotations

import dataclasses

_DEFLATE_FILTER = 1

_SHUFFLE_FILTER = 2

_FLETCHER32_FILTER = 3

_supported_filters = {_DEFLATE_FILTER, _SHUFFLE_FILTER, _FLETCHER32_FILTER}

# HDF5 drivers whose file addresses are byte offsets into a single ordinary file, which pynbody can check against
# the file descriptor HDF5 holds. ('windows' is an alias of sec2 in some builds.)
_supported_drivers = {'sec2', 'windows'}



@dataclasses.dataclass(frozen=True)
class ReadProperties:
    """What matters, for deciding how to perform them, about reading some rows of a dataset"""

    direct: bool
    """True if the rows are read directly, rather than through h5py"""

    compressed: bool = False
    """True if the rows are compressed, so that decoding them takes appreciable CPU time"""

    chunk_nbytes: int = 0
    """The size of the largest chunk to be decoded (0 if the data are not chunked)"""

    job_nbytes: int = 0
    """The most data a job reading the rows holds between fetching and processing it: the size of the chunks it
    decodes (small ones being decoded several to a job), or of a block of data to be converted or gathered (0 if
    nothing is held). Decoding needs roughly twice this."""

    num_chunks: int = 0
    """How many chunks are to be decoded"""


_THROUGH_H5PY = ReadProperties(direct=False)


class BulkReadFallbackWarning(UserWarning):
    """Issued when pynbody reads a dataset through h5py because it cannot be sure of reading it correctly itself.

    The data are still read correctly; but through h5py, reads cannot overlap."""


class _CannotReadDirectly(Exception):
    """Raised while planning a read when a dataset should be read through h5py instead.

    *warn* is False when that is merely the better choice (e.g. for tiny datasets) rather than a limitation."""

    def __init__(self, reason: str, warn: bool = True):
        super().__init__(reason)
        self.reason = reason
        self.warn = warn


class _UnexpectedData(Exception):
    """Raised during a read when the file does not contain what its metadata led us to expect."""
