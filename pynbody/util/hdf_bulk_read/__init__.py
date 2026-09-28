"""Bulk reads of HDF5 datasets that bypass libhdf5, so that reads of different files can overlap.

pynbody reads HDF5 snapshots through h5py, whose interpreter-wide lock serialises every call into libhdf5, even
calls that concern different files. This package lets the bulk particle data be read without going through
libhdf5. h5py is still used for everything it does cheaply: finding the dataset, its datatype, its storage layout,
its filters, and the position in the file of its data or of each of its chunks. pynbody then reads the bytes itself
and, for chunked datasets, undoes the filters itself (zlib, which releases the GIL, for deflate; numpy for shuffle
and for verifying fletcher32 checksums).

Reading bytes from a file behind HDF5's back is only correct if pynbody understands exactly how they are stored, so
:meth:`BulkReader.open` checks that before anything is read, and hands the dataset back to be read through h5py
unless every check passes. The checks are:

* the file is open read-only (and not in SWMR mode), through HDF5's default ``sec2`` driver (whose file addresses
  are plain byte offsets), and the path pynbody would read is the very file HDF5 has open -- which is checked
  again on every read;
* the dataset has a simple dataspace of at least one dimension, and its datatype is exactly the standard HDF5
  representation of a numpy integer or floating-point type of 1, 2, 4 or 8 bytes (so no padding, unusual
  precision, non-IEEE or extended-precision floats, enumerations, compound or string types);
* contiguous data lies wholly within the file, is not held in external files, and occupies exactly as many bytes
  as its elements;
* chunked data uses no filters other than deflate, shuffle and fletcher32, and a shuffle filter's element size is
  the datatype's (a partial chunk at the edge of a dataset whose stored size is that of an unfiltered chunk is read
  through h5py, since HDF5 can be told to leave such chunks unfiltered without recording that it has);
* virtual datasets map contiguous blocks of whole rows from source datasets that pass the same checks, and have
  the same datatype (see below);
* the fill value is defined and is written into unallocated space (the default), so that parts of a dataset never
  written read as that value.

Where a check fails because of how the file was written, a :class:`BulkReadFallbackWarning` says so, once per
reason per snapshot, and h5py is used. Datasets h5py is simply the better tool for (compact storage, which only
very small datasets use; scalars; datasets never written) are handed back without a warning.

Reads are checked as they happen, too: every read must return the number of bytes expected, every chunk must
decode to exactly the size of a chunk, and fletcher32 checksums are verified. If anything is amiss, a warning is
issued and that dataset is read through h5py from then on, so any disagreement between pynbody and HDF5 about a
file is settled by HDF5.

Virtual datasets, such as those in the single-file view SWIFT writes of a multi-file snapshot, are decomposed into
their source datasets, which are then read like any other. This is supported where every mapping places a
contiguous block of whole rows of the virtual dataset, which covers the layouts written by SWIFT and by
:class:`pynbody.util.hdf_vds.HdfVdsMaker`. Source files are found by following HDF5's rules (see
:func:`.virtual._resolve_virtual_source_filename`), except that if a search path has been configured (through the
``HDF5_VDS_PREFIX`` environment variable or a dataset access property) the virtual dataset is read through h5py,
since HDF5's handling of those paths varies between versions. Any other layout is read through h5py, as is any
source that cannot be found (HDF5 then fills its rows with the fill value) or that does not itself pass the checks
above.

Files must not be replaced on disk while they are open. Replacement of a file being read directly is detected on
every read, but a source of a virtual dataset that is replaced after HDF5 has opened it cannot be, since pynbody
opens sources by name.

Callers describe what they want as a list of :class:`ReadRequest` (rows of a dataset, and where to put them) and
pass it to :meth:`BulkReader.read`, which takes responsibility for everything else: deciding which datasets can be
read directly, splitting the work (for example, where a virtual dataset draws on one source file after another),
choosing how many threads to read with (see :mod:`.strategy`), and performing the reads.

The package is organised as:

* :mod:`.reader` -- :class:`BulkReader`, which opens datasets for reading and performs requests;
* :mod:`.plan` -- turning requests into units of work, and grouping those into tasks for threads;
* :mod:`.strategy` -- the rules deciding how many threads to use;
* :mod:`.execute` -- performing the tasks, serially or in threads;
* :mod:`.datasets` and :mod:`.virtual` -- direct readers of contiguous, chunked and virtual datasets;
* :mod:`.decode` -- undoing HDF5's filters;
* :mod:`.files` -- access to the bytes of the files HDF5 has open.
"""

from . import strategy
from .common import BulkReadFallbackWarning, ReadProperties
from .datasets import is_direct_reader
from .decode import decode_chunk, fletcher32
from .plan import ReadRequest
from .reader import BulkReader

__all__ = ['BulkReader', 'BulkReadFallbackWarning', 'ReadProperties', 'ReadRequest', 'decode_chunk', 'fletcher32',
           'is_direct_reader', 'strategy']
