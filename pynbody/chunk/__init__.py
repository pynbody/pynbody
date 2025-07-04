"""
Methods for describing parts of files to load.

This module provides generalized logic for getting parts of sequential data off disk.
It is for internal use. If you want to write a loader that supports partial loading,
it will make it a lot easier.

The steps for loading particle data are as follows

1. Set up a description of the particles you have on disk. This is a dictionary
   mapping a family type to a slice, e.g.

   .. code-block:: python

      on_disk = {pynbody.family.dm: slice(0,100), pynbody.family.gas: slice(100, 150)}

   describes a file with 100 dark matter particles followed by 50 gas particles.

2. Create a :class:`LoadControl` object.

   .. code-block:: python

      load_control = pynbody.chunk.LoadControl(on_disk, chunk_size, take)

   Here,

   * on_disk is the dictionary you set up in the last step.

   * chunk_size is the maximum number of particles you are willing to load off
     disk at once these will have to be stored in a temporary array, so you don't
     want it to be too large; but also reading a small number of large chunks is
     more efficient, so you don't want it to be too small. No careful experimentation
     has been done with this, but chunk_sizes of around 10000 seem to work OK.

   * take describes what to load in. Currently this is either ``None`` (= load the whole file)
     or a list of ids (= load the specified particles). However this may be expanded
     in future to a more comprehensive syntax. The idea is your code will not have to
     change when this happens, and will automatically support more advanced partial loading
     specifications.

3. Load your particle data. The :class:`LoadControl` object has an ``iterate`` method. This returns
   step-by-step instructions that take you through the file, specifying what to keep and what
   to throw away in a simple-to-use fashion. See the help for :func:`LoadControl.iterate` for details
   on how to implement this final step.

"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Callable, Iterable, Iterator

import numpy as np

import pynbody.util.indexing_tricks

from .. import util

if TYPE_CHECKING:
    from .. import family

class LoadControl:
    """LoadControl provides the logic required for partial loading.

    See the documentation for :mod:`pynbody.chunk` for more information."""

    def __init__(self, family_slice: dict[family.Family, slice], max_chunk: int, clauses: np.ndarray | None):
        """Initialize a LoadControl object.

        *Inputs:*

          *family_slice*: a dictionary of family slices describing the contiguous
            layout of families on disk

          *max_chunk*: the guaranteed maximum chunk of data to load in a single
            read operation. Larger values are likely more efficient, but also require
            bigger temporary buffers in your reader code.

          *clauses*: a description of the type of partial loading to implement. If None, all data is loaded.
            Otherwise, currently the only supported option is a numpy array of particle ids to load.
         """

        self._disk_family_slice = family_slice
        self._generate_family_order()

        # generate simulation-level ID list
        if hasattr(clauses, "__len__"):
            self._ids = np.asarray(clauses, dtype=np.int64) 
        else:
            self._ids = None  # no partial loading!

        self.generate_family_id_lists()
        self._generate_mem_slice()

        self.mem_num_particles = self.mem_family_slice[
            self._ordered_families[-1]].stop
        self.disk_num_particles = self._disk_family_slice[
            self._ordered_families[-1]].stop

        self._generate_chunks(max_chunk)

    @staticmethod
    def _scan_for_next_stop(ids, offset_start, id_maximum):
        from . import scan
        return scan.scan_for_next_stop(ids, offset_start, id_maximum)

    def generate_family_id_lists(self):
        if self._ids is None:
            self._family_ids = None
            return

        self._family_ids = {}
        offset = 0
        stop = 0
        for fam in self._ordered_families:
            sl = self._disk_family_slice[fam]
            self._family_ids[fam] = self._ids[
                (self._ids >= sl.start) * (self._ids < sl.stop)] - sl.start

    def _generate_family_order(self):
        famlist = []
        for fam, sl in self._disk_family_slice.items():
            famlist.append((fam, sl.start, sl.stop))

        famlist.sort(key=lambda x: x[1])
        famlist.sort(key=lambda x: x[2])
        self._ordered_families = [x[0] for x in famlist]

    def _generate_mem_slice(self):
        if self._ids is None:
            self.mem_family_slice = self._disk_family_slice
            return

        self.mem_family_slice = {}
        stop = 0
        for current_family in self._ordered_families:

            start = stop
            stop = stop + len(self._family_ids[current_family])
            self.mem_family_slice[current_family] = slice(start, stop)

    def _generate_null_chunks(self, max_chunk):
        """Generate internal chunk map in the special case that we are loading all data.

        See also :func:`_generate_chunks` for the general case.
        """

        self._family_chunks = {}

        for current_family in self._ordered_families:
            self._family_chunks[current_family] = []
            disk_sl = self._disk_family_slice[current_family]
            for i0 in range(0, disk_sl.stop - disk_sl.start, max_chunk):
                nread = min(disk_sl.stop - disk_sl.start - i0, max_chunk)
                buf_sl = slice(0, nread)
                mem_sl = slice(i0, i0 + nread)

                self._family_chunks[current_family].append(
                    (nread, buf_sl, mem_sl))

    def _generate_chunks(self, max_chunk):
        """Generate internal chunk map

        This must satisfy the requirements that:
          * maximum chunk size does not exceed *max_chunk*
          * chunks do not cross family boundaries

        """

        if self._ids is None:
            self._generate_null_chunks(max_chunk)
            return

        self._family_chunks = {}

        for current_family in self._ordered_families:
            self._family_chunks[current_family] = []
            disk_sl = self._disk_family_slice[current_family]
            ids = self._family_ids[current_family]
            i = 0

            disk_ptr = 0
            mem_ptr = 0

            while disk_ptr < disk_sl.stop - disk_sl.start:
                disk_ptr_end = disk_ptr + \
                    min(disk_sl.stop - disk_sl.start - disk_ptr, max_chunk - 1)
                j = self._scan_for_next_stop(ids, i, disk_ptr_end - 1)

                nread_disk = disk_ptr_end - disk_ptr

                assert (ids[i:j] < disk_ptr_end).all()

                if i != j:
                    mem_slice = slice(mem_ptr, mem_ptr + j - i)
                else:
                    mem_slice = None

                disk_mask = ids[i:j] - disk_ptr

                mem_ptr = mem_ptr + j - i
                i = j
                disk_ptr = disk_ptr_end

                self._family_chunks[current_family].append(
                    (nread_disk, disk_mask, mem_slice))

    def iterate_with_interrupts(self, families_on_disk: list[family.Family], families_in_memory: list[family.Family],
                                disk_interrupt_points: Iterable[int], disk_interrupt_fn: Callable,
                                multiskip: bool = False):
        """Yields instructions for loading an array with the specified families, breaking at specified file offsets

        Performs the same function as :func:`iterate` but additionally takes a list of exact file offsets
        *disk_interrupt_points* at which to interrupt the loading process and call a user-specified function
        *disk_interrupt_fn*.

        Parameters
        ----------

        disk_interrupt_points: Iterable
            List (or other iterable) of disk offsets at which to call the interrupt function, in ascending order

        disk_interrupt_fn: Callable
            Function which takes the file offset as an argument, and is called precisely at the point that the disk
            interrupt point is reached


        See :func:`iterate` for other parameters.

        """

        fpos = 0
        i = 0
        next_dip = disk_interrupt_points[i]
        for nread_disk, disk_slice, mem_slice in self.iterate(families_on_disk, families_in_memory, multiskip):
            while next_dip and fpos + nread_disk > next_dip:
                # an interrupt falls in the middle of our slice
                # work out what to read first
                len_pre = next_dip - fpos

                if disk_slice is not None:
                    d_slice_pre = pynbody.util.indexing_tricks.concatenate_indexing(
                        slice(0, len_pre), disk_slice)
                    len_m_pre = pynbody.util.indexing_tricks.indexing_length(d_slice_pre)
                    m_slice_pre = pynbody.util.indexing_tricks.concatenate_indexing(
                        mem_slice, slice(0, len_m_pre))

                    # work out what to read second
                    len_post = nread_disk - len_pre
                    d_slice_post = copy.copy(pynbody.util.indexing_tricks.concatenate_indexing(
                        disk_slice, slice(len_m_pre, pynbody.util.indexing_tricks.indexing_length(mem_slice))))
                    # the copy above is necessary to ensure we don't end up inadvertently modifying
                    # list of offsets somewhere else

                    # that's the disk slice relative to having read the whole thing continuously.
                    # Offset to reflect what we've missed.

                    if isinstance(d_slice_post, slice):
                        d_slice_post = slice(
                            d_slice_post.start - len_pre, d_slice_post.stop - len_pre, d_slice_post.step)
                    else:
                        d_slice_post -= len_pre

                    len_m_post = pynbody.util.indexing_tricks.indexing_length(d_slice_post)
                    m_slice_post = pynbody.util.indexing_tricks.concatenate_indexing(
                        mem_slice, slice(len_m_pre, len_m_pre + len_m_post))

                    if pynbody.util.indexing_tricks.indexing_length(d_slice_post) == 0:
                        d_slice_post = None
                        m_slice_post = None

                    if pynbody.util.indexing_tricks.indexing_length(d_slice_pre) == 0:
                        d_slice_pre = None
                        m_slice_pre = None

                else:
                    d_slice_pre = None
                    m_slice_pre = None
                    len_post = nread_disk - len_pre
                    d_slice_post = None
                    m_slice_post = None

                yield len_pre, d_slice_pre, m_slice_pre
                fpos += len_pre
                disk_interrupt_fn(disk_interrupt_points[i])

                # prepare for next interrupt

                i += 1
                if len(disk_interrupt_points) > i:
                    next_dip = disk_interrupt_points[i]
                else:
                    next_dip = None

                # update 'input' to reflect post reading slice (may still need further
                # decomposition)
                nread_disk, disk_slice, mem_slice = len_post, d_slice_post, m_slice_post


            yield nread_disk, disk_slice, mem_slice
            fpos += nread_disk

    def iterate(self, families_on_disk: list[family.Family], families_in_memory: list[family.Family],
                multiskip: bool = False) -> Iterator[tuple[int, slice | None, slice | None]]:
        """Yields step-by-step instructions for partial-loading an array with the specified families.

        A typical read loop should be as follows:

        .. code-block:: python

            for readlen, buffer_index, memory_index in ctl.iterate(fams_on_disk, fams_in_mem) :
              data = read_entries(count=readlen)
              if memory_index is not None :
                target_array[memory_index] = data[buffer_index]

        Obviously this can be optimized, for instance to skip through
        file data when memory_index is None rather than read and discard it.

        Parameters
        ----------
        families_on_disk : list
            List of families for which the array exists on disk
        families_in_memory : list
            List of families for which we want to read the array into memory
        multiskip : bool
            If True, skip commands (i.e. entries with buffer_index=None)
            can have readlen greater than the block length

        Yields
        ------
        readlen : int
            Number of entries to read from disk
        buffer_index : slice | None
            Slice to read from the resulting buffer, or None if this particular read is to be ignored (skipped)
        memory_index : slice | None
            Slice to write into memory, or None if ``buffer_index`` is None

        """

        mem_offset = 0

        skip_accumulation = 0

        for current_family in self._ordered_families:
            if skip_accumulation > 0:
                yield skip_accumulation, None, None
                skip_accumulation = 0

            if current_family not in families_on_disk:
                assert current_family not in families_in_memory
            else:
                if current_family in families_in_memory:
                    for nread_disk, disk_mask, mem_slice in self._family_chunks[current_family]:
                        if mem_slice is None:
                            if multiskip:
                                skip_accumulation += nread_disk
                            else:
                                yield nread_disk, None, None
                        else:
                            if skip_accumulation > 0:
                                yield skip_accumulation, None, None
                                skip_accumulation = 0
                            mem_slice_offset = slice(
                                mem_slice.start + mem_offset, mem_slice.stop + mem_offset)
                            yield nread_disk, disk_mask, mem_slice_offset

                    mem_fs = self.mem_family_slice[current_family]
                    mem_offset += mem_fs.stop - mem_fs.start
                else:
                    for nread_disk, disk_mask, mem_slice in self._family_chunks[current_family]:
                        yield nread_disk, None, None
