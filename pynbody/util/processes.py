"""Running an analysis in several processes, each with a read-only copy of snapshots and halo catalogues.

:class:`ProcessPool` gives each of its worker processes its own copy of the snapshots, halo catalogues and
KD-trees that it is given, whose arrays live in shared memory so that they are not duplicated. See
:ref:`portable_parallelism` for an introduction.

.. versionadded:: 2.9.0

"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import sys
import warnings

import numpy as np

from ..array import SimArray, shared
from ..halo.portable import map_arrays
from . import threads

#: Arrays smaller than this number of bytes are sent to each worker process directly, rather than through shared
#: memory, unless they are already in shared memory. Every shared array is a separate block of shared memory,
#: and there is little to gain from sharing small ones.
MIN_SHARED_BYTES = 1 << 16

# The objects that the present worker process has been given (see _initialise_worker), or None in a process
# that is not a worker
_worker_objects = None
# The snapshots (and families of them) that the present worker process has been given. Halo catalogues refer to
# their snapshots only weakly, so they are kept alive here, even if they were not themselves among the objects
# asked for
_worker_snapshots = None
_worker_setup_error = None


class ProcessPool:
    """A pool of processes, each of which has a read-only copy of the snapshots and halo catalogues it is given.

    For example, to calculate a quantity for each halo in a catalogue using four processes:

    .. sourcecode:: python

        def stellar_mass(f, h, halo_number):
            return h[halo_number].st['mass'].sum()

        if __name__ == '__main__':
            f = pynbody.load('my_snapshot')
            f.enable_shared_arrays()
            f.st['mass']
            h = f.halos()

            with pynbody.ProcessPool(f, h, processes=4) as pool:
                masses = pool.map(stellar_mass, h.keys())

    The objects given when creating the pool -- here ``f`` and ``h`` -- are passed to every call of the
    function, ahead of the item being worked on. In each process they are copies of the originals, made once
    when the process starts. Their arrays are in shared memory, so they take no further memory however many
    processes there are.

    The copies are frozen, holding only what had been loaded by the time the pool was created:

    * The arrays that had been loaded or derived are available, but are read-only: in particular, a copy cannot
      be centred or rotated (which would move it for every process). To work on a halo in its own frame, make a
      modifiable copy of it in your function using
      :meth:`~pynbody.snapshot.simsnap.SimSnap.get_copy_on_access_simsnap`.
    * Arrays that had not been loaded cannot be loaded, and asking for one raises a ``KeyError``
      (specifically a :class:`~pynbody.snapshot.portable.ArrayNotTransferredError`).
    * Arrays can still be derived, and new arrays created; these belong to the process that makes them.
    * Each process has only what it is given, so the function should use only its arguments. While the pool is
      open, the originals should not be modified (e.g. by converting units or centring), since the processes
      share their arrays.
    * Halo catalogues hold every halo's particles and the properties provided by the halo finder, without
      referring back to its files (see :meth:`~pynbody.halo.HaloCatalogue.get_portable_state`).
    * A KD-tree built with :meth:`~pynbody.snapshot.simsnap.SimSnap.build_tree` comes with its snapshot, and is
      used as normal (e.g. by :class:`~pynbody.filt.Sphere`) without being rebuilt.

    As well as snapshots (:class:`~pynbody.snapshot.simsnap.SimSnap`), halo catalogues
    (:class:`~pynbody.halo.HaloCatalogue`) and KD-trees (:class:`~pynbody.kdtree.KDTree`), the pool can be
    given numpy arrays, which are shared in the same way, and any other object that can be pickled, which each
    process receives a copy of.

    Arrays that are not already in shared memory are copied there when the pool is created, and the copy is
    freed when the pool is shut down. To avoid copying the arrays of a snapshot, call
    :meth:`~pynbody.snapshot.simsnap.SimSnap.enable_shared_arrays` before loading them.

    .. versionadded:: 2.9.0

    """

    def __init__(self, *objects, processes: int | None = None, threads_per_process: int = 1,
                 start_method: str = 'spawn'):
        """Prepare the objects to be shared, and create the pool.

        Parameters
        ----------

        *objects :
            The snapshots, halo catalogues and other objects that each process is to have a copy of. These are
            passed, in the same order, as the first arguments of every function run by the pool. A snapshot
            must be a whole snapshot rather than a view of one such as ``f.dm``. A halo catalogue brings its
            snapshot with it, which need not be passed separately (but if it is, the catalogue in each process
            refers to that process's copy of it).

        processes : int, optional
            The number of processes. If None, there is one for each CPU available to this process.

        threads_per_process : int
            The number of threads that each pynbody operation in a process may use (see
            :class:`~pynbody.util.threads.thread_limit`). The default, 1, is appropriate when there are as many
            processes as CPUs.

        start_method : str
            How to start the processes: ``'spawn'`` (default), ``'forkserver'`` or ``'fork'`` (see the
            documentation of python's :mod:`multiprocessing`). The default is the same on every platform, and
            starts each process afresh, so that it can only use what it is given. With ``'fork'``, which is not
            available on Windows, each process starts as a copy of this one, so a function defined interactively
            can be run, provided it was defined before the pool was created.
        """
        for obj in objects:
            if isinstance(obj, int) and not isinstance(obj, bool):
                warnings.warn(f"{obj} has been given to the pool as an object for its processes to share. If you "
                              f"meant it to be the number of processes, pass processes={obj} instead.",
                              UserWarning, stacklevel=2)
        self._context = multiprocessing.get_context(start_method)
        self._packer = _SharedMemoryPacker()
        packed = _pack(objects, self._packer)
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=processes, mp_context=self._context,
            initializer=_initialise_worker, initargs=(packed, threads_per_process))

    def submit(self, fn, /, *args, **kwargs) -> concurrent.futures.Future:
        """Run ``fn(*objects, *args, **kwargs)`` in one of the processes, returning a
        :class:`~concurrent.futures.Future` for the result.

        Here *objects* are the copies of the objects given when the pool was created."""
        self._check_function(fn)
        return self._executor.submit(_run, fn, args, kwargs)

    def map(self, fn, *iterables, chunksize: int = 1) -> list:
        """Run ``fn(*objects, item)`` for each item of an iterable, in parallel, and return a list of the results.

        Here *objects* are the copies of the objects given when the pool was created. As for python's built-in
        :func:`map`, if more than one iterable is given, the function is called with an item from each.

        Parameters
        ----------

        fn : callable
            The function to run. It must be defined at the top level of a module or script, so that the
            processes can find it.

        *iterables :
            The items to work on, for example the numbers of the halos to analyse.

        chunksize : int
            The number of items to send to a process at once. Increasing this reduces the overhead of
            communicating with the processes, which can be worthwhile if there are many items that are each
            processed quickly.
        """
        self._check_function(fn)
        return list(self._executor.map(_RunWithObjects(fn), *iterables, chunksize=chunksize))

    def shutdown(self, cancel_pending: bool = False):
        """Wait for the processes to finish their work, then stop them and free any shared memory made for them.

        This is called automatically at the end of a ``with`` block. If the block ends because of an exception
        (including a keyboard interrupt), work that has not yet started is cancelled.

        Parameters
        ----------

        cancel_pending : bool
            If True, work that has been submitted but not yet started is cancelled, rather than waited for.
        """
        self._executor.shutdown(wait=True, cancel_futures=cancel_pending)
        self._packer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown(cancel_pending=exc_type is not None)

    def _check_function(self, fn):
        """Give a clear error for a function that the worker processes will not be able to find.

        Without this, the failure would be an obscure one in the worker, which with some start methods also
        breaks the pool."""
        main = sys.modules.get('__main__')
        if (self._context.get_start_method() != 'fork' and getattr(fn, '__module__', None) == '__main__'
                and not hasattr(main, '__file__')):
            raise ValueError(
                f"The function {getattr(fn, '__name__', fn)!r} was defined interactively, so processes started by "
                f"the {self._context.get_start_method()!r} method cannot find it. Define it in a python file "
                f"(either a script, or a module you import) instead; or, except on Windows, create the pool with "
                f"start_method='fork' after defining the function.")


class _RunWithObjects:
    """Calls a function with the worker's objects ahead of its arguments. (A class rather than a closure, so that
    it can be pickled.)"""
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return _run(self.fn, args, {})


def _run(fn, args, kwargs):
    if _worker_setup_error is not None:
        raise RuntimeError(f"Setting up this worker process failed: {_worker_setup_error!r}") \
            from _worker_setup_error
    return fn(*_worker_objects, *args, **kwargs)


def _initialise_worker(packed, threads_per_process):
    global _worker_objects, _worker_snapshots, _worker_setup_error
    # After a fork, this process must neither clean up its parent's shared memory, nor (by inheriting the
    # parent's handler) ignore the SIGTERM with which the pool may need to stop it
    shared._forget_inherited_shared_memory()
    # The limit is never lifted; it lasts as long as the process
    threads.thread_limit(threads_per_process).__enter__()
    try:
        _worker_objects, _worker_snapshots = _unpack(packed)
    except Exception as e:
        # Raising here would break the whole pool, with an uninformative message. Instead, the error is reported
        # back to the parent when the process is asked to do some work.
        _worker_setup_error = e


class _SharedMemoryPacker:
    """Replaces arrays with references to shared memory, copying them there if necessary."""

    def __init__(self):
        self.copies = []
        self.bytes_copied = 0
        # Maps the memory an array occupies to (array, reference). The array is kept so that its memory cannot be
        # freed and reused by a different array, which would then be given the same reference
        self._references = {}

    def __call__(self, ar):
        key = (ar.__array_interface__['data'][0], ar.shape, ar.strides, ar.dtype.str)
        if key in self._references:
            return self._references[key][1]

        reference = None
        if isinstance(ar, SimArray):
            try:
                reference = shared.to_shared_reference(ar)
            except TypeError:
                pass  # not in shared memory

        if reference is None:
            if ar.nbytes < MIN_SHARED_BYTES or ar.dtype.hasobject:
                reference = np.asarray(ar)
            else:
                copy = shared.make_shared_array(ar.shape, ar.dtype)
                copy[...] = ar
                self.copies.append(copy)
                self.bytes_copied += ar.nbytes
                reference = shared.to_shared_reference(copy)

        self._references[key] = (ar, reference)
        return reference

    def pack(self, state):
        return map_arrays(state, self)

    def release(self):
        self._references = {}
        self.copies = []


def _pack(objects, packer: _SharedMemoryPacker):
    from ..halo import HaloCatalogue
    from ..kdtree import KDTree
    from ..snapshot import SimSnap
    from ..snapshot.subsnap import FamilySubSnap

    snapshots = []

    def catalogue_base(h):
        """Return the index of the snapshot a catalogue belongs to, and the name of its family if it is of one"""
        base = h.base
        if isinstance(base, FamilySubSnap):
            return snapshot_index(base.ancestor), base._unifamily.name
        elif base is not base.ancestor:
            raise TypeError(f"{h!r} belongs to {base!r}, which is part of a snapshot. A halo catalogue can only be "
                            f"shared with a ProcessPool if it belongs to a whole snapshot, or to a family.")
        return snapshot_index(base), None

    def snapshot_index(sim):
        if sim is not sim.ancestor:
            raise TypeError(f"{sim!r} is part of a snapshot. Only whole snapshots can be shared with a "
                            f"ProcessPool; share the whole snapshot, and select the part you need in your function.")
        for i, s in enumerate(snapshots):
            if s is sim:
                return i
        snapshots.append(sim)
        return len(snapshots) - 1

    # Snapshots are gathered first, so that a tree or catalogue belonging to one of them can refer to it,
    # whatever the order in which they were given
    for obj in objects:
        if isinstance(obj, SimSnap):
            snapshot_index(obj)
        elif isinstance(obj, HaloCatalogue):
            catalogue_base(obj)

    descriptions = []
    for obj in objects:
        if isinstance(obj, SimSnap):
            descriptions.append(('snapshot', snapshot_index(obj)))
        elif isinstance(obj, HaloCatalogue):
            descriptions.append(('halos', *catalogue_base(obj), packer.pack(obj.get_portable_state())))
        elif isinstance(obj, KDTree):
            owner = [i for i, s in enumerate(snapshots) if getattr(s, 'kdtree', None) is obj]
            if owner:
                descriptions.append(('tree_of_snapshot', owner[0]))
            else:
                descriptions.append(('tree', packer.pack(obj.get_portable_state())))
        elif isinstance(obj, np.ndarray):
            descriptions.append(('array', packer(obj)))
        else:
            descriptions.append(('object', obj))

    snapshot_states = []
    for sim in snapshots:
        bytes_before = packer.bytes_copied
        snapshot_states.append(packer.pack(sim.get_portable_state()))
        copied = packer.bytes_copied - bytes_before
        if copied > 0:
            warnings.warn(f"Arrays of {sim!r} have been copied into shared memory to share them between "
                          f"processes, using an extra {copied / 2**20:.1f} MB until the pool is shut down. To avoid "
                          f"this, call enable_shared_arrays() on the snapshot before loading arrays or building "
                          f"its tree.", RuntimeWarning, stacklevel=3)

    return snapshot_states, descriptions


def _unpack(packed):
    from .. import family
    from ..halo import HaloCatalogue
    from ..kdtree import KDTree
    from ..snapshot import SimSnap

    def unpack_array(x):
        if isinstance(x, shared.SharedArrayReference):
            x = shared.from_shared_reference(x)
        x.flags['WRITEABLE'] = False
        return x

    def unpack_state(state):
        return map_arrays(state, unpack_array, types=(shared.SharedArrayReference, np.ndarray))

    snapshot_states, descriptions = packed
    snapshots = [SimSnap.from_portable_state(unpack_state(state)) for state in snapshot_states]

    objects = []
    keep_alive = list(snapshots)
    for kind, *details in descriptions:
        if kind == 'snapshot':
            objects.append(snapshots[details[0]])
        elif kind == 'halos':
            snapshot_number, family_name, state = details
            base = snapshots[snapshot_number]
            if family_name is not None:
                base = base[family.get_family(family_name)]
                keep_alive.append(base)
            objects.append(HaloCatalogue.from_portable_state(unpack_state(state), base))
        elif kind == 'tree_of_snapshot':
            objects.append(snapshots[details[0]].kdtree)
        elif kind == 'tree':
            objects.append(KDTree.from_portable_state(unpack_state(details[0])))
        elif kind == 'array':
            objects.append(unpack_array(details[0]))
        else:
            objects.append(details[0])

    return tuple(objects), keep_alive
