"""The :class:`ProcessPool`, which runs functions in worker processes that share snapshots with the main process.

.. versionadded:: 2.9.0

"""

from __future__ import annotations

import concurrent.futures
import contextlib
import multiprocessing
import os
import signal
import sys
import warnings

from ..array import shared
from ..snapshot import SimSnap
from ..util import threads
from .channel import LocalChannel, LocalListener
from .served import ServedSimSnap
from .server import SnapshotServer
from .transfer import pack_objects, unpack_objects
from .transport import SharedMemoryTransport

# The objects that the present worker process has been given (see _initialise_worker), or None in a process
# that is not a worker
_worker_objects = None
# The snapshots (and families of them) that the present worker process has been given. Halo catalogues refer to
# their snapshots only weakly, so they are kept alive here, even if they were not themselves among the objects
# asked for
_worker_snapshots = None
# The worker's channel to the main process, if it is being served
_worker_channel = None
_worker_setup_error = None


class ProcessPool:
    """A pool of processes, each with a read-only copy of the snapshots and halo catalogues it is given.

    For example, to calculate a quantity for each halo in a catalogue using four processes:

    .. sourcecode:: python

        def stellar_mass(f, h, halo_number):
            return h[halo_number].st['mass'].sum()

        if __name__ == '__main__':
            f = pynbody.load('my_snapshot')
            h = f.halos()

            with pynbody.ProcessPool(f, h, processes=4) as pool:
                masses = pool.map(stellar_mass, h.keys())

    The objects given when creating the pool -- here ``f`` and ``h`` -- are passed to every call of the
    function, ahead of the item being worked on. In each process they are copies of the originals, made once when
    the process starts, whose arrays are in shared memory; so they take no further memory however many processes
    there are.

    * When a process needs an array that the snapshot has not loaded, it asks the main process, which loads it
      once, into shared memory, for every process to use. The same goes for derived arrays, and for the KD-tree
      (see :meth:`~pynbody.snapshot.simsnap.SimSnap.build_tree`). While the main process builds a tree or derives
      an array, using all the CPUs, the other processes are paused (except on Windows).
    * The copies are read-only: in particular, a copy cannot be centred or rotated (which would move it for every
      process). To work on a halo in its own frame, make a modifiable copy of it in your function using
      :meth:`~pynbody.snapshot.simsnap.SimSnap.get_copy_on_access_simsnap`. New arrays can be created in the
      copies; they belong to the process that makes them.
    * Each process has only what it is given, so the function should use only its arguments. While the pool is
      open, the originals should not be modified (e.g. by converting units or centring), since the processes
      share their arrays.
    * Halo catalogues hold every halo's particles and the properties provided by the halo finder, without
      referring back to its files (see :meth:`~pynbody.halo.HaloCatalogue.get_portable_state`).

    With ``serve=False``, the copies are instead frozen: they hold only what had been loaded or derived by the
    time the pool was created, and asking for any other array that could have been loaded raises a ``KeyError``
    (specifically a :class:`~pynbody.snapshot.portable.ArrayNotTransferredError`). The main process then does
    nothing while the pool is working.

    As well as snapshots (:class:`~pynbody.snapshot.simsnap.SimSnap`), halo catalogues
    (:class:`~pynbody.halo.HaloCatalogue`) and KD-trees (:class:`~pynbody.kdtree.KDTree`), the pool can be
    given numpy arrays, which are shared in the same way, and any other object that can be pickled, which each
    process receives a copy of.

    Arrays that are not already in shared memory when the pool is created are copied there, and the copies are
    freed when the pool is shut down. To avoid copying the arrays of a snapshot that you load before creating the
    pool, call :meth:`~pynbody.snapshot.simsnap.SimSnap.enable_shared_arrays` before loading them. (Arrays that the
    main process loads for the pool are put in shared memory to begin with.)

    .. versionadded:: 2.9.0

    """

    def __init__(self, *objects, processes: int | None = None, threads_per_process: int = 1,
                 start_method: str = 'spawn', serve: bool = True):
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

        serve : bool
            If True (default), the processes can ask the main process for arrays and KD-trees that their copies
            lack. If False, the copies hold only what had been loaded when the pool was created.
        """
        for obj in objects:
            if isinstance(obj, int) and not isinstance(obj, bool):
                warnings.warn(f"{obj} has been given to the pool as an object for its processes to share. If you "
                              f"meant it to be the number of processes, pass processes={obj} instead.",
                              UserWarning, stacklevel=2)
        self._context = multiprocessing.get_context(start_method)
        self._transport = SharedMemoryTransport()
        packed, snapshots = pack_objects(objects, self._transport)

        if serve:
            # The server's threads may create shared memory, but cannot register the handler that cleans it up
            shared._register_sigterm_handler()
            for sim in snapshots:
                sim.enable_shared_arrays()
            self._server = SnapshotServer(snapshots, self._transport, self._paused_workers)
            self._listener = LocalListener(self._server.handle)
            channel_address = (self._listener.address, self._listener.authkey)
        else:
            self._listener = None
            channel_address = None

        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=processes, mp_context=self._context, initializer=_initialise_worker,
            initargs=(packed, type(self._transport), channel_address, threads_per_process))

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
        if self._listener is not None:
            self._listener.close()
        self._transport.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown(cancel_pending=exc_type is not None)

    @contextlib.contextmanager
    def _paused_workers(self):
        """Stop the worker processes for the duration of a with block, so that the main process has all the CPUs.

        Nothing that the main process needs while they are stopped can be held by a worker: the server uses only
        its own lock and the snapshots in this process, and a worker part-way through a request finishes it later.
        Were this process to die with the workers stopped, they would be sent SIGHUP and SIGCONT, as members of an
        orphaned process group. Windows has no means of stopping a process, so there this does nothing."""
        if not hasattr(signal, 'SIGSTOP'):
            yield
            return
        pids = self._listener.worker_pids()
        for pid in pids:
            _signal_process(pid, signal.SIGSTOP)
        try:
            yield
        finally:
            for pid in pids:
                _signal_process(pid, signal.SIGCONT)

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


def _signal_process(pid, signum):
    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        pass  # the worker has exited


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


def _initialise_worker(packed, transport_class, channel_address, threads_per_process):
    global _worker_objects, _worker_snapshots, _worker_channel, _worker_setup_error
    # After a fork, this process must neither clean up its parent's shared memory, nor (by inheriting the
    # parent's handler) ignore the SIGTERM with which the pool may need to stop it
    shared._forget_inherited_shared_memory()
    # The limit is never lifted; it lasts as long as the process
    threads.thread_limit(threads_per_process).__enter__()
    try:
        transport = transport_class()
        if channel_address is None:
            def make_snapshot(state, view_key):
                return SimSnap.from_portable_state(state)
        else:
            _worker_channel = LocalChannel(*channel_address)

            def make_snapshot(state, view_key):
                return ServedSimSnap(state, _worker_channel, view_key, transport)

        _worker_objects, _worker_snapshots = unpack_objects(packed, transport, make_snapshot)
    except Exception as e:
        # Raising here would break the whole pool, with an uninformative message. Instead, the error is reported
        # back to the parent when the process is asked to do some work.
        _worker_setup_error = e
