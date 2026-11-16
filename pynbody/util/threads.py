"""Limiting the threads that pynbody uses for computation; see :class:`thread_limit`.

Code that runs threads of its own for computation asks :func:`compute_threads` how many to use, each time it does so.

.. versionadded:: 2.9.0

"""

from __future__ import annotations

import os
import threading

from ..configuration import config

_lock = threading.Lock()


def _reset_lock_after_fork():
    # A process forked while another thread held the lock (e.g. one serving a pynbody.ProcessPool) would otherwise
    # inherit it locked, with no thread to release it
    global _lock
    _lock = threading.Lock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_reset_lock_after_fork)

_active_limits = []
# One entry for each thread_limit block that any thread is currently inside

_threadpoolctl_limit = None
# The limit set through threadpoolctl (see _register_with_threadpoolctl), or None


class thread_limit:
    """Limit the number of threads that each pynbody operation uses for computation, inside a with block.

    This is intended for when you parallelise your own analysis across threads -- particularly on
    :ref:`free-threaded Python <free_threading>` -- and so do not also want pynbody to run threads of its own. For
    example:

    .. code-block:: python

        with pynbody.thread_limit():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                results = list(pool.map(analyse_halo, halos))

    Inside the block, each pynbody operation (such as building a KD-tree, SPH smoothing, rendering an image or
    decoding data read from a file) uses at most ``n`` threads for computation. Note that:

    * The limit applies to every thread while any thread is inside the block, so that it also applies to threads
      started inside it (such as those of the pool above). If blocks with different limits are entered at once,
      the smallest applies.

    * It limits each operation, not pynbody as a whole: operations running at the same time in different threads
      may each use ``n`` threads.

    * Threads that only read files (see :ref:`parallel_io`) are not limited, since they spend their time waiting for
      the files; but under a limit they leave decoding what they read to at most ``n`` other threads. To limit the
      number of reading threads too, use the ``io-threads`` options.

    * An operation already running when the block is entered keeps the threads it started with.

    * The limit is applied on top of the ``number_of_threads`` configuration option, and of any number of threads
      passed explicitly to a pynbody routine; it never increases them.

    If `threadpoolctl <https://github.com/joblib/threadpoolctl>`_ is installed, ``threadpoolctl.threadpool_limits``
    also limits pynbody in the same way, under the API name ``"pynbody"``.

    .. versionadded:: 2.9.0

    Parameters
    ----------
    n : int, optional
        The most threads each operation may use for computation. The default is 1, meaning that pynbody does its
        computation in the thread that asks for it.
    """

    def __init__(self, n: int = 1):
        if int(n) != n or n < 1:
            raise ValueError(f"The thread limit must be a whole number of at least 1, not {n!r}")
        self._n = int(n)

    @property
    def n(self) -> int:
        """The most threads each operation may use for computation, inside this block"""
        return self._n

    def __enter__(self):
        with _lock:
            _active_limits.append(self._n)
        return self

    def __exit__(self, *excp):
        with _lock:
            _active_limits.remove(self._n)

    def __repr__(self):
        return f"<thread_limit {self._n}>"


def current_limit() -> int | None:
    """The most threads each pynbody operation may use for computation now, or None if there is no limit"""
    with _lock:
        limits = list(_active_limits)
        if _threadpoolctl_limit is not None:
            limits.append(_threadpoolctl_limit)
    return min(limits) if limits else None


def compute_threads(requested: int | None = None) -> int:
    """The number of threads that an operation should use for computation.

    Parameters
    ----------
    requested : int, optional
        The number of threads asked for, if explicitly; otherwise, the ``number_of_threads`` configuration option.

    Returns
    -------
    int
        The number requested, reduced to the current :class:`thread_limit` if there is one, and at least 1.
    """
    if requested is None:
        requested = config['number_of_threads']
    requested = max(int(requested), 1)
    limit = current_limit()
    return requested if limit is None else min(requested, limit)


def _register_with_threadpoolctl():
    """Let ``threadpoolctl.threadpool_limits`` limit pynbody's threads, if threadpoolctl is installed"""
    try:
        import threadpoolctl
    except ImportError:
        return
    register = getattr(threadpoolctl, 'register', None)
    if register is None:
        return  # (threadpoolctl too old to accept other libraries' controllers)

    class PynbodyController(threadpoolctl.LibController):
        """threadpoolctl's view of the threads that pynbody uses for computation (see :class:`thread_limit`)"""

        user_api = "pynbody"
        internal_api = "pynbody"
        # threadpoolctl finds a library by the files loaded into the process; pynbody's KD-tree module is always
        # loaded along with pynbody
        filename_prefixes = ("kdmain",)
        check_symbols = ("PyInit_kdmain",)

        def get_num_threads(self):
            with _lock:
                limit = _threadpoolctl_limit
            # (not the effective number, which includes any thread_limit, since threadpoolctl sets this back on exit)
            return config['number_of_threads'] if limit is None else limit

        def set_num_threads(self, num_threads):
            global _threadpoolctl_limit
            # threadpoolctl restores the value it found on entering threadpool_limits by setting it again, so a
            # value no smaller than the configured number of threads must mean no limit, rather than become one
            if num_threads is None or num_threads >= config['number_of_threads']:
                num_threads = None
            else:
                num_threads = max(int(num_threads), 1)
            with _lock:
                _threadpoolctl_limit = num_threads

        def get_version(self):
            from .. import __version__
            return __version__

    register(PynbodyController)


_register_with_threadpoolctl()
