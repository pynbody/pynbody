"""How the workers of a :class:`~pynbody.parallel.ProcessPool` send requests to the main process.

A worker calls :meth:`LocalChannel.request`, giving the name of a request and its arguments. In the main process,
a :class:`LocalListener` receives it and passes it to a handler -- normally
:meth:`SnapshotServer.handle <pynbody.parallel.server.SnapshotServer.handle>` -- whose return value, or exception,
is sent back. Neither side knows anything about snapshots; another transport (e.g. between machines) can replace
these classes without changing the handler.

These classes use :mod:`multiprocessing.connection`, so they work between processes on the same machine. The
listener runs a thread for each worker, so the handler must be thread-safe.

.. versionadded:: 2.9.0

"""

from __future__ import annotations

import os
import threading
from multiprocessing.connection import Client, Listener


class LocalListener:
    """Receives requests from workers in the main process, and replies with the result of a handler"""

    def __init__(self, handler):
        """Start listening.

        Parameters
        ----------

        handler : callable
            Called as ``handler(name, args)`` for each request; its return value is sent back to the worker, as
            is any exception it raises.
        """
        self._handler = handler
        self.authkey = os.urandom(32)
        self._listener = Listener(authkey=self.authkey)
        self._worker_pids = set()
        self._pids_lock = threading.Lock()
        self._closing = False
        self._accept_thread = threading.Thread(target=self._accept_connections, daemon=True,
                                               name="pynbody-parallel-listener")
        self._accept_thread.start()

    @property
    def address(self):
        """The address that workers pass to :class:`LocalChannel`"""
        return self._listener.address

    def worker_pids(self) -> list[int]:
        """The process ids of the workers that are currently connected"""
        with self._pids_lock:
            return list(self._worker_pids)

    def close(self):
        """Stop accepting connections. Connections already made end when their workers disconnect."""
        if self._closing:
            return
        self._closing = True
        # Wake the thread blocked in accept(), which then sees that the listener is closing
        try:
            Client(self.address, authkey=self.authkey).close()
        except OSError:
            pass
        self._accept_thread.join()
        self._listener.close()

    def _accept_connections(self):
        while True:
            try:
                connection = self._listener.accept()
            except OSError:
                # including a failed authentication, which is not a connection we want anyway
                if self._closing:
                    return
                continue
            if self._closing:
                connection.close()
                return
            threading.Thread(target=self._serve, args=(connection,), daemon=True,
                             name="pynbody-parallel-connection").start()

    def _serve(self, connection):
        try:
            pid = connection.recv()
        except (EOFError, OSError):
            connection.close()
            return

        with self._pids_lock:
            self._worker_pids.add(pid)
        try:
            while True:
                try:
                    name, args = connection.recv()
                except (EOFError, OSError):
                    return
                try:
                    reply = ('ok', self._handler(name, args))
                except Exception as e:
                    reply = ('error', e)
                try:
                    connection.send(reply)
                except Exception as e:
                    # The reply could not be pickled. Nothing has been sent, so the worker can be told why.
                    connection.send(('error', RuntimeError(f"The reply to {name!r} could not be sent: {e!r}")))
        finally:
            with self._pids_lock:
                self._worker_pids.discard(pid)
            connection.close()


class LocalChannel:
    """Sends requests from a worker to a :class:`LocalListener` in the main process"""

    def __init__(self, address, authkey):
        self._connection = Client(address, authkey=authkey)
        self._connection.send(os.getpid())
        # A worker may itself use several threads, which must not interleave their requests and replies
        self._lock = threading.Lock()

    def request(self, name: str, *args):
        """Ask the main process to carry out the request *name*, and return its reply.

        If handling the request raised an exception in the main process, it is raised here."""
        with self._lock:
            self._connection.send((name, args))
            status, value = self._connection.recv()
        if status == 'error':
            raise value
        return value
