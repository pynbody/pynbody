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
import pickle
import threading
import traceback
from multiprocessing.connection import Client, Listener, answer_challenge, deliver_challenge


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
        # Authentication is done by the thread serving each connection (see _serve), rather than in accept(),
        # so that a connection which fails it, or never completes it, cannot stop others being accepted
        self._listener = Listener()
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
            Client(self.address).close()
        except OSError:
            pass
        self._accept_thread.join(timeout=10)
        self._listener.close()

    def _accept_connections(self):
        while True:
            try:
                connection = self._listener.accept()
            except Exception:
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
            # The same handshake as a Listener with an authkey would make in accept()
            deliver_challenge(connection, self.authkey)
            answer_challenge(connection, self.authkey)
            pid = connection.recv()
        except Exception:
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
                    reply = ('error', _transferable(e), traceback.format_exc())
                try:
                    connection.send(reply)
                except Exception as e:
                    # The reply could not be pickled. Nothing has been sent, so the worker can be told why.
                    try:
                        connection.send(('error', RuntimeError(f"The reply to {name!r} could not be sent: {e!r}"),
                                         traceback.format_exc()))
                    except OSError:
                        return
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
            status, value, *details = self._connection.recv()
        if status == 'error':
            value.__cause__ = RemoteTraceback(details[0])
            raise value
        return value


class RemoteTraceback(Exception):
    """Carries the traceback of an exception raised in the main process, as the cause of the exception raised in
    the worker, in the same way as :mod:`concurrent.futures` does for exceptions raised in workers"""

    def __str__(self):
        return f"\n\nIn the main process:\n{self.args[0]}"


def _transferable(exception):
    """Return the exception if it can be sent to another process; otherwise, a RuntimeError describing it.

    Some exceptions can be pickled but not unpickled, e.g. if their initialiser takes arguments that they do not
    pass on to Exception; so the whole round trip is tried."""
    try:
        pickle.loads(pickle.dumps(exception))
        return exception
    except Exception:
        return RuntimeError(f"{type(exception).__name__}: {exception}")
