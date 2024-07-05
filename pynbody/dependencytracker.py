"""Tools for tracking how derived arrays depend on other arrays

Users should not need to use this module directly; it is used by :class:`~pynbody.snapshot.simsnap.SimSnap`
to track dependencies between derived arrays.
"""

import threading


class DependencyError(RuntimeError):
    pass

class _DependencyContext:
    """Context manager for tracking dependencies between arrays"""
    def __init__(self, tracker, name):
        self.tracker = tracker
        self.name = name

    def __enter__(self):
        self.tracker._calculation_lock.__enter__()
        if self.name in self.tracker._current_calculation_stack:
            self.tracker._calculation_lock.__exit__(None, None, None)
            raise DependencyError("Circular dependency")
        self.tracker._push(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker._pop(self.name)
        if exc_type is None:
            self.tracker._add_me_to_dependents(self.name)
        self.tracker._calculation_lock.__exit__(exc_type, exc_val, exc_tb)


class DependencyTracker:
    """Class for tracking dependencies between arrays.

    This class is used by :class:`pynbody.snapshot.simsnap.SimSnap` to track how derived arrays
    depend on other arrays. Users should not need to use this class directly.

    As an example of how this class is used, see the following code.

    >>> my_tracker = DependencyTracker()
    >>> with my_tracker.calculating('my_array'):
    >>>    with my_tracker.calculating('my_other_array'):
    >>>       my_tracker.touching('source_array')
    >>> my_tracker.get_dependents('my_array') # -> {}
    >>> my_tracker.get_dependents('my_other_array') # -> {'my_array'}
    >>> my_tracker.get_dependents('source_array') # -> {'my_other_array', 'my_array'}
    >>> with my_tracker.calculating('my_array'):
    >>>     with my_tracker.calculating('my_array'): # -> raises DependencyError
    >>>        pass

    Note that the class is thread-safe, in the sense that if a second thread starts trying to use it
    while a first thread is already mid-way through a derivation, the second thread blocks until
    the first thread finishes.
    """
    def __init__(self):
        self._dependencies = {}
        self._current_calculation_stack = []
        self._calculation_lock = threading.RLock()

    def _setup_my_dependencies(self,name):
        if name not in self._dependencies:
            self._dependencies[name] = set()

    def _add_me_to_dependents(self, name):
        self._setup_my_dependencies(name)
        for other in self._current_calculation_stack:
            self._dependencies[name].add(other)

    def _push(self, name):
        self._current_calculation_stack.append(name)

    def _pop(self, name):
        assert self._current_calculation_stack[-1]==name
        del self._current_calculation_stack[-1]


    def calculating(self, name):
        """Return a context manager when calculating a named array."""
        return _DependencyContext(self, name)

    def touching(self,name):
        """Note that any ongoing calculations depend on the named array."""
        with self._calculation_lock:
            self._add_me_to_dependents(name)

    def get_dependents(self, name):
        """Return the set of arrays that are known to depend on the named array."""
        return self._dependencies.get(name, set())
