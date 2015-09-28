import threading


class DependencyError(RuntimeError):
    pass

class DependencyContext(object):
    def __init__(self, tracker, name):
        self.tracker = tracker
        self.name = name

    def __enter__(self):
        self.tracker._calculation_lock.__enter__()
        if self.name in self.tracker._current_calculation_stack:
            raise DependencyError, "Circular dependency"
        self.tracker._push(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker._pop(self.name)
        if exc_type is None:
            self.tracker._add_me_to_dependents(self.name)
        self.tracker._calculation_lock.__exit__(exc_type, exc_val, exc_tb)


class DependencyTracker(object):
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
        return DependencyContext(self, name)

    def touching(self,name):
        with self._calculation_lock:
            self._add_me_to_dependents(name)

    def get_dependents(self, name):
        return self._dependencies.get(name, set())
