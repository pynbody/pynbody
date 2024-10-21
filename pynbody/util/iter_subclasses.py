"""Provides a mix-in class for iterating over all subclasses of a given class, possibly in a user-specified order."""

from __future__ import annotations

from typing import Iterable


class IterableSubclasses:
    """A mixin for classes where we need to be able to iterate over their subclasses, possibly in a
    user-specified order. This is used by HaloCatalogue and SimSnap to find a suitable loader for
    a given file. """
    @classmethod
    def iter_subclasses(cls) -> Iterable[type]:
        """Iterate over all subclasses of this class, recursively.

        This is used by HaloCatalogue and SimSnap to find a suitable loader for a given file."""
        for c in cls.__subclasses__():
            yield from c.iter_subclasses()
            yield c

    @classmethod
    def iter_subclasses_with_priority(cls, priority: Iterable[str | type]) -> Iterable[type]:
        """Iterate over all subclasses, starting with the given priorities

        The priorities can be provided either as a string or a class"""
        all_subclasses = list(cls.iter_subclasses())
        all_subclasses_name = [s.__name__ for s in all_subclasses]
        for next_priority in priority:
            if isinstance(next_priority, type):
                if next_priority in all_subclasses:
                    index = all_subclasses.index(next_priority)
                    del all_subclasses[index]
                    del all_subclasses_name[index]
                yield next_priority
            else:
                if next_priority not in all_subclasses_name:
                    raise ValueError(f"Unknown class {next_priority}")
                next_priority_index = all_subclasses_name.index(next_priority)
                yield all_subclasses[next_priority_index]
                del all_subclasses[next_priority_index]
                del all_subclasses_name[next_priority_index]

        # now iterate the rest
        yield from all_subclasses
