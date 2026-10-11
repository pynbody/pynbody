"""Alias for :mod:`pynbody.plot.imaging`, which was previously named ``pynbody.plot.sph``.

.. versionchanged:: 2.9.0

  The plotting routines for images have moved to :mod:`pynbody.plot.imaging`, since they render AMR simulations as
  well as SPH simulations. This module remains available as an alias, so that existing code continues to work;
  attributes accessed through it are those of :mod:`pynbody.plot.imaging`.
"""

from . import imaging as _imaging

__all__ = [name for name in dir(_imaging) if not name.startswith('_')]

def __getattr__(name):
    try:
        return getattr(_imaging, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

def __dir__():
    return __all__
