"""Deprecated location of :mod:`pynbody.imaging.renderers`.

.. deprecated:: 2.9.0

  The rendering pipeline has moved to :mod:`pynbody.imaging.renderers`, since it now renders AMR simulations as well
  as SPH simulations. Attributes accessed through this module are those of the new module, with a deprecation
  warning.
"""

import warnings

from ..imaging import renderers as _renderers

__all__ = [name for name in dir(_renderers) if not name.startswith('_')]

def __getattr__(name):
    try:
        value = getattr(_renderers, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    warnings.warn(f"pynbody.sph.renderers is deprecated; use pynbody.imaging.renderers instead",
                  DeprecationWarning, stacklevel=2)
    return value
