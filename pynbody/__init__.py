"""
N-body and hydro analysis tools for Python

For more information, see https://pynbody.readthedocs.io/

"""

import importlib
import sys

# We need to import configuration first, so prevent isort from reordering
# isort: off
from .configuration import config, logger, config_parser
# isort: on
from . import (
    analysis,
    array,
    bridge,
    configuration,
    derived,
    family,
    filt,
    gravity,
    halo,
    snapshot,
    sph,
    transformation,
    util,
)


# The PlotModuleProxy serves to delay import of pynbody.plot until it's accessed.
# Importing pynbody.plot imports pylab which in turn is quite slow and cause problem
# for terminal-only applications. However, since pynbody always auto-imported
# pynbody.plot, it would seem to be too destructive to stop this behaviour.
# So this hack is the compromise and should be completely transparent to most
# users.
class PlotModuleProxy:
    def _do_import(self):
        # import_module goes through the import system even while this proxy is still bound to
        # pynbody.plot, and so is safe for several threads to call at once. (Deleting the proxy
        # first, then using "from . import plot", is not: one thread could delete the module that
        # another thread's import had just put in place.)
        global plot
        plot = importlib.import_module(".plot", __name__)

    def __hasattr__(self, key):
        self._do_import()
        return hasattr(plot, key)

    def __getattr__(self, key):
        self._do_import()
        global plot
        return getattr(plot, key)

    def __setattr__(self, key, value):
        self._do_import()
        global plot
        return setattr(plot, key, value)

    def __dir__(self):
        self._do_import()
        global plot
        return dir(plot)

    def __repr__(self):
        return "<Unloaded plot module>"

plot = PlotModuleProxy()

from .snapshot import load, new
from .util.threads import thread_limit

derived_array = snapshot.simsnap.SimSnap.derived_array

__version__ = '2.8.0'

__all__ = ['load', 'new', 'derived_array']
