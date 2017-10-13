"""
pynbody
=======

A light-weight, portable, format-transparent analysis framework
for N-body and SPH astrophysical simulations.

For more information, either build the latest documentation included
in our git repository, or view the online version here:
http://pynbody.github.io/pynbody/

"""

import imp

from . import backcompat
from . import configuration

from .configuration import config, config_parser, logger

from . import util, filt, array, family, snapshot
from .snapshot import tipsy, gadget, gadgethdf, ramses, grafic, nchilada, ascii
from . import analysis, halo, derived, bridge, gravity, sph, transformation


# The PlotModuleProxy serves to delay import of pynbody.plot until it's accessed.
# Importing pynbody.plot imports pylab which in turn is quite slow and cause problem
# for terminal-only applications. However, since pynbody always auto-imported
# pynbody.plot, it would seem to be too destructive to stop this behaviour.
# So this hack is the compromise and should be completely transparent to most
# users.
class PlotModuleProxy(object):
    def _do_import(self):
        global plot
        del plot
        from . import plot

    def __hasattr__(self, key):
        self._do_import()
        return hasattr(plot, key)

    def __getattr__(self, key):
        self._do_import()
        return getattr(plot, key)

    def __setattr__(self, key, value):
        self._do_import()
        return setattr(plot, key, value)

    def __repr__(self):
        return "<Unloaded plot module>"

plot = PlotModuleProxy()

from .snapshot import new, load

configuration.configure_snapshot_and_halo_loading_priority()

derived_array = snapshot.SimSnap.derived_quantity

__all__ = ['load', 'new', 'derived_array']
