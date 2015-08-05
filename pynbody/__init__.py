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

try:
    from . import plot
except:
    warnings.warn(
        "Unable to import plotting package (missing matplotlib or running from a text-only terminal? Plotting is disabled.", RuntimeWarning)

from .snapshot import new, load

configuration.configure_snapshot_and_halo_loading_priority()

derived_array = snapshot.SimSnap.derived_quantity

__all__ = ['load', 'new', 'derived_array']
