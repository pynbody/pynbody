"""Useful plotting routines, using matplotlib to display pynbody's calculations

.. versionchanged :: 2.0

    The plotting routines have been simplified and pared back in version 2.0.

    Call signatures have been simplified, and keywords such as *filename*, *axes*, *clear*, *subplot*
    have been removed. Use the matplotlib functions directly to save the figure or modify the axes.

    The *metals* module has been removed. To plot metallicity-related quantities, use generic functions
    appropriately. Specific examples of [Fe/H] and [O/Fe] vs [Fe/H] plots have been provided in the
    quick-start tutorial, under the section :ref:`metals_histogram_tutorial`.
"""

import importlib

from . import gas, generic, profile, stars, util

importlib.reload(profile)
importlib.reload(generic)
importlib.reload(stars)
importlib.reload(gas)
importlib.reload(util)

from .gas import rho_T
from .generic import fourier_map, gauss_kde, hist2d
from .profile import density_profile, rotation_curve
from .sph import faceon_image, image, sideon_image
from .stars import sbprofile, schmidtlaw, sfh
