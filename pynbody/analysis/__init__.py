"""Tools for scientific analysis with pynbody

This sub-package contains a number of modules that enable scientific analysis beyond the basic capabilities of pynbody.
The most essential tools are imported into the sub-package itself, so that they can be accessed directly from
pynbody.analysis. For example, :meth:`pynbody.analysis.halo.center` is accessible as :meth:`pynbody.analysis.center`,
while :meth:`pynbody.analysis.angmom.faceon` is accessible as :meth:`pynbody.analysis.faceon`.

"""

from . import (
    cosmology,
    halo,
    hifrac,
    hmf,
    interpolate,
    ionfrac,
    luminosity,
    pkdgrav_cosmo,
    profile,
    ramses_util,
    theoretical_profiles,
)
from .angmom import faceon, sideon
from .halo import center
from .hmf import halo_mass_function
from .profile import Profile
