"""
Functions that derive arrays (e.g. radius) from others (e.g. position)

Users do not need to call these functions directly. They are called automatically when a
derived array is requested from any :class:`~pynbody.snapshot.simsnap.SimSnap`.

.. seealso::
  Not all derived arrays that are provided by the pynbody framework are defined in this
  module. In particular, see :class:`~pynbody.analysis.luminosity` for arrays related to
  stellar luminosity.

  For more information about how the derived array system operates, see :ref:`derived`.

"""

import logging
import time
import warnings

import numpy as np

from . import array, config, units
from .dependencytracker import DependencyError
from .snapshot import SimSnap

logger = logging.getLogger('pynbody.derived')


@SimSnap.derived_array
def r(self):
    """Radial position"""
    return ((self['pos'] ** 2).sum(axis=1)) ** (1, 2)


@SimSnap.derived_array
def rxy(self):
    """Cylindrical radius in the x-y plane"""
    return ((self['pos'][:, 0:2] ** 2).sum(axis=1)) ** (1, 2)


@SimSnap.derived_array
def vr(self):
    """Radial velocity"""
    return (self['pos'] * self['vel']).sum(axis=1) / self['r']


@SimSnap.derived_array
def v2(self):
    """Squared velocity"""
    return (self['vel'] ** 2).sum(axis=1)


@SimSnap.derived_array
def vt(self):
    """Tangential velocity"""
    return np.sqrt(self['v2'] - self['vr'] ** 2)


@SimSnap.derived_array
def ke(self):
    """Specific kinetic energy"""
    return 0.5 * (self['vel'] ** 2).sum(axis=1)


@SimSnap.derived_array
def te(self):
    """Specific total energy"""
    return self['ke'] + self['phi']


@SimSnap.derived_array
def j(self):
    """Specific angular momentum"""
    angmom = np.cross(self['pos'], self['vel']).view(array.SimArray)
    angmom.units = self['pos'].units * self['vel'].units
    return angmom


@SimSnap.derived_array
def j2(self):
    """Square of the specific angular momentum"""
    return (self['j'] ** 2).sum(axis=1)


@SimSnap.derived_array
def jz(self):
    """z-component of the angular momentum"""
    return self['j'][:, 2]


@SimSnap.derived_array
def vrxy(self):
    """Cylindrical radial velocity in the x-y plane"""
    return (self['pos'][:, 0:2] * self['vel'][:, 0:2]).sum(axis=1) / self['rxy']


@SimSnap.derived_array
def vcxy(self):
    """Cylindrical tangential velocity in the x-y plane"""
    f = (self['x'] * self['vy'] - self['y'] * self['vx']) / self['rxy']
    f[np.where(f != f)] = 0
    return f


@SimSnap.derived_array
def vphi(self):
    """Azimuthal velocity (synonym for vcxy)"""
    return self['vcxy']


@SimSnap.derived_array
def vtheta(self):
    """Velocity projected to polar direction"""
    return (np.cos(self['az']) * np.cos(self['theta']) * self['vx'] +
            np.sin(self['az']) * np.cos(self['theta']) * self['vy'] -
            np.sin(self['theta']) * self['vz'])


_op_dict = {"mean": "mean velocity",
            "disp": "velocity dispersion",
            "curl": "velocity curl",
            "div": "velocity divergence",
            }


def _v_sph_operation(self, op):
    """SPH-smoothed velocity operations"""
    self.build_tree()

    nsmooth = config['sph']['smooth-particles']

    logger.info('Calculating %s with %d nearest neighbours' % (_op_dict[op], nsmooth))

    if op in ['mean', 'curl']:
        sm = array.SimArray(np.empty_like(self['vel']), self['vel'].units)
    else:
        sm = array.SimArray(np.empty(len(self['vel']), dtype=self['vel'].dtype), self['vel'].units)

    if op in ['div', 'curl']:
        sm.units /= self['pos'].units

    self.kdtree.set_array_ref('rho', self['rho'])
    self.kdtree.set_array_ref('smooth', self['smooth'])
    self.kdtree.set_array_ref('mass', self['mass'])
    self.kdtree.set_array_ref('qty', self['vel'])
    self.kdtree.set_array_ref('qty_sm', sm)

    start = time.time()
    self.kdtree.populate('qty_%s' % op, nsmooth)
    end = time.time()

    logger.info(f'{_op_dict[op]} done in {end - start:5.3g} s')

    return sm


@SimSnap.derived_array
def v_mean(self):
    """SPH-smoothed mean velocity"""
    return _v_sph_operation(self, "mean")

@SimSnap.derived_array
def v_disp(self):
    """SPH-smoothed velocity dispersion"""
    return _v_sph_operation(self, "disp")

@SimSnap.derived_array
def v_curl(self):
    """SPH-smoothed curl of velocity"""
    return _v_sph_operation(self, "curl")

@SimSnap.derived_array
def vorticity(self):
    """SPH-smoothed vorticity"""
    return _v_sph_operation(self, "curl")

@SimSnap.derived_array
def v_div(self):
    """SPH-smoothed divergence of velocity"""
    return _v_sph_operation(self, "div")

@SimSnap.derived_array
def age(self):
    """Stellar age determined from formation time and current snapshot time"""
    return self.properties['time'].in_units(self['tform'].units, **self.conversion_context()) - self['tform']


@SimSnap.derived_array
def theta(self):
    """Angle from the z axis, from [0:pi]"""
    return np.arccos(self['z'] / self['r'])


@SimSnap.derived_array
def alt(self):
    """Angle from the horizon, from [-pi/2:pi/2]"""
    return np.pi / 2 - self['theta']


@SimSnap.derived_array
def az(self):
    """Angle in the xy plane from the x axis, from [-pi:pi]"""
    return np.arctan2(self['y'], self['x'])


@SimSnap.derived_array
def cs(self):
    """Sound speed"""
    return np.sqrt(5.0 / 3.0 * units.k * self['temp'] / self['mu'] / units.m_p)



@SimSnap.derived_array
def mu(sim, t0=None, Y=0.245):
    """Mean molecular mass, i.e. the mean atomic mass per particle. Assumes primordial abundances."""
    try:
        x = _mu_from_electron_frac(sim, Y)
    except (KeyError, DependencyError):
        try:
            x = _mu_from_HI_HeI_HeII_HeIII(sim)
        except KeyError:
            x = _mu_from_temperature_threshold(sim, Y, t0)

    x.units = units.Unit("1")
    return x


def _mu_from_temperature_threshold(sim, Y, t0):
    warnings.warn("No ionization fractions found, assuming fully ionised gas above 10^4 and neutral below 10^4K. "
                  "This is a very crude approximation.")
    x = np.empty(len(sim)).view(array.SimArray)
    if t0 is None:
        t0 = sim['temp']
    x[np.where(t0 >= 1e4)[0]] = 4. / (8 - 5 * Y)
    x[np.where(t0 < 1e4)[0]] = 4. / (4 - 3 * Y)
    return x


def _mu_from_HI_HeI_HeII_HeIII(sim):
    x = sim["HI"] + 2 * sim["HII"] + sim["HeI"] + \
        2 * sim["HeII"] + 3 * sim["HeIII"]
    x = x ** -1
    return x

def _mu_from_electron_frac(sim, Y):
    return 4./(4.-3.*Y+4*(1.-Y)*sim['ElectronAbundance'])



@SimSnap.derived_array
def p(sim):
    """Pressure"""
    p = sim["u"] * sim["rho"] * (2. / 3)
    p.convert_units("Pa")
    return p


@SimSnap.derived_array
def u(self):
    """Gas internal energy derived from temperature"""
    gamma = 5. / 3
    return self['temp'] * units.k / (self['mu'] * units.m_p * (gamma - 1))


@SimSnap.derived_array
def temp(self):
    """Gas temperature derived from internal energy

    Note that to perform this derivation requires the mean molecular mass of the gas to be
    known. This  depends on the ionisation state, which not all simulations store explicitly.

    This requires an iterative approach, repeatedly estimating the mean molecular
    mass for a best-guess temperature, then refining the temperature estimate.


    """
    gamma = 5. / 3
    mu_est = np.ones(len(self))
    for i in range(5):
        temp = (self['u'] * units.m_p / units.k) * (mu_est * (gamma - 1))
        temp.sim = self # to allow use of conversion context, e.g. scalefactor
        temp.convert_units("K")
        mu_est = mu(self, temp)
    return temp


@SimSnap.derived_array
def zeldovich_offset(self):
    """The position offset in the current snapshot according to
    the Zel'dovich approximation applied to the current velocities.
    (Only useful in the generation or analysis of initial conditions.)"""
    from . import analysis
    bdot_by_b = analysis.cosmology.rate_linear_growth(
        self, unit='km Mpc^-1 s^-1') / analysis.cosmology.linear_growth_factor(self)

    a = self.properties['a']

    offset = self['vel'] / (a * bdot_by_b)
    offset.units = self['vel'].units / units.Unit('km Mpc^-1 s^-1 a^-1')
    return offset


@SimSnap.derived_array
def aform(self):
    """The expansion factor at the time specified by the tform array."""

    from . import analysis
    z = analysis.cosmology.redshift(self, self['tform'])
    a = 1. / (1. + z)
    return a

@SimSnap.derived_array
def tform(self):
    """The time of the specified expansion factor in the aform"""
    from . import analysis
    t = analysis.cosmology.age(self, 1./self['aform'] - 1.)
    return t

@SimSnap.derived_array
def iord_argsort(self):
    """Indices so that particles are ordered by increasing ids"""
    return np.argsort(self['iord'])
