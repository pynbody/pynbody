from .. import units
from .. import array
from .. import config
from ..util import get_eps, eps_as_simarray

import math
import tree
import numpy as np

import warnings

from ._gravity import direct


def all_direct(f, eps=None):
    phi, acc = direct(f, f['pos'].view(np.ndarray), eps)
    f['phi'] = phi
    f['acc'] = acc


def all_pm(f, eps=None, ngrid=10):
    phi, acc = pm(f, f['pos'].view(np.ndarray), eps, ngrid=ngrid)
    f['phi'] = phi
    f['acc'] = acc


def pm(f, ipos, eps=None, ngrid=10, x0=-1, x1=1):
    dx = float(x1 - x0) / ngrid
    grid, edges = np.histogramdd(f['pos'],
                                 bins=ngrid,
                                 range=[(x0, x1), (x0, x1), (x0, x1)],
                                 normed=False,
                                 weights=f['mass'])
    grid /= dx ** 3
    recip_rho_grid = np.fft.rfftn(grid)

    freqs = np.fft.fftfreq(ngrid, d=dx)

    kvecs = np.zeros((ngrid, ngrid, ngrid / 2 + 1, 3))
    kvecs[:, :,:, 0] = freqs.reshape((1, ngrid, 1, 1))
    kvecs[:, :,:, 1] = freqs.reshape((1, 1, ngrid, 1))
    kvecs[:, :,:, 2] = abs(freqs[:ngrid/2+1].reshape((1, 1, 1, ngrid/2+1)))

    k = (kvecs ** 2).sum(axis=3)
    assert k.shape == recip_rho_grid.shape

    recip_phi_grid = 4 * math.pi * recip_rho_grid / k ** 2
    recip_phi_grid[np.where(k == 0)] = 0

    phi_grid = np.fft.irfftn(recip_phi_grid, grid.shape)
    grad_phi_grid = np.concatenate((np.fft.irfftn(-1.j*kvecs[:, :,:, 0]*recip_phi_grid, grid.shape)[:,:,:, np.newaxis],
                                    np.fft.irfftn(-1.j*kvecs[:, :,:, 1]*recip_phi_grid, grid.shape)[:,:,:, np.newaxis],
                                    np.fft.irfftn(-1.j*kvecs[:, :,:, 2]*recip_phi_grid, grid.shape)[:,:,:, np.newaxis]),
                                   axis=3)

    ipos_I = np.array((ipos - x0) / dx, dtype=int)

    phi = np.array([phi_grid[x, y, z] for x, y, z in ipos_I])
    grad_phi = np.array([grad_phi_grid[x, y, z, :] for x, y, z in ipos_I])

    phi = phi.view(array.SimArray)
    phi.units = units.G * f['mass'].units / f['pos'].units

    grad_phi = grad_phi.view(array.SimArray)
    grad_phi.units = units.G * f['mass'].units / f['pos'].units ** 2

    return phi, -grad_phi

def treecalc(f, rs, eps=None):
    gtree = tree.GravTree(
        f['pos'].view(np.ndarray), f['mass'].view(np.ndarray), eps=f['eps'], rs=rs)
    a, p = gtree.calc(rs, eps=eps)
    return p, a


def midplane_rot_curve(f, rxy_points, eps=None, mode=config['gravity_calculation_mode']):

    direct_omp = None

    if mode == 'direct_omp':
        mode = 'direct'  # deprecated
        warnings.warn(
            "OpenMP module is now selected at install time", DeprecationWarning)

    if eps is None:
        eps = get_eps(f)
    elif isinstance(eps, (str, units.UnitBase)):
        eps = eps_as_simarray(f, eps)

    # u_out = (units.G * f['mass'].units / f['pos'].units)**(1,2)

    # Do four samples like Tipsy does
    rs = [pos for r in rxy_points for pos in [
        (r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]]

    try:
        fn = {'direct': direct,
              'tree': treecalc,
              }[mode]
    except KeyError:
        fn = mode

    pot, accel = fn(f, np.array(rs, dtype=f['pos'].dtype), eps=eps)

    u_out = (accel.units * f['pos'].units) ** (1, 2)

    # accel = array.SimArray(m_by_r2,units.G * f['mass'].units / (f['pos'].units**2) )

    vels = []

    i = 0
    for r in rxy_points:
        r_acc_r = []
        for pos in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]:
            r_acc_r.append(np.dot(-accel[i, :], pos))
            i = i + 1

        vel2 = np.mean(r_acc_r)
        if vel2 > 0:
            vel = math.sqrt(vel2)
        else:
            vel = 0

        vels.append(vel)

    x = array.SimArray(vels, units=u_out)
    x.sim = f.ancestor
    return x


def midplane_potential(f, rxy_points, eps=None, mode=config['gravity_calculation_mode']):
    direct_omp = None

    if mode == 'direct_omp':
        try:
            from pynbody.grav_omp import direct as direct_omp
        except ImportError:
            mode = 'direct'

    if eps is None:
        eps = get_eps(f)
    elif isinstance(eps, (str, units.UnitBase)):
        eps = eps_as_simarray(f, eps)

    u_out = units.G * f['mass'].units / f['pos'].units

    try:
        fn = {'direct': direct,
              'direct_omp': direct_omp,
              'tree': tree,
              }[mode]
    except KeyError:
        fn = mode

    # Do four samples like Tipsy does
    rs = [pos for r in rxy_points for pos in [
        (r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]]

    m_by_r, m_by_r2 = fn(f, np.array(rs, dtype=f['pos'].dtype), eps=eps)

    potential = units.G * m_by_r * f['mass'].units / f['pos'].units

    pots = []

    i = 0
    for r in rxy_points:
        # Do four samples like Tipsy does
        pot = []
        for pos in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]:
            pot.append(potential[i])
            i = i + 1

        pots.append(np.mean(pot))

    x = array.SimArray(pots, units=u_out)
    x.sim = f.ancestor
    return x
