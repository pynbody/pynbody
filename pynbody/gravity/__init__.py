"""Routines for calculating gravitational potential and accelerations.

.. seealso::

    :mod:`pynbody.analysis.profile` provides a user-friendly interface to rotation curves.
"""


from __future__ import annotations

import math

import numpy as np

from .. import array, config, units
from ..array import SimArray
from ..snapshot.simsnap import SimSnap


def direct(f: SimSnap, ipos: np.ndarray, eps: float | SimArray | None = None, num_threads: int | None = None):
    """Calculate the gravitational acceleration and potential at the specified positions

    The gravitational softening length is determined by (in order of preference):

    1. The parameter ``eps`` (scalar, unit or array of length len(f))

    2. The array ``f['eps']``

    3. ``f.properties['eps']`` (scalar or unit)


    Parameters
    ----------

    f : :class:`pynbody.snapshot.SimSnap`
        The snapshot containing the particles to be used in the calculation.

    ipos : array_like
        The position at which the potential is to be calculated.

    eps : float, :class:`pynbody.units.Unit`, or array_like, optional
        The gravitational softening length. See above for what happens if this is not specified.

    num_threads : int, optional
        The number of threads to use. If not specified, the number of threads is determined by the
        configuration parameter ``number_of_threads``.

    Returns
    -------

    pot : :class:`pynbody.array.SimArray`
        The gravitational potential at the specified positions, with units.

    accel : :class:`pynbody.array.SimArray`
        The gravitational acceleration at the specified positions, with units.

    """
    from ._gravity import direct
    return direct(f, ipos, eps, num_threads or 0)


def all_direct(f: SimSnap, eps: float | SimArray | None = None):
    """Calculate the potential and acceleration for all particles in the snapshot using a direct summation algorithm.

    The results are stored inside the snapshot itself, as f['phi'] and f['acc'].

    .. warning::
       The direct summation algorithm is implemented in Cython and parallelised. Nonetheless, given the O(N^2) scaling
       of the algorithm, it quickly becomes prohibitive for large numbers of particles.

    Parameters
    ----------

    f :
        The snapshot to calculate the potential and acceleration for

    eps :
        The gravitational softening length. See :func:`pynbody.gravity.direct` for details of
        how this is used, or what happens when it is not specified.

    """
    phi, acc = direct(f, f['pos'].view(np.ndarray), eps)
    f['phi'] = phi
    f['acc'] = acc


def all_pm(f: SimSnap, ngrid: int = 10):
    """Calculate the potential and acceleration for all particles in the snapshot using a Particle-Mesh algorithm.
    This is faster than, but much less accurate than, :func:`pynbody.gravity.all_direct`. It also takes into account
    periodicity of the box.

    The results are stored inside the snapshot itself, as ``f['phi']`` and ``f['acc']``.

    .. warning::
       PM calculations assume periodic boundary conditions, and are only accurate on large scales (much larger than


    Parameters
    ----------

    f :
        The snapshot to calculate the potential and acceleration for

    ngrid :
        The number of grid points to use in each dimension for the Particle-Mesh calculation.

    """
    phi, acc = pm(f, f['pos'].view(np.ndarray), eps, ngrid=ngrid)
    f['phi'] = phi
    f['acc'] = acc


def pm(f: SimSnap, ipos: np.ndarray, ngrid:int = 10, x0=None, x1=None):
    """Calculate the potential and acceleration for a set of particles using a Particle-Mesh algorithm.

    For large numbers of particles, this is faster than, but much less accurate than, :func:`pynbody.gravity.direct`.
    It also takes into account periodicity of the box.

    Parameters
    ----------

    f :
        The snapshot to calculate the potential and acceleration for

    ipos :
        The positions of the particles to calculate the potential and acceleration for

    x0 :
        The lower bound of the grid in each dimension. If ``None``, the minimum of the snapshot's positions will be
        used.

    x1 :
        The upper bound of the grid in each dimension. If ``None``, ``x0 + f.properties['boxsize']`` will be used.

    Returns
    -------

    phi : array.SimArray
        The gravitational potential at the specified positions

    grad_phi : array.SimArray
        The gravitational acceleration at the specified positions

    """

    if x0 is None:
        x0 = f['pos'].min()
    if x1 is None:
        x1 = x0 + f.properties['boxsize']

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

def midplane_rot_curve(f: SimSnap, rxy_points: np.ndarray, eps: float | SimArray | None = None):
    """Calculate the rotation curve of a disk galaxy in the x-y midplane (with z=0)

    Parameters
    ----------
    f : SimSnap
        The snapshot to calculate the rotation curve for
    rxy_points : array_like
        Radii at which to calculate the rotation curve, in the xy-plane
    eps:
        The gravitational softening length. See :func:`pynbody.gravity.direct` for details of
        how this is used, and what happens when it is not specified.

    Returns
    -------

    v : array.SimArray
        The rotation curve at the specified radii
    """


    # u_out = (units.G * f['mass'].units / f['pos'].units)**(1,2)

    # Do four samples like Tipsy does
    rs = [pos for r in rxy_points for pos in [
        (r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]]

    pot, accel = direct(f, np.array(rs, dtype=f['pos'].dtype), eps=eps)

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


def midplane_potential(f, rxy_points, eps=None):
    """Calculate the potential of a disk galaxy in the x-y midplane (with z=0)

    Parameters
    ----------
    f :
        The snapshot to calculate the potential for
    rxy_points :
        A list or array of radii at which to calculate the potential, in the xy-plane
    eps :
        The gravitational softening length. See :func:`pynbody.gravity.direct` for details of
        how this is used, or what happens when it is not specified.

    Returns
    -------

    v : array.SimArray
        The potential at the specified radii
    """

    u_out = units.G * f['mass'].units / f['pos'].units


    # Do four samples like Tipsy does
    rs = [pos for r in rxy_points for pos in [
        (r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]]

    m_by_r, m_by_r2 = direct(f, np.array(rs, dtype=f['pos'].dtype), eps=eps)

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
