"""Plots for radial profiles

For more information, see the :ref:`profile` tutorial.

.. versionchanged :: 2.0

  The ``fourier_profile`` routine has been removed.

  Call signatures for the other two functions have been cleared up (see below).

"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from .. import transformation
from ..analysis import angmom, halo, profile

logger = logging.getLogger('pynbody.plot.profile')


def rotation_curve(sim, center=True, r_units='kpc',
                   v_units='km s^-1', nbins=50, bin_spacing='log', quick=False,
                   rmin=None, rmax=None, parts=False, **kwargs):
    """Generate and plot a rotation curve.

    This routine centres and aligns the disk into the x-y plane, then uses
    the potential in that plane to generate and plot a rotation curve.

    The transformation of the simulation is then reverted to its original
    state.

    .. versionchanged :: 2.0

       The transformation of the simulation is now reverted when the routine exits. Previously,
       the transformation was left in place.

       The *clear*, *axes*, *legend*, *filename* and *yrange* keywords have been removed for consistency with
       the rest of the plotting routines. Use the matplotlib functions directly to save the figure
       or modify the axes.

    Parameters
    ----------

    sim : pynbody.snapshot.SimSnap
        The simulation snapshot to be used.

    center : bool, optional
        If True (default), the simulation is centered and rotated so that the disk is in the x-y plane. If False,
        the simulation is assumed to be pre-centred and already aligned.

    quick : bool, optional
        If True, the rotation curve is calculated using a spherical approximation to the circular velocity.
        If False (default), the rotation curve is calculated using 3D forces.

    bin_spacing : str, optional
        The type of bin spacing to use in the profile. See :class:`~pynbody.analysis.profile.Profile` for details.

    rmin : float, optional
        The minimum radius to use in the profile. Default is the minimum radius in the simulation.

    rmax : float, optional
        The maximum radius to use in the profile. Default is the maximum radius in the simulation.

    nbins : int, optional
        The number of bins to use in the profile. Default is 50.

    r_units : str, optional
        The units in which to plot the radial axis. Default is 'kpc'.

    v_units : str, optional
        The units in which to plot the velocity axis. Default is 'km s^-1'.

    parts : bool, optional
        If True, the rotation curve is calculated and plotted for each particle type separately. Default is False.

    min : float, optional
        Deprecated. Use rmin instead.

    max : float, optional
        Deprecated. Use rmax instead.

    Returns
    -------

    r : pynbody.array.SimArray
        The radial bins used in the profile.

    v : pynbody.array.SimArray
        The circular velocity profile

    """

    if 'min' in kwargs:
        rmin = kwargs.pop('min')
        logger.warning("The 'min' keyword is deprecated. Use 'rmin' instead.", DeprecationWarning)

    if 'max' in kwargs:
        rmax = kwargs.pop('max')
        logger.warning("The 'max' keyword is deprecated. Use 'rmax' instead.", DeprecationWarning)


    if center:
        trans = angmom.faceon(sim)
    else:
        trans = transformation.NullTransformation(sim)

    with trans:

        if rmin is None:
            rmin = sim['rxy'].min()
        if rmax is None:
            rmax = sim['rxy'].max()

        pro = profile.Profile(sim, type=bin_spacing, nbins=nbins,
                              rmin =rmin, rmax =rmax)

        r = pro['rbins'].in_units(r_units)
        if quick:
            v = pro['rotation_curve_spherical'].in_units(v_units)
        else:
            v = pro['v_circ'].in_units(v_units)

        if parts:
            plt.plot(r, v, label='total', **kwargs)
            gpro = profile.Profile(sim.gas, type=bin_spacing, nbins=nbins,
                                   rmin =rmin, rmax =rmax)
            dpro = profile.Profile(sim.dark, type=bin_spacing, nbins=nbins,
                                   rmin =rmin, rmax =rmax)
            spro = profile.Profile(sim.star, type=bin_spacing, nbins=nbins,
                                   rmin =rmin, rmax =rmax)
            if quick:
                gv = gpro['rotation_curve_spherical'].in_units(v_units)
                dv = dpro['rotation_curve_spherical'].in_units(v_units)
                sv = spro['rotation_curve_spherical'].in_units(v_units)
            else:
                gv = gpro['v_circ'].in_units(v_units)
                dv = dpro['v_circ'].in_units(v_units)
                sv = spro['v_circ'].in_units(v_units)
            plt.plot(gpro['rbins'].in_units(r_units), gv, "--", label="gas")
            plt.plot(dpro['rbins'].in_units(r_units), dv, label="dark")
            plt.plot(spro['rbins'].in_units(r_units), sv, linestyle="dotted", label="star")
        else:
            plt.plot(r, v, **kwargs)


        plt.xlabel("r / $" + r.units.latex() + "$", fontsize='large')
        plt.ylabel("v$_c / " + v.units.latex() + '$', fontsize='large')

    return r, v


def density_profile(sim, center=True, r_units=None, rho_units=None,
                    rmin=None, rmax=None, nbins=50, bin_spacing='log', **kwargs):
    """Generate and plot a 3D density profile with error-bars.

    This routine centers the simulation and generates a 3D density profile with error-bars. It then undoes the
    centering transformation before returning.

    .. versionchanged :: 2.0

      The transformation of the simulation is now reverted when the routine exits. Previously,
      the transformation was left in place.

      The y-axis is no longer scaled to the critical density by default. Specify the units using the *in_units* keyword.

      The x-axis is no longer scaled to kpc by default. Specify the units using the *r_units* keyword.

      The *clear*, *axes*, *filename*, *fit* and *fit_factor* keywords have been removed for consistency with
      the rest of the plotting routines. Use the matplotlib functions directly to save the figure
      or modify the axes.

    Parameters
    ----------

    sim : pynbody.snapshot.SimSnap
        The simulation snapshot to be used.

    center : bool, optional
        If True (default), the simulation is centered. If False, the simulation is assumed to be pre-centered.

    r_units : str, optional
        The units in which to plot the radial axis. Default is None (no conversion).

    rho_units : str, optional
        The units in which to plot the density axis. Default is None (no conversion).

    rmin : float, optional
        The minimum radius to use in the profile. Default is the minimum radius in the simulation.

    rmax : float, optional
        The maximum radius to use in the profile. Default is the maximum radius in the simulation.

    nbins : int, optional
        The number of bins to use in the profile. Default is 50.

    bin_spacing : str, optional
        The type of bin spacing to use in the profile. See :class:`~pynbody.analysis.profile.Profile` for details.

    **kwargs : dict
        Additional keyword arguments are passed to :func:`matplotlib.pyplot.errorbar`.

    """

    if 'min' in kwargs:
        rmin = kwargs.pop('min')
        logger.warning("The 'min' keyword is deprecated. Use 'rmin' instead.", DeprecationWarning)

    if 'max' in kwargs:
        rmax = kwargs.pop('max')
        logger.warning("The 'max' keyword is deprecated. Use 'rmax' instead.", DeprecationWarning)

    if 'in_units' in kwargs:
        rho_units = kwargs.pop('in_units')
        logger.warning("The 'in_units' keyword is deprecated. Use 'rho_units' instead.", DeprecationWarning)

    if center:
        trans = halo.center(sim, mode='ssc')
    else:
        trans = transformation.NullTransformation(sim)

    with trans:
        ps = profile.Profile(sim, ndim=3, type=bin_spacing, nbins=nbins, rmin =rmin, rmax=rmax)

        r = ps['rbins']
        den = ps['density']

        if r_units is not None:
            r = r.in_units(r_units)

        if rho_units is not None:
            den = den.in_units(rho_units)

    plt.errorbar(r, den, yerr=den / np.sqrt(ps['n']), fmt='o', **kwargs)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$r / '+r.units.latex()+'$')
    plt.ylabel(r'$\rho / '+den.units.latex()+'$')
