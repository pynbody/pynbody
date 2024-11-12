"""Plots for radial profiles

For more information, see the :ref:`profile` tutorial.

"""

import logging

import numpy as np
import pylab as p

from .. import config, filt, transformation, units
from ..analysis import angmom, halo, profile

logger = logging.getLogger('pynbody.plot.profile')


def rotation_curve(sim, center=True, r_units='kpc',
                   v_units='km s^-1', nbins=50, bin_spacing='equaln', quick=False,
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

        import pylab as p

        if parts:
            p.plot(r, v, label='total', **kwargs)
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
            p.plot(gpro['rbins'].in_units(r_units), gv, "--", label="gas")
            p.plot(dpro['rbins'].in_units(r_units), dv, label="dark")
            p.plot(spro['rbins'].in_units(r_units), sv, linestyle="dotted", label="star")
        else:
            p.plot(r, v, **kwargs)


        p.xlabel("r / $" + r.units.latex() + "$", fontsize='large')
        p.ylabel("v$_c / " + v.units.latex() + '$', fontsize='large')

    return r, v


def fourier_profile(sim, center=True, disk_height='2 kpc', nbins=50,
                    pretime='2 Gyr', r_units='kpc', bin_spacing='equaln',
                    clear=True, min=False, max=False, filename=None, **kwargs):
    """
    Centre on potential minimum, align so that the disk is in the
    x-y plane, then plot the amplitude of the 2nd fourier mode as a
    function of radius.

    **needs description of the keyword arguments**

    """

    if center:
        angmom.faceon(sim)

    if min:
        min_r = min
    else:
        min_r = sim['rxy'].min()
    if max:
        max_r = max
    else:
        max_r = sim['rxy'].max()

    if isinstance(pretime, str):
        pretime = units.Unit(pretime)

    diskstars = sim.star[filt.Disc(max_r, disk_height)]
    youngstars = np.where(diskstars['tform'].in_units("Myr") >
                          sim.properties['time'].in_units(
                              "Myr", **sim.conversion_context())
                          - pretime.in_units('Myr'))[0]

    pro = profile.Profile(diskstars[youngstars], type=bin_spacing,
                          nbins=nbins, rmin =min_r, rmax =max_r)

    r = pro['rbins'].in_units(r_units)
    fourierprof = pro['fourier']
    a2 = fourierprof['amp'][2]

    if clear:
        p.clf()

    p.plot(r, a2, **kwargs)

    p.xlabel("r / $" + r.units.latex() + "$")
    p.ylabel("Amplitude of 2nd Fourier Mode")
    if (filename):
        logger.info("Saving %s", filename)
        p.savefig(filename)


def density_profile(sim, linestyle=False, center=True, clear=True, fit=False,in_units=None,
                    filename=None, fit_factor=0.02, axes=False, **kwargs):
    '''

    3d density profile

    **Options:**

    *filename* (None):  name of file to which to save output

    **Usage:**

    >>> import pynbody.plot as pp
    >>> halos = s.halos()
    >>> pp.density_profile(halos[1],linestyle='dashed',color='k')


    '''
    if axes: plt = axes
    else: import matplotlib.pyplot as plt

    global config

    logger.info("Centering...")
    if center:
        halo.center(sim, mode='ssc')

    logger.info("Creating profile...")

    if 'min' in kwargs:
        ps = profile.Profile(
            sim, ndim=3, type='log', nbins=40, rmin =kwargs['min'])
        del kwargs['min']
    else:
        ps = profile.Profile(sim, ndim=3, type='log', nbins=40)

    if clear and not axes:
        plt.clf()
    critden = (units.Unit('100 km s^-1 Mpc^-1')
               * sim.properties['h']) ** 2 / 8.0 / np.pi / units.G
    r = ps['rbins'].in_units('kpc')
    if in_units is None:
        den = ps['density'].in_units(critden)
    else:
        den = ps['density'].in_units(in_units)

    if linestyle:
        plt.errorbar(r, den, yerr=den / np.sqrt(ps['n']),
                     linestyle=linestyle, **kwargs)
    else:
        plt.errorbar(r, den, yerr=den / np.sqrt(ps['n']),
                     fmt='o', **kwargs)

    if in_units is None:
        ylabel=r'$\rho / \rho_{cr}$'  # +den.units.latex()+'$]')
    else:
        ylabel=r'$\rho / '+den.units.latex()+'$'
    if axes:
        plt.set_yscale('log')
        plt.set_xscale('log')
        plt.set_xlabel('r [kpc]')
        plt.set_ylabel(ylabel) #r'$\rho / \rho_{cr}$')  # +den.units.latex()+'$]')
    else:
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('r [kpc]')
        plt.ylabel(ylabel) #r'$\rho / \rho_{cr}$')  # +den.units.latex()+'$]')


    if (filename):
        logger.info("Saving %s", filename)
        plt.savefig(filename)

    if fit:
        fit_inds = np.where(r < fit_factor*sim['r'].max())
        alphfit = np.polyfit(np.log10(r[fit_inds]),
                             np.log10(den[fit_inds]), 1)

        fit = np.poly1d(alphfit)
        plt.plot(r[fit_inds], 10**fit(np.log10(r[fit_inds])),
                 color='k',linestyle='dashed',
                 label=r'$\alpha$=%.1f'%alphfit[0])
        plt.legend(loc=3)

        return alphfit[0]
