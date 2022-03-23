"""

profile
=======

"""

import logging

import numpy as np
import pylab as p

from .. import config, filt, units
from ..analysis import angmom, halo, profile

logger = logging.getLogger('pynbody.plot.profile')


def rotation_curve(sim, center=True, r_units='kpc',
                   v_units='km s^-1', disk_height='100 pc', nbins=50,
                   bin_spacing='equaln', clear=True, quick=False,
                   filename=None, min=False, max=False, yrange=False,
                   legend=False, parts=False, axes=False, **kwargs):
    """

    Centre on potential minimum, align so that the disk is in the
    x-y plane, then use the potential in that plane to generate and
    plot a rotation curve.

    **needs documentation/description of the keyword arguments**

    """
    import pylab as p

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

    pro = profile.Profile(sim, type=bin_spacing, nbins=nbins,
                          rmin =min_r, rmax =max_r)

    r = pro['rbins'].in_units(r_units)
    if quick:
        v = pro['rotation_curve_spherical'].in_units(v_units)
    else:
        v = pro['v_circ'].in_units(v_units)

    if axes:
        p = axes
    else:
        import pylab as p
        if clear:
            p.clf()

    if parts:
        p.plot(r, v, label='total', **kwargs)
        gpro = profile.Profile(sim.gas, type=bin_spacing, nbins=nbins,
                               rmin =min_r, rmax =max_r)
        dpro = profile.Profile(sim.dark, type=bin_spacing, nbins=nbins,
                               rmin =min_r, rmax =max_r)
        spro = profile.Profile(sim.star, type=bin_spacing, nbins=nbins,
                               rmin =min_r, rmax =max_r)
        if quick:
            gv = gpro['rotation_curve_spherical'].in_units(v_units)
            dv = dpro['rotation_curve_spherical'].in_units(v_units)
            sv = spro['rotation_curve_spherical'].in_units(v_units)
        else:
            gv = gpro['v_circ'].in_units(v_units)
            dv = dpro['v_circ'].in_units(v_units)
            sv = spro['v_circ'].in_units(v_units)
        p.plot(r, gv, "--", label="gas")
        p.plot(r, dv, label="dark")
        p.plot(r, sv, linestyle="dotted", label="star")
    else:
        p.plot(r, v, **kwargs)

    if yrange:
        p.axis(
            [min_r, units.Unit(max_r).in_units(r.units), yrange[0], yrange[1]])

    if not axes:
        p.xlabel("r / $" + r.units.latex() + "$", fontsize='large')
        p.ylabel("v$_c / " + v.units.latex() + '$', fontsize='large')

    if legend:
        p.legend(loc=0)

    if (filename):
        logger.info("Saving %s", filename)
        p.savefig(filename)

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
    >>> h = s.halos()
    >>> pp.density_profile(h[1],linestyle='dashed',color='k')


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

#        print "alpha: ", alphfit[0], "  norm:", alphfit[1]

        fit = np.poly1d(alphfit)
        plt.plot(r[fit_inds], 10**fit(np.log10(r[fit_inds])),
                 color='k',linestyle='dashed',
                 label=r'$\alpha$=%.1f'%alphfit[0])
        plt.legend(loc=3)

        return alphfit[0]
