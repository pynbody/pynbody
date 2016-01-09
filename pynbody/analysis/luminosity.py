"""

luminosity
==========

Calculates luminosities -- NEEDS DOCUMENTATION

"""

import numpy as np

import os
from ..array import SimArray

from interpolate import interpolate2d


def calc_mags(simstars, band='v'):
    """Calculating visible magnitudes

    Using Padova Simple stellar populations (SSPs) from Girardi
    http://stev.oapd.inaf.it/cgi-bin/cmd
    Marigo+ (2008), Girardi+ (2010)

    pynbody includes a grid of SSP luminosities for many bandpasses for
    various stellar ages and metallicities.  This function linearly 
    interpolates to the desired value and returns the value as a magnitude.

    **Usage:**

    >>> import pynbody
    >>> pynbody.analysis.luminosity.calc_mags(h[1].s)

    **Optional keyword arguments:**

       *band* (default='v'): Which observed bandpass magnitude in which 
            magnitude should be calculated

    """

    # find data file in PYTHONPATH
    # data is from http://stev.oapd.inaf.it/cgi-bin/cmd
    # Padova group stellar populations Marigo et al (2008), Girardi et al
    # (2010)
    lumfile = os.path.join(os.path.dirname(__file__), "cmdlum.npz")
    if os.path.exists(lumfile):
        lums = np.load(lumfile)
    else:
        raise IOError, "cmdlum.npz (magnitude table) not found"

    age_star = simstars['age'].in_units('yr')
    # allocate temporary metals that we can play with
    metals = simstars['metals']
    # get values off grid to minmax
    age_star[np.where(age_star < np.min(lums['ages']))] = np.min(lums['ages'])
    age_star[np.where(age_star > np.max(lums['ages']))] = np.max(lums['ages'])
    metals[np.where(metals < np.min(lums['mets']))] = np.min(lums['mets'])
    metals[np.where(metals > np.max(lums['mets']))] = np.max(lums['mets'])

    age_grid = np.log10(lums['ages'])
    met_grid = lums['mets']
    mag_grid = lums[band]

    output_mags = interpolate2d(
        metals, np.log10(age_star), met_grid, age_grid, mag_grid)

    try:
        vals = output_mags - 2.5 * \
            np.log10(simstars['massform'].in_units('Msol'))
    except KeyError, ValueError:
        vals = output_mags - 2.5 * np.log10(simstars['mass'].in_units('Msol'))

    vals.units = None
    return vals


def halo_mag(sim, band='v'):
    """Calculating halo magnitude

    Calls pynbody.analysis.luminosity.calc_mags for ever star in passed
    in simulation, converts those magnitudes back to luminosities, adds
    those luminosities, then converts that luminosity back to magnitudes,
    which are returned.

    **Usage:**

    >>> import pynbody
    >>> pynbody.analysis.luminosity.halo_mag(h[1].s)

    **Optional keyword arguments:**

       *band* (default='v'): Which observed bandpass magnitude in which 
            magnitude should be calculated
    """
    if (len(sim.star) > 0):
        return -2.5 * np.log10(np.sum(10.0 ** (-0.4 * sim.star[band + '_mag'])))
    else:
        return np.nan


def halo_lum(sim, band='v'):
    """Calculating halo luminosiy

    Calls pynbody.analysis.luminosity.calc_mags for every star in passed
    in simulation, converts those magnitudes back to luminosities, adds
    those luminosities, which are returned.  Uses solar magnitudes from
    http://www.ucolick.org/~cnaw/sun.html.

    **Usage:**

    >>> import pynbody
    >>> pynbody.analysis.luminosity.halo_mag(h[1].s)

    **Optional keyword arguments:**

       *band* (default='v'): Which observed bandpass magnitude in which 
            magnitude should be calculated
    """
    sun_abs_mag = {'u':5.56,'b':5.45,'v':4.8,'r':4.46,'i':4.1,'j':3.66,
                   'h':3.32,'k':3.28}[band]
    return np.sum(10.0 ** ((sun_abs_mag - sim.star[band + '_mag']) / 2.5))


def half_light_r(sim, band='v'):
    '''Calculate half light radius

    Calculates entire luminosity of simulation, finds half that, sorts
    stars by distance from halo center, and finds out inside which radius
    the half luminosity is reached.
    '''
    import pynbody
    import pynbody.filt as f
    half_l = halo_lum(sim, band=band) * 0.5

    max_high_r = np.max(sim.star['r'])
    test_r = 0.5 * max_high_r
    testrf = f.LowPass('r', test_r)
    min_low_r = 0.0
    test_l = halo_lum(sim[testrf], band=band)
    it = 0
    while ((np.abs(test_l - half_l) / half_l) > 0.01):
        it = it + 1
        if (it > 20):
            break

        if (test_l > half_l):
            test_r = 0.5 * (min_low_r + test_r)
        else:
            test_r = (test_r + max_high_r) * 0.5
        testrf = f.LowPass('r', test_r)
        test_l = halo_lum(sim[testrf], band=band)

        if (test_l > half_l):
            max_high_r = test_r
        else:
            min_low_r = test_r

    return test_r * sim.star['r'].units
