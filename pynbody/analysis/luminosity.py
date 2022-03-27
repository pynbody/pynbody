"""

luminosity
==========

Calculates luminosities -- NEEDS DOCUMENTATION

"""

import os

import numpy as np

from .. import filt
from .interpolate import interpolate2d

_cmd_lum_file = os.path.join(os.path.dirname(__file__), "cmdlum.npz")

def use_custom_cmd(path):
    """Use a custom set of stellar populations to calculate magnitudes.

    The path is to a numpy archive with a suitable grid of ages/metallicities and corresponding magnitudes.

    The following script from Stephanie De Beer should help you make a suitable file starting
    from ugriz SSPs downloaded from http://stev.oapd.inaf.it/cgi-bin/cmd.

    import numpy as np

    metals = np.linspace(0.002, 0.05, 25)
    ages = np.logspace(5.67, 10.13, 25)

    mags_bol = np.zeros((len(metals),len(ages)))
    mags_u = np.zeros((len(metals),len(ages)))
    mags_g = np.zeros((len(metals),len(ages)))
    mags_r = np.zeros((len(metals),len(ages)))
    mags_i = np.zeros((len(metals),len(ages)))
    mags_z = np.zeros((len(metals),len(ages)))

    bands = ['bol', 'u', 'g', 'r', 'i', 'z']
    k=2
    for b in bands:
        for x in range(1,26):
            with open('/users/sdebeer/render_stuff/PGSP_files/'+str(x)+'_output.txt', 'r') as f:
                output = f.readlines()
            i = 0
            for line in output:
                magnitudes = line.split()
                if magnitudes[0]=='#':
                    continue
                vars()['mags_'+b][i,x-1]=magnitudes[k]
                i +=1
        k+=1

    np.savez('my_cmd.npz', ages=ages, metals=metals, bol=mags_bol, u=mags_u, g=mags_g, r=mags_r, i=mags_i, z=mags_z)
    """
    global _cmd_lum_file
    _cmd_lum_file = path

def calc_mags(simstars, band='v', cmd_path=None):
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

       *path* (default=None): Path to the CMD grid. If None, use the
            default or a path specified by use_custom_cmd. For more information
            about generating a custom CMD grid, see use_custom_cmd.

    """

    # find data file in PYTHONPATH
    # data is from http://stev.oapd.inaf.it/cgi-bin/cmd
    # Padova group stellar populations Marigo et al (2008), Girardi et al
    # (2010)
    if cmd_path is not None:
        lums = np.load(cmd_path)
    else:
        lums = np.load(_cmd_lum_file)


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
    except KeyError as ValueError:
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


def half_light_r(sim, band='v', cylindrical=False):
    '''Calculate half light radius

    Calculates entire luminosity of simulation, finds half that, sorts
    stars by distance from halo center, and finds out inside which radius
    the half luminosity is reached.

    If cylindrical is True compute the half light radius as seen from the z-axis.
    '''
    half_l = halo_lum(sim, band=band) * 0.5

    if cylindrical:
        coord = 'rxy'
    else:
        coord = 'r'
    max_high_r = np.max(sim.star[coord])
    test_r = 0.5 * max_high_r
    testrf = filt.LowPass(coord, test_r)
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
        testrf = filt.LowPass(coord, test_r)
        test_l = halo_lum(sim[testrf], band=band)

        if (test_l > half_l):
            max_high_r = test_r
        else:
            min_low_r = test_r

    return test_r
