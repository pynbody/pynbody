"""
Routines and derived arrays for calculating luminosities and magnitudes.

This module provides a number of routines for calculating luminosities and magnitudes
starting from the stellar populations in a simulation.

Two sets of derived arrays are programmatically generated. The first set is the
magnitude of a stellar particle in a given bandpass, e.g. 'v_mag'. The second set is
the luminosity density, encoded as 10^{-0.4 * mag} per unit volume, e.g. 'v_lum_den'.
The purpose of the luminosity density array is that, when integrated over a line of sight,
it becomes a number of magnitudes per unit area (e.g. mag/kpc^2), which can then be
turned into an astronomical surface brightness (mag/arcsec^2) -- this approach is taken
by the :func:`pynbody.plot.stars.render` routine.

The above arrays are generated for all known bandpasses in the CMD.

Origin of the luminosity tables
-------------------------------

Luminosities in pynbody are calculated by treating each star particle as a single
stellar population (SSP) of a known age and metallicity, assuming a fixed initial
mass function (IMF). Magntiudes for each star particle can then be interpolated
from a table, for the known UBVRIJHK bandpasses.

The SSP tables provided are computed using CMD 3.7, via the web interface at
http://stev.oapd.inaf.it/cgi-bin/cmd_3.7 . There are a number of reasons why you
may wish to use a custom set of SSPs:

 * You wish to use a different set of stellar evolution assumptions;
 * You wish to use a different set of bandpasses;
 * You wish to assume a different IMF;
 * You wish to include interstellar dust extinctions (by default, only circumstellar
   dust is included, and dust extinction is optionally applied separately when rendering
   images; see the :func:`pynbody.plot.stars.render` documentation for more information).

If making your own tables using CMD, ensure that you opt for a grid of ages and metallicities
and a single-burst stellar population for 1 Msol of stars. These crucial options are near the
bottom of the web interface (as of May 2024). For the default table included with pynbody,
the requested table has log(age/yr) between 6.6 and 10.2 dex in steps of 0.1 dex, and the
metallicities between -2 and 0.2 dex solar in steps of 0.2 dex.  Note that ages and metallicities lying
outside the tabulated range are clamped to the edge of the table.

All other options are left as per the CMD 3.7 defaults in the default table.

"""

import functools
import os
import warnings

import numpy as np

from .. import filt, snapshot
from .interpolate import interpolate2d

_ssp_table = None
_default_ssp_file = os.path.join(os.path.dirname(__file__), "default_ssp.txt")

class SSPTable:
    """An SSP table for interpolating magnitudes from stellar populations"""

    def __init__(self, ages, metallicities, magnitudes, case_insensitive=True):
        """Initialise an SSP table

        Parameters
        ----------

        ages : array-like
            Array of ages in log10 years, length N

        metallicities : array-like
            Array of log10 metal mass fractions, length M

        magnitudes : dict[str, array-like]
            Dictionary of bandpass names to 2D arrays of magnitudes, size N x M

        case_insensitive : bool, optional
            If True, the bandpass names are treated as case-insensitive.


        """
        self._ages = np.asarray(ages)
        self._metallicities = np.asarray(metallicities)

        self._case_insensitive = case_insensitive
        self._magnitudes = {k.lower() if self._case_insensitive else k: np.asarray(v) for k, v in magnitudes.items()}

    def __repr__(self):
        return f"<SSPTable; bands={list(self._magnitudes.keys())}>"

    def interpolate(self, ages, metallicities, band):
        """Interpolate the magnitude for a given age, metallicity and bandpass

        Parameters
        ----------

        age : float or array-like
             Age in log10 years

        metallicities : float or array-like
            Metallicity in log10 mass fraction

        band : str
            Bandpass name

        Returns
        -------

        float or array-like
            Magnitude(s) per solar mass interpolated from the SSP table

        """
        if self._case_insensitive:
            band = band.lower()
        metallicities = np.copy(metallicities)
        ages = np.copy(ages)
        # clamp to the edge of the table
        metallicities = self._clamp_value(metallicities, self._metallicities)
        ages = self._clamp_value(ages, self._ages)

        return interpolate2d(metallicities, ages, self._metallicities, self._ages, self._magnitudes[band])

    def __call__(self, snapshot, band):
        """Interpolate the magnitude for a given snapshot and bandpass

        Parameters
        ----------

        snapshot : pynbody.SimSnap
            Snapshot containing the stars

        band : str
            Bandpass name

        Returns
        -------

        array-like
            Magnitudes of star particles interpolated from the SSP table

        """

        age_star = snapshot['age'].in_units('yr')
        metals = snapshot['metals']
        try:
            masses = snapshot['massform'].in_units('Msol')
        except KeyError:
            masses = snapshot['mass'].in_units('Msol')

        with np.errstate(invalid='ignore'):
            output_mags = self.interpolate(np.log10(age_star), metals, band)


        vals = output_mags - 2.5 * np.log10(masses)

        vals.units = None
        return vals

    def get_central_wavelength(self, band):
        """Get the estimated central wavelength of a bandpass

        The values are approximate, based on the tabulated central wavelengths from CMD. If you use a different
        bandpass set, you may wish to override this method. It is used by the :func:`pynbody.plot.stars.render` routine
        to estimate the central wavelength of a bandpass for the purposes of estimating dust extinction.

        Parameters
        ----------

        band : str
            Bandpass name

        Returns
        -------

        float
            Central wavelength in Angstroms

        """

        return {'u': 3598.54, 'b': 4385.92, 'v': 5490.56, 'r': 6594.72, 'i': 8059.88, 'j': 12369.26,
                'h': 16464.45, 'k': 22105.45}[band.lower()]



    @classmethod
    def _clamp_value(cls, value, values):
        value = np.atleast_1d(value)
        value[value < np.min(values)] = np.min(values)
        value[value > np.max(values)] = np.max(values)
        return value

class ArchivedSSPTable(SSPTable):
    """An SSP table from a pynbody v1 archive"""

    def __init__(self, path):
        """Initialise an SSP table from a pynbody v1 archive

        Parameters
        ----------

        path : str
            Path to the archive file

        """
        data = np.load(path)

        super().__init__(np.log10(data['ages']), data['mets'],
                         {k: data[k] for k in data.files if k not in ['ages', 'mets']})

class StevSSPTable(SSPTable):
    """An SSP table from the output of the STEV/CMD web interface"""

    def __init__(self, path):
        """Initialise an SSP table from the text output of the STEV/CMD web interface

        """
        with open(path) as f:
            lines = f.readlines()

        column_names = None

        # find a line starting with '# age' and parse it for the column names:
        for line in lines:
            if line.startswith('# age'):
                column_names = line[1:].strip().split()
                break

        if column_names is None:
            raise ValueError("Could not find column names in the file")

        # use numpy to extract the values in each column:
        data = np.genfromtxt(lines, comments='#', names=column_names)

        ages = np.log10(data['age'])
        ages1d = np.unique(ages)

        metallicities = np.log10(data['Z'])
        metallicities1d = np.unique(metallicities)

        # check that the ages and metallicities are in the correct order
        try:
            ages2d = ages.reshape((-1, len(ages1d)))
            fail = False
        except ValueError:
            fail = True

        if fail or np.any(ages1d != ages2d):
            raise ValueError("Ages don't follow expected grid pattern") from None

        try:
            metallicities2d = metallicities.reshape((len(metallicities1d), -1))
            fail = False
        except ValueError:
            fail = True

        if fail or np.any(metallicities1d[:,np.newaxis] != metallicities2d):
            raise ValueError("Metallicities don't follow expected grid pattern") from None


        super().__init__(ages1d, metallicities1d,
                         {k[:-3]: data[k].reshape((len(metallicities1d), len(ages1d)))
                          for k in column_names if k.endswith('mag')})


def _load_ssp_table(path_or_table):
    if isinstance(path_or_table, SSPTable):
        return path_or_table
    elif isinstance(path_or_table, str):
        if path_or_table.endswith('.npz'):
            return ArchivedSSPTable(path_or_table)
        else:
            return StevSSPTable(path_or_table)
    else:
        raise ValueError("Invalid path or table")

def get_current_ssp_table() -> SSPTable:
    """Get the current preferred SSP table for calculating magnitudes

    This will either be pynbody's default table or a custom table specified by :func:`use_custom_ssp_table`."""
    global _ssp_table
    if _ssp_table is None:
        _ssp_table = _load_ssp_table(_default_ssp_file)
    return _ssp_table

class SSPTableContext:
    """Context manager for temporarily using a custom SSP table"""
    def __init__(self, ssp_table):
        global _ssp_table
        self._new_table = ssp_table
        self._old_table = _ssp_table
        _ssp_table = self._new_table

    def __enter__(self):
        pass

    def __exit__(self, *args):
        global _ssp_table
        _ssp_table = self._old_table

def use_custom_ssp_table(path_or_table : SSPTable):
    """Specify a custom SSP table for calculating magnitudes.

    This function allows you to specify a custom SSP table for calculating magnitudes.
    The specified table will be used for all future calls to :func:`calc_mags`, and by extension,
    for all derived arrays that depend on magnitudes. However, be aware that this will not affect
    any existing arrays that have already been derived.

    A context manager is returned, so you can use this function in a ``with`` block to temporarily
    use a custom table, i.e.

    >>> with use_custom_ssp_table('mytable.npz'):
    >>>     print(pynbody.analysis.luminosity.calc_mags(s, 'v'))
    >>> print(pynbody.analysis.luminosity.calc_mags(s, 'v'))

    Here, the first call to calc_mags will use 'mytable.npz', and the second call will use the default
    table again.

    However, you do not need to enter the context manager if you want to permanently change the table, i.e.

    >>> use_custom_ssp_table('mytable.npz')
    >>> print(pynbody.analysis.luminosity.calc_mags(s, 'v'))

    Here, the call to calc_mags will use 'mytable.npz' until the table is changed again.

    Parameters
    ----------

    path_or_table : str or SSPTable
        Path to the SSP table file, or an :class:`SSPTable` object. Alternatively, you can pass
        either 'default' (for the default table included with pynbody) or 'v1' (for the
        table included with pynbody v1).

    Returns
    -------

    SSPTableContext
        A context manager that can be used to temporarily use a custom SSP table. This is useful
        for situations where you want to use a custom table for a specific calculation, but then
        revert to the default table.

    """
    if path_or_table == 'default':
        path_or_table = _default_ssp_file
    elif path_or_table == 'v1':
        path_or_table = os.path.join(os.path.dirname(__file__), "cmdlum.npz")

    return SSPTableContext(_load_ssp_table(path_or_table))


def use_custom_cmd(path):
    """Deprecated alias for :func:`use_custom_ssp_table`"""
    warnings.warn("use_custom_cmd is deprecated; use use_custom_ssp_table instead", DeprecationWarning)
    use_custom_ssp_table(path)

def calc_mags(simstars, band='v', cmd_path=None):
    """Calculate the magnitude of stars in a simulation

    This makes use of SSP tables, as described in the module documentation (see :mod:`pynbody.analysis.luminosity`).

    Parameters
    ----------

    simstars : pynbody.SimSnap
        Snapshot containing the stars (only). If you have a snapshot with non-star particles, pass
        ``sim.s`` to this function.

    band : str
        Bandpass name. Can be any that is defined in the SSP table (which by default includes
        'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K'). See the module documentation (:mod:`pynbody.analysis.luminosity`).

    cmd_path : str, optional
        Path to the SSP table file. If not provided, the default table will be used. This is either the
        default table included with pynbody, or the table specified by :func:`use_custom_cmd`.

    """

    # find data file in PYTHONPATH
    # data is from http://stev.oapd.inaf.it/cgi-bin/cmd
    # Padova group stellar populations Marigo et al (2008), Girardi et al
    # (2010)
    if cmd_path is None:
        table = get_current_ssp_table()
    else:
        table = _load_ssp_table(cmd_path)

    return table(simstars, band)




def halo_mag(sim, band='v'):
    """Calculate the absolute magnitude of the provided halo (or other collection of particles)

    Parameters
    ----------

    sim : pynbody.SimSnap
        Halo (or other subsnap, or even a whole simulation) for which to calculate the absolute magnitude.
        Any non-star particles are ignored.

    band : str
        Bandpass name. Can be any that is defined in the SSP table. See the module documentation
        (:mod:`pynbody.analysis.luminosity`).
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


def _setup_derived_arrays():

    bands_available = ['u', 'b', 'v', 'r', 'i', 'j', 'h', 'k', 'U', 'B', 'V', 'R', 'I',
                       'J', 'H', 'K']

    def _lum_den_template(band, s):
        val = (10 ** (-0.4 * s[band + "_mag"])) * s['rho'] / s['mass']
        val.units = s['rho'].units/s['mass'].units
        return val

    for band in bands_available:
        X = lambda s, b=str(band): calc_mags(s, band=b)
        X.__name__ = band + "_mag"
        X.__doc__ = band + " magnitude from analysis.luminosity.calc_mags"""
        snapshot.SimSnap.derived_array(X)

        lum_den = functools.partial(_lum_den_template, band)

        lum_den.__name__ = band + "_lum_den"
        lum_den.__doc__ = "Luminosity density in astronomy-friendly units: 10^(-0.4 %s_mag) per unit volume. " \
                          "" \
                          "The magnitude is taken from analysis.luminosity.calc_mags."%band
        snapshot.SimSnap.derived_array(lum_den)


_setup_derived_arrays()
