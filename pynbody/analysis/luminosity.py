"""
Routines and derived arrays for calculating luminosities and magnitudes.

.. versionchanged:: 2.0

  Luminosity tables are now generated directly from the output of the STEV/CMD web interface.
  The default tables are updated to more modern stellar population tracks (May 2024). This
  will result in different magnitudes being calculated compared to pynbody v1. Furthermore,
  the default tables now include the AB-calibrated LSST bandpasses (ugrizy) in addition to the
  Vega-calibrated Johnson-Cousins bandpasses (UBVRIJHK). As a result, the bandpass names are
  now case-sensitive.

  To obtain pynbody v1 behaviour, you can use the 'v1' table. See :func:`use_custom_ssp_table`.

This module provides a number of routines for calculating luminosities and magnitudes
starting from the stellar populations in a simulation.

Two sets of derived arrays are programmatically generated. The first set is the
magnitude of a stellar particle in a given bandpass, e.g. ``V_mag``. The second set is
the luminosity density, encoded as :math:`10^{-0.4 * {\\rm mag}}` per unit volume, e.g. ``V_lum_den``.
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
from a table, for the known UBVRIJHK and ugrizy bandpasses. Note that lower-case
ugrizy are LSST bandpasses with an AB calibration, while upper-case UBVRIJHK are
Johnson-Cousins bandpasses with a Vega calibration. For many applications, the
distinction is unimportant, but pynbody can also generate absolute fluxes from
AB-calibrated bandpasses.

Customizing the SSP tables
---------------------------

The SSP tables provided are computed using CMD 3.7 from the Padova group, via the web interface
at http://stev.oapd.inaf.it/cgi-bin/cmd_3.7 . There are a number of reasons why you
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
metallicities between -2 and 0.0 dex solar in steps of 0.2 dex.  Note that ages and
metallicities lying outside the tabulated range are clamped to the edge of the table.

All other options are left as per the CMD 3.7 defaults in the default table.

To use your own table, you can use the :func:`use_custom_ssp_table` function. This can also
be used as a context manager, so you can temporarily use a custom table for a specific calculation.
See the documentation for :func:`use_custom_ssp_table` for more information.

"""

import functools
import os
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import pynbody

from .. import filt, snapshot, units, util

_ssp_table = None
_default_ssp_file = [os.path.join(os.path.dirname(__file__), "default_ssp.txt"),
                        os.path.join(os.path.dirname(__file__), "lsst_ssp.txt")]
class SSPTable:
    """An SSP table for interpolating magnitudes from stellar populations"""

    def __init__(self, ages, metallicities, magnitudes, case_insensitive=False, ignore_bands=None, is_ab_system=False):
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

        ignore_bands : list[str], optional
            List of bandpasses to ignore. These bandpasses will not be available in the table.

        is_ab_system : bool, optional
            If True, the magnitudes are in the AB system. If False, the magnitudes are in the Vega system.


        """
        self._ages = np.asarray(ages)
        self._metallicities = np.asarray(metallicities)

        self._case_insensitive = case_insensitive
        self._magnitudes = {k.lower() if self._case_insensitive else k: np.asarray(v) for k, v in magnitudes.items()}
        self._is_ab_system = is_ab_system
        if ignore_bands is not None:
            for band in ignore_bands:
                if self._case_insensitive:
                    band = band.lower()
                self._magnitudes.pop(band, None)

    def __repr__(self):
        return f"<{type(self).__name__}; bands={', '.join(self.bands)}>"

    @property
    def bands(self):
        """List of bandpasses available in this table"""
        return list(self._magnitudes.keys())

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

        nan_mask = np.isnan(metallicities) | np.isnan(ages)

        ages[nan_mask ] = self._ages[0]
        metallicities[nan_mask] = self._metallicities[0]

        interpolator = RegularGridInterpolator((self._ages, self._metallicities), self._magnitudes[band].T,
                                               method='linear', bounds_error=False, fill_value=np.nan)
        result = interpolator(np.array([ages, metallicities]).T)
        result[nan_mask] = np.nan

        return result

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
        age_star[age_star<1.0] = 1.0
        metals = snapshot['metals']
        try:
            masses = snapshot['massform'].in_units('Msol')
        except KeyError:
            masses = snapshot['mass'].in_units('Msol')

        with np.errstate(invalid='ignore'):
            output_mags = self.interpolate(np.log10(age_star), metals, band)


        vals = output_mags - 2.5 * np.log10(masses)
        vals = vals.view(pynbody.array.SimArray)
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

        table = {'u': 3598.54, 'b': 4385.92, 'v': 5490.56, 'g': 4858.82,
                'r': 6594.72, 'i': 8059.88, 'z': 8669.25, 'j': 12369.26,
                'h': 16464.45, 'k': 22105.45, 'y': 9738.60}

        # we always treat band names as case-insensitive for getting a wavelength
        band = band.lower()
        if band in table:
            return table[band]
        else:
            raise ValueError("The central wavelength is not known for this band")

    def get_flux_normalization(self, band):
        """Get the normalization factor for converting relative magnitudes to fluxes

        The normalization factor is the flux of a star with a magnitude of 0 in the given bandpass.
        Currently, this is only implemented for AB-calibrated systems, where the normalization
        factor is 3631 Jy."""

        if self._is_ab_system:
            return 3631. * units.Jy
        else:
            raise ValueError("The flux normalization is not known for this band")

    def get_spectral_density_normalization(self, band):
        """Get the normalization factor for converting absolute magnitudes to spectral density

        This uses the flux normalization and then (as per definition of absolute magnitude)
        considers the source to be at a distance of 10 pc. The output units are
        power per unit frequency (or, eqivalently, energy).
        """

        return self.get_flux_normalization(band) * (4 * np.pi * (10 * units.pc)**2)


    @classmethod
    def _clamp_value(cls, value, values):
        value = np.atleast_1d(value)
        value[value <= np.min(values)] = np.nextafter(np.min(values), np.inf)
        value[value >= np.max(values)] = np.nextafter(np.max(values), -np.inf)
        return value

class MultiSSPTable(SSPTable):
    """Combines multiple SSP tables, each of which must offer different bandpasses"""
    def __init__(self, *tables):
        self._bandpass_to_table = {}
        for table in tables:
            for band in table.bands:
                if band in self._bandpass_to_table:
                    raise ValueError(f"Bandpass {band} is present in multiple tables")
                self._bandpass_to_table[band] = table

    @property
    def bands(self):
        return list(self._bandpass_to_table.keys())

    def interpolate(self, ages, metallicities, band):
        return self._bandpass_to_table[band].interpolate(ages, metallicities, band)

    def __call__(self, snapshot, band):
        return self._bandpass_to_table[band](snapshot, band)

    def get_central_wavelength(self, band):
        return self._bandpass_to_table[band].get_central_wavelength(band)

    def get_flux_normalization(self, band):
        return self._bandpass_to_table[band].get_flux_normalization(band)

    def get_power_normalization(self, band):
        return self._bandpass_to_table[band].get_power_normalization(band)


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
                         {k: data[k] for k in data.files if k not in ['ages', 'mets']},
                         case_insensitive=True)

class StevSSPTable(SSPTable):
    """An SSP table from the output of the STEV/CMD web interface"""

    def __init__(self, path, ignore_bands=None):
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

        is_ab_system = any('ABmags' in l for l in lines)

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
                          for k in column_names if k.endswith('mag')},
                         ignore_bands=ignore_bands, is_ab_system=is_ab_system)


def _load_ssp_table(path_or_table):
    if isinstance(path_or_table, list) or isinstance(path_or_table, tuple):
        return MultiSSPTable(*(_load_ssp_table(p) for p in path_or_table))
    if isinstance(path_or_table, SSPTable):
        return path_or_table
    elif isinstance(path_or_table, str):
        if path_or_table.endswith('.npz'):
            return ArchivedSSPTable(path_or_table)
        else:
            return StevSSPTable(path_or_table, ignore_bands=["mbol"])
    else:
        raise ValueError("Invalid path or table")

def get_current_ssp_table() -> SSPTable:
    """Get the current preferred SSP table for calculating magnitudes

    This will either be pynbody's default table or a custom table specified by :func:`use_custom_ssp_table`."""
    global _ssp_table
    if _ssp_table is None:
        _ssp_table = _load_ssp_table(_default_ssp_file)
    return _ssp_table

class SSPTableContext(util.SettingControl):
    """Context manager for temporarily using a custom SSP table"""
    def __init__(self, ssp_table):
        super().__init__(globals(), "_ssp_table", ssp_table)

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

    if cmd_path is None:
        table = get_current_ssp_table()
    else:
        table = _load_ssp_table(cmd_path)

    return table(simstars, band)


def halo_mag(sim, band='V'):
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


def halo_lum(sim, band, physical_units=True):
    """Calculate a total spectral density in a given bandpass for a halo.

    Note that this requires an absolute calibration to be known for the requested bandpass,
    which presently is only coded for the AB system.

    Spectral density is here defined as the power emitted per unit frequency.

    Parameters
    ----------

    sim : pynbody.SimSnap
        Halo (or other subsnap, or even a whole simulation) for which to calculate the luminosity.
        Any non-star particles are ignored.

    band : str
        Bandpass name. Must be in the current SSP table and have an absolute calibration (unless
        normalized=False).

    physical_units : bool
        If True, the luminosity is normalized with physical units. This requires an absolute calibration
        to be known for the requested bandpass. If False, the luminosity is normalized to a reference
        star with a magnitude of 0 in the given bandpass.
    """

    if physical_units:
        norm = get_current_ssp_table().get_spectral_density_normalization(band)
    else:
        norm = 1.0
    return np.sum(10.0 ** ((- sim.star[band + '_mag']) / 2.5)) * norm


def half_light_r(sim, band='V', cylindrical=False):
    '''Calculate half-light radius

    Calculates entire luminosity of the provided snapshot, finds half that, sorts
    stars by distance from halo center, and finds out inside which radius the half luminosity
    is reached.

    Parameters
    ----------

    sim : pynbody.SimSnap
        Halo (or other subsnap, or even a whole simulation) for which to calculate the half-light radius.
        Any non-star particles are ignored.

    band : str
        Bandpass name. Can be any that is defined in the SSP table. See the module documentation
        (:mod:`pynbody.analysis.luminosity`).

    cylindrical : bool
        If True, the radius is calculated in the cylindrical xy-plane coordinates.

    '''
    half_l = halo_lum(sim, band=band, physical_units=False) * 0.5

    if cylindrical:
        coord = 'rxy'
    else:
        coord = 'r'
    max_high_r = np.max(sim.star[coord])
    test_r = 0.5 * max_high_r
    testrf = filt.LowPass(coord, test_r)
    min_low_r = 0.0
    test_l = halo_lum(sim[testrf], band=band, physical_units=False)
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
        test_l = halo_lum(sim[testrf], band=band, physical_units=False)

        if (test_l > half_l):
            max_high_r = test_r
        else:
            min_low_r = test_r

    return test_r


def _setup_derived_arrays():

    bands_available = 'UBVRIJHKugrizy'

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
