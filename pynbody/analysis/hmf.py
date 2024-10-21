"""
Routines related to halo mass function and other cosmological calculations.

.. warning ::
    These routines were implemented in 2012 when there were no other python libraries that could do this.
    Since then, cosmology-focussed libraries such as `hmf <https://hmf.readthedocs.io/en/latest/index.html>`_,
    `Colossus <https://bdiemer.bitbucket.io/colossus/index.html>`_ and
    `CCL <https://ccl.readthedocs.io/en/latest/>`_ have been developed.
    For precision cosmology applications, we recommend using these libraries, since the routines here
    have not been tested for precision applications (and are known to use approximations).

"""

import copy
import logging
import math
import os
import warnings

import numpy as np
import scipy
import scipy.interpolate

import pynbody

from .. import configuration, units, util
from . import cosmology

logger = logging.getLogger('pynbody.analysis.hmf')


#######################################################################
# Filters
#######################################################################

class FieldFilter:
    """Represents a filter acting on a field"""

    def M_to_R(self, M):
        """Return the mass scale (Msol h^-1) for a given length (Mpc h^-1 comoving)"""
        return (M / (self.gammaF * self.rho_bar)) ** 0.3333

    def R_to_M(self, R):
        """Return the length scale (Mpc h^-1 comoving) for a given spherical mass (Msol h^-1)"""
        return self.gammaF * self.rho_bar * R ** 3

    @staticmethod
    def Wk(kR):
        """Return the Fourier-space filter function, as a function of kR where R is the scale of the filter"""
        raise NotImplementedError("Not implemented")


class TophatFilter(FieldFilter):
    """A top-hat filter in real space"""
    def __init__(self, context):
        self.gammaF = 4 * math.pi / 3
        self.rho_bar = cosmology.rho_M(context, unit="Msol Mpc^-3 h^2 a^-3")

    @staticmethod
    def Wk(kR):
        return 3 * (np.sin(kR) - kR * np.cos(kR)) / (kR) ** 3


class GaussianFilter(FieldFilter):
    """A Gaussian filter"""

    def __init__(self, context):
        self.gammaF = (2 * math.pi) ** 1.5
        self.rho_bar = cosmology.rho_M(context, unit="Msol Mpc^-3 h^2 a^-3")

    @staticmethod
    def Wk(kR):
        return np.exp(-(kR) ** 2 / 2)


class HarmonicStepFilter(FieldFilter):
    """A step filter in harmonic space"""

    def __init__(self, context):
        self.gammaF = 6 * math.pi ** 2
        self.rho_bar = cosmology.rho_M(context, unit="Msol Mpc^-3 h^2 a^-3")

    @staticmethod
    def Wk(kR):
        return (kR < 1)

#######################################################################
# Power spectrum management / normalization
#######################################################################


class PowerSpectrum:
    """A power spectrum object, which can be called to return the power spectrum at a given wavenumber."""

    def __init__(self, context, filename=None, k=None, Pk=None, log_interpolation=True):
        """Set up a power spectrum object.

        Either a tabulated power spectrum or the filename of a tabulated power spectrum can be passed.
        If neither is provided, a default power spectrum is used which assumes Planck 2018 values.

        Parameters
        ----------

        context : SimSnap
            The simulation snapshot from which to pull the cosmological context. This is used for
            the linear growth factor and sigma8 normalization.

        filename : str, optional
            The name of a file containing the power spectrum.

        k : array, optional
            The wavenumbers of the power spectrum in comoving h/Mpc

        Pk : array, optional
            The power spectrum in Mpc^3 h^-3

        log_interpolation : bool, optional
            If True, interpolate in log space. This is generally more accurate for power spectra.

        """

        if k is None or Pk is None:
            if filename is None:
                warnings.warn(
                    "Using the default power-spectrum spectrum which assumes Planck 2018 values: "
                    "Omega_c h^2 = 0.120; Omega_b h^2 = 0.0224; h = 0.676; n_s = 0.9667. Sigma8 will be correctly scaled to the simulation value. ",
                    RuntimeWarning)
                filename = os.path.join(os.path.dirname(__file__), "CAMB_Planck18")
            k, Pk = np.loadtxt(filename, unpack=True)

        self._orig_k_min = k.min()
        self._orig_k_max = k.max()

        bot_k = 1.e-4

        if k[0] > bot_k:
            # extrapolate out
            n = math.log10(Pk[1] / Pk[0]) / math.log10(k[1] / k[0])

            Pkinterp = 10 ** (math.log10(Pk[0]) - math.log10(k[0] / bot_k) * n)
            k = np.hstack((bot_k, k))
            Pk = np.hstack((Pkinterp, Pk))

        top_k = 1.e4

        if k[-1] < top_k:
            # extrapolate out
            n = math.log10(Pk[-1] / Pk[-2]) / math.log10(k[-1] / k[-2])

            Pkinterp = 10 ** (math.log10(Pk[-1]
                                         ) - math.log10(k[-1] / top_k) * n)
            k = np.hstack((k, top_k))
            Pk = np.hstack((Pk, Pkinterp))

        self.k = k.view(pynbody.array.SimArray)
        self.k.units = "Mpc^-1 h a^-1"

        self.Pk_z0_unnormalised = Pk.view(pynbody.array.SimArray)
        self.Pk_z0_unnormalised.units = "Mpc^3 h^-3"

        self.k.sim = context
        self.Pk_z0_unnormalised.sim = context

        self._lingrowth = 1

        if context.properties['z'] != 0:
            self._lingrowth = cosmology.linear_growth_factor(context) ** 2

        self._default_filter = TophatFilter(context)
        self.min_k = self.k.min()*1.000001
        self.max_k = self.k.max()*0.999999
        self._norm = 1

        self._log_interp = log_interpolation

        self._init_interpolation()

        self.set_sigma8(context.properties['sigma8'])

    def _init_interpolation(self):
        if self._log_interp:
            self._interp = scipy.interpolate.interp1d(
                np.log(self.k), np.log(self.Pk_z0_unnormalised))
        else:
            self._interp = scipy.interpolate.interp1d(np.log(self.k), self.Pk_z0_unnormalised)

    def set_sigma8(self, sigma8):
        """Update the normalisation for a given sigma8 value at z=0"""
        current_sigma8_2 = self.get_sigma8() ** 2
        self._norm *= sigma8 ** 2 / current_sigma8_2

    def get_sigma8(self):
        """Calculate the current sigma8 value at z=0"""
        current_sigma8 = math.sqrt(
            variance(8.0, self._default_filter, self, True) / self._lingrowth)
        return current_sigma8

    def __call__(self, k):
        """Evaluate the power spectrum at the snapshot redshift"""
        if self._log_interp:
            return self._norm * self._lingrowth * np.exp(self._interp(np.log(k)))
        else:
            return self._norm * self._lingrowth * self._interp(np.log(k))


class PowerSpectrumCAMB(PowerSpectrum):
    """A power spectrum object that calculates the power spectrum on the fly using CAMB.

    This is slower than using a precomputed power spectrum, but is more flexible."""

    def __init__(self, context, use_context=True, camb_params={}, log_interpolation=True):
        """Runs CAMB to calculate a power spectrum for the provided context.

        Parameters
        ----------
        context : SimSnap
            The simulation snapshot from which to pull the cosmological context.

        use_context : bool, optional
            If True, use the context to set the cosmological parameters in CAMB.
            If False, use only the parameters in ``camb_params``.

        camb_params : dict, optional
            A dictionary of parameters to pass to CAMB's ``set_cosmology`` method. Note that the
            scalar spectral index ``ns`` is not included in ``set_cosmology``; if provided here, it is
            extracted and passed to ``InitPower.set_params``.

        log_interpolation : bool, optional
            If True, interpolate in log space. This is generally more accurate for power spectra.

        """

        try:
            import camb
        except ImportError:
            warnings.warn("The camb module is not installed. Falling back to using a pre-computed power spectrum"
                          "which may not be appropriate for your cosmology.")
            super().__init__(context)
            return

        camb_params_obj = camb.CAMBparams()
        camb_params = copy.copy(camb_params)

        # default value for ns, in case use_context is false:
        ns = configuration.config_parser.get('default-cosmology', 'ns')

        if use_context:
            h0 = context.properties['h']
            omBh2 = context.properties['omegaB0'] * h0 ** 2
            omMh2 = context.properties['omegaM0'] * h0 ** 2
            if omBh2 == 0.0 and 'ombh2' not in camb_params_obj:
                warnings.warn("OmegaB0 is zero, presumably because this is a DMO run. For the power spectrum, setting ombh2 to 0.0224 (Planck 2018)."
                              "To override, set 'omegaB0' in the simulation parameters, or 'ombh2' in the camb_params argument.")
                omBh2 = 0.0224
            OmCh2 = omMh2 - omBh2

            if 'H0' not in camb_params:
                camb_params['H0'] = h0 * 100
            if 'ombh2' not in camb_params:
                camb_params['ombh2'] = omBh2
            if 'omch2' not in camb_params:
                camb_params['omch2'] = OmCh2

            ns = context.properties['ns']

        if 'ns' in camb_params:
            ns = camb_params.pop('ns')

        camb_params_obj.set_cosmology(**camb_params)

        camb_params_obj.InitPower.set_params(ns=ns)

        camb_params_obj.WantTransfer = True

        # always get the matter power spectrum at z=0; growth factor will be scaled later if needed
        camb_params_obj.set_matter_power(redshifts=[0.0], kmax=1e2, k_per_logint=0,
                                     silent=True, nonlinear=False)
        results = camb.get_results(camb_params_obj)

        k, z, Pk = results.get_matter_power_spectrum(minkh=1.e-4, maxkh=1e2, npoints=1000)

        # filename = os.path.join(os.path.dirname(__file__), "CAMB_Planck18")
        # np.savetxt(filename, np.array([k, Pk[0]]).T)

        assert len(z)==1
        assert z[0] == 0.0

        super().__init__(context, k=k, Pk=Pk[0], log_interpolation=log_interpolation)



class BiasedPowerSpectrum(PowerSpectrum):
    """A power spectrum object with linear bias"""

    def __init__(self, bias, pspec):
        """Set up a biased power spectrum.

        Parameters
        ----------
        bias : float  or function
            Either a constant linear bias factor, or a function of wavenumber that returns the bias at that wavenumber.

        pspec : PowerSpectrum
            The power spectrum object to wrap with a bias
        """

        if not hasattr(bias, '__call__'):
            bias = lambda x: bias

        self._bias = bias
        self._pspec = pspec
        self._norm = 1.0
        self.min_k = pspec.min_k
        self.max_k = pspec.max_k
        self.k = pspec.k
        self.Pk_z0_unnormalised = pspec.Pk_z0_unnormalised * self._bias(self.k) ** 2

    def __call__(self, k):
        return self._norm * self._pspec(k) * self._bias(k) ** 2


#######################################################################
# Variance calculation
#######################################################################

def variance(M_or_R, f_filter=TophatFilter, powspec=PowerSpectrum, arg_is_R=False):
    """Calculate the variance of the density field smoothed on a mass scale M, or optionally a length scale R.

    Parameters
    ----------
    M_or_R : float or array
        The mass scale in Msol/h or the radius of the filter in Mpc/h comoving. If an array, the
        variance is calculated for each element of the array.

    f_filter : FieldFilter, optional
        The filter to use. Default is a top-hat filter.

    powspec : PowerSpectrum, optional
        The power spectrum object to use. Default is a Planck 2018 power spectrum at z=0.

    arg_is_R : bool, optional
        If True, interpret the input as a length scale R rather than a mass scale M.

    Returns
    -------
    float or array
        The variance of the density field smoothed on the given scale(s).

    """
    if hasattr(M_or_R, '__len__'):
        ax = pynbody.array.SimArray(
            [variance(Mi, f_filter, powspec, arg_is_R) for Mi in M_or_R])
        # hopefully dimensionless
        ax.units = powspec.Pk_z0_unnormalised.units * powspec.k.units ** 3
        return ax

    if arg_is_R:
        R = M_or_R
    else:
        R = f_filter.M_to_R(M_or_R)

    integrand = lambda k: k ** 2 * powspec(k) * f_filter.Wk(k * R) ** 2
    integrand_ln_k = lambda k: np.exp(k) * integrand(np.exp(k))
    v =  scipy.integrate.quad(integrand_ln_k, math.log(powspec.min_k), math.log(
        1. / R) + 3, epsrel=1.e-6)[0] / (2 * math.pi ** 2)

    return v


def get_neffm(mass, sigma):
    """Calculate the effective spectral index of the power spectrum at a given mass scale."""
    dlnm = np.diff(np.log(mass))
    dlnsigmainv = np.diff(np.log(1. / sigma))
    neff = 6. * dlnsigmainv / dlnm - 3.
    return neff


@units.takes_arg_in_units((0, "Mpc h^-1"))
def correlation(r, powspec=PowerSpectrum):
    """Calculate the correlation function of the density field at a specified radius."""

    if hasattr(r, '__len__'):
        ax = pynbody.array.SimArray([correlation(ri,  powspec) for ri in r])
        ax.units = powspec.Pk_z0_unnormalised.units * powspec.k.units ** 3
        return ax

    # Because sin kr becomes so highly oscilliatory, normal
    # quadrature is slow/inaccurate for this problem. The following
    # is the best way I could come up with to overcome that.
    #
    # For small kr, sin kr/kr is represented as a Taylor expansion and
    # each segment of the power spectrum is integrated over, summing
    # over the Taylor series to convergence.
    #
    # When the convergence of this starts to fail, each segment of the
    # power spectrum is still represented by a power law, but the
    # exact integral boils down to a normal incomplete gamma function
    # extended into the complex plane.
    #
    # Originally, we had:
    #
    # integrand = lambda k: k**2 * powspec(k) * (np.sin(k*r)/(k*r))
    # integrand_ln_k = lambda k: np.exp(k)*integrand(np.exp(k))
    #

    tot = 0
    defer = False

    k = powspec.k

    gamma_method = False

    for k_bot, k_top in zip(k[:-1], k[1:]):

        if k_bot >= k_top:
            continue

        # express segment as P(k) = P0*k^n
        Pk_top = powspec(k_top)
        Pk_bot = powspec(k_bot)

        n = np.log(Pk_top / Pk_bot) / np.log(k_top / k_bot)
        P0 = Pk_top / k_top ** n

        if n != n or abs(n) > 2:
            # looks nasty in log space, so interpolate linearly instead
            grad = (Pk_top - Pk_bot) / (k_top - k_bot)
            segment = ((-2 * grad + k_bot * Pk_bot * r ** 2) * np.cos(k_bot * r) +
                       (2 * grad - k_top * (-(grad * k_bot) + grad * k_top + Pk_bot) * r ** 2) * np.cos(k_top * r) -
                       (grad * k_bot + Pk_bot) * r * np.sin(k_bot * r) +
                       (-(grad * k_bot) + 2 * grad * k_top + Pk_bot) * r * np.sin(k_top * r)) / r ** 4

        elif k_top * r < 6.0 and not gamma_method:
            # approximate sin y/y as polynomial = \sum_m coeff_m y^m

            segment = 0
            term = 0

            m = 0
            coeff = 1
            while m == 0 or (abs(term / segment) > 1.e-7 and m < 50):
                if m > 0:
                    coeff *= (-1.0) / (m * (m + 1))

                # integral is P0 * r^m * int_(k_bot)^(k_top) k^(2+n+m) dk = P0
                # r^m [k^(3+n+m)/(3+n+m)]
                top_val = k_top ** (3 + n + m) / (3 + n + m)
                bot_val = k_bot ** (3 + n + m) / (3 + n + m)
                term = P0 * (r ** m) * (top_val - bot_val) * coeff
                segment += term
                m += 2

            if m >= 50:
                raise RuntimeError("Convergence failure in sin y/y series integral")

            if m > 18:
                gamma_method = True
                # experience suggests when you have to sum beyond m=18, it's faster
                # to switch to the method below

        else:

            # now integral of this segment is exactly
            # P0 * int_(k_bot)^(k_top) k^(2+n) sin(kr)/(kr) = (P0/r^(n+3)) Im[ (i)^(-n-2) Gamma(n+2,i k_bot r, i k_top r)]
            # First we need to evaluate the Gamma integral sufficiently
            # accurately

            top_val = util.gamma_inc(n + 2, (1.0j) * r * k_top)
            bot_val = util.gamma_inc(n + 2, (1.0j) * r * k_bot)
            segment = - \
                ((1.0j) ** (-n - 2) * P0 *
                 (top_val - bot_val) / r ** (n + 3)).imag

        tot += segment

    tot /= (2 * math.pi ** 2)

    return tot


def correlation_func(context, log_r_min=-3, log_r_max=2, delta_log_r=0.2,
                     pspec=PowerSpectrumCAMB):
    """
    Calculate the linear density field correlation function.

    Parameters
    ----------

    context : SimSnap
        The snapshot from which to pull the cosmological context

    log_r_min : float, optional
        log10 of the minimum separation (Mpc h^-1) to consider

    log_r_max : float, optional
        log10 of the maximum separation (Mpc h^-1) to consider

    delta_log_r : float, optional
        The value spacing in dex

    pspec : PowerSpectrum, optional
        The power spectrum class to use, or if an instance is provided, the power spectrum to use.
        The default is to use PowerSpectrumCAMB which will attempt to calculate a power spectrum
        from the simulation context; if it fails, it will fall back to using Planck2018 power spectrum
        and issue a warning.

    Returns
    -------

    r : array
        The separation values in Mpc h^-1

    Xi : array
        The correlation function at each separation


    """

    if isinstance(pspec, type):
        pspec = pspec(context)

    r = (10.0 ** np.arange(log_r_min, log_r_max + delta_log_r / 2,
                           delta_log_r)).view(pynbody.array.SimArray)
    r.sim = context
    r.units = "Mpc h^-1 a"

    Xi_r = np.array([correlation(ri, pspec)
                     for ri in r]).view(pynbody.array.SimArray)
    Xi_r.sim = context
    Xi_r.units = ""

    return r, Xi_r

#######################################################################
# Default kernels for halo mass function
#######################################################################


def _f_press_schechter(nu):
    """

    The Press-Schechter kernel used by halo_mass_function

    """

    f = math.sqrt(2. / math.pi) * nu * np.exp(-nu * nu / 2.)
    return f


def _f_sheth_tormen(nu, Anorm=0.3222, a=0.707, p=0.3):
    """
    Sheth & Tormen (1999) fit (see also Sheth Mo & Tormen 2001)
    """
    #  Anorm: normalization, set so all mass is in halos (integral [f nu dn]=1)
    #  a: affects mainly the number of massive halo,
    #  a=0.75 is favored by Sheth & Tormen (2002)

    f = Anorm * math.sqrt(2. * a / math.pi) * \
        (1. + np.power((1. / a / nu / nu), p))
    f *= nu * np.exp(-a * nu * nu / 2.)
    return f


def _f_jenkins(nu, deltac=1.68647):
    # Jenkins et al (2001) fit   ##  valid for  -1.2 << ln(1/sigma) << 1.05
    sigma = deltac / nu
    lnsigmainv = np.log(1. / sigma)
    if ((np.any(lnsigmainv < -1.2)) or (np.any(lnsigmainv > 1.05))):
        logger.warning(
            "Jenkins mass function is outsie of valid mass range.  Continuing calculations anyway.")
    f = 0.315 * np.exp(-np.power((np.fabs(lnsigmainv + 0.61)), 3.8))
    return f


def _f_warren(nu, deltac=1.68647):
    #  Warren et al. 2006  -- valid for (10**10 - 10**15 Msun/h)
    sigma = deltac / nu
    A = 0.7234
    a = 1.625
    b = 0.2538
    c = 1.1982
    f = A * (np.power(sigma, -a) + b) * np.exp(-c / sigma ** 2)
    return f


def _f_reed_no_z(nu, deltac=1.68647):  # universal form
    # Reed et al. (2007) fit, eqn. 9 -- with no redshift depedence (simple
    # universal form)
    """ modified S-T fit  by the G1 gaussian term and c"""
    sigma = deltac / nu
    # normalization that all mass is in halos not strictly conserved here
    Anorm = 0.3222
    a = 0.707  # affects mostly the number of massive halos
    # a=0.75    #  favored by Sheth & Tormen (2002)
    p = 0.3
    c = 1.08
    nu = deltac / sigma
    lnsigmainv = np.log(1. / sigma)
    G1 = np.exp(-np.power((lnsigmainv - 0.4), 2) / (2. * 0.6 * 0.6))
    f = Anorm * np.sqrt(2. * a / np.pi) * \
        (1. + np.power((1. / a / nu / nu), p) + 0.2 * G1)
    f *= nu * np.exp(-c * a * nu * nu / 2.)
    return f


def _f_reed_z_evo(nu, neff, deltac=1.68647):  # non-universal form
    # Reed et al. (2007) fit, eqn. 11 -- with redshift depedence for accuracy
    # at z >~ z_reion
    """ modified S-T fit  by the n_eff dependence and the G1 and G2 gaussian terms and c
    where   P(k) proportional to k_halo**(n_eff)  and
    k_halo = Mhalo / r_halo_precollapse.
    eqn 13 of Reed et al 2007   estimtes neff = 6 d ln(1/sigma(M))/ d ln M  - 3 """
    sigma = deltac / nu
    # normalization that all mass is in halos not strictly conserved here
    Anorm = 0.3222
    a = 0.707  # affects mostly the number of massive halos
    # a=0.75    #  favored by Sheth & Tormen (2002)
    p = 0.3
    c = 1.08
    nu = deltac / sigma
    lnsigmainv = np.log(1. / sigma)
    G1 = np.exp(- np.power((lnsigmainv - 0.4), 2) / (2. * 0.6 * 0.6))
    G2 = np.exp(- np.power((lnsigmainv - 0.75), 2) / (2. * 0.2 * 0.2))
    f = Anorm * np.sqrt(2. * a / np.pi) * (1. +
                                           np.power((1. / a / nu / nu), p) + 0.6 * G1 + 0.4 * G2)
    f *= nu * \
        np.exp(-c * a * nu * nu / 2. - 0.03 / (neff + 3)
               ** 2 * np.power(nu, 0.6))
    return f


def _f_bhattacharya(nu, red, deltac=1.68647):
    # Bhattacharya et al. 2010  -- 6x10**11 - 310**15 Msun/h  z=0-2
    sigma = deltac / nu
    A = 0.333 / pow((1. + red), 0.11)
    a = 0.788 / pow((1. + red), 0.01)
    p = 0.807 / pow((1. + red), 0.0)
    q = 1.795 / pow((1. + red), 0.0)
    f = A * np.sqrt(2. / np.pi) * (1. + np.power((1. / a / nu / nu), p))
    f *= np.power(nu * math.sqrt(a), q) * np.exp(-a * nu * nu / 2.)
    return f

def _cole_kaiser_bias(nu, delta_c):
    """

    The Cole-Kaiser (1989) bias function. Also in Mo & White 1996.

    """
    return 1 + (nu ** 2 - 1) / delta_c


def _sheth_tormen_bias(nu, delta_c,
                       a=0.707, b=0.5, c=0.6):
    """

    The Sheth-Tormen (1999) bias function [eq 8]

    """

    root_a = math.sqrt(a)

    return 1. + (root_a * a * nu ** 2 + root_a * b * (a * nu ** 2) ** (1. - c)
                 - (a * nu ** 2) ** c / ((a * nu ** 2) ** c + b * (1 - c) * (1 - c / 2))) \
        / (root_a * delta_c)



def halo_mass_function(context,
                       log_M_min=8.0, log_M_max=15.0, delta_log_M=0.1,
                       kern="ST",
                       pspec=PowerSpectrum,
                       delta_crit=1.686):
    """
    Returns the halo mass function, dN/d log_{10} M in units of Mpc^-3 h^3.

    See :ref:`hmf_tutorial` for an example and more information.


    Parameters
    ----------

    context : SimSnap
        The snapshot from which to pull the cosmological context. Sigma8 normalization and growth function
        redshift dependence is always taken into account. If `camb <https://camb.readthedocs.io/en/latest/>`_ is
        installed, an appropriate power spectrum is calculated for other cosmological parameters (e.g. baryon
        density, Hubble constant) from the simulation. Otherwise, a default Planck18 power spectrum is
        used.

    log_M_min : float, optional
        The minimum halo mass (Msol h^-1) to consider. Default is 8.

    log_M_max : float, optional
        The maximum halo mass (Msol h^-1) to consider. Default is 15.

    delta_log_M : float, optional
        The bin spacing of halo masses. Default is 0.1.

    kern : str or function, optional
        The kernel function which dictates what type of mass function to calculate. Default is "ST" (Sheth-Tormen).
        Other options are "PS" (Press-Schechter), "J" (Jenkins), "W" (Warren), "REEDZ" (Reed et al 2007 with redshift
        dependence), "REEDU" (Reed et al 2007 without redshift dependence), "B" (Bhattacharya).

    pspec : PowerSpectrumCAMB, optional
        A power spectrum object (which also defines the window function), overriding the cosmological
        parameters of the simulation.

    delta_crit : float, optional
        The critical overdensity for collapse. Default is 1.686.

    Returns
    -------

    M : SimArray
        The centre of the mass bins, in Msol h^-1.

    sigma : SimArray
        The linear variance of the corresponding sphere.

    N : SimArray
        The abundance of halos of that mass (Mpc^-3 h^3 comoving, per decade of mass).

    Notes
    -----

    Because numerical derivatives are involved, the value of ``delta_log_M`` affects the accuracy.

    The halo mass function code in pynbody was implemented in 2012 when there
    were no other python libraries that could do this.
    Since then, cosmology-focussed libraries such as `hmf <https://hmf.readthedocs.io/en/latest/index.html>`_,
    and `CCL <https://halotools.readthedocs.io/en/latest/>`_ have been developed.
    For precision cosmology applications, we recommend using these libraries.

    The functionality here is retained for quick cross-checks of simulations.

    """

    if isinstance(kern, str):
        kern = {'PS': _f_press_schechter,
                'ST': _f_sheth_tormen,
                'J': _f_jenkins,
                'W': _f_warren,
                'REEDZ': _f_reed_z_evo,
                # Reed et al 2007 without redshift dependence
                'REEDU': _f_reed_no_z,
                'B': _f_bhattacharya}[kern]

    rho_bar = cosmology.rho_M(context, unit="Msol Mpc^-3 h^2")
    red = context.properties['z']  # redshift is always set by simulation

    M = np.arange(log_M_min, log_M_max, delta_log_M)
    M_mid = np.arange(
        log_M_min + delta_log_M / 2, log_M_max - delta_log_M / 2, delta_log_M)

    if isinstance(pspec, type):
        pspec = pspec(context)

    sig = variance(10 ** M, pspec._default_filter, pspec)  # sigma(m)**2

    nu = delta_crit / np.sqrt(sig)
    nu.units = "1"

    nu_mid = (nu[1:] + nu[:-1]) / 2

    d_ln_nu_d_ln_M = np.diff(np.log10(nu)) / delta_log_M

    dM = np.diff(10 ** M)

    if (kern == _f_reed_z_evo):
        neff = get_neffm(10. ** M, sig ** 0.5)
        out = (rho_bar / (10 ** M_mid)) * kern(nu_mid, neff) * \
            d_ln_nu_d_ln_M * math.log(10.) * context.properties['a'] ** 3
    elif (kern == _f_bhattacharya):
       # eq 7.46, Mo, van den Bosch and White
        out = (rho_bar / (10 ** M_mid)) * kern(nu_mid, red) * \
            d_ln_nu_d_ln_M * math.log(10.) * context.properties['a'] ** 3
    else:
        # eq 7.46, Mo, van den Bosch and White
        out = (rho_bar / (10 ** M_mid)) * kern(nu_mid) * \
            d_ln_nu_d_ln_M * math.log(10.) * context.properties['a'] ** 3
    out= out.view(pynbody.array.SimArray)
    out.units = "Mpc^-3 h^3 a^-3"
    out.sim = context

    M_mid = (10 ** M_mid).view(pynbody.array.SimArray)
    M_mid.units = "Msol h^-1"
    M_mid.sim = context

    # interpolate sigma for output checking purposes
    sig = (sig[1:] + sig[:-1]) / 2

    return M_mid, np.sqrt(sig), out


@units.takes_arg_in_units((1, "Msol h^-1"), context_arg=0)
def halo_bias(context, M, kern="CK", pspec=PowerSpectrumCAMB,
              delta_crit=1.686):
    """
    Return the halo bias for the given halo mass.

    Parameters
    ----------

    context : SimSnap
        The snapshot from which to pull the cosmological context

    M : float
        The halo mass in Msol h^-1

    kern : str or function, optional
        The kernel function describing the halo bias. Default is "CK" (Cole-Kaiser).
        Other options are "ST" (Sheth-Tormen). Alternatively a function can be provided.

    pspec : PowerSpectrum, optional
        The power spectrum class to use, or if an instance is provided, the power spectrum to use.
        Default is PowerSpectrumCAMB which will attempt to calculate a power spectrum from the simulation context;
        if it fails, it will fall back to using Planck2018 power spectrum and issue a warning.

    delta_crit : float, optional
        The critical overdensity for collapse. Default is 1.686.

    Returns
    -------

    float
        The halo bias for the given halo mass.

    """

    if isinstance(kern, str):
        kern = {'CK': _cole_kaiser_bias,
                'ST': _sheth_tormen_bias}[kern]

    if isinstance(pspec, type):
        pspec = pspec(context)

    sig = variance(M, pspec._default_filter, pspec)
    nu = delta_crit / np.sqrt(sig)

    return kern(nu, delta_crit)


def simulation_halo_mass_function(snapshot_or_cat,
                                  log_M_min=8.0, log_M_max=15.0, delta_log_M=0.1,
                                  masses=None, mass_property=None, calculate_err=True,
                                  subsample_catalogue=None):
    """Estimate a halo mass function from a simulation, via binning haloes in mass.

    Parameters
    ----------

    snapshot_or_cat : SimSnap or HaloCatalogue
        The snapshot from which to calculate the halo mass function, or the halo catalogue itself.
        If a snapshot is provided, ``.halos()`` is called to get the first available catalogue.

    log_M_min : float, optional
        The minimum halo mass (Msol h^-1) to consider. Default is 8.

    log_M_max : float, optional
        The maximum halo mass (Msol h^-1) to consider. Default is 15.

    delta_log_M : float, optional
        The bin spacing of halo masses. Default is 0.1.

    masses : array, optional
        Provide an array of halo masses in Msol h^-1. If None, this is calculated from the snapshot.

    mass_property : str, optional
        The property name giving the mass of each halo. If None, sums the mass in particles in each halo
        according to the particle data in the catalogue. Summing masses can be slow for large catalogues.

    calculate_err : bool, optional
        If True, estimates error bars according to Poisson statistics.

    subsample_catalogue : int, optional
        Subsample the halo catalogue by the specified factor. Not recommended except for debugging.

    Returns
    -------

    bin_centers : SimArray
        The centre of the mass bins, in Msol h^-1.

    num_den : SimArray
        The binned number density of haloes in this snapshot in comoving Mpc**-3 h**3 per decade of mass.

    err : SimArray
        The error on the number density, if ``calculate_err`` is True.

    """

    if isinstance(snapshot_or_cat, pynbody.snapshot.SimSnap):
        snapshot = snapshot_or_cat
        halo_catalogue = snapshot.halos()
    elif isinstance(snapshot_or_cat, pynbody.halo.HaloCatalogue):
        halo_catalogue = snapshot_or_cat
        snapshot = halo_catalogue.base
    else:
        raise TypeError("snapshot_or_cat must be a SimSnap or HaloCatalogue")


    #Check that the mass resolution is uniform, this method does not handle otherwise
    if len(set(snapshot.d['mass'])) > 1:
        warnings.warn( "The mass resolution of the snapshot is not uniform (e.g. zooms). This method"
                       "will not generate a correct HMF in this case.")

    nbins = int(1 + (log_M_max - log_M_min)/delta_log_M)
    bins = np.logspace(log_M_min, log_M_max, num=nbins, base=10)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    halo_catalogue.load_all()

    if subsample_catalogue is not None:
        halo_catalogue_underlying = halo_catalogue # keep alive for the lifetime of the function
        halo_catalogue = halo_catalogue[::subsample_catalogue]

    if masses is None:
        if mass_property is None:
            warnings.warn("Halo finder masses not provided. Calculating them (might take a while...)")
            masses = np.array([h['mass'].sum().in_units('1 h**-1 Msol') for h in halo_catalogue])
        else:
            masses = halo_catalogue.get_properties_all_halos(with_units=True)[mass_property]
            if units.has_unit(masses):
                masses = masses.in_units('1 h**-1 Msol')

    if np.amax(masses) > 10**log_M_max or np.amin(masses) < 10**log_M_min :
        warnings.warn("Your bin range does not encompass the full range of halo masses")

    # Calculate number of halos in each bin
    num_halos = np.histogram(masses, bins)[0]

    normalisation = (snapshot.properties['boxsize'].in_units(' a h**-1 Mpc', **snapshot.conversion_context()) ** 3)\
                    * delta_log_M

    if calculate_err:
        # Calculate error bars assuming Poisson distribution in each bin
        err = np.sqrt(num_halos )/normalisation

    num_halos = num_halos  / normalisation


    # Make sure units are consistent
    bin_centers = bin_centers.view(pynbody.array.SimArray)
    bin_centers.units = "Msol h**-1"
    bin_centers.sim = snapshot

    num_halos = num_halos.view(pynbody.array.SimArray)
    num_halos.units = "a**-3 Mpc**-3 h**3"
    num_halos.sim = snapshot

    if calculate_err:
        err = err.view(pynbody.array.SimArray)
        err.units = "a**-3 Mpc**-3 h**3"
        err.sim = snapshot
        return bin_centers, num_halos, err
    else:
        return bin_centers, num_halos
