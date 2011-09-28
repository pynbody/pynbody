import numpy as np
from . import cosmology
from .. import units
from .. import util
import pynbody
import os
import scipy, scipy.interpolate
import math
import warnings
import cmath

#######################################################################
# Filters
#######################################################################

class FieldFilter(object) :
    def __init__(self) :
        raise RuntimeError, "Cannot instantiate directly, use a subclass instead"
    
    def M_to_R(self, M) :
        """Return the mass scale (Msol h^-1) for a given length (Mpc h^-1 comoving)"""
        return (M/(self.gammaF*self.rho_bar))**0.3333

    def R_to_M(self, R) :
        """Return the length scale (Mpc h^-1 comoving) for a given spherical mass (Msol h^-1)"""
        return self.gammaF * self.rho_bar * R**3

    @staticmethod
    def Wk(kR) :
        raise RuntimeError, "Not implemented"
    
class TophatFilter(FieldFilter) :
    def __init__(self, context) :
        self.gammaF = 4*math.pi/3
        self.rho_bar = cosmology.rho_M(context, unit="Msol Mpc^-3 h^2 a^-3")

    @staticmethod
    def Wk(kR) :
        return 3*(np.sin(kR)-kR*np.cos(kR))/(kR)**3
    
class GaussianFilter(FieldFilter) :
    def __init__(self, context) :
        self.gammaF = (2*math.pi)**1.5
        self.rho_bar = cosmology.rho_M(context, unit="Msol Mpc^-3 h^2 a^-3")

    @staticmethod
    def Wk(kR) :
        return np.exp(-(kR)**2/2)

class HarmonicStepFilter(FieldFilter) :
    def __init__(self, context) :
        self.gammaF = 6*math.pi**2
        self.rho_bar = cosmology.rho_M(context, unit="Msol Mpc^-3 h^2 a^-3")

    @staticmethod
    def Wk(kR) :
        return (kR<1)
  
#######################################################################
# Power spectrum management / normalization
#######################################################################


class PowerSpectrumCAMB(object) :
    def __init__(self, context, filename=None) :
        if filename is None :
            warnings.warn("Using the default power-spectrum spectrum which assumes ns=0.96 and WMAP7+H0+BAO transfer function.", RuntimeWarning)
            filename = os.path.join(os.path.dirname(__file__),"CAMB_WMAP7")

        k, Pk = np.loadtxt(filename,unpack=True)

        self._orig_k_min = k.min()
        self._orig_k_max = k.max()
        
        bot_k = 1.e-5
        
        if k[0]>bot_k :
            # extrapolate out
            n = math.log10(Pk[1]/Pk[0])/math.log10(k[1]/k[0])
            
            
            Pkinterp = 10**(math.log10(Pk[0])-math.log10(k[0]/bot_k)*n)
            k = np.hstack((bot_k,k))
            Pk = np.hstack((Pkinterp,Pk))


      
        top_k = 1.e7
        
        if k[-1]<top_k :
            # extrapolate out
            n = math.log10(Pk[-1]/Pk[-2])/math.log10(k[-1]/k[-2])
            
            Pkinterp = 10**(math.log10(Pk[-1])-math.log10(k[-1]/top_k)*n)
            k = np.hstack((k,top_k))
            Pk = np.hstack((Pk,Pkinterp))

            
        self.k = k.view(pynbody.array.SimArray)
        self.k.units = "Mpc^-1 h"
        
        self.Pk = Pk.view(pynbody.array.SimArray)
        self.Pk.units = "Mpc^3 h^-3"

        
        
        self.k.sim = context
        self.Pk.sim = context
        

        self._lingrowth = 1
        
        if context.properties['z']!=0 :
            self._lingrowth =cosmology.linear_growth_factor(context)**2

        self._default_filter = TophatFilter(context)
        self.min_k = self.k.min()
        self.max_k = self.k.max()
        self._norm = 1
        
        self._interp = scipy.interpolate.interp1d(np.log(self.k), np.log(self.Pk))

        self.set_sigma8(context.properties['sigma8'])    
            

    def set_sigma8(self, sigma8) :
        current_sigma8_2 = self.get_sigma8()**2
        self._norm*=sigma8**2/current_sigma8_2

    def get_sigma8(self) :
        current_sigma8 = math.sqrt(variance(8.0, self._default_filter, self, True)/self._lingrowth)
        return current_sigma8
    
    def __call__(self, k) :
        if np.any(k<self._orig_k_min) :
            warnings.warn("Power spectrum does not extend to low enough k; using power-law extrapolation (this is likely to be fine)", RuntimeWarning)
        if np.any(k>self._orig_k_max) :
            warnings.warn("Power spectrum does not extend to high enough k; using power-law extrapolation. This is bad but your results are unlikely to be sensitive to it unless they relate directly to very small scales or you have run CAMB with inappropriate settings.", RuntimeWarning)
        return self._norm*self._lingrowth*np.exp(self._interp(np.log(k)))

    

#######################################################################
# Variance calculation
#######################################################################

def variance(M, f_filter=TophatFilter, powspec=PowerSpectrumCAMB, arg_is_R=False) :
    if hasattr(M,'__len__') :
        ax =  pynbody.array.SimArray([variance(Mi, f_filter, powspec, arg_is_R) for Mi in M])
        ax.units = powspec.Pk.units * powspec.k.units**3 # hopefully dimensionless
        return ax
        
    if arg_is_R :
        R = M
    else :
        R = f_filter.M_to_R(M)
        
    
    integrand = lambda k: k**2 * powspec(k) * f_filter.Wk(k*R)**2
    integrand_ln_k = lambda k: np.exp(k)*integrand(np.exp(k))
    v = scipy.integrate.romberg(integrand_ln_k, math.log(powspec.min_k), math.log(1./R)+3,divmax=10, rtol=1.e-4)/(2*math.pi**2)
    
    return v





        
def correlation(r, powspec=PowerSpectrumCAMB) :
    
    if hasattr(r,'__len__') :
        ax = pynbody.array.SimArray([correlation(ri,  powspec) for ri in r])
        ax.units = powspec.Pk.units*powspec.k.units**3
        return ax
    
    # Because sin kr becomes so highly oscilliatory, normal
    # quadrature is slow/inaccurate for this problem. The following
    # is the best way I could come up with to overcome that.
    #
    # Each segment of the power spectrum is represented by a power law,
    # over which the integral boils down to a normal incomplete gamma
    # function extended into the complex plane.
    #
    # Originally, we had:
    #
    # integrand = lambda k: k**2 * powspec(k) * (np.sin(k*r)/(k*r))
    # integrand_ln_k = lambda k: np.exp(k)*integrand(np.exp(k))
    #
    
    tot=0
    defer = False

    k = powspec.k
    
    for k_bot, k_top in zip(k[:-1],k[1:]) :
        if defer :
            k_bot = k_bot_defer
            
        # express segment as P(k) = P0*k^n 
        Pk_top = powspec(k_top)
        Pk_bot = powspec(k_bot)

        n = np.log(Pk_top/Pk_bot)/np.log(k_top/k_bot)
        P0 = Pk_top/k_top**n

        
        # now integral of this segment is exactly
        # P0 * int_(k_bot)^(k_top) k^(2+n) sin(kr)/(kr) = (P0/r^(n+3)) Im[ (i)^(-n-2) Gamma(n+2,i k_bot r, i k_top r)]
        # First we need to evaluate the Gamma integral sufficiently accurately

    
        top_val = util.gamma_inc(n+2,(1.0j) * r * k_top)
        bot_val = util.gamma_inc(n+2, (1.0j)*r*k_bot)
        segment = -((1.0j)**(-n-2) *P0* (top_val-bot_val) / r**(n+3)).imag

        # accuracy monitoring
        f_acc = (np.abs(top_val-bot_val)/np.abs(top_val))
        # N.B. for large r, we can see from this that we're subtracting two big numbers
        # to get something very small, and the accuracy of the gamma procedure
        # becomes bad. We need to do something about this

        tot+=segment
        
        #if abs(segment)/tot>1.e-6 :
        #    print k_bot, n, segment, "|",abs(segment)/tot, f_acc, tot
        
        
        
    tot/= (2*math.pi**2)

    return tot

def correlation_func(pspec, log_r_min=-3, log_r_max=2, delta_log_r=0.2) :
    r = 10.0**np.arange(log_r_min,log_r_max+delta_log_r/2,delta_log_r)

    Xi_r = np.array([correlation(ri,pspec) for ri in r])

    return r, Xi_r
    
#######################################################################
# Default kernels for halo mass function
#######################################################################

def press_schechter(nu) :
    """The Press-Schechter kernel used by halo_mass_function"""
    return 0.7978845 * nu * np.exp(-(nu**2)/2)

def sheth_tormen(nu, A=0.322, q=0.3, p=0.) :
    """The Sheth-Tormen kernel used by halo_mass_function.

    Default shape values are taken from eq 7.67 of Mo, van den Bosch and White (CUP)."""
    
    nu_bar = nu*0.84
    return A*(1+1./nu_bar**(q))*press_schechter(nu_bar)

#######################################################################
# Bias functions
#######################################################################

def cole_kaiser(nu, delta_c) :
    """The Cole-Kaiser (1989) bias function"""
    return 1+(nu**2-1)/delta_c

#######################################################################
# The most useful function: halo_mass_function
#######################################################################

def halo_mass_function(context,
                       log_M_min=8.0, log_M_max=15.0, delta_log_M=0.1,
                       kern = "ST",
                       pspec = PowerSpectrumCAMB,
                       delta_crit = 1.686,
                       no_h = False) :
    """Returns the halo mass function, dN/d log_{10} M in units of Mpc^-3 h^3.

    Args:
       context (SimSnap): The snapshot from which to pull the cosmological context
          (includes sigma8 normalization and growth function integrations, but does
          not currently affect transfer function)

    Kwargs:
       log_M_min: The minimum halo mass (Msol h^-1) to consider
       log_M_max: The maximum halo mass (Msol h^-1) to consider
       delta_log_M: The bin spacing of halo masses (see warning below)
       kern: The kernel function which dictates what type of mass function to calculate;
             or a string ("PS" or "ST") for one of the defaults
       pspec: A power spectrum object (which also defines the window function);
             default is a WMAP7 cosmology calculated by CAMB, and a top hat window
       delta_crit: The critical overdensity for collapse

    Returns:       
       M: The centre of the mass bins, in Msol h^-1
       sigma: The linear variance of the corresponding sphere 
       N: The abundance of halos of that mass (Mpc^-3 h^3 comoving, per decade of mass)

    Because numerical derivatives are involved, the value of delta_log_M affects
    the accuracy. Numerical experiments suggest that delta_log_M=0.1 gives more than
    enough accuracy, but you should check for your own use case.
       
    """

    if isinstance(kern, str) :
        kern = {'PS': press_schechter,
                'ST': sheth_tormen}[kern]
    
    rho_bar = cosmology.rho_M(context, unit="Msol Mpc^-3 h^2")


    M = np.arange(log_M_min, log_M_max, delta_log_M)
    M_mid = np.arange(log_M_min+delta_log_M/2, log_M_max-delta_log_M/2, delta_log_M)

    if isinstance(pspec, type) :
        pspec = pspec(context)

        
    sig = variance(10**M, pspec._default_filter, pspec)
    
    nu = delta_crit/np.sqrt(sig)
    nu.units = "1"

    nu_mid = (nu[1:]+nu[:-1])/2
    
    d_ln_nu_d_ln_M = np.diff(np.log10(nu))/delta_log_M

    dM = np.diff(10**M)

    # eq 7.46, Mo, van den Bosch and White
    out = (rho_bar/(10**M_mid)) * kern(nu_mid) * d_ln_nu_d_ln_M * math.log(10.) * context.properties['a']**3
    out.units = "Mpc^-3 h^3 a^-3"
    out.sim = context

    M_mid = (10**M_mid).view(pynbody.array.SimArray)
    M_mid.units = "Msol h^-1"

    # interpolate sigma for output checking purposes
    sig = (sig[1:]+sig[:-1])/2

    return M_mid, sig, out

@units.takes_arg_in_units((1, "Msol h^-1"), context_arg=0)
def halo_bias(context, M, kern=cole_kaiser, pspec = PowerSpectrumCAMB,
              delta_crit = 1.686) :
    """Return the halo bias for the given halo mass.

    Args:
       context (SimSnap): The snapshot from which to pull the cosmological context
       M: float, unit or string describing the halo mass. If a float, units are Msol h^-1.

    Kwargs:
       kern: The kernel function describing the halo bias (default Cole-Kaiser).
       pspec: A power spectrum object (which also defines the window function);
              default is a WMAP7 cosmology calculated by CAMB, and a top hat window
       delta_crit: The critical overdensity for collapse

    Returns:
       The halo bias (single float)

    """

    if isinstance(pspec, type) :
        pspec = pspec(context)

    sig = variance(M, pspec._default_filter, pspec)
    nu = delta_crit / np.sqrt(sig)

    return kern(nu, delta_crit)
