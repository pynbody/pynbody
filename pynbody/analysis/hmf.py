import numpy as np
from . import cosmology
import pynbody
import os
import scipy, scipy.interpolate
import math
import warnings

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

        
        bot_k = 1.e-5
        
        if k[0]>bot_k :
            # extrapolate out
            n = math.log10(Pk[1]/Pk[0])/math.log10(k[1]/k[0])
            warnings.warn("Power spectrum does not extend to low enough k; extrapolating as power law assuming ns=%.2f"%n, RuntimeWarning)
            
            Pkinterp = 10**(math.log10(Pk[0])-math.log10(k[0]/bot_k)*n)
            k = np.hstack((bot_k,k))
            Pk = np.hstack((Pkinterp,Pk))


      
        top_k = 1.e7
        
        if k[-1]<top_k :
            # extrapolate out
            n = math.log10(Pk[-1]/Pk[-2])/math.log10(k[-1]/k[-2])
            warnings.warn("Power spectrum does not extend to high enough k; extrapolating as power law assuming ns=%.2f"%n, RuntimeWarning)
            
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
# The most usefu function: halo_mass_function
#######################################################################

def halo_mass_function(context,
                       log_M_min=8.0, log_M_max=12.0, delta_log_M=0.1,
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
                'ST': sheth_tormen}
    
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
