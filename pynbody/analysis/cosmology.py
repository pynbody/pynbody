import math, numpy as np
numpy=np ## alias the alias 
from .. import units

def _a_dot(a, h0, om_m, om_l) :
    om_k = 1.0-om_m-om_l
    return h0*a*np.sqrt(om_m*(a**-3) + om_k*(a**-2) + om_l)

def _a_dot_recip(*args) :
    return 1./_a_dot(*args)

def hzoverh0(a, omegam0): 
    """ returns: H(a) / H0  = [omegam/a**3 + (1-omegam)]**0.5 """
    return numpy.sqrt(omegam0*numpy.power(a,-3) + (1.-omegam0))

def lingrowthintegral(a,omegam0):   
    """ (e.g. eq. 8 in lukic et al. 2008)   returns: da / [a*H(a)/H0]**3 """
    return numpy.power((a * hzoverh0(a,omegam0)),-3)

def lingrowthfac(red,omegam0,omegal0):
    """
    returns: linear growth factor, b(a) normalized to 1 at z=0, good for flat lambda only
    a = 1/1+z
    b(a) = Delta(a) / Delta(a=1)   [ so that b(z=0) = 1 ]
    (and b(a) [Einstein de Sitter, omegam=1] = a)
    
    Delta(a) = 5 omegam / 2 H(a) / H(0) * integral[0:a] [da / [a H(a) H0]**3]
    equation  from  peebles 1980 (or e.g. eq. 8 in lukic et al. 2008) """
##  need to add w ~= , nonflat, -1 functionality
    
    import scipy.integrate

    if ((omegam0 + omegal0) != 1.):
        print "WARNING -- omegam + lambda not equal 1, solution NOT VALID"
    
    a = 1/(1.+red)
    
    ## 1st calc. for z=z
    lingrowth = scipy.integrate.quad(lingrowthintegral,0.,a, (omegam0))[0]
    lingrowth *= 5./2. * omegam0 * hzoverh0(a,omegam0)

    ## then calc. for z=0 (for normalization)
    a0 = 1.
    lingrowtha0 =  scipy.integrate.quad(lingrowthintegral,0.,a0, (omegam0))[0]
    lingrowtha0 *= 5./2. * omegam0  * hzoverh0(a0,omegam0)

    lingrowthfactor = lingrowth / lingrowtha0
    return lingrowthfactor


def getlingrowthfactor(f,z=None):  ##  this is just a wrapper for pynbody.
    ## returns: linear growth factor, b(a) normalized to 1 at z=0, good for flat lambda only
    if z is None :
        red = f.properties['z']
    omegam0 = f.properties['omegaM0']
    omegal0 = f.properties['omegaL0']
    return lingrowthfac(red,omegam0,omegal0)


def age(f, z=None, unit='Gyr') :
    """Calculate the age of the universe in the snapshot f
    by integrating the Friedmann equation.

    The output is given in the specified units. If a redshift
    z is specified, it is used in place of the redshift in the
    output f."""

    import scipy, scipy.integrate

    if z is None :
        z = f.properties['z']

    a = 1.0/(1.0+z)
    h0 = f.properties['h']
    omM = f.properties['omegaM0']
    omL = f.properties['omegaL0']

    conv = units.Unit("0.01 s Mpc km^-1").ratio(unit, **f.conversion_context())

    age = scipy.integrate.quad(_a_dot_recip,0,a, (h0, omM, omL))[0]

    return age*conv


def rho_crit(f, z=None, unit=None) :
    """Calculate the critical density of the universe in
    the snapshot f.

    z specifies the redshift. If z is none, the redshift of the
    provided snapshot is used.

    unit specifies the units of the returned density. If unit is None,
    the returned density will be in the units of
    f["mass"].units/f["pos"].units**3. If that unit cannot be calculated,
    the returned units are Msol kpc^-3 comoving.

    Note that you can get slightly confusing results if your
    simulation is in comoving units and you specify a different
    redshift z. Specifically, the physical density for the redshift
    you specify is calulated, but expressed as a comoving density *at
    the redshift of the snapshot*. This is intentional behaviour."""


    if z is None :
        z = f.properties['z']

    if unit is None :
        try :
            unit = f["mass"].units/f["pos"].units**3
        except units.UnitsException :
            unit = units.NoUnit()

    if hasattr(unit, "_no_unit") :
        unit = units.Unit("Msol kpc^-3 a^-3")

    omM = f.properties['omegaM0']
    omL = f.properties['omegaL0']
    h0 = f.properties['h']
    a = 1.0/(1.0+z)

    H_z = _a_dot(a, h0, omM, omL)/a
    H_z = units.Unit("100 km s^-1 Mpc^-1")*H_z

    rho_crit = (3*H_z**2)/(8*math.pi*units.G)

    return rho_crit.ratio(unit, **f.conversion_context())
