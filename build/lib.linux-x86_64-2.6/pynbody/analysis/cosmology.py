"""

cosmology
=========

A set of functions for common cosmological calculations. 

"""

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

def _lingrowthintegrand(a,omegam0):   
    """ (e.g. eq. 8 in lukic et al. 2008)   returns: da / [a*H(a)/H0]**3 """
    return numpy.power((a * hzoverh0(a,omegam0)),-3)

def _lingrowthfac(red,omegam0,omegal0, return_norm=False):
    """
    returns: linear growth factor, b(a) normalized to 1 at z=0, good for flat lambda only
    a = 1/1+z
    b(a) = Delta(a) / Delta(a=1)   [ so that b(z=0) = 1 ]
    (and b(a) [Einstein de Sitter, omegam=1] = a)
    
    Delta(a) = 5 omegam / 2 H(a) / H(0) * integral[0:a] [da / [a H(a) H0]**3]
    equation  from  peebles 1980 (or e.g. eq. 8 in lukic et al. 2008) """
##  need to add w ~= , nonflat, -1 functionality
    
    import scipy.integrate

    if (abs(omegam0 + omegal0-1.) >1.e-4):
        raise RuntimeError, "Linear growth factors can only be calculated for flat cosmologies"
    
    a = 1/(1.+red)
    
    ## 1st calc. for z=z
    lingrowth = scipy.integrate.quad(_lingrowthintegrand,0.,a, (omegam0))[0]
    lingrowth *= 5./2. * omegam0 * hzoverh0(a,omegam0)

    ## then calc. for z=0 (for normalization)
    a0 = 1.
    lingrowtha0 =  scipy.integrate.quad(_lingrowthintegrand,0.,a0, (omegam0))[0]
    lingrowtha0 *= 5./2. * omegam0  * hzoverh0(a0,omegam0)

    lingrowthfactor = lingrowth / lingrowtha0
    if return_norm :
        return lingrowthfactor, lingrowtha0
    else :
        return lingrowthfactor
    
def linear_growth_factor(f,z=None):
    """Calculate the linear growth factor b(a), normalized to 1
    at z=0, for the cosmology of snapshot f.

    The output is dimensionless. If a redshift z is
    specified, it is used in place of the redshift in
    output f.
    """
    if z is None :
        z = f.properties['z']
    omegam0 = f.properties['omegaM0']
    omegal0 = f.properties['omegaL0']
    return _lingrowthfac(z,omegam0,omegal0)

def rate_linear_growth(f, z=None, unit='h Gyr^-1') :
    """Calculate the linear growth rate b'(a), normalized
    to 1 at z=0, for the cosmology of snapshot f.

    The output is in 'h Gyr^-1' by default. If a redshift z is specified,
    it is used in place of the redshift in output f."""

    if z is None :
        z = f.properties['z']
    a = 1./(1.+z)
    omegam0 = f.properties['omegaM0']
    omegal0 = f.properties['omegaL0']
    
    
    b,X = _lingrowthfac(z,omegam0,omegal0,return_norm=True)
    I = _lingrowthintegrand(a,omegam0)
    
    term1 = -(1.5*omegam0 * a**-3)*b / math.sqrt(1.-omegam0 + omegam0*a**-3)
    term2 = (2.5*omegam0) * hzoverh0(a, omegam0)**2 * a * I / X


    res = units.h * (term1+term2) * 100. * units.Unit("km s^-1 Mpc^-1")

    return res.in_units(unit, **f.conversion_context())

def _test_rate_linear_growth(f, z=None, unit='h Gyr^-1') :
    # coded up by AP to test linear growth *rate* equation above
    if z is None :
        z = f.properties['z']
    a0 = 1./(1.+z)
    a1 = a0*0.999
    z0=1./a0-1
    z1=1./a1-1
    
    b0 = linear_growth_factor(f,z0)
    b1 = linear_growth_factor(f,z1)

    db = b1-b0

    unit = units.Unit(unit)
    dt = age(f, z1, unit**-1)-age(f,z0, unit**-1)

    return db/dt
    
def age(f, z=None, unit='Gyr') :
    """
    Calculate the age of the universe in the snapshot f
    by integrating the Friedmann equation.

    The output is given in the specified units. If a redshift
    z is specified, it is used in place of the redshift in the
    output f.

    **Input**:
    
    *f*: SimSnap

    **Optional Keywords**:
    
    *z (None)*: desired redshift. Can be a single number, a list, or a
    numpy.ndarray.

    *unit ('Gyr')*: desired units for age output

    """

    import scipy, scipy.integrate

    if z is None :
        z = f.properties['z']


    h0 = f.properties['h']
    omM = f.properties['omegaM0']
    omL = f.properties['omegaL0']

    conv = units.Unit("0.01 s Mpc km^-1").ratio(unit, **f.conversion_context())

    def get_age(x) : 
        x = 1.0/(1.0 + x)
        return scipy.integrate.quad(_a_dot_recip,0,x, (h0, omM, omL))[0]*conv

    if isinstance(z,np.ndarray) or isinstance(z,list): 
        return np.array(map(get_age,z))
    else : 
        return get_age(z)

@units.takes_arg_in_units((1,"Gyr"),context_arg=0)
def redshift(f, time) : 
    """ 
    Calculate the redshift given a snapshot and a time since Big Bang
    in Gyr.

    Uses scipy.optimize.newton to do the root finding if number of
    elements in the time array is less than 1000, otherwise uses a linear
    interpolation.


    **Input**:

    *f*: SimSnap with cosmological parameters defined

    *time*: time since the Big Bang in Gyr for which a redshift should
     be returned. float, list, or numpy.ndarray

    """

    from scipy.optimize import newton
    from scipy.interpolate import interp1d
    from .. import array

    def func(x,sim,time) : 
        return age(sim,x) - time

    if isinstance(time,list) or isinstance(time,np.ndarray):
        if len(time) > 1000 : 
            zs = np.logspace(3,-10,1000)
            ages = age(f,zs)
            i = interp1d(ages,zs)
            return i(time)
        else : 
            return np.array(map(lambda x: newton(func,1,args=(f,x)),time))
    else : 
        return newton(func,1,args=(f,time))

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
            unit = f.dm["mass"].units/f.dm["pos"].units**3
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

def rho_M(f, z=None, unit=None) :
    """Calculate the matter density of the universe in snapshot f.

    unit and z are used if not None, as by rho_crit. See also the note in
    rho_crit about confusion over comoving units in this case."""

    if z is None :
        z = f.properties['z']
        
    return f.properties['omegaM0']*rho_crit(f,0,unit)*(1.0+z)**3

def H(f) :
    """Calculate the Hubble parameter of the universe in snapshot f"""
    return f.properties['h']*hzoverh0(f.properties['a'], f.properties['omegaM0'])*units.Unit("100 km s^-1 Mpc^-1")
    
def add_hubble(f) :
    """Add the hubble flow to velocities in snapshot f"""

    f['vel']+=f['pos']*H(f)

def comoving_to_physical(ar) :
    """Given an array, modify it to be in physical units (remove any
    dependencies on a or aform)."""

    a_power = ar.units._power_of("a")
    aform_power = ar.units._power_of("aform")

    if a_power!=0 :
        a = ar.sim.properties['a']
        ar*=a**a_power
        ar/=units.Unit("a")**a_power
    if aform_power!=0 :
        aform = ar.sim['aform']
        ar*=aform**aform_power
        ar/=units.Unit("aform")
    
