import math, numpy as np

from .. import units

def _a_dot(a, h0, om_m, om_l) :
    om_k = 1.0-om_m-om_l
    return h0*a*np.sqrt(om_m*(a**-3) + om_k*(a**-2) + om_l)

def _a_dot_recip(*args) :
    return 1./_a_dot(*args)

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
