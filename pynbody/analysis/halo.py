from .. import filt, util, config
from . import cosmology
import numpy as np
import math


def center_of_mass(sim) : 
    """Return the centre of mass of the SimSnap"""
    mtot = sim["mass"].sum()
    p = np.sum(sim["mass"]*sim["pos"].transpose(), axis=1)/mtot
    v = np.sum(sim["mass"]*sim["vel"].transpose(), axis=1)/mtot

    p.units, v.units = sim["pos"].units, sim["vel"].units # otherwise behaviour is numpy version dependent

    return p,v
    # return np.average(sim["pos"],axis=0,weights=sim["mass"]), np.average(sim["vel"],axis=0,weights=sim["mass"])

def shrink_sphere_center(sim, r=None, shrink_factor = 0.7, min_particles = 100, verbose=False) :
    """Return the center according to the shrinking-sphere method
    of Power et al (2003)"""
    x = sim

    if r is None :
        # use rough estimate for a maximum radius
        # results will be insensitive to the exact value chosen
        r = (sim["x"].max()-sim["x"].min())/2
    com=None
    while len(x)>min_particles or com is None :
        com, cov = center_of_mass(x)
        r*=shrink_factor
        x = sim[filt.Sphere(r, com)]
        if verbose:
            print com,r,len(x)
    return com

def virial_radius(sim, cen=(0,0,0), overden=178, r_max=None) :
    """Calculate the virial radius of the halo centerd on the given
    coordinates.

    This is here defined by the sphere centerd on cen which contains a mean
    density of overden * rho_c_0 * (1+z)^3. """

    if r_max is None :
        r_max = (sim["x"].max()-sim["x"].min())
    else :
        sim = sim[filt.Sphere(r_max,cen)]


    r_min = 0.0

    sim["r"] = ((sim["pos"]-cen)**2).sum(axis=1)**(1,2)

    rho = lambda r : sim["mass"][np.where(sim["r"]<r)].sum()/(4.*math.pi*(r**3)/3)
    target_rho = overden * sim.properties["omegaM0"] * cosmology.rho_crit(sim, z=0) * (1.0+sim.properties["z"])**3

    return util.bisect(r_min, r_max, lambda r : target_rho-rho(r), epsilon=0, eta=1.e-3*target_rho, verbose=True)


def potential_minimum(sim) :
    i = sim["phi"].argmin()
    return sim["pos"][i].copy()

def hybrid_center(sim, r='3 kpc', **kwargs) :
    """Determine the center of the halo by finding the
    shrink-sphere -center inside the specified distance
    of the potential minimum"""

    cen_a = potential_minimum(sim)
    return shrink_sphere_center(sim[filt.Sphere(r, cen_a)], **kwargs)

def index_center(sim, **kwargs) :
    """Determine the center of mass based on specific particles"""
    if 'ind' in kwargs :
        ind = kwargs['ind']
        return center_of_mass(sim[ind])
        # return np.average(sim["pos"][ind],axis=0,weights=sim["mass"][ind])
    else :  
        raise RuntimeError("Need to supply indices for centering")
    

def center(sim, mode=None, retcen=False, **kwargs) :
    """Determine the center of mass using the specified mode
    and recenter the particles accordingly

    Accepted values for mode are
      'pot': potential minimum
      'com': center of mass
      'ssc': shrink sphere center
      'ind': center on specific particles
    or a function returning the COM.

    retcen: if True only return the center without centering
            the snapshot (default = False)
    """

    global config
    if mode is None:
        mode=config['centering-scheme']

    try:
        fn = {'pot': potential_minimum,
              'com': center_of_mass,
              'ssc': shrink_sphere_center,
              'hyb': hybrid_center,
              'ind': index_center}[mode]
    except KeyError :
        fn = mode

    if retcen:  return fn(sim, **kwargs)
    else:
        cen = fn(sim, **kwargs)
        sim["pos"]-=cen

