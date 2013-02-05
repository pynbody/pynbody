"""

halo
====

Functions for dealing with and manipulating halos in simulations.


"""

from .. import filt, util, config
from . import cosmology
import numpy as np
import math


def center_of_mass(sim) : 
    """

    Return the centre of mass of the SimSnap

    """
    mtot = sim["mass"].sum()
    p = np.sum(sim["mass"]*sim["pos"].transpose(), axis=1)/mtot

    p.units = sim["pos"].units # otherwise behaviour is numpy version dependent

    return p # only return position to be consistent with other functions in halo.py

def center_of_mass_velocity(sim) :
    """

    Return the center of mass velocity of the SimSnap

    """
    mtot = sim["mass"].sum()
    v = np.sum(sim["mass"]*sim["vel"].transpose(), axis=1)/mtot
    v.units = sim["vel"].units # otherwise behaviour is numpy version dependent

    return v

def shrink_sphere_center(sim, r=None, shrink_factor = 0.7, min_particles = 100, verbose=False) :
    """
    
    Return the center according to the shrinking-sphere method of
    Power et al (2003)
    
    """
    x = sim

    if r is None :
        # use rough estimate for a maximum radius
        # results will be insensitive to the exact value chosen
        r = (sim["x"].max()-sim["x"].min())/2
    com=None
    while len(x)>min_particles or com is None :
        com = center_of_mass(x)#, cov = center_of_mass(x)
        r*=shrink_factor
        x = sim[filt.Sphere(r, com)]
        if verbose:
            print com,r,len(x)
    return com

def virial_radius(sim, cen=None, overden=178, r_max=None) :
    """
    
    Calculate the virial radius of the halo centerd on the given
    coordinates.

    This is here defined by the sphere centerd on cen which contains a
    mean density of overden * rho_c_0 * (1+z)^3.

    """

    if r_max is None :
        r_max = (sim["x"].max()-sim["x"].min())
    else :
        if cen is not None :
            sim = sim[filt.Sphere(r_max,cen)]
        else :
            sim = sim[filt.Sphere(r_max)]

    r_min = 0.0

    if cen is not None :
        sim['pos']-=cen
        
    # sim["r"] = ((sim["pos"]-cen)**2).sum(axis=1)**(1,2)

    rho = lambda r : sim["mass"][np.where(sim["r"]<r)].sum()/(4.*math.pi*(r**3)/3)
    target_rho = overden * sim.properties["omegaM0"] * cosmology.rho_crit(sim, z=0) * (1.0+sim.properties["z"])**3

    result = util.bisect(r_min, r_max, lambda r : target_rho-rho(r), epsilon=0, eta=1.e-3*target_rho, verbose=True)
    if cen is not None :
        sim['pos']+=cen

    return result

def potential_minimum(sim) :
    i = sim["phi"].argmin()
    return sim["pos"][i].copy()

def hybrid_center(sim, r='3 kpc', **kwargs) :
    """

    Determine the center of the halo by finding the shrink-sphere
    -center inside the specified distance of the potential minimum

    """

    try:
        cen_a = potential_minimum(sim)
    except KeyError:
        cen_a = center_of_mass(sim)
    return shrink_sphere_center(sim[filt.Sphere(r, cen_a)], **kwargs)

def index_center(sim, ind = None, **kwargs) :
    """

    Determine the center of mass based on specific particles.

    Supply a list of indices using the ``ind`` keyword.

    """

    if 'ind' is not None :
        ind = kwargs['ind']
        return center_of_mass(sim[ind])
    else :  
        raise RuntimeError("Need to supply indices for centering")
    

def vel_center(sim, mode=None, cen_size = "1 kpc", retcen=False, **kwargs) :
    """

    Use stars from a spehre to calculate center of velocity. The size
    of the sphere is given by the ``cen_size`` keyword and defaults to
    1 kpc.


    """


    if config['verbose'] :
        print "Finding halo velocity center..."
    cen = sim.star[filt.Sphere(cen_size)]
    if len(cen)<5 :
        # fall-back to DM
        cen = sim.dm[filt.Sphere(cen_size)]
    if len(cen)<5 :
        # fall-back to gas
        cen = sim.gas[filt.Sphere(cen_size)]
    if len(cen)<5 :
        # very weird snapshot, or mis-centering!
        raise ValueError, "Insufficient particles around center to get velocity"

    vcen = (cen['vel'].transpose()*cen['mass']).sum(axis=1)/cen['mass'].sum()
    vcen.units = cen['vel'].units
    if config['verbose'] :
        print "vcen=",vcen

    if retcen:  return vcen
    else:  sim.ancestor["vel"]-=vcen

def center(sim, mode=None, retcen=False, vel=True, **kwargs) :
    """

    Determine the center of mass of the given particles using the
    specified mode, then recenter the particles (of the entire
    ancestor snapshot) accordingly

    Accepted values for *mode* are

      *pot*: potential minimum

      *com*: center of mass

      *ssc*: shrink sphere center

      *ind*: center on specific particles; supply the list of particles using the ``ind`` keyword.

      *hyb*: for sane halos, returns the same as ssc, but works faster by
             starting iteration near potential minimum

    or a function returning the COM.

    **Other keywords:**

    *retcen*: if True only return the center without centering the
     snapshot (default = False)

    *ind*: only used when *mode=ind* -- specifies the indices of
     particles to be used for centering

    *vel*: if True, translate velocities so that the velocity of the
    central 1kpc is zeroed
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
        sim.ancestor["pos"]-=cen

    if vel :
        vel_center(sim, cen_size = "1 kpc")
