"""

angmom
======

"""

import numpy as np
from .. import array, filt, units, config
from . import halo

def ang_mom_vec(snap) :
    """

    Return the angular momentum vector of the specified snapshot.

    The return units are [mass]*[dist]*[vel] as per the units of the
    snapshot.

    """
    
    angmom = (snap['mass'].reshape((len(snap),1))*np.cross(snap['pos'], snap['vel'])).sum(axis=0).view(np.ndarray)
    return angmom

def ang_mom_vec_units(snap) :
    """

    Return the angular momentum vector of the specified snapshot
    with correct units.
    Note that the halo has to be aligned such that the disk
    is in the x-y-plane and its center must be the coordinate
    origin.

    """

    angmom = ang_mom_vec(snap)
    return array.SimArray(angmom, snap['mass'].units*snap['pos'].units*snap['vel'].units)

def spin_parameter(snap) :
    """

    Return the spin parameter \lambda' of a centered halo
    as defined in eq. (5) of Bullock et al. 2001
    (2001MNRAS.321..559B).
    Note that the halo has to be aligned such that the disk
    is in the x-y-plane and its center must be the coordinate
    origin.

    """

    m3 = snap['mass'].sum()
    m3 = m3*m3*m3
    l = np.sqrt(((ang_mom_vec_units(snap)**2).sum())/(2*units.G*m3*snap['r'].max()))
    return float(l.in_units('1', **snap.conversion_context()))

def calc_sideon_matrix(angmom_vec) :
    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in/np.sum(vec_in**2).sum()**0.5
    vec_p1 = np.cross([1,0,0],vec_in)
    vec_p1 = vec_p1/np.sum(vec_p1**2).sum()**0.5
    vec_p2 = np.cross(vec_in,vec_p1)

    matr = np.concatenate((vec_p2,vec_in,vec_p1)).reshape((3,3))

    return matr

def calc_faceon_matrix(angmom_vec, up=[0.0,1.0,0.0]) :
    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in/np.sum(vec_in**2).sum()**0.5
    vec_p1 = np.cross(up,vec_in)
    vec_p1 = vec_p1/np.sum(vec_p1**2).sum()**0.5
    vec_p2 = np.cross(vec_in,vec_p1)

    matr = np.concatenate((vec_p1,vec_p2,vec_in)).reshape((3,3))

    return matr


def sideon(h, vec_to_xform=calc_sideon_matrix, cen_size = "1 kpc", 
           disk_size = "5 kpc", cen = None, vcen=None, top=None,
           return_transform = False, **kwargs ) :
    """

    Reposition and rotate the simulation containing the halo h to see
    h's disk edge on.

    Given a simulation and a subview of that simulation (probably the
    halo of interest), this routine centers the simulation and rotates
    it so that the disk lies in the x-z plane. This gives a side-on
    view for SPH images, for instance.

    """

    global config

    if top is None :
        top = h
        while hasattr(top,'base') : top = top.base

    # Top is the top-level view of the simulation, which will be
    # transformed

    if cen is None :
        if config['verbose'] :
            print "Finding halo center..."
        cen = halo.center(h,retcen=True,**kwargs) # or h['pos'][h['phi'].argmin()]
        if config['verbose'] :
            print "cen=",cen

    top['pos']-=cen

    if vcen is None :
        vcen = halo.vel_center(h,retcen=True)

    top['vel']-=vcen

    # Use gas from inner 10kpc to calculate angular momentum vector
    if (len(h.gas) > 0):
        cen = h.gas[filt.Sphere(disk_size)]
    else:
        cen = h[filt.Sphere(disk_size)]

    if config['verbose'] :
        print "Calculating angular momentum vector..."
    trans = vec_to_xform(ang_mom_vec(cen))

    if config['verbose'] :
        print "Transforming simulation..."
    top.transform(trans)

    if return_transform :
        return trans


def faceon(h, **kwargs) :
    """

    Reposition and rotate the simulation containing the halo h to see
    h's disk face on.

    Given a simulation and a subview of that simulation (probably the
    halo of interest), this routine centers the simulation and rotates
    it so that the disk lies in the x-y plane. This gives a face-on
    view for SPH images, for instance.

    """

    sideon(h, calc_faceon_matrix, **kwargs)
