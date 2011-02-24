import numpy as np
from .. import filt
from . import halo

def ang_mom_vec(snap) :
    """Return the angular momentum vector of the specified snapshot.

    The return units are [mass]*[dist]*[vel] as per the units of the snapshot."""
    angmom = (snap['mass'].reshape((len(snap),1))*np.cross(snap['pos'], snap['vel'])).sum(axis=0).view(np.ndarray)
    return angmom

def calc_sideon_matrix(angmom_vec) :
    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in/np.sum(vec_in**2).sum()**0.5
    vec_p1 = np.cross([1,0,0],vec_in)
    vec_p1 = vec_p1/np.sum(vec_p1**2).sum()**0.5
    vec_p2 = np.cross(vec_in,vec_p1)

    matr = np.concatenate((vec_p2,vec_in,vec_p1)).reshape((3,3))

    return matr

def calc_faceon_matrix(angmom_vec) :
    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in/np.sum(vec_in**2).sum()**0.5
    vec_p1 = np.cross([1,0,0],vec_in)
    vec_p1 = vec_p1/np.sum(vec_p1**2).sum()**0.5
    vec_p2 = np.cross(vec_in,vec_p1)

    matr = np.concatenate((vec_p1,vec_p2,vec_in)).reshape((3,3))

    return matr


def sideon(h, vec_to_xform=calc_sideon_matrix, cen_size = "1 kpc", disk_size = "5 kpc",
           cen = None, vcen=None, verbose=False ) :
    """Reposition and rotate the simulation containing the halo h to
    see h's disk edge on.

    Given a simulation and a subview of that simulation (probably
    the halo of interest), this routine centers the simulation and
    rotates it so that the disk lies in the x-z plane. This gives
    a side-on view for SPH images, for instance."""

    top = h
    while hasattr(top,'base') : top = top.base

    # Top is the top-level view of the simulation, which will be
    # transformed

    if cen is None :
        if verbose :
            print "Finding halo center..."
        cen = halo.hybrid_center(h) # or h['pos'][h['phi'].argmin()]
        if verbose :
            print "cen=",cen

    top['pos']-=cen

    if vcen is None :
        # Use stars from inner 1kpc to calculate center of velocity
        if verbose :
            print "Finding halo velocity center..."
        cen = h.star[filt.Sphere(cen_size)]
        if len(cen)<5 :
            # fall-back to DM
            cen = h.dm[filt.Sphere(cen_size)]
        if len(cen)<5 :
            # fall-back to gas
            cen = h.gas[filt.Sphere(cen_size)]
        if len(cen)<5 :
            # very weird snapshot, or mis-centering!
            raise ValueError, "Insufficient particles around center to get velocity"

        vcen = (cen['vel'].transpose()*cen['mass']).sum(axis=1)/cen['mass'].sum()
        if verbose :
            print "vcen=",vcen

    top['vel']-=vcen

    # Use gas from inner 10kpc to calculate angular momentum vector
    cen = h.gas[filt.Sphere(disk_size)]

    if verbose :
        print "Calculating angular momentum vector..."
    trans = vec_to_xform(ang_mom_vec(cen))

    if verbose :
        print "Transforming simulation..."
    top.transform(trans)


def faceon(h, **kwargs) :
    """Reposition and rotate the simulation containing the halo h to
    see h's disk face on.

    Given a simulation and a subview of that simulation (probably
    the halo of interest), this routine centers the simulation and
    rotates it so that the disk lies in the x-y plane. This gives
    a face-on view for SPH images, for instance."""

    sideon(h, calc_faceon_matrix, **kwargs)
