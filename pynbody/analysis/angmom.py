import numpy as np
from .. import filt

def ang_mom_vec(snap) :
    """Return the angular momentum vector of the specified snapshot.

    The return units are [mass]*[dist]*[vel] as per the units of the snapshot."""
    angmom = (snap['mass'].reshape(len(snap),1)*np.cross(snap['pos'], snap['vel'])).sum(axis=0).view(np.ndarray)
    return angmom

def calc_sideon_matrix(angmom_vec) :
    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in/np.sum(vec_in**2).sum()**0.5
    vec_p1 = np.cross([1,0,0],vec_in)
    vec_p1 = vec_p1/np.sum(vec_p1**2).sum()**0.5
    vec_p2 = np.cross(vec_in,vec_p1)

    matr = np.concatenate((vec_in,vec_p1,vec_p2)).reshape((3,3))
  
    return matr

def calc_faceon_matrix(angmom_vec) :
    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in/np.sum(vec_in**2).sum()**0.5
    vec_p1 = np.cross([1,0,0],vec_in)
    vec_p1 = vec_p1/np.sum(vec_p1**2).sum()**0.5
    vec_p2 = np.cross(vec_in,vec_p1)

    matr = np.concatenate((vec_p1,vec_p2,vec_in)).reshape((3,3))
  
    return matr


def sideon(h, vec_to_xform=calc_sideon_matrix) :
    """Reposition and rotate the simulation containing the halo h to
    see h's disk edge on.

    Given a simulation and a subview of that simulation (probably
    the halo of interest), this routine centres the simulation and
    rotates it so that the disk lies in the x-z plane. This gives
    a side-on view for SPH images, for instance."""

    top = h
    while hasattr(top,'base') : top = top.base

    # Top is the top-level view of the simulation, which will be
    # transformed

    cen = h['pos'][h['phi'].argmin()]
    top['pos']-=cen

    # Use gas from inner 1kpc to calculate centre of velocity
    cen = h.gas[filt.Sphere("1 kpc")]
    top['vel']-=cen['vel'].mean(axis=0)
    
    # Use gas from inner 10kpc to calculate angular momentum vector
    cen = h.gas[filt.Sphere("10 kpc")]

    trans = vec_to_xform(ang_mom_vec(cen))
    print trans
    
    top.transform(trans)
    
    
def faceon(h) :
    """Reposition and rotate the simulation containing the halo h to
    see h's disk face on.

    Given a simulation and a subview of that simulation (probably
    the halo of interest), this routine centres the simulation and
    rotates it so that the disk lies in the x-z plane. This gives
    a face-on view for SPH images, for instance."""

    sideon(h, calc_faceon_matrix)
