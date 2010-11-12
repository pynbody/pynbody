import numpy as np

def centre_of_mass(sim) : # shared-code names should be explicit, not short
    """Return the centre of mass of the SimSnap"""
    return np.average(sim["pos"],axis=0,weights=sim["mass"])

def shrink_sphere_centre(sim) :
    """Return the centre according to the shrinking-sphere method
    of Power et al (2003)"""
    raise RuntimeError("Not implemented")
    
def potential_minimum(sim) :
    i = sim["phi"].argmin()
    return sim["pos"][i]


def centre(sim, mode='pot') :
    """Determine the centre of mass using the specified mode
    and recentre the particles accordingly

    Accepted values for mode are
      'pot': potential minimum
      'com': centre of mass
      'ssc': shrink sphere centre
    or a function returning the COM."""
    
    try:
	fn = {'pot': potential_minimum,
	      'com': centre_of_mass,
	      'ssc': shrink_sphere_centre}[mode]
    except KeyError :
	fn = mode

    cen = fn(sim)
    sim["pos"]-=cen
    
