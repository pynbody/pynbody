"""

gravity
=======

Functions for calculating the gravitational potential and accelerations. 

"""


from .. import units
from .. import array

import math
import numpy as np

def potential(f, pos_vec, eps=None, unit=None) :
    """

    Calculates the gravitational potential at the specified position
    using all particles in the specified snapshot.

    The gravitational softening length is determined by (in order of
    preference):

    1. The parameter eps (scalar, unit or array)

    2. The array f['eps']

    3. f.properties['eps'] (scalar or unit) 

    """

    if eps is None :
        try :
            eps = f['eps']
        except KeyError :
            eps = f.properties['eps']
            
    if isinstance(eps, str) :
        eps = units.Unit(eps)

    if isinstance(eps, units.UnitBase) :
        eps = eps.in_units(f['pos'].units, **f.conversion_context())
        
    d_pos = f['pos'] - pos_vec
    
    GM_by_r = units.G * f['mass'] / ((d_pos**2).sum(axis=1) + eps**2)**(1,2)

    if unit is not None :
        GM_by_r.convert_units(unit)

    return  -(GM_by_r.sum())

    
def accel(f, pos_vec, eps=None) :
    """

    Calculates the gravitational acceleration vector at the specified
    position using all particles in the specified snapshot.

    The gravitational softening length is determined by (in order of
    preference):

    1. The parameter eps (scalar, unit or array)

    2. The array f['eps']

    3. f.properties['eps'] (scalar or unit) 

    """
    
    if eps is None :
        try :
            eps = f['eps']
        except KeyError :
            eps = f.properties['eps']
            
    if isinstance(eps, str) :
        eps = units.Unit(eps)

    if isinstance(eps, units.UnitBase) :
        eps = eps.in_units(f['pos'].units, **f.conversion_context())
        
    d_pos = f['pos'] - pos_vec
    
    GM_by_r3 = units.G * f['mass'] / ((d_pos**2).sum(axis=1) + eps**2)**(3,2)

    return  -(d_pos.T*GM_by_r3).sum(axis=1)


def midplane_rot_curve(f, rxy_points, eps = None) :
    """

    Calculates the midplane rotation curve at a specified set of
    points. Used primarily by :func:`pynbody.analysis.profile.v_circ`

    """

    u_out = (units.G * f['mass'].units / f['pos'].units)**(1,2)
    
    vels = []
    
    for r in rxy_points :
        # Do four samples like Tipsy does
        r_acc_r = []
        for pos in [(r,0,0), (0,r,0), (-r,0,0), (0,-r,0)] :
            acc = accel(f, pos, eps)
            r_acc_r.append(np.dot(acc,pos))
        
        vel2 = np.mean(r_acc_r)
        if vel2>0 :
            vel = math.sqrt(vel2)
        else :
            vel = 0

        vels.append(vel)

    x = array.SimArray(vels, units = u_out)
    x.sim = f.ancestor
    return x

def midplane_potential(f, rxy_points, eps = None) :
    """

    Calculates the midplane potential at a specified set of
    points. Used primarily by :func:`pynbody.analysis.profile.pot`

    """

    u_out = units.G * f['mass'].units / f['pos'].units
    
    pots = []

    for r in rxy_points :
        # Do four samples like Tipsy does
        pot = []
        for pos in [(r,0,0), (0,r,0), (-r,0,0), (0,-r,0)] :
            pot.append(potential(f, pos, eps))
            
        pots.append(np.mean(pot))

    x = array.SimArray(pots, units = u_out)
    x.sim = f.ancestor
    return x
