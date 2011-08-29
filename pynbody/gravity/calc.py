from .. import units
from .. import array

import math
import tree
import numpy as np

def direct(f, ipos, eps= None) :
    try :
        import scipy, scipy.weave
        from scipy.weave import inline
        import os
    except ImportError :
        pass
    nips = len(ipos)
    m_by_r2 = np.zeros((nips,3))
    m_by_r = np.zeros(nips)
    pos = f['pos'].view(np.ndarray)
    mass = f['mass'].view(np.ndarray)
    n = len(mass)
    epssq = np.float(eps*eps)

    code = file(os.path.join(os.path.dirname(__file__),'direct.c')).read()
    inline(code,['nips','n','pos','mass','ipos','epssq','m_by_r','m_by_r2'])
    return m_by_r, m_by_r2

def treecalc(f, rs, eps= None) :
    gtree = tree.GravTree(f['pos'].view(np.ndarray),f['mass'].view(np.ndarray),eps=np.min(eps))
    a, p = gtree.calc(rs,eps=eps)
    return p, a

def midplane_rot_curve(f, rxy_points, eps = None, mode='tree') :
    
    if eps is None :
        try :
            eps = f['eps']
        except KeyError :
            eps = f.properties['eps']
            
    if isinstance(eps, str) :
        eps = units.Unit(eps)

    u_out = (units.G * f['mass'].units / f['pos'].units)**(1,2)
    
    # Do four samples like Tipsy does
    rs = [pos for r in rxy_points for pos in [(r,0,0), (0,r,0), (-r,0,0), (0,-r,0)]]

    try:
        fn = {'direct': direct,
              'tree': treecalc,
              }[mode]
    except KeyError :
        fn = mode

    #if mode == 'tree':
    #else:
    #    m_by_r, m_by_r2 = fn(f,np.array(rs), eps=np.min(eps))

    m_by_r, m_by_r2 = fn(f,np.array(rs), eps=np.min(eps))
    
    accel = array.SimArray(m_by_r2,units.G * f['mass'].units / (f['pos'].units**2) )

    vels = []

    i=0
    for r in rxy_points:
        r_acc_r = []
        for pos in [(r,0,0), (0,r,0), (-r,0,0), (0,-r,0)]:
            r_acc_r.append(np.dot(accel[i,:],pos))
            i = i+1
        
        vel2 = np.mean(r_acc_r)
        if vel2>0 :
            vel = math.sqrt(vel2)
        else :
            vel = 0

        import pdb; pdb.set_trace()
        vels.append(vel)

    x = array.SimArray(vels, units = u_out)
    x.sim = f.ancestor
    return x

def midplane_potential(f, rxy_points, eps = None) :
    
    u_out = units.G * f['mass'].units / f['pos'].units
    
    try:
        fn = {'dir': direct,
              'tree': tree,
              }[mode]
    except KeyError :
        fn = mode

    m_by_r, m_by_r2 = fn(f,np.array(rs),eps=eps)

    potential = units.G * m_by_r * f['mass'].units / f['pos'].units

    pots = []

    i=0
    for r in rxy_points :
        # Do four samples like Tipsy does
        pot = []
        for pos in [(r,0,0), (0,r,0), (-r,0,0), (0,-r,0)] :
            pot.append(potential[i])
            i=i+1
            
        pots.append(np.mean(pot))

    x = array.SimArray(pots, units = u_out)
    x.sim = f.ancestor
    return x
