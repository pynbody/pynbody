"""
Handy utilities for using RAMSES outputs in pynbody.
"""

import pynbody

hop = 

def load_hop(s, hop='~/ramses/galaxy_formation/script_hop.sh'): 
    """
    Loads the hop catalog for the given RAMSES snapshot. If the
    catalog doesn't exist, it tries to run hop to create one via the
    'script_hop.sh' script found in the RAMSES distribution. The hop
    output should be in a 'hop' directory in the base directory of the
    simulation.

    Input:
    ------

    *s* : loaded RAMSES snapshot

    Optional Keywords:
    ------------------

    *hop* : path to `script_hop.sh`

    """

    if s.filename[-1] == '/' : 
        name = s.filename[-6:-1] 
        filename = s.filename[:-13]+'hop/grp%s.pos'%name
    else: 
        name = s.filename[-5:]
        filename = s.filename[:-12]+'hop/grp%s.pos'%name
    
    try : 
        data = np.genfromtxt(filename,unpack=True)
    except IOError : 
        import os
        dir = s.filename[:-12] if len(s.filename[:-12]) else './'
        
        os.system('cd %s;/home/itp/roskar/ramses/galaxy_formation/script_hop.sh %d;cd ..'%(dir,int(name)))
        data = np.genfromtxt(filename,unpack=True)

    return data
