import numpy as np

import pynbody as pyn


# Generate a fake little slab with a gas fraction of 0.5.  Size should be power of 8.
# The inner 10 kpc has stars that are 10 Myr old, and the outer 10 kpc has stars that are 90 Myr old.
def make_disc(size=8**5):
    size = int(size)
    s = pyn.new(dm=0, g=size, s=size)
    s.properties['time'] = pyn.units.Unit('100 Myr')
    prange = np.linspace(-1,1,int(np.round(size**(1/3))))
    s.s['mass'] = np.ones(size)
    s.g['mass'] = np.ones(size)
    s.g['x'], s.g['y'], s.g['z'] = map(np.ravel, np.meshgrid(prange,prange,0.1*prange)) # 1x1x0.1 slab
    s.s['x'], s.s['y'], s.s['z'] = map(np.ravel, np.meshgrid(prange,prange,0.1*prange)) # 1x1x0.1 slab
    s['pos'].units = 'kpc'
    s['mass'].units = 'Msol'

    # Set stellar ages
    s.s['tform'] = 90*np.ones(size)
    s.s['tform'][s.s['r'] > 0.5 ] /= 9
    s.s['tform'].units = 'Myr'

    # Scale disk
    s['x'] *= 20
    s['y'] *= 20
    s['z'] *= 3

    s['vel'] = 0.0
    s['vx'] = -s['y']
    s['vy'] = s['x']

    return s
