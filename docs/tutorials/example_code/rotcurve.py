from os import environ

import matplotlib.pylab as plt

import pynbody

# load the snapshot and set to physical units
s = pynbody.load('testdata/gasoline_ahf/g15784.lr.01024.gz')

# load the halos
h = s.halos()

# center on the largest halo and align the disk
pynbody.analysis.angmom.faceon(h[1])

# create a profile object for the stars
s.physical_units()
p = pynbody.analysis.profile.Profile(h[1],rmin=.01,rmax =250,type='log',ndim=3)
pg = pynbody.analysis.profile.Profile(h[1].g,rmin=.01,rmax =250,type='log',ndim=3)
ps = pynbody.analysis.profile.Profile(h[1].s,rmin=.01,rmax =250,type='log',ndim=3)
pd = pynbody.analysis.profile.Profile(h[1].d,rmin=.01,rmax =250,type='log',ndim=3)

# make the plot
plt.plot(p['rbins'],p['v_circ'],label='total')
plt.plot(pg['rbins'],pg['v_circ'],label='gas')
plt.plot(ps['rbins'],ps['v_circ'],label='stars')
plt.plot(pd['rbins'],pd['v_circ'],label='dm')

plt.xlabel('R [kpc]')
plt.ylabel(r'$v_c$ [km/s]')
plt.legend()
