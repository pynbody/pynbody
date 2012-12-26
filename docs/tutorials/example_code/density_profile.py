import pynbody
import matplotlib.pylab as plt
from os import environ
# load the snapshot and set to physical units
s = pynbody.load('%s/nose/testdata/g15784.lr.01024.gz'%environ['PYNBODY_SRC'])

# load the halos
h = s.halos()

# center on the largest halo and align the disk
pynbody.analysis.angmom.faceon(h[1])

# create a profile object for the stars
s.physical_units()
p = pynbody.analysis.profile.Profile(h[1].s,min=.01,max=50)

# make the plot
plt.plot(p['rbins'],p['density'])
plt.semilogy()
plt.xlabel('R [kpc]')
plt.ylabel(r'$\rho$ [M$_{\odot}$ kpc$^{-3}$]')
