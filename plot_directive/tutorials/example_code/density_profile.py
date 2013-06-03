import pynbody
import matplotlib.pylab as plt

# load the snapshot and set to physical units
s = pynbody.load('testdata/g15784.lr.01024.gz')

# load the halos
h = s.halos()

# center on the largest halo and align the disk
pynbody.analysis.angmom.faceon(h[1])

# convert all units to something reasonable (kpc, Msol etc)
s.physical_units()

# create a profile object for the stars (by default this is a 2D profile)
p = pynbody.analysis.profile.Profile(h[1].s,min=.01,max=50)

# make the plot
plt.subplot(211)
plt.plot(p['rbins'],p['density'])
plt.semilogy()
plt.xlabel('R [kpc]')
plt.ylabel(r'$\Sigma_{\star}$ [M$_{\odot}$ kpc$^{-2}$]')

# make a 3D density plot of the dark matter (note ndim=3 in the constructor below)
p = pynbody.analysis.profile.Profile(h[1].d,min=.01,max=50,ndim=3)
plt.subplot(212)
plt.plot(p['rbins'],p['density'])
plt.semilogy()
plt.xlabel('R [kpc]')
plt.ylabel(r'$\rho_{DM}$ [M$_{\odot}$ kpc$^{-3}$]')
