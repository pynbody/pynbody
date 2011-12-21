import pynbody, sys
import matplotlib.pyplot as plt

tfile = sys.argv[1]
s = pynbody.load(tfile)
h = s.halos()

pynbody.analysis.angmom.faceon(h[1])
stars = h[1].s[pynbody.filt.LowPass('rxy','2 kpc')]
rxyhist, rxybins = np.histogram(stars['rxy'],bins=20)
rxyinds = np.digitize(stars['rxy'], rxybins)
nrbins = len(numpy.unique(rxyinds))
sigvz = np.zeros(nrbins)
sigvr = np.zeros(nrbins)
sigvt = np.zeros(nrbins)
rxy = np.zeros(nrbins)

for i, ind in enumerate(numpy.unique(rxyinds)):
    bininds = np.where(rxyinds == ind)
    sigvz[i] = np.std(stars[bininds]['vz'].in_units('km s^-1'))
    sigvr[i] = np.std(stars[bininds]['vr'].in_units('km s^-1'))
    sigvt[i] = np.std(stars[bininds]['vt'].in_units('km s^-1'))
    rxy[i] = np.mean(stars[bininds]['rxy'].in_units('kpc'))

plt.plot(rxy,sigvz,'o',label='$\sigma_{vz}$')
plt.plot(rxy,sigvr,'o',label='$\sigma_{vr}$')
plt.plot(rxy,sigvt,'o',label=r'$\sigma_{v \theta}$')
plt.xlabel('R$_{xy}$ [kpc]')
plt.ylabel('$\sigma$ [km s$^{-1}$]')

plt.legend(loc=0)

plt.savefig(tfile+'.vdisp.png')
