import numpy as np
import matplotlib.pyplot as plt
from ..analysis import angmom
from ..analysis import profile

def sfh(t,filename=None,**kwargs):
    nbins=100
    sfhist, bins, patches = plt.hist(t.star['tform'].in_units("Gyr"),
                                     weights=t.star['mass'].in_units('Msol')*1e-9*nbins / (t.star['tform'].in_units("Gyr").max() - t.star['tform'].in_units("Gyr").min()),
                                     bins=nbins,histtype='step',color='k')
    plt.xlabel('Time [Gyr]')
    plt.ylabel('SFR [M$_\odot$ yr$^{-1}$]')
    if (filename): plt.savefig(filename,**kwargs)


def schmidtlaw(t,center=True,filename=None,pretime=50,diskheight=3,maxr=20,radial=True,**kwargs):
    if center :
        angmom.faceon(t)
    
    # select stuff
    youngstars = np.where(t.star['tform'].in_units("Myr") > t.properties['time'].in_units("Myr") - pretime)
    diskstars = np.where((np.abs(t.star['z'].in_units("kpc")) < diskheight) and (t.star['rxy'] < rmax))
    youngdiskstars = np.intersect1d(youngstars, diskstars)
    
    diskgas = np.where((np.abs(t.gas['z'].in_units("kpc")) < diskheight) and (t.gas['rxy'] < rmax))

    # calculate surface densities
    if radial :
        ps = profile.Profile(t.star[youngdiskstars])
#        ps.rho
        pg = profile.Profile(t.gas[diskgas])
#        pg.rho
    else :
        gashist, x, y = np.histogram2d(sim.gas['x'][diskgas], sim.gas['y'][diskgas],bins=nbins,range=[t_range,rho_range])
        starhist, x, y = np.histogram2d(sim.star['x'][youngdiskstars], sim.star['y'][youngdiskstars],bins=nbins,range=[t_range,rho_range])

    plt.plot(pg['rho'],ps['rho'],mark="+")
    ysigma=2.5e-4*xsigma**1.5        # Kennicutt (1998)
    ysigmanew=10.**(-2.1)*xsigma**1.0   # Bigiel et al (2007)
    plt.plot(xsigma,ysigma)
    plt.plot(xsigma,ysigmanew,line="--")

