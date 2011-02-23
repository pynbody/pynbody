import numpy as np
import matplotlib.pyplot as plt
from ..analysis import angmom
from ..analysis import profile

def sfh(t,filename=None,massform=True,**kwargs):
    '''star formation history
    Usage:
    import pynbody.plot as pp
    pp.sfh(s,linestyle='dashed',color='k')

    By default, sfh will use the formation mass of the star.  In tipsy, this will be
    taken from the starlog file.  Set massform=False if you want the final (observed)
    star formation history
    '''
    nbins=100
    binnorm = 1e-9*nbins / (t.star['tform'].in_units("Gyr").max() - t.star['tform'].in_units("Gyr").min())
    if massform :
        try:
            weight = t.star['massform'].in_units('Msol') * binnorm
        except KeyError :
            weight = t.star['mass'].in_units('Msol') * binnorm
    else:
        weight = t.star['mass'].in_units('Msol') * binnorm
                                                               
    sfhist, bins, patches = plt.hist(t.star['tform'].in_units("Gyr"),weights=weight,
                                     bins=nbins,histtype='step',**kwargs)
    plt.xlabel('Time [Gyr]')
    plt.ylabel('SFR [M$_\odot$ yr$^{-1}$]')
    if (filename): plt.savefig(filename)


def schmidtlaw(t,center=True,filename=None,pretime=50,diskheight=3,rmax=20,radial=True,**kwargs):
    if center :
        angmom.faceon(t)
    
    # select stuff
    youngstars = np.where(t.star['tform'].in_units("Myr") > t.properties['time'].in_units("Myr") - pretime)[0]
    # * means "and"
    diskstars = np.where((np.abs(t.star['z'].in_units("kpc")) < diskheight) * (t.star['rxy'] < rmax))[0]
    youngdiskstars = np.intersect1d(youngstars, diskstars)
    
    diskgas = np.where((np.abs(t.gas['z'].in_units("kpc")) < diskheight) * (t.gas['rxy'] < rmax))[0]

    # calculate surface densities
    if radial :
        ps = profile.Profile(t.star[youngdiskstars])
        pg = profile.Profile(t.gas[diskgas])
    else :
        gashist, x, y = np.histogram2d(sim.gas['x'][diskgas], sim.gas['y'][diskgas],bins=nbins,range=[t_range,rho_range])
        starhist, x, y = np.histogram2d(sim.star['x'][youngdiskstars], sim.star['y'][youngdiskstars],bins=nbins,range=[t_range,rho_range])

    plt.loglog(pg['den'].in_units('Msol pc^-2'),ps['den'].in_units('Msol kpc^-2') / pretime/1e6,"+")
    xsigma = 10.0**np.linspace(np.log10(pg['den'].min()),np.log10(pg['den'].max()),100)
    ysigma=2.5e-4*xsigma**1.5        # Kennicutt (1998)
    ysigmanew=10.**(-2.1)*xsigma**1.0   # Bigiel et al (2007)
    plt.loglog(xsigma,ysigma,label='Kennicutt (1998)')
    plt.loglog(xsigma,ysigmanew,linestyle="dashed",label='Bigiel et al (2007)')
    plt.xlabel('$\Sigma_{gas}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    plt.ylabel('$\Sigma_{SFR}$ [M$_\odot$ pc$^{-2}$]')
    plt.legend(loc=2)

