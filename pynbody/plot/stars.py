import numpy as np
import matplotlib.pyplot as plt
from ..analysis import profile, angmom

def sfh(sim,filename=None,massform=True,**kwargs):
    '''star formation history
    Usage:
    import pynbody.plot as pp
    pp.sfh(s,linestyle='dashed',color='k')

    By default, sfh will use the formation mass of the star.  In tipsy, this will be
    taken from the starlog file.  Set massform=False if you want the final (observed)
    star formation history
    '''
    nbins=100
    binnorm = 1e-9*nbins / (sim.star['tform'].in_units("Gyr").max() - sim.star['tform'].in_units("Gyr").min())
    if massform :
        try:
            weight = sim.star['massform'].in_units('Msol') * binnorm
        except KeyError :
            weight = sim.star['mass'].in_units('Msol') * binnorm
    else:
        weight = sim.star['mass'].in_units('Msol') * binnorm
                                                               
    sfhist, bins, patches = plt.hist(sim.star['tform'].in_units("Gyr"),weights=weight,
                                     bins=nbins,histtype='step',**kwargs)
    plt.xlabel('Time [Gyr]')
    plt.ylabel('SFR [M$_\odot$ yr$^{-1}$]')
    if (filename): plt.savefig(filename)


def schmidtlaw(sim,center=True,filename=None,pretime=50,diskheight=3,rmax=20,radial=True,**kwargs):
   
    if not radial :
        print 'only radial Schmidt Law supported at the moment'
        return
    
    if center :
        angmom.faceon(sim)

    # select stuff
    diskgas = sim.gas[filt.Disc(rmax,diskheight)]
    diskstars = sim.star[filt.Disc(rmax,diskheight)]

    youngstars = np.where(diskstars['tform'].in_units("Myr") > t.properties['time'].in_units("Myr") - pretime)[0]

    # calculate surface densities
    if radial :
        ps = profile.Profile(diskstars[youngstars])
        pg = profile.Profile(diskgas)
    else :
        # make bins 2 kpc
        nbins = rmax * 2 / binsize
        pg, x, y = np.histogram2d(diskgas['x'], diskgas['y'],bins=nbins,
                                  weights=diskgas['mass'],
                                  range=[(-rmax,rmax),(-rmax,rmax)])
        ps, x, y = np.histogram2d(diskstars[youngstars]['x'], diskstars[youngstars]['y'],
                                  weights=diskstars['mass'],
                                  bins=nbins,range=[(-rmax,rmax),(-rmax,rmax)])

    plt.loglog(pg['den'].in_units('Msol pc^-2'),ps['den'].in_units('Msol kpc^-2') / pretime/1e6,"+")
    xsigma = 10.0**np.linspace(np.log10(pg['den'].min()),np.log10(pg['den'].max()),100)
    ysigma=2.5e-4*xsigma**1.5        # Kennicutt (1998)
    ysigmanew=10.**(-2.1)*xsigma**1.0   # Bigiel et al (2007)
    plt.loglog(xsigma,ysigma,label='Kennicutt (1998)')
    plt.loglog(xsigma,ysigmanew,linestyle="dashed",label='Bigiel et al (2007)')
    plt.xlabel('$\Sigma_{gas}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    plt.ylabel('$\Sigma_{SFR}$ [M$_\odot$ pc$^{-2}$]')
    plt.legend(loc=2)

