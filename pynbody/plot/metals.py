import numpy as np
import matplotlib.pyplot as plt
from ..analysis import profile

def mdf(sim,filename=None,**kwargs):
    '''Metallicity Distribution Function
    Usage:
    import pynbody.plot as pp
    pp.mdf(s,linestyle='dashed',color='k')
    '''
    nbins=100
    metpdf, bins, patches = plt.hist(t.star['feh'],weights=t.star['mass'],
                                     bins=nbins,histtype='step',normed=True,
                                     **kwargs)
    plt.xlabel('[Fe / H]')
    plt.ylabel('PDF')
    if (filename): plt.savefig(filename)


def schmidtlaw(t,center=True,filename=None,pretime=50,diskheight=3,rmax=20,radial=True,**kwargs):
   
    if not radial :
        print 'only radial Schmidt Law supported at the moment'
        return
    
    if center :
        angmom.faceon(t)

    # select stuff
    diskgas = t.gas[filt.Disc(rmax,diskheight)]
    diskstars = t.star[filt.Disc(rmax,diskheight)]

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

