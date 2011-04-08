import numpy as np
import matplotlib.pyplot as plt
from ..analysis import profile, angmom, halo
from .. import filt, units

def sfh(sim,filename=None,massform=True,clear=True,**kwargs):
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
        except (KeyError, units.UnitsException) :
            weight = sim.star['mass'].in_units('Msol') * binnorm
    else:
        weight = sim.star['mass'].in_units('Msol') * binnorm
                                                               
    if clear : plt.clf()
    sfhist, bins, patches = plt.hist(sim.star['tform'].in_units("Gyr"),
                                     weights=weight, bins=nbins,
                                     histtype='step',**kwargs)
    plt.xlabel('Time [Gyr]')
    plt.ylabel('SFR [M$_\odot$ yr$^{-1}$]')
    if (filename): 
        print "Saving "+filename
        plt.savefig(filename)


def schmidtlaw(sim,center=True,filename=None,pretime='50 Myr',
               diskheight='3 kpc',rmax='20 kpc',
               radial=True,clear=True,**kwargs):
    '''Schmidt Law
    Usage:
    import pynbody.plot as pp
    pp.schmidtlaw(h[1])

    Plots star formation surface density vs. gas surface density including
    the observed relationships.  Currently, only plots densities found in
    radial annuli.
    '''
    
    if not radial :
        print 'only radial Schmidt Law supported at the moment'
        return
    
    if center :
        angmom.faceon(sim)

    if isinstance(pretime, str):
        pretime = units.Unit(pretime)

    # select stuff
    diskgas = sim.gas[filt.Disc(rmax,diskheight)]
    diskstars = sim.star[filt.Disc(rmax,diskheight)]

    youngstars = np.where(diskstars['tform'].in_units("Myr") > 
                          sim.properties['time'].in_units("Myr", **sim.conversion_context()) 
                          - pretime.in_units('Myr'))[0]

    # calculate surface densities
    if radial :
        ps = profile.Profile(diskstars[youngstars],nbins=10)
        pg = profile.Profile(diskgas,nbins=10)
    else :
        # make bins 2 kpc
        nbins = rmax * 2 / binsize
        pg, x, y = np.histogram2d(diskgas['x'], diskgas['y'],bins=nbins,
                                  weights=diskgas['mass'],
                                  range=[(-rmax,rmax),(-rmax,rmax)])
        ps, x, y = np.histogram2d(diskstars[youngstars]['x'], 
                                  diskstars[youngstars]['y'],
                                  weights=diskstars['mass'],
                                  bins=nbins,range=[(-rmax,rmax),(-rmax,rmax)])

    if clear : plt.clf()
    plt.loglog(pg['density'].in_units('Msol pc^-2'),
               ps['density'].in_units('Msol kpc^-2') / pretime/1e6,"+",
               **kwargs)
    xsigma = np.logspace(np.log10(pg['density'].in_units('Msol pc^-2')).min(),
                         np.log10(pg['density'].in_units('Msol pc^-2')).max(),
                         100)
    ysigma=2.5e-4*xsigma**1.5        # Kennicutt (1998)
    xbigiel=np.logspace(1,2,10)
    ybigiel=10.**(-2.1)*xbigiel**1.0   # Bigiel et al (2007)
    plt.loglog(xsigma,ysigma,label='Kennicutt (1998)')
    plt.loglog(xbigiel,ybigiel,linestyle="dashed",label='Bigiel et al (2007)')
    plt.xlabel('$\Sigma_{gas}$ [M$_\odot$ pc$^{-2}$]')
    plt.ylabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    plt.legend(loc=2)
    if (filename): 
        print "Saving "+filename
        plt.savefig(filename)


def satlf(sim,band='R',filename=None, compare=True,clear=True,**kwargs) :
    '''satellite luminosity function
    Usage:
    import pynbody.plot as pp
    h = s.halos()
    pp.satlf(h[1],linestyle='dashed',color='k')

    Options:
    * band='v'       which Johnson band to use. available filters:  
                     U, B, V, R, I, J, H, K
    * filename=None  name of file to which to save output
    * compare=True   whether to plot comparison lines to MW

    By default, satlf will use the formation mass of the star.  
    In tipsy, this will be taken from the starlog file. 
    '''
    from ..analysis import luminosity as lum
    import os

    halomags = []
    #try :
    for haloid in sim.properties['children'] :
        if (sim._halo_catalogue.contains(haloid)) :
            halo = sim._halo_catalogue[haloid]
            try:
                halo.properties[band+'_mag'] = lum.halo_mag(halo,band=band)
                halomags.append(halo.properties[band+'_mag'])
            except IndexError:
                pass  # no stars in satellite
    #except KeyError:
        #raise KeyError, str(sim)+' properties have no children key as a halo type would'
    
    if clear : plt.clf()
    plt.semilogy(sorted(halomags),np.arange(len(halomags)), label='Simulation',
                 **kwargs)
    plt.xlabel('M'+band)
    plt.ylabel('Cumulative LF')
    if (compare):
        # compare with observations of MW
        tolfile = os.path.join(os.path.dirname(__file__),"tollerud2008mw")
        if os.path.exists(tolfile) :
            tolmags = [float(q) for q in file(tolfile).readlines()]
        else :
            raise IOError, "cmdlum.npz not found"
        plt.semilogy(sorted(tolmags),np.arange(len(tolmags)),
                     label='MW (Tollerud et al 1998)')

        halomags = np.array(halomags)
        halomags = halomags[np.asarray(np.where(np.isfinite(halomags)))]
        xmag = np.linspace(halomags.min(),halomags.max(),100)
        # Trentham + Tully (2009) equation 6
        # number of dwarfs between -11>M_R>-17 is well correlated with mass
        logNd = 0.91*np.log10(sim.properties['mass'])-10.2
        # set Nd from each equal to combine Trentham + Tully with Koposov
        coeff = 10.0**logNd / (10**-0.6 - 10**-1.2)

        print 'Koposov coefficient:'+str(coeff)
        # Analytic expression for MW from Koposov
        yn=coeff * 10**((xmag+5.0)/10.0) # Koposov et al (2007)
        plt.semilogy(xmag,yn,linestyle="dashed",
                     label='Koposov et al (2007) + Trentham & Tully (2009)')

    plt.legend(loc=2)
    if (filename): 
        print "Saving "+filename
        plt.savefig(filename)


def sbprofile(sim, band='v',diskheight='3 kpc', rmax='20 kpc', 
              center=True, clear=True, filename=None, **kwargs) :
    
    if center :
        print "Centering"
        angmom.faceon(sim)

    print "Selecting disk stars"
    diskstars = sim.star[filt.Disc(rmax,diskheight)]
    print "Creating profile"
    ps = profile.Profile(diskstars)
    print "Plotting"
    if clear : plt.clf()
    r=ps['rbins'].in_units('kpc')
    plt.plot(r,ps['sb'],**kwargs)
    plt.axis([min(r),max(r),max(ps['sb']),min(ps['sb'])])
    plt.xlabel('R [kpc]')
    plt.ylabel('Surface brightness [mag as$^{-2}$]')
    if (filename): 
        print "Saving "+filename
        plt.savefig(filename)
