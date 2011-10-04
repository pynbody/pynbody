"""

stars
=====

"""

import numpy as np
import matplotlib.pyplot as plt
from ..analysis import profile, angmom, halo
from .. import filt, units, config, array
import warnings

def sfh(sim,filename=None,massform=True,clear=True,legend=False,
        subplot=False, trange=False, nbins=100, **kwargs):
    '''
    star formation history

    **Optional keyword arguments:**

       *trange*: list, array, or tuple
         size(t_range) must be 2. Specifies the time range.

       *nbins*: int
         number of bins to use for the SFH

       *massform*: bool
         decides whether to use original star mass (massform) or final star mass

       *subplot*: subplot object
         where to plot SFH

       *legend*: boolean
         whether to draw a legend or not

    By default, sfh will use the formation mass of the star.  In tipsy, this will be
    taken from the starlog file.  Set massform=False if you want the final (observed)
    star formation history

    **Usage:**

    >>> import pynbody.plot as pp
    >>> pp.sfh(s,linestyle='dashed',color='k')

    
    '''

    if subplot:
        plt = subplot
    else :
        import matplotlib.pyplot as plt

    if trange:
        assert len(trange) == 2
    else:
        trange = [sim.star['tform'].in_units("Gyr").min(),sim.star['tform'].in_units("Gyr").max()]
    binnorm = 1e-9*nbins / (trange[1] - trange[0])

    trangefilt = filt.And(filt.HighPass('tform',str(trange[0])+' Gyr'), 
                          filt.LowPass('tform',str(trange[1])+' Gyr'))
    tforms = sim.star[trangefilt]['tform'].in_units('Gyr')

    if massform :
        try:
            weight = sim.star[trangefilt]['massform'].in_units('Msol') * binnorm
        except (KeyError, units.UnitsException) :
            warnings.warn("Could not load massform array -- falling back to current stellar masses", RuntimeWarning)
            weight = sim.star[trangefilt]['mass'].in_units('Msol') * binnorm
    else:
        weight = sim.star[trangefilt]['mass'].in_units('Msol') * binnorm
                                                               
    if clear : plt.clf()
    sfhist, bins, patches = plt.hist(tforms, weights=weight, bins=nbins,
                                     histtype='step',**kwargs)
    if not subplot:
        plt.xlabel('Time [Gyr]',fontsize='large')
        plt.ylabel('SFR [M$_\odot$ yr$^{-1}$]',fontsize='large')

    if legend: plt.legend(loc=1)
    if (filename): 
        if config['verbose']: print "Saving "+filename
        plt.savefig(filename)

    return array.SimArray(sfhist, "Msol yr**-1"), array.SimArray(bins, "Gyr")

def schmidtlaw(sim,center=True,filename=None,pretime='50 Myr',
               diskheight='3 kpc',rmax='20 kpc', compare=True,
               radial=True,clear=True,legend=True,**kwargs):
    '''Schmidt Law

    Plots star formation surface density vs. gas surface density including
    the observed relationships.  Currently, only plots densities found in
    radial annuli.

    **Usage:**
    
    >>> import pynbody.plot as pp
    >>> pp.schmidtlaw(h[1])

    **needs a description of keywords**
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

    if compare:
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
    if legend : plt.legend(loc=2)
    if (filename): 
        print "Saving "+filename
        plt.savefig(filename)


def oneschmidtlawpoint(sim,center=True,pretime='50 Myr',
                       diskheight='3 kpc',rmax='20 kpc',**kwargs):
    '''One Schmidt Law Point

    Determines values for star formation surface density and gas surface 
    density for the entire galaxy based on the half mass cold gas radius.

    **Usage:**
    import pynbody.plot as pp
    pp.oneschmidtlawpoint(h[1])


    **needs a description of keywords**
    '''
    
    if center :
        angmom.faceon(sim)

    cg = h[1].gas[filt.LowPass('temp', 1e5)]
    cgd = cg[filt.Disc('30 kpc','3 kpc')]
    cgs = np.sort(cgd['rxy'].in_units('kpc'))
    rhgas = cgs[len(cgs)/2]
    instars = h[1].star[filt.Disc(str(rhgas)+' kpc', '3 kpc')]
    minstars = np.sum(instars[filt.LowPass('age','100 Myr')]['mass'].in_units('Msol'))
    ingasm = np.sum(cg[filt.Disc(str(rhgas)+' kpc', '3 kpc')]['mass'].in_units('Msol'))
    rpc = rhgas * 1000.0
    rkpc = rhgas
    xsigma = ingasm / (np.pi*rpc*rpc)
    ysigma = minstars / (np.pi*rkpc*rkpc*1e8)
    return xsigma, ysigma


def satlf(sim,band='V',filename=None, MWcompare=True, Trentham=True, 
          clear=True, legend=True,
          label='Simulation',**kwargs) :
    '''

    satellite luminosity function

    **Options:**

    *band* ('v'): which Johnson band to use. available filters: U, B,
    V, R, I, J, H, K

    *filename* (None): name of file to which to save output

    *MWcompare* (True): whether to plot comparison lines to MW

    *Trentham* (True): whether to plot comparison lines to Trentham +
                     Tully (2009) combined with Koposov et al (2007)

    By default, satlf will use the formation mass of the star.  In
    tipsy, this will be taken from the starlog file.

    **Usage:**

    >>> import pynbody.plot as pp
    >>> h = s.halos()
    >>> pp.satlf(h[1],linestyle='dashed',color='k')

    
    '''
    from ..analysis import luminosity as lum
    import os

    halomags = []
    try :
        for haloid in sim.properties['children'] :
            if (sim._halo_catalogue.contains(haloid)) :
                halo = sim._halo_catalogue[haloid]
                try:
                    halo.properties[band+'_mag'] = lum.halo_mag(halo,band=band)
                    halomags.append(halo.properties[band+'_mag'])
                except IndexError:
                    pass  # no stars in satellite
    except KeyError:
        raise KeyError, str(sim)+' properties have no children key as a halo type would'
    
    if clear : plt.clf()
    plt.semilogy(sorted(halomags),np.arange(len(halomags))+1, label=label,
                 **kwargs)
    plt.xlabel('M$_{'+band+'}$')
    plt.ylabel('Cumulative LF')
    if MWcompare:
        # compare with observations of MW
        tolfile = os.path.join(os.path.dirname(__file__),"tollerud2008mw")
        if os.path.exists(tolfile) :
            tolmags = [float(q) for q in file(tolfile).readlines()]
        else :
            raise IOError, tolfile+" not found"
        plt.semilogy(sorted(tolmags),np.arange(len(tolmags)),
                     label='Milky Way')

    if Trentham:
        halomags = np.array(halomags)
        halomags = halomags[np.asarray(np.where(np.isfinite(halomags)))]
        xmag = np.linspace(halomags.min(),halomags.max(),100)
        # Trentham + Tully (2009) equation 6
        # number of dwarfs between -11>M_R>-17 is well correlated with mass
        logNd = 0.91*np.log10(sim.properties['mass'])-10.2
        # set Nd from each equal to combine Trentham + Tully with Koposov
        coeff = 10.0**logNd / (10**-0.6 - 10**-1.2)

        #print 'Koposov coefficient:'+str(coeff)
        # Analytic expression for MW from Koposov
        #import pdb; pdb.set_trace()
        yn=coeff * 10**((xmag+5.0)/10.0) # Koposov et al (2007)
        plt.semilogy(xmag,yn,linestyle="dashed",
                     label='Trentham & Tully (2009)')

    if legend : plt.legend(loc=2)
    if (filename): 
        print "Saving "+filename
        plt.savefig(filename)


def sbprofile(sim, band='v',diskheight='3 kpc', rmax='20 kpc', binning='equaln',
              center=True, clear=True, filename=None, **kwargs) :
    '''

    surface brightness profile
    
    **Options:**

    *band* ('v'): which Johnson band to use. available filters: U, B,
                     V, R, I, J, H, K

    *filename* (None): name of file to which to save output

    **needs a description of all keywords**

    By default, sbprof will use the formation mass of the star.  
    In tipsy, this will be taken from the starlog file. 

    **Usage:**

    >>> import pynbody.plot as pp
    >>> h = s.halos()
    >>> pp.sbprofile(h[1],linestyle='dashed',color='k')
    
    '''
    
    if center :
        if config['verbose']: print "Centering"
        angmom.faceon(sim)

    if config['verbose']: print "Selecting disk stars"
    diskstars = sim.star[filt.Disc(rmax,diskheight)]
    if config['verbose']: print "Creating profile"
    ps = profile.Profile(diskstars, type=binning)
    if config['verbose']: print "Plotting"
    if clear : plt.clf()
    r=ps['rbins'].in_units('kpc')
    plt.plot(r,ps['sb,'+band],'o',**kwargs)
    plt.axis([min(r),max(r),max(ps['sb,'+band]),min(ps['sb,'+band])])
    plt.xlabel('R [kpc]')
    plt.ylabel(band+'-band Surface brightness [mag as$^{-2}$]')
    if (filename): 
        if config['verbose']: print "Saving "+filename
        plt.savefig(filename)


def guo(halo_catalog, clear=False, compare=True, baryfrac=False,
        filename=False,**kwargs):
    '''Stellar Mass vs. Halo Mass
    
    Takes a halo catalogue and plots the member stellar masses as a
    function of halo mass.

    Options: 

    *filename* (None): name of file to which to save output
    
    **needs a description of all keyword arguments**

    Usage:
    
    >>> import pynbody.plot as pp
    >>> h = s.halos()
    >>> pp.guo(h,marker='+',markerfacecolor='k')
    
    '''

    #if 'marker' not in kwargs :
    #    kwargs['marker']='o'

    starmasshalos = []
    totmasshalos = []

    halo_catalog._halos[1]['mass'].convert_units('Msol')

    for i in np.arange(len(halo_catalog._halos))+1 :
        halo = halo_catalog[i]
        halostarmass = np.sum(halo.star['mass'])
        if halostarmass :
            starmasshalos.append(halostarmass)
            totmasshalos.append(np.sum(halo['mass']))

    if clear: plt.clf()

    plt.loglog(totmasshalos,starmasshalos,'o',**kwargs)
    plt.xlabel('Total Halo Mass')
    plt.ylabel('Halo Stellar Mass')

    if compare :
        xmasses = np.logspace(np.log10(min(totmasshalos)),1+np.log10(max(totmasshalos)),20)
        if compare == 'guo':
            # from Sawala et al (2011) + Guo et al (2009)
            ystarmasses = xmasses*0.129*((xmasses/2.5e11)**-0.926 + (xmasses/2.5e11)**0.261)**-2.44
        else :
            # from Moster et al (2010)
            mM0 = 0.0282
            beta = -1.057
            gamma = 0.5560
            M1 = 10**(11.884)
            mu = 0.019
            nu = -0.72
            gamma1 = -0.26
            beta1 = 0.17
            z=halo[1].properties['z']

            M1_z = 10**(np.log10(M1)*(z+1)**mu)
            mM0_z = mM0*(z+1)**nu
            gamma_z = gamma*(z+1)**gamma1
            beta_z  = beta+ beta1*z
            
            ratio = 2.0*mM0_z/((xmasses/M1_z)**beta_z + (xmasses/M1_z)**gamma_z)
            ystarmasses = ratio*xmasses
        plt.loglog(xmasses,ystarmasses,linestyle='dashed',label='Moster et al (2009)')

    if baryfrac :
        xmasses = np.logspace(np.log10(min(totmasshalos)),1+np.log10(max(totmasshalos)),20)
        ystarmasses = xmasses*0.04/0.24
        plt.loglog(xmasses,ystarmasses,linestyle='dotted',label='f_b = 0.16')
   
    plt.axis([0.8*min(totmasshalos),1.2*max(totmasshalos),
              0.8*min(starmasshalos),1.2*max(starmasshalos)])

    if (filename): 
        if config['verbose']: print "Saving "+filename
        plt.savefig(filename)

