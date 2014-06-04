"""

stars
=====

"""

import numpy as np
import matplotlib,matplotlib.pyplot as plt
from ..analysis import profile, angmom, halo
from .. import filt, units, config, array
from .sph import image
import warnings
from .. import units as _units

def bytscl(arr,mini=0,maxi=10000):
    X= (arr-mini)/(maxi-mini)
    X[X>1] = 1
    X[X<0] = 0
    return X

def nw_scale_rgb(r,g,b,scales=[4,3.2,3.4]):
    return r*scales[0], g*scales[1], b*scales[2]

def nw_arcsinh_fit(r,g,b,nonlinearity=3):
    radius = r+g+b
    val=np.arcsinh(radius*nonlinearity)/nonlinearity/radius
    return r*val,g*val,b*val

def combine(r,g,b,dynamic_range):
    maxi = []
    
    # find something close to the maximum that is not quite the maximum
    for x in r,g,b :
        ordered = np.sort(x.flatten())
        maxi.append(ordered[-len(ordered)/5000])

    maxi = np.log10(max(maxi))
    
    rgbim=np.zeros((r.shape[0],r.shape[1],3))
    rgbim[:,:,0]=bytscl(np.log10(r),maxi-dynamic_range,maxi)
    rgbim[:,:,1]=bytscl(np.log10(g),maxi-dynamic_range,maxi)
    rgbim[:,:,2]=bytscl(np.log10(b),maxi-dynamic_range,maxi)
    return rgbim

def render(sim,filename=None,
           r_band='i',g_band='v',b_band='u',
           r_scale=0.5, g_scale = 1.0, b_scale = 1.0,
           dynamic_range=2.0,
           width=50,
           starsize=None, 
           plot=True, axes=None, ret_im=False,clear=True):
    '''
    Make a 3-color image of stars.

    The colors are based on magnitudes found using stellar Marigo
    stellar population code.  However there is no radiative transfer
    to account for dust.
    
    Returns: If ret_im=True, an NxNx3 array representing an RGB image
    
    **Optional keyword arguments:**
    
       *filename*: string (default: None)
         Filename to be written to (if a filename is specified)

       *r_band*: string (default: 'i')
         Determines which Johnston filter will go into the image red channel

       *g_band*: string (default: 'v')
         Determines which Johnston filter will go into the image green channel

       *b_band*: string (default: 'b')
         Determines which Johnston filter will go into the image blue channel

       *r_scale*: float (default: 0.5)
         The scaling of the red channel before channels are combined

       *g_scale*: float (default: 1.0)
         The scaling of the green channel before channels are combined

       *b_scale*: float (default: 1.0)
         The scaling of the blue channel before channels are combined

       *width*: float in kpc (default:50)
         Sets the size of the image field in kpc

       *starsize*: float in kpc (default: None)
         If not None, sets the maximum size of stars in the image

       *ret_im*: bool (default: False)
         if True, the NxNx3 image array is returned

       *plot*: bool (default: True)
         if True, the image is plotted

       *axes*: matplotlib axes object (deault: None)
         if not None, the axes object to plot to

       *dynamic_range*: float (default: 2.0)
         The number of dex in luminosity over which the image brightness ranges
    '''
    
    if isinstance(width,str) or issubclass(width.__class__,_units.UnitBase) : 
        if isinstance(width,str) : 
            width = _units.Unit(width)
        width = width.in_units(sim['pos'].units,**sim.conversion_context())
    
    if starsize is not None :
        smf = filt.HighPass('smooth',str(starsize)+' kpc')
        sim.s[smf]['smooth'] = array.SimArray(starsize, 'kpc', sim=sim)
    
    r=image(sim.s,qty=r_band+'_lum_den',width=width,log=False,
                         av_z=True,clear=False,noplot=True) * r_scale
    g=image(sim.s,qty=g_band+'_lum_den',width=width,log=False,
                         av_z=True,clear=False,noplot=True) * g_scale
    b=image(sim.s,qty=b_band+'_lum_den',width=width,log=False,
                         av_z=True,clear=False,noplot=True) * b_scale

    #r,g,b = nw_scale_rgb(r,g,b)
    #r,g,b = nw_arcsinh_fit(r,g,b)

    rgbim=combine(r,g,b,dynamic_range)

    if plot :
        if clear: plt.clf()
        if axes is None:
            axes=plt.gca()

        if axes:
            axes.imshow(rgbim[::-1,:],extent=(-width/2,width/2,-width/2,width/2))
            axes.set_xlabel('x ['+str(sim.s['x'].units)+']')
            axes.set_ylabel('y ['+str(sim.s['y'].units)+']')
            plt.draw()
        
    if filename : 
        plt.savefig(filename)
        
    if ret_im:
        return rgbim

def sfh(sim,filename=None,massform=True,clear=False,legend=False,
        subplot=False, trange=False, bins=100, **kwargs):
    '''
    star formation history

    **Optional keyword arguments:**

       *trange*: list, array, or tuple
         size(t_range) must be 2. Specifies the time range.

       *bins*: int
         number of bins to use for the SFH

       *massform*: bool
         decides whether to use original star mass (massform) or final star mass

       *subplot*: subplot object
         where to plot SFH

       *legend*: boolean
         whether to draw a legend or not

       *clear*: boolean
         if False (default), plot on the current axes. Otherwise, clear the figure first.

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
    
    if "nbins" in kwargs: bins = kwargs['nbins']

    if 'nbins' in kwargs: 
        bins=kwargs['nbins']
        del kwargs['nbins']

    if trange:
        assert len(trange) == 2
    else:
        trange = [sim.star['tform'].in_units("Gyr").min(),sim.star['tform'].in_units("Gyr").max()]
    binnorm = 1e-9*bins / (trange[1] - trange[0])

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
    sfhist, thebins, patches = plt.hist(tforms, weights=weight, bins=bins,
                                     histtype='step',**kwargs)
    if not subplot:
        plt.ylim(0.0,1.2*np.max(sfhist))
        plt.xlabel('Time [Gyr]',fontsize='large')
        plt.ylabel('SFR [M$_\odot$ yr$^{-1}$]',fontsize='large')
    else:
        plt.set_ylim(0.0,1.2*np.max(sfhist))

    # Make both axes have the same start and end point.
    if subplot: x0,x1 = plt.get_xlim()
    else: x0,x1 = plt.gca().get_xlim()
    from pynbody.analysis import pkdgrav_cosmo as cosmo
    c = cosmo.Cosmology(sim=sim)
    pz = plt.twiny()
    labelzs = np.arange(5,int(sim.properties['z'])-1,-1)
    times = [13.7*c.Exp2Time(1.0 / (1+z))/c.Exp2Time(1) for z in labelzs]
    pz.set_xticks(times)
    pz.set_xticklabels([str(x) for x in labelzs])
    pz.set_xlim(x0, x1)
    pz.set_xlabel('z')

    if legend: plt.legend(loc=1)
    if filename : 
        if config['verbose']: print "Saving "+filename
        plt.savefig(filename)

    return array.SimArray(sfhist, "Msol yr**-1"), array.SimArray(thebins, "Gyr")

def schmidtlaw(sim,center=True,filename=None,pretime='50 Myr',
               diskheight='3 kpc',rmax='20 kpc', compare=True,
               radial=True,clear=True,legend=True,bins=10,**kwargs):
    '''Schmidt Law

    Plots star formation surface density vs. gas surface density including
    the observed relationships.  Currently, only plots densities found in
    radial annuli.

    **Usage:**
    
    >>> import pynbody.plot as pp
    >>> pp.schmidtlaw(h[1])

    **Optional keyword arguments:**

       *center*: bool
         center and align the input simulation faceon.

       *filename*: string
         Name of output file

       *pretime* (default='50 Myr'): age of stars to consider for SFR

       *diskheight* (default='3 kpc'): height of gas and stars above
          and below disk considered for SF and gas densities.
         
       *rmax* (default='20 kpc'): radius of disk considered

       *compare* (default=True):  whether to include Kennicutt (1998) and
            Bigiel+ (2008) for comparison

       *radial* (default=True):  should bins be annuli or a rectangular grid?

       *bins* (default=10):  How many radial bins should there be?

       *legend*: boolean
         whether to draw a legend or not
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
        ps = profile.Profile(diskstars[youngstars],nbins=bins)
        pg = profile.Profile(diskgas,nbins=bins)
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

       *pretime* (default='50 Myr'): age of stars to consider for SFR

       *diskheight* (default='3 kpc'): height of gas and stars above
          and below disk considered for SF and gas densities.
         
       *rmax* (default='20 kpc'): radius of disk considered
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


def satlf(sim,band='v',filename=None, MWcompare=True, Trentham=True, 
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
    #try :
    for haloid in sim.properties['children'] :
        if (sim._halo_catalogue.contains(haloid)) :
            halo = sim._halo_catalogue[haloid]
            try:
                halo.properties[band+'_mag'] = lum.halo_mag(halo,band=band)
                if np.isfinite(halo.properties[band+'_mag']):
                    halomags.append(halo.properties[band+'_mag'])
            except IndexError:
                pass  # no stars in satellite
    #except KeyError:
        #raise KeyError, str(sim)+' properties have no children key as a halo type would'
    
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
              center=True, clear=True, filename=None, axes=False, fit_exp=False,
              print_ylabel=True, fit_sersic=False, **kwargs) :
    '''

    surface brightness profile
    
    **Usage:**

    >>> import pynbody.plot as pp
    >>> h = s.halos()
    >>> pp.sbprofile(h[1],exp_fit=3,linestyle='dashed',color='k')

    **Options:**

    *band* ('v'): which Johnson band to use. available filters: U, B,
                     V, R, I, J, H, K

    *fit_exp*(False): Fits straight exponential line outside radius specified.

    *fit_sersic*(False): Fits Sersic profile outside radius specified.

    *diskheight('3 kpc')*
    *rmax('20 kpc')*:  Size of disk to be profiled

    *binning('equaln')*:  How show bin sizes be determined? based on 
          pynbody.analysis.profile

    *center(True)*:  Automatically align face on and center?

    *axes(False)*: In which axes (subplot) should it be plotted?

    *filename* (None): name of file to which to save output

    **needs a description of all keywords**

    By default, sbprof will use the formation mass of the star.  
    In tipsy, this will be taken from the starlog file. 
    
    '''
    
    if center :
        if config['verbose']: print "Centering"
        angmom.faceon(sim)

    if config['verbose']: print "Selecting disk stars"
    diskstars = sim.star[filt.Disc(rmax,diskheight)]
    if config['verbose']: print "Creating profile"
    ps = profile.Profile(diskstars, type=binning)
    if config['verbose']: print "Plotting"
    r=ps['rbins'].in_units('kpc')

    if axes: plt=axes
    else: import matplotlib.pyplot as plt
    if clear : plt.clf()

    plt.plot(r,ps['sb,'+band],linewidth=2,**kwargs)
    if axes:
        plt.set_ylim(max(ps['sb,'+band]),min(ps['sb,'+band]))
    else:
        plt.ylim(max(ps['sb,'+band]),min(ps['sb,'+band]))
    if fit_exp:
        exp_inds = np.where(r.in_units('kpc') > fit_exp)
        expfit = np.polyfit(np.array(r[exp_inds]), 
                          np.array(ps['sb,'+band][exp_inds]), 1)
        # 1.0857 is how many magnitudes a 1/e decrease is
        print "h: ",1.0857/expfit[0],"  u_0:",expfit[1]
        fit = np.poly1d(expfit)
        if 'label' in kwargs:
            del kwargs['label']
        if 'linestyle' in kwargs:
            del kwargs['linestyle']
        plt.plot(r,fit(r),linestyle='dashed',**kwargs)
    if fit_sersic:
        sersic_inds = np.where(r.in_units('kpc') < fit_sersic)
        sersicfit = np.polyfit(np.log10(np.array(r[sersic_inds])),
                               np.array(ps['sb,'+band][sersic_inds]), 1)
        fit = np.poly1d(sersicfit)
        print "n: ",sersicfit[0],"  other: ",sersicfit[1]
        if 'label' in kwargs:
            del kwargs['label']
        if 'linestyle' in kwargs:
            del kwargs['linestyle']
        plt.plot(r,fit(r),linestyle='dashed',**kwargs)
        #import pdb; pdb.set_trace()
    if axes:
        if print_ylabel:
            plt.set_ylabel(band+'-band Surface brightness [mag as$^{-2}$]')
    else:
        plt.xlabel('R [kpc]')
        plt.ylabel(band+'-band Surface brightness [mag as$^{-2}$]')
    if filename: 
        if config['verbose']: print "Saving "+filename
        plt.savefig(filename)


def moster(xmasses,z):
    '''Based on Moster+ (2013) return what stellar mass corresponds to the
    halo mass passed in.

    **Usage**
    
       >>> from pynbody.plot.stars import moster
       >>> xmasses = np.logspace(np.log10(min(totmasshalos)),1+np.log10(max(totmasshalos)),20)
       >>> ystarmasses, errors = moster(xmasses,halo_catalog._halos[1].properties['z'])
       >>> plt.fill_between(xmasses,np.array(ystarmasses)/np.array(errors),
                         y2=np.array(ystarmasses)*np.array(errors),
                         facecolor='#BBBBBB',color='#BBBBBB')
    '''
    hmp = np.log10(xmasses)
    # from Moster et al (2013)                                                  
    M10a=11.590470
    M11a=1.194913
    R10a=0.035113
    R11a=-0.024729
    B10a=1.376177
    B11a=-0.825820
    G10a=0.608170
    G11a=0.329275

    M10e = 0.236067
    M11e = 0.353477
    R10e = 0.00577173
    R11e = 0.00693815
    B10e = 0.153
    B11e = 0.225
    G10e = 0.059
    G11e = 0.173

    a = 1.0/(z+1.0)
    m1 = M10a+M11a*(1.0-a)
    r  = R10a+R11a*(1.0-a)
    b  = B10a+B11a*(1.0-a)
    g  = G10a+G11a*(1.0-a)
    smp= hmp+np.log10(2.0*r)-np.log10((10.0**(hmp-m1))**(-b)+(10.0**(hmp-m1))**
(g))
    eta = np.exp(np.log(10.)*(hmp-m1))
    alpha = eta**(-b)+eta**g
    dmdm10 = (g*eta**g+b*eta**(-b))/alpha
    dmdm11 = (g*eta**g+b*eta**(-b))/alpha*(1.0-a)
    dmdr10 = np.log10(np.exp(1.0))/r
    dmdr11 = np.log10(np.exp(1.0))/r*(1.0-a)
    dmdb10 = np.log10(np.exp(1.0))/alpha*np.log(eta)*eta**(-b)
    dmdb11 = np.log10(np.exp(1.0))/alpha*np.log(eta)*eta**(-b)*(1.0-a)
    dmdg10 = -np.log10(np.exp(1.0))/alpha*np.log(eta)*eta**g
    dmdg11 = -np.log10(np.exp(1.0))/alpha*np.log(eta)*eta**g*(1.0-a)
    sigma = np.sqrt(dmdm10*dmdm10*M10e*M10e+dmdm11*dmdm11*M11e*M11e+dmdr10*dmdr10*R10e*R10e+dmdr11*dmdr11*R11e*R11e+dmdb10*dmdb10*B10e*B10e+dmdb11*dmdb11*B11e
*B11e+dmdg10*dmdg10*G10e*G10e+dmdg11*dmdg11*G11e*G11e)
    return 10**smp, 10**sigma


def guo(halo_catalog, clear=False, compare=True, baryfrac=False,
        filename=False,**kwargs):
    '''Stellar Mass vs. Halo Mass
    
    Takes a halo catalogue and plots the member stellar masses as a
    function of halo mass.

    Usage:
    
    >>> import pynbody.plot as pp
    >>> h = s.halos()
    >>> pp.guo(h,marker='+',markerfacecolor='k')
    
    **Options:**

    *compare* (True): Should comparison line be plotted?
         If compare = 'guo', Guo+ (2010) plotted instead of Moster+ (2012)

    *baryfrac* (False):  Should line be drawn for cosmic baryon fraction?

    *filename* (None): name of file to which to save output
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
            ystarmasses, errors = moster(xmasses,halo_catalog._halos[1].properties['z'])
        plt.fill_between(xmasses,np.array(ystarmasses)/np.array(errors),
                         y2=np.array(ystarmasses)*np.array(errors),
                         facecolor='#BBBBBB',color='#BBBBBB')
        plt.loglog(xmasses,ystarmasses,label='Moster et al (2012)')

    if baryfrac :
        xmasses = np.logspace(np.log10(min(totmasshalos)),1+np.log10(max(totmasshalos)),20)
        ystarmasses = xmasses*0.04/0.24
        plt.loglog(xmasses,ystarmasses,linestyle='dotted',label='f_b = 0.16')
   
    plt.axis([0.8*min(totmasshalos),1.2*max(totmasshalos),
              0.8*min(starmasshalos),1.2*max(starmasshalos)])

    if (filename): 
        if config['verbose']: print "Saving "+filename
        plt.savefig(filename)

