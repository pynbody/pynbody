"""

profile
=======

"""

import numpy as np
from ..analysis import angmom, profile, halo
from .. import filt, units, config

import pylab as p

def rotation_curve(sim, center=True, r_units = 'kpc',
                   v_units = 'km s^-1', disk_height='100 pc', nbins=50,
                   bin_spacing = 'equaln', clear = True, quick=False,
                   filename=None,min=False,max=False,yrange=False,
                   legend=False, parts=False, axes=False, **kwargs) :
    """

    Centre on potential minimum, align so that the disk is in the
    x-y plane, then use the potential in that plane to generate and
    plot a rotation curve.

    **needs documentation/description of the keyword arguments**
    
    """
    import pylab as p

    if center :
        angmom.faceon(sim)

    if min :
        min_r = min
    else:
        min_r = sim['rxy'].min()
    if max :
        max_r = max
    else:
        max_r = sim['rxy'].max()

    pro = profile.Profile(sim, type=bin_spacing, nbins = nbins, 
                         min = min_r, max = max_r)

    r = pro['rbins'].in_units(r_units)
    if quick :
        v = pro['rotation_curve_spherical'].in_units(v_units)
    else :
        v = pro['v_circ'].in_units(v_units)

    if axes: p=axes
    else: 
        import pylab as p
        if clear : p.clf()


    if parts :
        p.plot(r, v, label='total',**kwargs)
        gpro = profile.Profile(sim.gas, type=bin_spacing, nbins = nbins,
                               min = min_r, max = max_r)
        dpro = profile.Profile(sim.dark, type=bin_spacing, nbins = nbins,
                               min = min_r, max = max_r)
        spro = profile.Profile(sim.star, type=bin_spacing, nbins = nbins,
                               min = min_r, max = max_r)
        if quick :
            gv = gpro['rotation_curve_spherical'].in_units(v_units)
            dv = dpro['rotation_curve_spherical'].in_units(v_units)
            sv = spro['rotation_curve_spherical'].in_units(v_units)
        else :
            gv = gpro['v_circ'].in_units(v_units)
            dv = dpro['v_circ'].in_units(v_units)
            sv = spro['v_circ'].in_units(v_units)
        p.plot(r,gv,"--",label="gas")
        p.plot(r,dv,label="dark")
        p.plot(r,sv,linestyle="dotted",label="star")
    else:
        p.plot(r, v,**kwargs)

    

    if yrange :
        p.axis([min_r,units.Unit(max_r).in_units(r.units),yrange[0],yrange[1]])
    
    if not axes:
        p.xlabel("r / $"+r.units.latex()+"$",fontsize='large')
        p.ylabel("v$_c / "+v.units.latex()+'$',fontsize='large')

    if legend :
        p.legend(loc=0)

    if (filename): 
        print "Saving "+filename
        p.savefig(filename)
    
    return r,v

def fourier_profile(sim, center=True, disk_height='2 kpc', nbins=50,
                    pretime='2 Gyr',r_units='kpc', bin_spacing = 'equaln', 
                    clear = True,min=False,max=False,filename=None,**kwargs) :
    """
    Centre on potential minimum, align so that the disk is in the
    x-y plane, then plot the amplitude of the 2nd fourier mode as a 
    function of radius.

    **needs description of the keyword arguments**

    """

    if center :
        angmom.faceon(sim)

    if min :
        min_r = min
    else:
        min_r = sim['rxy'].min()
    if max :
        max_r = max
    else:
        max_r = sim['rxy'].max()

    if isinstance(pretime, str):
        pretime = units.Unit(pretime)

    diskstars = sim.star[filt.Disc(max_r,disk_height)]
    youngstars = np.where(diskstars['tform'].in_units("Myr") > 
                          sim.properties['time'].in_units("Myr", **sim.conversion_context()) 
                          - pretime.in_units('Myr'))[0]

    pro = profile.Profile(diskstars[youngstars], type=bin_spacing, 
                          nbins = nbins, min = min_r, max = max_r)

    r = pro['rbins'].in_units(r_units)
    fourierprof = pro['fourier']
    a2 = fourierprof['amp'][2]

    if clear : p.clf()

    p.plot(r, a2, **kwargs)

    p.xlabel("r / $"+r.units.latex()+"$")
    p.ylabel("Amplitude of 2nd Fourier Mode")
    if (filename): 
        print "Saving "+filename
        p.savefig(filename)

def density_profile(sim, linestyle=False, center=True, clear=True, 
                    filename=None, **kwargs) :
    '''

    3d density profile
    
    **Options:**

    *filename* (None):  name of file to which to save output

    **Usage:**

    >>> import pynbody.plot as pp
    >>> h = s.halos()
    >>> pp.density_profile(h[1],linestyle='dashed',color='k')

    
    '''
    import matplotlib.pyplot as plt
    global config
    
    if center :
        if config['verbose']: print "Centering"
        halo.center(sim,mode='ssc')

    if config['verbose']: print "Creating profile"
    if 'min' in kwargs:
        ps = profile.Profile(sim,ndim=3,type='log',nbins=40,min=kwargs['min'])
        del kwargs['min']
    else:
        ps = profile.Profile(sim,ndim=3,type='log',nbins=40)
    if config['verbose']: print "Plotting"
    if clear : plt.clf()
    critden = (units.Unit('100 km s^-1 Mpc^-1')*sim.properties['h'])**2 /8.0/np.pi/units.G
    r=ps['rbins'].in_units('kpc')
    den=ps['density'].in_units(critden)
    if linestyle:
        plt.errorbar(r,den,yerr=den/np.sqrt(ps['n']),
                     linestyle=linestyle,**kwargs)
    else:
        plt.errorbar(r,den,yerr=den/np.sqrt(ps['n']),
                     fmt='o',**kwargs)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('r [kpc]')
    plt.ylabel(r'$\rho / \rho_{cr}$')#+den.units.latex()+'$]')
    if (filename): 
        if config['verbose']: print "Saving "+filename
        plt.savefig(filename)

