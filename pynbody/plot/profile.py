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
                   legend=False, parts=False, **kwargs) :
    """
    
    Centre on potential minimum, align so that the disk is in the
    x-y plane, then use the potential in that plane to generate and
    plot a rotation curve.

    **needs documentation/description of the keyword arguments**
    
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

    pro = profile.Profile(sim, type=bin_spacing, nbins = nbins, 
                         min = min_r, max = max_r)

    r = pro['rbins'].in_units(r_units)
    if quick :
        v = pro['rotation_curve_spherical'].in_units(v_units)
    else :
        v = pro['v_circ'].in_units(v_units)

    if clear : p.clf()

    p.plot(r, v, **kwargs)

    if parts :
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
        p.plot(r,gv,linestyle="dotted",label="gas")
        p.plot(r,dv,label="dark")
        p.plot(r,sv,"--",label="star")


    if yrange :
        p.axis([min_r,units.Unit(max_r).in_units(r.units),yrange[0],yrange[1]])
    p.xlabel("r / $"+r.units.latex()+"$",fontsize='large')
    p.ylabel("v$_c / "+v.units.latex()+'$',fontsize='large')

    if legend :
        p.legend(loc=0)

    if (filename): 
        print "Saving "+filename
        p.savefig(filename)


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
        halo.center(sim)

    if config['verbose']: print "Creating profile"
    ps = profile.Profile(sim,dim=3,type='log')
    if config['verbose']: print "Plotting"
    if clear : plt.clf()
    r=ps['rbins'].in_units('kpc')
    if linestyle:
        plt.loglog(r,ps['density'],linestyle=linestyle,**kwargs)
    else:
        plt.loglog(r,ps['density'],'o',**kwargs)
    plt.xlabel('r [kpc]')
    plt.ylabel('Density [$'+ps['density'].units.latex()+'$]')
    if (filename): 
        if config['verbose']: print "Saving "+filename
        plt.savefig(filename)

