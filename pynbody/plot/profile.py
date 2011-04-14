import numpy as np
from ..analysis import angmom, profile
from .. import filt, units

import pylab as p

def rotation_curve(sim, center=True, r_units = 'kpc',
                   v_units = 'km s^-1', disk_height='100 pc', nbins=50,
                   bin_spacing = 'equaln', clear = True, quick=False,
                   filename=None,min=False,max=False,**kwargs) :
    """Centre on potential minimum, align so that the disk is in the
    x-y plane, then use the potential in that plane to generate and
    plot a rotation curve."""

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

    p.xlabel("r / $"+r.units.latex()+"$")
    p.ylabel("v_c / $"+v.units.latex()+'$')
    if (filename): 
        print "Saving "+filename
        p.savefig(filename)


def fourier_profile(sim, center=True, disk_height='2 kpc', nbins=50,
                    pretime='2 Gyr',r_units='kpc', bin_spacing = 'equaln', 
                    clear = True,min=False,max=False,filename=None,**kwargs) :
    """Centre on potential minimum, align so that the disk is in the
    x-y plane, then plot the amplitude of the 2nd fourier mode as a 
    function of radius."""

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
