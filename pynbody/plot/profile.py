from .. import analysis
from ..analysis import angmom
from ..analysis import profile
from .. import filt

import pylab as p

def rotation_curve(s, center=True, r_units = 'kpc',
                   v_units = 'km s^-1', disk_height='100 pc', nbins=50,
                   bin_spacing = 'equaln', **kwargs) :
    """Centre on potential minimum, align so that the disk is in the
    x-y plane, then use the potential in that plane to generate and
    plot a rotation curve."""

    if center :
        angmom.faceon(s)

    if 'min' in kwargs :
        min_r = kwargs['min']
    else:
        min_r = s['rxy'].min()
    if 'max' in kwargs :
        max_r = kwargs['max']
    else:
        max_r = s['rxy'].max()

    pro = profile.Profile(s, min = min_r, max = max_r, nbins = nbins, type=bin_spacing)

    r = pro['rbins'].in_units(r_units)
    v = pro['v_circ'].in_units(v_units)

    p.plot(r, v)

    p.xlabel("$r/"+r.units.latex()+"$")
    p.ylabel("$v_c/"+v.units.latex()+'$')
