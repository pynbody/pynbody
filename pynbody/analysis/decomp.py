"""

decomp
======

Tools for bulge/disk/halo decomposition

"""

from . import angmom
from .. import array
from .. import filt, util
from .. import config
from . import profile
import numpy as np
import sys

def decomp(h, aligned=False, j_disk_min = 0.8, j_disk_max=1.1, E_cut = None, j_circ_from_r=False,
           cen=None, vcen=None, log_interp=False, angmom_size="3 kpc") :
    """
    Creates an array 'decomp' for star particles in the simulation, with an
    integer specifying a kinematic decomposition. The possible values are:

    1 -- thin disk

    2 -- halo

    3 -- bulge

    4 -- thick disk

    5 -- pseudo bulge

    This routine is based on an original IDL procedure by Chris Brook.


    **Parameters:**

    *h* -- the halo to work on

    *j_disk_min* -- the minimum angular momentum as a proportion of
                  the circular angular momentum which a particle must
                  have to be part of the 'disk'

    *j_disk_max* -- the maximum angular momentum as a proportion of
                  the circular angular momentum which a particle can
                  have to be part of the 'disk'

    *E_cut* -- the energy boundary between bulge and spheroid. If
             None, this is taken to be the median energy of the stars.

    *aligned* -- if False, the simulation is recenterd and aligned so
               the disk is in the xy plane as required for the rest of
               the analysis.

    *cen* -- if not None, specifies the center of the halo. Otherwise
           it is found.  This has no effect if aligned=True

    *vcen* -- if not None, specifies the velocity center of the
            halo. Otherwise it is found.  This has no effect if
            aligned=True

    *j_circ_from_r* -- if True, the maximum angular momentum is
    determined as a function of radius, rather than as a function of
    orbital energy. Default False (determine as function of energy).

    *angmom_size* -- the size of the gas sphere used to determine the
     plane of the disk

    """

    import scipy.interpolate as interp
    global config
        
    # Center, eliminate proper motion, rotate so that
    # gas disk is in X-Y plane
    if not aligned :
        angmom.faceon(h,cen=cen,vcen=vcen, disk_size=angmom_size)

    # Find KE, PE and TE
    ke = h['ke']
    pe = h['phi']

    h['phi'].convert_units(ke.units) # put PE and TE into same unit system

    te = ke+pe
    h['te'] = te
    te_star = h.star['te']

    te_max = te_star.max()

    # Add an arbitrary offset to the PE to reflect the idea that
    # the group is 'fully bound'.
    te-=te_max
    if config['verbose'] : print>>sys.stderr, "te_max = ",te_max

    h['te']-=te_max

    
    if config['verbose'] : print>>sys.stderr, "Making disk rotation curve..."
    
    # Now make a rotation curve for the disk. We'll take everything
    # inside a vertical height of eps*3

    d = h[filt.Disc('1 Mpc', h['eps'].min()*3)]
    
    try :
        
        # attempt to load rotation curve off disk
        r, v_c = np.loadtxt(h.ancestor.filename+".rot."+str(h.properties["halo_id"]), skiprows=1, unpack=True)
        
        pro_d = profile.Profile(d, nbins=100, type='log')
        r_me = pro_d["rbins"].in_units("kpc")
        r_x = np.concatenate(([0], r, [r.max()*2]))
        v_c = np.concatenate(([v_c[0]], v_c, [v_c[-1]]))
        v_c = interp.interp1d(r_x, v_c, bounds_error=False)(r_me)

            
        if config['verbose'] : 
            print>>sys.stderr, "   -- found existing rotation curve on disk, using that"
            
        v_c = v_c.view(array.SimArray)
        v_c.units = "km s^-1"
        v_c.sim = d

        v_c.convert_units(h['vel'].units)
        
        pro_d._profiles['v_circ'] = v_c
        pro_d.v_circ_loaded = True
        
    except :
        pro_d = profile.Profile(d, nbins=100, type='log') #.D()
        # Nasty hack follows to force the full halo to be used in calculating the
        # gravity (otherwise get incorrect rotation curves)
        pro_d._profiles['v_circ'] = profile.v_circ(pro_d, h)  
                                                    
    pro_phi = pro_d['phi']
    #import pdb; pdb.set_trace()
    # offset the potential as for the te array
    pro_phi-=te_max

    # (will automatically be reflected in E_circ etc)
    # calculating v_circ for j_circ and E_circ is slow

    if j_circ_from_r :
        pro_d.create_particle_array("j_circ", out_sim=h)
        pro_d.create_particle_array("E_circ", out_sim=h)
    else :
        
        if log_interp :
            j_from_E  = interp.interp1d(np.log10(-pro_d['E_circ'].in_units(ke.units))[::-1], np.log10(pro_d['j_circ'])[::-1], bounds_error=False)
            h['j_circ'] = 10**j_from_E(np.log10(-h['te']))
        else :
#            j_from_E  = interp.interp1d(-pro_d['E_circ'][::-1], (pro_d['j_circ'])[::-1], bounds_error=False)
            j_from_E  = interp.interp1d(pro_d['E_circ'].in_units(ke.units), pro_d['j_circ'], bounds_error=False)
            h['j_circ'] = j_from_E(h['te'])

        # The next line forces everything close-to-unbound into the
        # spheroid, as per CB's original script ('get rid of weird
        # outputs', it says).
        h['j_circ'][np.where(h['te']>pro_d['E_circ'].max())] = np.inf

        # There are only a handful of particles falling into the following category:
        h['j_circ'][np.where(h['te']< pro_d['E_circ'].min())] = pro_d['j_circ'][0]


    h['jz_by_jzcirc'] = h['j'][:,2]/h['j_circ']
    h_star = h.star

    if not h_star.has_key('decomp') :
        h_star._create_array('decomp', dtype=int)
    disk = np.where((h_star['jz_by_jzcirc']>j_disk_min)*(h_star['jz_by_jzcirc']<j_disk_max))

    h_star['decomp', disk[0]] = 1
    # h_star = h_star[np.where(h_star['decomp']!=1)]
    

    # Find disk/spheroid angular momentum cut-off to make spheroid
    # rotational velocity exactly zero.

    V = h_star['vcxy']
    JzJcirc = h_star['jz_by_jzcirc']
    te = h_star['te']


    if config['verbose'] : 
        print>>sys.stderr, "Finding spheroid/disk angular momentum boundary..."
    j_crit = util.bisect(0.,5.0,
                         lambda c : np.mean(V[np.where(JzJcirc<c)]))


        
    if config['verbose'] :
        print>>sys.stderr, "j_crit = ",j_crit
        if j_crit>j_disk_min :
            print>>sys.stderr, "!! j_crit exceeds j_disk_min. This is usually a sign that something is going wrong (train-wreck galaxy?) !!"
            print>>sys.stderr, "!! j_crit will be reset to j_disk_min =",j_disk_min,"!!"
            
    if j_crit>j_disk_min :
        j_crit = j_disk_min

    sphere = np.where(h_star['jz_by_jzcirc']<j_crit)


    if E_cut is None :
        E_cut = np.median(h_star['te'])

    if config['verbose'] : 
        print>>sys.stderr, "E_cut = ",E_cut


    halo =np.where((te>E_cut) * (JzJcirc<j_crit))
    bulge = np.where((te<=E_cut) * (JzJcirc<j_crit))
    pbulge = np.where((te<=E_cut) * (JzJcirc>j_crit) * ((JzJcirc<j_disk_min) + (JzJcirc>j_disk_max)) )
    thick = np.where((te>E_cut) * (JzJcirc>j_crit) * ((JzJcirc<j_disk_min) + (JzJcirc>j_disk_max)) )



    # h_star['decomp', disk] = 1
    h_star['decomp', halo] = 2
    h_star['decomp', bulge] = 3
    h_star['decomp', thick] = 4
    h_star['decomp', pbulge] = 5

    # Return profile object for informational purposes
    return pro_d
