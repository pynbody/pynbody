"""decomp.py

Tools for bulge/disk/halo decomposition"""

from . import angmom
from .. import filt, util
from . import profile
import numpy as np

def decomp(h, aligned=False, j_disk_min = 0.8, j_disk_max=1.1, E_cut = None, j_circ_from_r=False,
	   cen=None, vcen=None, verbose=True, log_interp=False) :
    """
    Creates an array 'decomp' for star particles in the simulation, with an
    integer specifying a kinematic decomposition. The possible values are:

    1 -- thin disk
    2 -- halo
    3 -- bulge
    4 -- thick disk
    5 -- pseudo bulge

    This routine is based on an original IDL procedure by Chris Brook.
    
    
    Parameters:

    h -- the halo to work on
    j_disk_min -- the minimum angular momentum as a proportion of the circular
                  angular momentum which a particle must have to be part of the 'disk'
    j_disk_max -- the maximum angular momentum as a proportion of the circular
                  angular momentum which a particle can have to be part of the 'disk'
    E_cut -- the energy boundary between bulge and spheroid. If None, this is taken
             to be the median energy of the stars.
    aligned -- if False, the simulation is recentred and aligned so the disk is
               in the xy plane as required for the rest of the analysis.
    cen -- if not None, specifies the centre of the halo. Otherwise it is found.
           This has no effect if aligned=True
    vcen -- if not None, specifies the velocity centre of the halo. Otherwise it is found.
            This has no effect if aligned=True
    j_circ_from_r -- if True, the maximum angular momentum is determined as a
    function of radius, rather than as a function of orbital energy. Default
    False (determine as function of energy).
    verbose -- if True, print information
    """
    
    # Centre, eliminate proper motion, rotate so that
    # gas disk is in X-Y plane
    if not aligned :
	angmom.faceon(h,cen=cen,vcen=vcen, verbose=verbose)
	
	# Derive or rederive quantities of interest
	h.derive('ke')
	h.derive('j')


	
    
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
    print "te_max = ",te_max

    h['te']-=te_max
    
    
    
    print "Making disk rotation curve..."
    
    # Now make a rotation curve for the disk. We'll take everything
    # inside a vertical height of 100pc.
    
    d = h[filt.Disc('1 Mpc', '100 pc')]
    pro_d = profile.Profile(d, nbins=100, type='equaln').D()
    pro_phi = pro_d['phi']

    # offset the potential as for the te array
    pro_phi-=te_max
    
    # (will automatically be reflected in E_circ etc)
    
    if j_circ_from_r :
	pro_d.create_particle_array("j_circ", out_sim=h)
	pro_d.create_particle_array("E_circ", out_sim=h)
    else :
	import scipy.interpolate as interp
        if log_interp :
            j_from_E  = interp.interp1d(np.log10(-pro_d['E_circ'])[::-1], np.log10(pro_d['j_circ'])[::-1], bounds_error=False)
            h['j_circ'] = 10**j_from_E(np.log10(-h['te']))
        else :
            j_from_E  = interp.interp1d(-pro_d['E_circ'][::-1], (pro_d['j_circ'])[::-1], bounds_error=False)
            h['j_circ'] = j_from_E(-h['te'])
        
	# The next line forces everything close-to-unbound into the
	# spheroid, as per CB's original script ('get rid of weird
	# outputs', it says). 
	h['j_circ'][np.where(h['te']>pro_d['E_circ'].max())] = np.inf

	# There are only a handful of particles falling into the following category:
	h['j_circ'][np.where(h['te']<pro_d['E_circ'].min())] = pro_d['j_circ'][0]


    h['jz_by_jzcirc'] = h['j'][:,2]/h['j_circ']
    h_star = h.star
    
    if not h_star.has_key('decomp') :
	h_star._create_array('decomp', dtype=int)
    disk = np.where((h_star['jz_by_jzcirc']>j_disk_min)*(h_star['jz_by_jzcirc']<j_disk_max))
    h_star['decomp', disk[0]] = 1

    # Find disk/spheroid angular momentum cut-off to make spheroid
    # rotational velocity exactly zero.

    V = h_star['vcxy']
    JzJcirc = h_star['jz_by_jzcirc']
    te = h_star['te']
    

    if verbose:
        print "Finding spheroid/disk angular momentum boundary..."
    j_crit = util.bisect(0.,1.0,
			 lambda c : np.mean(V[np.where(JzJcirc<c)]))

    if verbose:
        print "j_crit = ",j_crit
    sphere = np.where(h_star['jz_by_jzcirc']<j_crit)
    

    if E_cut is None :
        E_cut = np.median(h_star['te'])

    if verbose :
        print "E_cut = ",E_cut

    

    halo =np.where((te>E_cut) * (JzJcirc<j_crit))
    bulge = np.where((te<=E_cut) * (JzJcirc<j_crit))
    pbulge = np.where((te<=E_cut) * (JzJcirc>j_crit) * ((JzJcirc<j_disk_min) + (JzJcirc>j_disk_max)) )
    thick = np.where((te>E_cut) * (JzJcirc>j_crit) * ((JzJcirc<j_disk_min) + (JzJcirc>j_disk_max)) )
    
		    
			   
    h_star['decomp', disk] = 1
    h_star['decomp', halo] = 2
    h_star['decomp', bulge] = 3
    h_star['decomp', thick] = 4
    h_star['decomp', pbulge] = 5
    
