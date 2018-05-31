"""

halo
====

Functions for dealing with and manipulating halos in simulations.


"""

from .. import filt, util, config, array, units, transformation
from . import cosmology, _com, profile
import numpy as np
import math
import logging
logger = logging.getLogger('pynbody.analysis.halo')


def center_of_mass(sim):
    """

    Return the centre of mass of the SimSnap

    """
    mtot = sim["mass"].sum()
    p = np.sum(sim["mass"] * sim["pos"].transpose(), axis=1) / mtot

    # otherwise behaviour is numpy version dependent
    p.units = sim["pos"].units

    # only return position to be consistent with other functions in halo.py
    return p


def center_of_mass_velocity(sim):
    """

    Return the center of mass velocity of the SimSnap

    """
    mtot = sim["mass"].sum()
    v = np.sum(sim["mass"] * sim["vel"].transpose(), axis=1) / mtot
    # otherwise behaviour is numpy version dependent
    v.units = sim["vel"].units

    return v


def shrink_sphere_center(sim, r=None, shrink_factor=0.7, min_particles=100, verbose=False, num_threads = config['number_of_threads'],**kwargs):
    """

    Return the center according to the shrinking-sphere method of
    Power et al (2003)


    **Input**:

    *sim* : a simulation snapshot - this can be any subclass of SimSnap

    **Optional Keywords**:

    *r* (default=None): initial search radius. This can be a string
     indicating the unit, i.e. "200 kpc", or an instance of
     :func:`~pynbody.units.Unit`.

    *shrink_factor* (default=0.7): the amount to shrink the search
     radius by on each iteration

    *min_particles* (default=100): minimum number of particles within
     the search radius. When this number is reached, the search is
     complete.

    *verbose* (default=False): if True, prints out the diagnostics at
     each iteration. Useful to determine whether the centering is
     zeroing in on the wrong part of the simulation.

    """

    if r is None:

        # use rough estimate for a maximum radius
        # results will be insensitive to the exact value chosen
        r = (sim["x"].max() - sim["x"].min()) / 2

    elif isinstance(r, str) or issubclass(r.__class__, units.UnitBase):
        if isinstance(r, str):
            r = units.Unit(r)
        r = r.in_units(sim['pos'].units, **sim.conversion_context())

    mass = np.asarray(sim['mass'], dtype='double')
    pos = np.asarray(sim['pos'], dtype='double')

    if shrink_factor == 1.0:
        tol = sim['eps'].in_units(sim['pos'].units, **sim.conversion_context()).min()*0.1
        com = _com.shrink_sphere_center(pos, mass, min_particles, shrink_factor, r, num_threads)
        com = _com.move_sphere_center(pos, mass, min_particles, shrink_factor, r, tol)
    else:
        com = _com.shrink_sphere_center(pos, mass, min_particles, shrink_factor, r, num_threads)

    logger.info("Final SSC=%s", com)

    return array.SimArray(com, sim['pos'].units)


def virial_radius(sim, cen=None, overden=178, r_max=None):
    """Calculate the virial radius of the halo centered on the given
    coordinates.

    This is here defined by the sphere centered on cen which contains a
    mean density of overden * rho_M_0 * (1+z)^3.

    """

    if r_max is None:
        r_max = (sim["x"].max() - sim["x"].min())
    else:
        if cen is not None:
            sim = sim[filt.Sphere(r_max, cen)]
        else:
            sim = sim[filt.Sphere(r_max)]

    r_min = 0.0

    if cen is not None:
        tx = transformation.inverse_translate(sim, cen)
    else:
        tx = transformation.null(sim)

    target_rho = overden * \
        sim.properties[
            "omegaM0"] * cosmology.rho_crit(sim, z=0) * (1.0 + sim.properties["z"]) ** 3
    logger.info("target_rho=%s", target_rho)

    with tx:
        sim = sim[filt.Sphere(r_max)]
        with sim.immediate_mode:
            mass_ar = np.asarray(sim['mass'])
            r_ar = np.asarray(sim['r'])

        """
        #pure numpy implementation
        rho = lambda r: np.dot(
            mass_ar, r_ar < r) / (4. * math.pi * (r ** 3) / 3)

        #numexpr alternative - not much faster because sum is not threaded
        def rho(r) :
            r_ar; mass_ar; # just to get these into the local namespace
            return ne.evaluate("sum((r_ar<r)*mass_ar)")/(4.*math.pi*(r**3)/3)
        """
        rho = lambda r: util.sum_if_lt(mass_ar,r_ar,r)/(4. * math.pi * (r ** 3) / 3)
        result = util.bisect(r_min, r_max, lambda r: target_rho -
                             rho(r), epsilon=0, eta=1.e-3 * target_rho, verbose=False)

    return result


def potential_minimum(sim):
    i = sim["phi"].argmin()
    return sim["pos"][i].copy()


def hybrid_center(sim, r='3 kpc', **kwargs):
    """

    Determine the center of the halo by finding the shrink-sphere
    -center inside the specified distance of the potential minimum

    """

    try:
        cen_a = potential_minimum(sim)
    except KeyError:
        cen_a = center_of_mass(sim)
    return shrink_sphere_center(sim[filt.Sphere(r, cen_a)], **kwargs)


def index_center(sim, **kwargs):
    """

    Determine the center of mass based on specific particles.

    Supply a list of indices using the ``ind`` keyword.

    """

    try:
        ind = kwargs['ind']
        return center_of_mass(sim[ind])
    except KeyError:
        raise RuntimeError("Need to supply indices for centering")


def vel_center(sim, mode=None, cen_size="1 kpc", retcen=False, move_all=True, **kwargs):
    """Use stars from a sphere to calculate center of velocity. The size
    of the sphere is given by the ``cen_size`` keyword and defaults to
    1 kpc.

    **Keyword arguments:**

    *mode*: reserved for future use; currently ignored

    *move_all*: if True (default), move the entire snapshot. Otherwise only move
    the particles in the halo passed in.

    *retcen*: if True only return the velocity center without moving the
     snapshot (default = False)

    """

    logger.info("Finding halo velocity center...")

    if move_all:
        target = sim.ancestor
    else:
        target = sim

    cen = sim.star[filt.Sphere(cen_size)]
    if len(cen) < 5:
        # fall-back to DM
        cen = sim.dm[filt.Sphere(cen_size)]
    if len(cen) < 5:
        # fall-back to gas
        cen = sim.gas[filt.Sphere(cen_size)]
    if len(cen) < 5:
        # very weird snapshot, or mis-centering!
        raise ValueError, "Insufficient particles around center to get velocity"

    vcen = (cen['vel'].transpose() * cen['mass']).sum(axis=1) / \
        cen['mass'].sum()
    vcen.units = cen['vel'].units
    if config['verbose']:
        logger.info("vcen=%s", vcen)

    if retcen:
        return vcen
    else:
        return transformation.v_translate(target, -vcen)


def center(sim, mode=None, retcen=False, vel=True, cen_size="1 kpc", move_all=True, wrap=False, **kwargs):
    """

    Determine the center of mass of the given particles using the
    specified mode, then recenter the particles (of the entire
    ancestor snapshot) accordingly

    Accepted values for *mode* are

      *pot*: potential minimum

      *com*: center of mass

      *ssc*: shrink sphere center

      *ind*: center on specific particles; supply the list of particles using the ``ind`` keyword.

      *hyb*: for sane halos, returns the same as ssc, but works faster by
             starting iteration near potential minimum

    or a function returning the COM.

    **Other keywords:**

    *retcen*: if True only return the center without centering the
     snapshot (default = False)

    *ind*: only used when *mode=ind* -- specifies the indices of
     particles to be used for centering

    *vel*: if True, translate velocities so that the velocity of the
    central 1kpc (default) is zeroed. Other values can be passed with cen_size.

    *move_all*: if True (default), move the entire snapshot. Otherwise only move
    the particles in the halo passed in.

    *wrap*: if True, pre-centre and wrap the simulation so that halos on the edge
    of the box are handled correctly. Default False.
    """

    global config
    if mode is None:
        mode = config['centering-scheme']

    try:
        fn = {'pot': potential_minimum,
              'com': center_of_mass,
              'ssc': shrink_sphere_center,
              'hyb': hybrid_center,
              'ind': index_center}[mode]
    except KeyError:
        fn = mode

    if move_all:
        target = sim.ancestor
    else:
        target = sim

    if wrap:
        # centre on something within the halo and wrap
        target = transformation.inverse_translate(target, sim['pos'][0])
        target.sim.wrap()

    if retcen:
        return fn(sim, **kwargs)
    else:
        cen = fn(sim, **kwargs)
        tx = transformation.inverse_translate(target, cen)

    if vel:
        velc = vel_center(sim, cen_size=cen_size, retcen=True)
        tx = transformation.inverse_v_translate(tx, velc)

    return tx

def halo_shape(sim, N=100, rin=0, rout=0, bins='equal'):

    """

    Return the axis ratios b/a and c/a, and the alignment angle, for homeoidal shells
    over a range of N radii. Set 'bins' to 'equal' for an equal number of particles
    per bin, and 'log' for logarithmic bins.
    The central radii of each bin are also returned.
    Output is in the order: bin radius, b/a, c/a, alignment angle, rotation matrix

    Caution is advised when assigning large number of bins and radial ranges with many
    particles, as the algorithm becomes very slow.

    """

    #--------------------------------FUNCTIONS--------------------------------------
    # Define an ellipsoid shell with lengths a,b,c and orientation E:
    def Ellipsoid(r, a,b,c, E):
      x,y,z = np.dot(E,[r[:,0],r[:,1],r[:,2]])
      return (x/a)**2 + (y/b)**2 + (z/c)**2

    # Define moment of inertia tensor:
    MoI = lambda r,m: np.array([[np.sum(m*r[:,i]*r[:,j]) for j in range(3)] for i in range(3)])

    # Splits data into number of steps N:
    split = lambda r,N: np.append([r[i*len(r)/N:(1+i)*len(r)/N][0] for i in range(N)],r[-1])

    # Retrieves alignment angle:
    almnt = lambda E: np.arccos(np.dot(np.dot(E,[1.,0.,0.]),[1.,0.,0.]))
    #--------------------------------FUNCTIONS--------------------------------------

    posr = np.array(sim.dm['r'])[np.where(sim.dm['r']<rout)[0]]
    pos  = np.array(sim.dm['pos'])[np.where(sim.dm['r']<rout)[0]]
    mass = np.array(sim.dm['mass'])[np.where(sim.dm['r']<rout)[0]]

    rotx = [[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]
    roty = [[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
    rotz = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]

    # Define bins:
    if (rout == 0): rout = np.max(posr)
    if (rin == 0):  rin  = rout/1E3

    if (bins == 'equal'): # Each bin contains equal number of particles
        mid  = split(np.sort(posr[np.where((posr>=rin) & (posr<=rout))[0]]),N*2)
        rbin = mid[1:N*2+1:2] ; mid = mid[0:N*2+1:2]

    elif (bins == 'log'): # Bins are logarithmically spaced
        mid  = profile.Profile(sim.dm, type='log', ndim=3, min=rin, max=rout, nbins=N+1)['rbins']
        rbin = np.sqrt(mid[0:N]*mid[1:N+1])

    # Define b/a and c/a ratios and angle arrays:
    ba,ca,angle = np.zeros(N),np.zeros(N),np.zeros(N)
    Es = [0]*N

    # Begin loop through radii:
    for i in range(0,N):

        # Initialise convergence criterion:
        tol   = 1E-3
        count = 0

        # Define initial spherical shell:
        a=b=c = rbin[i]
        E     = np.identity(3)
        L1,L2 = rbin[i]-mid[i],mid[i+1]-rbin[i]

        # Begin iterative procedure to fit data to shell:
        while True:
            count+= 1

            # Collect all particle positions and masses within homoeoid ellipsoidal shell:
            r     = pos[np.where((posr<a+L2) & (posr>c-L1*c/a))[0]]
            inner = Ellipsoid(r, a-L1,b-L1*b/a,c-L1*c/a, E)
            outer = Ellipsoid(r, a+L2,b+L2*b/a,c+L2*c/a, E)
            r     =    r[np.where((inner>1.) & (outer<1.))]
            m     = mass[np.where((inner>1.) & (outer<1.))]

            # End iterations if there is no data in range: [Either due to extreme axis ratios or bad data]
            if (len(r)==0):
                ba[i],ca[i],angle[i],Es[i] = b/a,c/a,almnt(E),E
                logger.info('No data in range after %i iterations. Ratios b/a, c/a = %.3f %.3f' %(count,b/a,c/a))
                break

            # Calculate shape tensor & diagonalise for eigenvalues and eigenvectors:
            D = list(np.linalg.eig(MoI(r,m)/np.sum(m)))

            # Purge complex numbers [This will produce unrealistic parameters for this iteration]:
            if isinstance(D[1][0,0],complex):
                D[0] = D[0].real ; D[1] = D[1].real
                logger.info('Complex numbers in D removed...')

            # Compute ratios a,b,c from moment of intertia principles [eigenvalues]:
            anew,bnew,cnew = np.sqrt(abs(D[0])*3.0)

            # a,b,c do not necessarily remain a>b>c. The rotation matrix must be reoriented:
            if ((anew>bnew) & (bnew>=cnew)): E=D[1]
            if ((bnew>anew) & (anew>=cnew)): E=np.dot(D[1],rotz)
            if ((cnew>anew) & (anew>=bnew)): E=np.dot(np.dot(D[1],rotz),rotx)
            if ((bnew>cnew) & (cnew>=anew)): E=np.dot(np.dot(D[1],rotz),roty)
            if ((anew>cnew) & (cnew>=bnew)): E=np.dot(D[1],rotx)
            if ((cnew>bnew) & (bnew>=anew)): E=np.dot(D[1],roty)
            if (almnt(-E)<almnt(E)): E=-E
            cnew,bnew,anew = np.sort(np.sqrt(abs(D[0])*3.0))

            # Keep a as semi-major axis and distort b,c by b/a and c/a:
            div   = rbin[i]/anew
            anew *= div
            bnew *= div
            cnew *= div

            # Convergence criterion: Fractional difference between old and new axis ratios:
            if (np.abs(b/a - bnew/anew)<tol) & (np.abs(c/a - cnew/anew)<tol):
                ba[i],ca[i],angle[i],Es[i] = bnew/anew,cnew/anew,almnt(E),E
                break

            # Increase tolerance if convergence has stagnated [multiply by 5 every 10 iterations]:
            elif (count%10 == 0): tol *= 5.

            # Reset a,b,c for the next iteration:
            a,b,c = anew,bnew,cnew

    return [array.SimArray(rbin, sim.d['pos'].units), ba, ca, angle, Es]
