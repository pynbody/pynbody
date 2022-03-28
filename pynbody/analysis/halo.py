"""

halo
====

Functions for dealing with and manipulating halos in simulations.


"""

import logging
import math

import numpy as np

from .. import array, config, filt, transformation, units, util
from . import _com, cosmology, profile

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


def virial_radius(sim, cen=None, overden=178, r_max=None, rho_def='matter'):
    """Calculate the virial radius of the halo centered on the given
    coordinates.

    The default is here defined by the sphere centered on cen which contains a
    mean density of overden * rho_M_0 * (1+z)^3.

    **Input**:

    *sim* : a simulation snapshot - this can be any subclass of SimSnap, especially a halo.

    **Optional Keywords**:

    *cen* (default=None): Provides the halo center. If None, assumes that the snapshot is already centered.

    *rmax (default=None): Maximum radius to start the search. If None, inferred from the halo particle positions.

    *overden (default=178): Overdensity corresponding to the required halo boundary definition.
    178 is the virial criterion for spherical collapse in an EdS Universe. Common possible values are 200, 500 etc

    *rho_def (default='matter'): Physical density used to define the overdensity. Default is the matter density at
    the redshift of the simulation. An other choice is "critical" for the critical density at this redshift.

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

    if rho_def == 'matter':
       ref_density = sim.properties["omegaM0"] * cosmology.rho_crit(sim, z=0) * (1.0 + sim.properties["z"]) ** 3
    elif rho_def == 'critical':
        ref_density = cosmology.rho_crit(sim, z=sim.properties["z"])
    else:
        raise ValueError(rho_def + "is not a valid definition for the reference density")

    target_rho = overden * ref_density
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
        raise ValueError("Insufficient particles around center to get velocity")

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

def halo_shape(sim, N=100, rin=None, rout=None, bins='equal'):
    """
    Returns radii in units of ``sim['pos']``, axis ratios b/a and c/a,
    the alignment angle of axis a in radians, and the rotation matrix
    for homeoidal shells over a range of N halo radii.

    **Keyword arguments:**

    *N* (100): The number of homeoidal shells to consider. Shells with
    few particles will take longer to fit.

    *rin* (None): The minimum radial bin in units of ``sim['pos']``.
    Note that this applies to axis a, so particles within this radius
    may still be included within homeoidal shells. By default this is
    taken as rout/1000.

    *rout* (None): The maximum radial bin in units of ``sim['pos']``.
    By default this is taken as the largest radial value in the halo
    particle distribution.

    *bins* (equal): The spacing scheme for the homeoidal shell bins.
    ``equal`` initialises radial bins with equal numbers of particles,
    with the exception of the final bin which will accomodate remainders.
    This number is not necessarily maintained during fitting.
    ``log`` and ``lin`` initialise bins with logarithmic and linear
    radial spacing.

    Halo must be in a centered frame.
    Caution is advised when assigning large number of bins and radial
    ranges with many particles, as the algorithm becomes very slow.
    """

    #-----------------------------FUNCTIONS-----------------------------
    # Define an ellipsoid shell with lengths a,b,c and orientation E:
    def Ellipsoid(r, a,b,c, E):
        x,y,z = np.dot(np.transpose(E),[r[:,0],r[:,1],r[:,2]])
        return (x/a)**2 + (y/b)**2 + (z/c)**2

    # Define moment of inertia tensor:
    MoI = lambda r,m: np.array([[np.sum(m*r[:,i]*r[:,j]) for j in range(3)]\
                               for i in range(3)])

    # Splits 'r' array into N groups containing equal numbers of particles.
    # An array is returned with the radial bins that contain these groups.
    sn = lambda r,N: np.append([r[i*int(len(r)/N):(1+i)*int(len(r)/N)][0]\
                               for i in range(N)],r[-1])

    # Retrieves alignment angle:
    almnt = lambda E: np.arccos(np.dot(np.dot(E,[1.,0.,0.]),[1.,0.,0.]))
    #-----------------------------FUNCTIONS-----------------------------

    if (rout == None): rout = sim.dm['r'].max()
    if (rin == None): rin = rout/1E3

    posr = np.array(sim.dm['r'])[np.where(sim.dm['r'] < rout)]
    pos = np.array(sim.dm['pos'])[np.where(sim.dm['r'] < rout)]
    mass = np.array(sim.dm['mass'])[np.where(sim.dm['r'] < rout)]

    rx = [[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]
    ry = [[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
    rz = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]

    # Define bins:
    if (bins == 'equal'): # Each bin contains equal number of particles
        mid = sn(np.sort(posr[np.where((posr >= rin) & (posr <= rout))]),N*2)
        rbin = mid[1:N*2+1:2]
        mid = mid[0:N*2+1:2]

    elif (bins == 'log'): # Bins are logarithmically spaced
        mid = profile.Profile(sim.dm, type='log', ndim=3, rmin=rin, rmax=rout, nbins=N+1)['rbins']
        rbin = np.sqrt(mid[0:N]*mid[1:N+1])

    elif (bins == 'lin'): # Bins are linearly spaced
        mid = profile.Profile(sim.dm, type='lin', ndim=3, rmin=rin, rmax=rout, nbins=N+1)['rbins']
        rbin = 0.5*(mid[0:N]+mid[1:N+1])

    # Define b/a and c/a ratios and angle arrays:
    ba,ca,angle = np.zeros(N),np.zeros(N),np.zeros(N)
    Es = [0]*N

    # Begin loop through radii:
    for i in range(0,N):

        # Initialise convergence criterion:
        tol = 1E-3
        count = 0

        # Define initial spherical shell:
        a=b=c = rbin[i]
        E = np.identity(3)
        L1,L2 = rbin[i]-mid[i],mid[i+1]-rbin[i]

        # Begin iterative procedure to fit data to shell:
        while True:
            count += 1

            # Collect all particle positions and masses within shell:
            r = pos[np.where((posr < a+L2) & (posr > c-L1*c/a))]
            inner = Ellipsoid(r, a-L1,b-L1*b/a,c-L1*c/a, E)
            outer = Ellipsoid(r, a+L2,b+L2*b/a,c+L2*c/a, E)
            r = r[np.where((inner > 1.) & (outer < 1.))]
            m = mass[np.where((inner > 1.) & (outer < 1.))]

            # End iterations if there is no data in range:
            if (len(r) == 0):
                ba[i],ca[i],angle[i],Es[i] = b/a,c/a,almnt(E),E
                logger.info('No data in range after %i iterations' %count)
                break

            # Calculate shape tensor & diagonalise:
            D = list(np.linalg.eig(MoI(r,m)/np.sum(m)))

            # Purge complex numbers:
            if isinstance(D[1][0,0],complex):
                D[0] = D[0].real ; D[1] = D[1].real
                logger.info('Complex numbers in D removed...')

            # Compute ratios a,b,c from moment of intertia principles:
            anew,bnew,cnew = np.sqrt(abs(D[0])*3.0)

            # The rotation matrix must be reoriented:
            E = D[1]
            if ((bnew > anew) & (anew >= cnew)): E = np.dot(E,rz)
            if ((cnew > anew) & (anew >= bnew)): E = np.dot(np.dot(E,ry),rx)
            if ((bnew > cnew) & (cnew >= anew)): E = np.dot(np.dot(E,rz),rx)
            if ((anew > cnew) & (cnew >= bnew)): E = np.dot(E,rx)
            if ((cnew > bnew) & (bnew >= anew)): E = np.dot(E,ry)
            cnew,bnew,anew = np.sort(np.sqrt(abs(D[0])*3.0))

            # Keep a as semi-major axis and distort b,c by b/a and c/a:
            div = rbin[i]/anew
            anew *= div
            bnew *= div
            cnew *= div

            # Convergence criterion:
            if (np.abs(b/a-bnew/anew) < tol) & (np.abs(c/a-cnew/anew) < tol):
                if (almnt(-E) < almnt(E)): E = -E
                ba[i],ca[i],angle[i],Es[i] = bnew/anew,cnew/anew,almnt(E),E
                break

            # Increase tolerance if convergence has stagnated:
            elif (count%10 == 0): tol *= 5.

            # Reset a,b,c for the next iteration:
            a,b,c = anew,bnew,cnew

    return [array.SimArray(rbin, sim.d['pos'].units), ba, ca, angle, Es]
