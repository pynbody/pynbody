"""

halo
====

Functions for dealing with and manipulating halos in simulations.


"""

import functools
import logging
import math
import operator
import warnings

import numpy as np

from .. import array, config, filt, transformation, units, util
from . import _com, cosmology, profile

logger = logging.getLogger('pynbody.analysis.halo')


def shrink_sphere_center(sim, r=None, shrink_factor=0.7, min_particles=100,
                         num_threads = None, particles_for_velocity = 0,
                         families_for_velocity = ['dm', 'star']):
    """
    Return the center according to the shrinking-sphere method of Power et al (2003)

    Most users will want to use the higher-level :func:`center` function, which actually performs the centering operation.
    This function is a lower-level interface that calculates the center but does not move the particles.

    Parameters
    ----------

    sim : SimSnap
        The simulation snapshot to center

    r : float | str, optional
        Initial search radius. If None, a rough estimate is used.

    shrink_factor : float, optional
        The amount to shrink the search radius by on each iteration

    min_particles : int, optional
        Minimum number of particles within the search radius. When this number is reached, the search is complete.

    num_threads : int, optional
        Number of threads to use for the calculation. If None, the number of threads is taken from the configuration.

    particles_for_velocity : int, optional
        If > min_particles, a velocity centre is calculated when the number of particles falls below this threshold,
        and returned.

    families_for_velocity : list, optional
        The families to use for the velocity centering. Default is ['dm', 'star'], because gas particles may
        be involved in violent outflows making them a risky choice for velocity centering.

    Returns
    -------
    com : SimArray
        The center of mass of the final sphere

    vel : SimArray
        The center of mass velocity of the final sphere. Only returned if particles_for_velocity > min_particles.

    """

    if num_threads is None:
        num_threads = config['number_of_threads']

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

    R = _com.shrink_sphere_center(pos, mass, min_particles, particles_for_velocity,
                                  shrink_factor, r, num_threads)

    com, final_radius, velocity_radius = R

    logger.info("Radius for velocity measurement = %s", velocity_radius)
    logger.info("Final SSC=%s", com)

    com_to_return = array.SimArray(com, sim['pos'].units)

    if particles_for_velocity > min_particles:
        fam_filter = functools.reduce(operator.or_, (filt.FamilyFilter(f) for f in families_for_velocity))
        final_sphere = sim[filt.Sphere(velocity_radius, com) & fam_filter]
        logger.info("Particles in sphere = %d", len(final_sphere))
        if len(final_sphere) == 0:
            warnings.warn("Final sphere is empty; cannot return a velocity. This probably implies something is "
                          "wrong with the position centre too.", RuntimeWarning)
            return com_to_return, np.array([0., 0., 0.])
        else:
            vel = final_sphere.mean_by_mass('vel')
            logger.info("Final velocity=%s", vel)
            return com_to_return, vel
    else:
        return com_to_return



def virial_radius(sim, cen=None, overden=178, r_max=None, rho_def='matter'):
    """Calculate the virial radius of the halo centered on the given coordinates.

    The default is here defined by the sphere centered on cen which contains a
    mean density of overden * rho_M_0 * (1+z)^3.

    Parameters
    ----------

    sim : SimSnap
        The simulation snapshot for which to calculate the virial radius.

    cen : array_like, optional
        The center of the halo. If None, the halo is assumed to be already centered.

    overden : float, optional
        The overdensity of the halo. Default is 178.

    r_max : float, optional
        The maximum radius to search for the virial radius. If None, the maximum radius of any
        particle in *sim* is used

    rho_def : str, optional
        The density definition to use.
        Default is 'matter', which uses the matter density at the redshift of the simulation. Alternatively,
        'critical' can be used for the critical density at this redshift.

    Returns
    -------

    float
        The virial radius of the halo in the position units of *sim*.

    """

    if r_max is None:
        r_max = (sim["x"].max() - sim["x"].min())
    else:
        if cen is not None:
            sim = sim[filt.Sphere(r_max, cen)]
        else:
            sim = sim[filt.Sphere(r_max)]

    r_min = 0.0

    if rho_def == 'matter':
       ref_density = sim.properties["omegaM0"] * cosmology.rho_crit(sim, z=0) * (1.0 + sim.properties["z"]) ** 3
    elif rho_def == 'critical':
        ref_density = cosmology.rho_crit(sim, z=sim.properties["z"])
    else:
        raise ValueError(rho_def + "is not a valid definition for the reference density")

    target_rho = overden * ref_density
    logger.info("target_rho=%s", target_rho)

    if cen is not None:
        transform = sim.translate(-np.asanyarray(cen))
    else:
        transform = transformation.NullTransformation(sim)

    with transform:
        sim = sim[filt.Sphere(r_max)]
        with sim.immediate_mode:
            mass_ar = np.asarray(sim['mass'])
            r_ar = np.asarray(sim['r'])

        rho = lambda r: util.sum_if_lt(mass_ar,r_ar,r)/(4. * math.pi * (r ** 3) / 3)
        result = util.bisect(r_min, r_max, lambda r: target_rho -
                             rho(r), epsilon=0, eta=1.e-3 * target_rho, verbose=False)

    return result


def _potential_minimum(sim):
    i = sim["phi"].argmin()
    return sim["pos"][i].copy()


def hybrid_center(sim, r='3 kpc', **kwargs):
    """Determine the center of the halo by finding the shrink-sphere-center near the potential minimum

    Most users will want to use the general :func:`center` function, which actually performs the centering operation.
    This function is a lower-level interface that calculates the center but does not move the particles.

    Parameters
    ----------

    sim : SimSnap
        The simulation snapshot of which to find the center
    r : float | str, optional
        Radius from the potential minimum to search for the center. Default is 3 kpc.

    Remaining parameters are passed onto :func:`shrink_sphere_center`.

    Returns
    -------

    com : SimArray
        The center of mass of the final sphere
    vel : SimArray
        The center of mass velocity of the final sphere. Only returned if particles_for_velocity > min_particles.


    """

    try:
        cen_a = _potential_minimum(sim)
    except KeyError:
        cen_a = center_of_mass(sim)
    return shrink_sphere_center(sim[filt.Sphere(r, cen_a)], **kwargs)



def vel_center(sim, cen_size="1 kpc", return_cen=False, move_all=True, **kwargs):
    """Recenter the snapshot on the center of mass velocity inside a sphere of specified radius

    Attempts to use the star particles, falling back to gas or dark matter if necessary.

    Parameters
    ----------

    sim : SimSnap
        The simulation snapshot to center

    cen_size : str or float, optional
        The size of the sphere to use for the velocity centering. Default is 1 kpc.

    return_cen : bool, optional
        If True, only return the velocity center without actually moving the snapshot. Default is False.

    move_all : bool, optional
        If True (default), move the entire snapshot. Otherwise only move the particles in the halo passed in.

    retcen : bool, optional
        Deprecated alias for return_cen

    Returns
    -------

    Transformation | SimArray
        Normally, a transformation object that can be used to revert the transformation.
        However, if return_cen is True, a SimArray containing the velocity center
        coordinates is returned instead, and the snapshot is not transformed.

    """

    if "retcen" in kwargs:
        return_cen = kwargs.pop("retcen")
        warnings.warn("The 'retcen' keyword is deprecated. Use 'return_cen' instead.", DeprecationWarning)

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

    logger.info("vcen=%s", vcen)

    if return_cen:
        return vcen
    else:
        return target.offset_velocity(-vcen)


def center(sim, mode=None, return_cen=False, with_velocity=True, cen_size="1 kpc",
           cen_num_particles=10000, move_all=True, wrap=False, **kwargs):
    """Transform the ancestor snapshot so that the provided snapshot is centred

    The centering scheme is determined by the ``mode`` keyword. As well as the
    position, the velocity can also be centred.

    The following centring modes are available:

    *  *pot*: potential minimum

    *  *com*: center of mass

    *  *ssc*: shrink sphere center

    *  *hyb*: for most halos, returns the same as ssc, but works faster by starting iteration near potential minimum

    Before the main centring routine is called, the snapshot is translated so that the
    halo is already near the origin. The box is then wrapped so that halos on the edge
    of the box are handled correctly.


    Parameters
    ----------

    sim : SimSnap
        The simulation snapshot from which to derive a centre. The ancestor snapshot is
        then transformed.

    mode : str or function, optional
        The method to use to determine the centre. If None, the default is taken from the configuration.
        Accepted values are discussed above. A function returning the centre, or a pair of
        centres (position and velocity) can also be passed.

    cen_size : str or float, optional
        The size of the sphere to use for the velocity centering. Default is 1 kpc.
        Note that this is only used if velocity centring is requested but the underlying
        method does not return a velocity centre. For example, if using the 'ssc' method,
        the cen_num_particles keyword should be used instead.

    cen_num_particles : int, optional
        The number of particles to use for the velocity centering. Default is 5000.
        This is passed to the 'ssc' method, which then finds the sphere with approximately
        this number of particles in it for the velocity centering.

    with_velocity: bool, optional
        If True, also center the velocity. Default is True.

    return_cen: bool, optional
        If True, only return the center without actually centering the snapshot.
        Default is False.

    move_all: bool, optional
        If True (default), move the entire snapshot. Otherwise only move the particles
        in the halo/subsnap passed into this function.

    vel: bool, optional
        Deprecated alias for with_velocity. Default is True.

    retcen: bool, optional
        If True, only return the center without centering the snapshot. Default is False.

    Returns
    -------
    Transformation | SimArray
        Normally, a transformation object that can be used to revert the transformation.
        However, if return_cen is True, a SimArray containing the center
        coordinates is returned instead, and the snapshot is not transformed.



    """

    if 'vel' in kwargs:
        warnings.warn("The 'vel' keyword is deprecated. Use 'with_velocity' instead.", DeprecationWarning)
        with_velocity = kwargs.pop('vel')

    if 'retcen' in kwargs:
        warnings.warn("The 'retcen' keyword is deprecated. Use 'return_cen' instead.", DeprecationWarning)
        return_cen = kwargs.pop('retcen')


    global config
    if mode is None:
        mode = config['centering-scheme']

    try:
        fn = {'pot': _potential_minimum,
              'com': lambda s : s.mean_by_mass('pos'),
              'ssc': functools.partial(shrink_sphere_center, particles_for_velocity=cen_num_particles),
              'hyb': hybrid_center}[mode]
    except KeyError:
        fn = mode

    if move_all:
        target = sim.ancestor
    else:
        target = sim

    if wrap:
        # centre on something within the halo and wrap
        initial_offset = -sim['pos'][0]
        transform = target.translate(initial_offset)
        target.wrap()
    else:
        transform = transformation.NullTransformation(target)
        initial_offset = np.array([0., 0., 0.])

    try:
        centre = fn(sim, **kwargs)
        if len(centre) == 2:
            # implies we have a velocity centre as well
            centre, vel_centre = centre
        else:
            vel_centre = None

        if return_cen:
            transform.revert()
            return centre - initial_offset

        transform = transform.translate(-centre)

        if with_velocity:
            if vel_centre is None :
                vel_centre = vel_center(sim, cen_size=cen_size, retcen=True)
            logger.info("vel_centre=%s", vel_centre)
            transform = transform.offset_velocity(-vel_centre)

    except:
        transform.revert()
        raise

    return transform

def halo_shape(sim, N=100, rin=None, rout=None, bins='equal'):
    """
    Computes the shape of a halo as a function of radius by fitting homeoidal shells.

    The halo must be pre-centred, e.g. using :func:`center`.

    Caution is advised when assigning large number of bins and radial
    ranges with many particles, as the algorithm becomes very slow.

    Parameters
    ----------

    N : int
        The number of homeoidal shells to consider. Shells with few particles will take longer to fit.

    rin : float
        The minimum radial bin in units of sim['pos']. By default this is taken as rout/1000.
        Note that this applies to axis a, so particles within this radius may still be included within
        homeoidal shells.

    rout : float
        The maximum radial bin in units of sim['pos']. By default this is taken as the largest radial value
        in the halo particle distribution.

    bins : str
        The spacing scheme for the homeoidal shell bins. 'equal' initialises radial bins with equal numbers
        of particles, with the exception of the final bin which will accomodate remainders. This
        number is not necessarily maintained during fitting. 'log' and 'lin' initialise bins
        with logarithmic and linear radial spacing.

    Returns
    -------

    rbin : SimArray
        The radial bins used for the fitting.

    ba : array
        The axial ratio b/a as a function of radius.

    ca : array
        The axial ratio c/a as a function of radius.

    angle : array
        The angle of the a-direction with respect to the x-axis as a function of radius.

    Es : array
        The rotation matrices for each shell.

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

    if (rout is None): rout = sim.dm['r'].max()
    if (rin is None): rin = rout/1E3

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
