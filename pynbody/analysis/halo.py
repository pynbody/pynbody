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
                                                     rho(r), epsilon=0, eta=1.e-3 * target_rho)

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
                vel_centre = vel_center(sim, cen_size=cen_size, return_cen=True)
            logger.info("vel_centre=%s", vel_centre)
            transform = transform.offset_velocity(-vel_centre)

    except:
        transform.revert()
        raise

    return transform

@util.deprecated("halo_shape is deprecated. Use shape instead.")
def halo_shape(sim, N=100, rin=None, rout=None, bins='equal'):
    """Deprecated wrapper around :func:`shape`, for backwards compatibility.

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

    angle = lambda E: np.arccos(abs(E[:,0,0]))

    rbin, axis_lengths, num_particles, rotation_matrices = shape(sim.dm, nbins=N, rmin=rin, rmax=rout, bins=bins)

    ba = axis_lengths[:, 1] / axis_lengths[:, 0]
    ca = axis_lengths[:, 2] / axis_lengths[:, 0]

    return rbin, ba.view(np.ndarray), ca.view(np.ndarray), angle(rotation_matrices), rotation_matrices

def shape(sim, nbins=100, rmin=None, rmax=None, bins='equal',
          ndim=3, max_iterations=10, tol=1e-3, justify=False):
    """Calculates the shape of the provided particles in homeoidal shells, over a range of nbins radii.

    Homeoidal shells maintain a fixed area (ndim=2) or volume (ndim=3). Note that all provided particles are used in
    calculating the shape, so e.g. to measure dark matter halo shape from a halo with baryons, you should pass
    only the dark matter particles.

    The simulation must be pre-centred, e.g. using :func:`center`.

    The algorithm is sensitive to substructure, which should ideally be removed.

    Caution is advised when assigning large number of bins and radial ranges with many particles, as the
    algorithm becomes very slow.

    Parameters
    ----------

      nbins : int
          The number of homeoidal shells to consider. Shells with few particles will take longer to fit.

      rmin : float
          The minimum radial bin in units of sim['pos']. By default this is taken as rout/1000.
          Note that this applies to axis a, so particles within this radius may still be included within
          homeoidal shells.

      rmax : float
          The maximum radial bin in units of sim['pos']. By default this is taken as the largest radial value
          in the halo particle distribution.

      bins : str
          The spacing scheme for the homeoidal shell bins. 'equal' initialises radial bins with equal numbers
          of particles, with the exception of the final bin which will accomodate remainders. This
          number is not necessarily maintained during fitting. 'log' and 'lin' initialise bins
          with logarithmic and linear radial spacing.

      ndim : int
          The number of dimensions to consider; either 2 or 3 (default). If ndim=2, the shape is calculated
          in the x-y plane. If using ndim=2, you may wish to make a cut in the z direction before
          passing the particles to this routine (e.g. using :class:`pynbody.filt.BandPass`).

      max_iterations : int
          The maximum number of shape calculations (default 10). Fewer iterations will result in a speed-up,
          but with a bias towards spheroidal results.

      tol : float
          Convergence criterion for the shape calculation. Convergence is achieved when the axial ratios have
          a fractional change <=tol between iterations.

      justify : bool
          Align the rotation matrix directions such that they point in a single consistent direction
          aligned with the overall halo shape. This can be useful if working with slerps.

    Returns
    -------

      rbin : SimArray
          The radial bins used for the fitting

      axis_lengths : SimArray
          A nbins x ndim array containing the axis lengths of the ellipsoids in each shell

      num_particles : np.ndarray
          The number of particles within each bin

      rotation_matrices : np.ndarray
          The rotation matrices for each shell

    """

    # Sanitise inputs:
    if (rmax == None): rmax = sim['r'].max()
    if (rmin == None): rmin = rmax / 1E3
    assert ndim in [2, 3]
    assert max_iterations > 0
    assert tol > 0
    assert rmin >= 0
    assert rmax > rmin
    assert nbins > 0
    if ndim == 2:
        assert np.sum((sim['rxy'] >= rmin) & (sim['rxy'] < rmax)) > nbins * 2
    elif ndim == 3:
        assert np.sum((sim['r'] >= rmin) & (sim['r'] < rmax)) > nbins * 2
    if bins not in ['equal', 'log', 'lin']: bins = 'equal'

    # Handy 90 degree rotation matrices:
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # -----------------------------FUNCTIONS-----------------------------
    sn = lambda r, N: np.append([r[i * int(len(r) / N):(1 + i) * int(len(r) / N)][0] \
                                 for i in range(N)], r[-1])

    # General equation for an ellipse/ellipsoid:
    def Ellipsoid(pos, a, R):
        x = np.dot(R.T, pos.T)
        return np.sum(np.divide(x.T, a) ** 2, axis=1)

    # Define moment of inertia tensor:
    def MoI(r, m, ndim=3):
        return np.array([[np.sum(m * r[:, i] * r[:, j]) for j in range(ndim)] for i in range(ndim)])

    # Calculate the shape in a single shell:
    def shell_shape(r, pos, mass, a, R, r_range, ndim=3):

        # Find contents of homoeoidal shell:
        mult = r_range / np.mean(a)
        in_shell = (r > min(a) * mult[0]) & (r < max(a) * mult[1])
        pos, mass = pos[in_shell], mass[in_shell]
        inner = Ellipsoid(pos, a * mult[0], R)
        outer = Ellipsoid(pos, a * mult[1], R)
        in_ellipse = (inner > 1) & (outer < 1)
        ellipse_pos, ellipse_mass = pos[in_ellipse], mass[in_ellipse]

        # End if there is no data in range:
        if not len(ellipse_mass):
            return a, R, np.sum(in_ellipse)

        # Calculate shape tensor & diagonalise:
        D = list(np.linalg.eigh(MoI(ellipse_pos, ellipse_mass, ndim) / np.sum(ellipse_mass)))

        # Rescale axis ratios to maintain constant ellipsoidal volume:
        R2 = np.array(D[1])
        a2 = np.sqrt(abs(D[0]) * ndim)
        div = (np.prod(a) / np.prod(a2)) ** (1 / float(ndim))
        a2 *= div

        return a2, R2, np.sum(in_ellipse)

    # Re-align rotation matrix:
    def realign(R, a, ndim):
        if ndim == 3:
            if a[0] > a[1] > a[2] < a[0]:
                pass  # abc
            elif a[0] > a[1] < a[2] < a[0]:
                R = np.dot(R, Rx)  # acb
            elif a[0] < a[1] > a[2] < a[0]:
                R = np.dot(R, Rz)  # bac
            elif a[0] < a[1] > a[2] > a[0]:
                R = np.dot(np.dot(R, Rx), Ry)  # bca
            elif a[0] > a[1] < a[2] > a[0]:
                R = np.dot(np.dot(R, Rx), Rz)  # cab
            elif a[0] < a[1] < a[2] > a[0]:
                R = np.dot(R, Ry)  # cba
        elif ndim == 2:
            if a[0] > a[1]:
                pass  # ab
            elif a[0] < a[1]:
                R = np.dot(R, Rz[:2, :2])  # ba
        return R

    # Calculate the angle between two vectors:
    def angle(a, b):
        return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # Flip x,y,z axes of R2 if they provide a better alignment with R1.
    def flip_axes(R1, R2):
        for i in range(len(R1)):
            if angle(R1[:, i], -R2[:, i]) < angle(R1[:, i], R2[:, i]):
                R2[:, i] *= -1
        return R2

    # -----------------------------FUNCTIONS-----------------------------

    # Set up binning:
    r = np.array(sim['r']) if ndim == 3 else np.array(sim['rxy'])
    pos = np.array(sim['pos'])[:, :ndim]
    mass = np.array(sim['mass'])

    if (bins == 'equal'):  # Bins contain equal number of particles
        full_bins = sn(np.sort(r[(r >= rmin) & (r <= rmax)]), nbins * 2)
        bin_edges = full_bins[0:nbins * 2 + 1:2]
        rbins = full_bins[1:nbins * 2 + 1:2]
    elif (bins == 'log'):  # Bins are logarithmically spaced
        bin_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
        rbins = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    elif (bins == 'lin'):  # Bins are linearly spaced
        bin_edges = np.linspace(rmin, rmax, nbins + 1)
        rbins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Initialise the shape arrays:
    rbins = array.SimArray(rbins, sim['pos'].units)
    axis_lengths = array.SimArray(np.zeros([nbins, ndim]), sim['pos'].units)
    N_in_bin = np.zeros(nbins).astype('int')
    rotations = [0] * nbins

    # Loop over all radial bins:
    for i in range(nbins):

        # Initial spherical shell:
        a = np.ones(ndim) * rbins[i]
        a2 = np.zeros(ndim)
        a2[0] = np.inf
        R = np.identity(ndim)

        # Iterate shape estimate until a convergence criterion is met:
        iteration_counter = 0
        while ((np.abs(a[1] / a[0] - np.sort(a2)[-2] / max(a2)) > tol) & \
               (np.abs(a[-1] / a[0] - min(a2) / max(a2)) > tol)) & \
                (iteration_counter < max_iterations):
            a2 = a.copy()
            a, R, N = shell_shape(r, pos, mass, a, R, bin_edges[[i, i + 1]], ndim)
            iteration_counter += 1

        # Adjust orientation to match axis ratio order:
        R = realign(R, a, ndim)

        # Ensure consistent coordinate system:
        if np.sign(np.linalg.det(R)) == -1:
            R[:, 1] *= -1

        # Update profile arrays:
        a = np.flip(np.sort(a))
        axis_lengths[i], rotations[i], N_in_bin[i] = a, R, N

    # Ensure the axis vectors point in a consistent direction:
    if justify:
        _, _, _, R_global = shape(sim, nbins=1, rmin=rmin, rmax=rmax, ndim=ndim)
        rotations = np.array([flip_axes(R_global, i) for i in rotations])

    return rbins, np.squeeze(axis_lengths.T).T, N_in_bin, np.squeeze(rotations)
