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


def shrink_sphere_center(sim, r=None, shrink_factor=0.7, min_particles=100, verbose=False, num_threads = None,**kwargs):
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

def shape(sim, nbins=100, rmin=None, rmax=None, bins='equal', ndim=3, max_iterations=10, tol=1e-3, justify=False):

  """
  Calculates the shape of sim in homeoidal shells, over a range of nbins radii.
  Homeoidal shells maintain a fixed area (ndim=2) or volume (ndim=3).
  The algorithm is sensitive to substructure, which should ideally be removed.
  Particles must be in a centered frame.
  Caution is advised when assigning large number of bins and radial
  ranges with many particles, as the algorithm becomes very slow.

  **Input:**

  *nbins* (default=100): The number of homeoidal shells to consider.

  *rmin* (default=None): The minimum initial radial bin in units of
  ``sim['pos']``. By default this is taken as rmax/1000.

  *rmax* (default=None): The maximum initial radial bin in units of
  ``sim['pos']``. By default this is taken as the greatest radial value.

  *bins* (default='equal'): The spacing scheme for the homeoidal shell bins.
  ``equal`` initialises radial bins with equal numbers of particles.
  This number is not necessarily maintained during fitting.
  ``log`` and ``lin`` initialise bins with logarithmic and linear
  radial spacing.

  *ndim* (default=3): The number of dimensions to consider. If ndim=2,
  the shape is calculated in the x-y plane. The user is advised to make
  their own cut in ``z`` if using ndim=2.

  *max_iterations* (default=10): The maximum number of shape calculations.
  Fewer iterations will result in a speed-up, but with a bias towards
  increadingly spheroidal shape calculations.

  *tol* (default=1e-3): Convergence criterion for the shape calculation.
  Convergence is achieved when the axial ratios have a fractional
  change <=tol between iterations

  *justify* (default=False): Align the rotation matrix directions
  such that they point in a singular consistent direction aligned
  with the overall halo shape. This can be useful if working with slerps.

  **Output**:

  *rbins*: The radii of the initial spherical bins in units
  of ``sim['pos']``.

  *axis lengths*: The axis lengths of each homoeoidal shell in
  order a>b>c with units of ``sim['pos']``.

  *N*: The number of particles within each bin.

  *R*: The rotation matrix of each homoeoidal shell.

  """

  # Sanitise inputs:
  if (rmax == None): rmax = sim['r'].max()
  if (rmin == None): rmin = rmax/1E3
  assert ndim in [2, 3]
  assert max_iterations > 0
  assert tol > 0
  assert rmin >= 0
  assert rmax > rmin
  assert nbins > 0
  if ndim==2:
    assert np.sum((sim['rxy'] >= rmin) & (sim['rxy'] < rmax)) > nbins*2
  elif ndim==3:
    assert np.sum((sim['r'] >= rmin) & (sim['r'] < rmax)) > nbins*2
  if bins not in ['equal', 'log', 'lin']: bins = 'equal'

  # Handy 90 degree rotation matrices:
  Rx = np.array([[1,0,0], [0,0,-1], [0,1,0]])
  Ry = np.array([[0,0,1], [0,1,0], [-1,0,0]])
  Rz = np.array([[0,-1,0], [1,0,0], [0,0,1]])

  #-----------------------------FUNCTIONS-----------------------------
  sn = lambda r,N: np.append([r[i*int(len(r)/N):(1+i)*int(len(r)/N)][0]\
                              for i in range(N)],r[-1])

  # General equation for an ellipse/ellipsoid:
  def Ellipsoid(pos, a, R):
      x = np.dot(R.T, pos.T)
      return np.sum(np.divide(x.T, a)**2, axis=1)

  # Define moment of inertia tensor:
  def MoI(r, m, ndim=3):
    return np.array([[np.sum(m*r[:,i]*r[:,j]) for j in range(ndim)] for i in range(ndim)])

  # Calculate the shape in a single shell:
  def shell_shape(r,pos,mass, a,R, r_range, ndim=3):

    # Find contents of homoeoidal shell:
    mult = r_range / np.mean(a)
    in_shell = (r > min(a)*mult[0]) & (r < max(a)*mult[1])
    pos, mass = pos[in_shell], mass[in_shell]
    inner = Ellipsoid(pos, a*mult[0], R)
    outer = Ellipsoid(pos, a*mult[1], R)
    in_ellipse = (inner > 1) & (outer < 1)
    ellipse_pos, ellipse_mass = pos[in_ellipse], mass[in_ellipse]

    # End if there is no data in range:
    if not len(ellipse_mass):
      return a, R, np.sum(in_ellipse)

    # Calculate shape tensor & diagonalise:
    D = list(np.linalg.eigh(MoI(ellipse_pos,ellipse_mass,ndim) / np.sum(ellipse_mass)))

    # Rescale axis ratios to maintain constant ellipsoidal volume:
    R2 = np.array(D[1])
    a2 = np.sqrt(abs(D[0]) * ndim)
    div = (np.prod(a) / np.prod(a2))**(1/float(ndim))
    a2 *= div

    return a2, R2, np.sum(in_ellipse)

  # Re-align rotation matrix:
  def realign(R, a, ndim):
    if ndim == 3:
      if a[0]>a[1]>a[2]<a[0]: pass                          # abc
      elif a[0]>a[1]<a[2]<a[0]: R = np.dot(R,Rx)            # acb
      elif a[0]<a[1]>a[2]<a[0]: R = np.dot(R,Rz)            # bac
      elif a[0]<a[1]>a[2]>a[0]: R = np.dot(np.dot(R,Rx),Ry) # bca
      elif a[0]>a[1]<a[2]>a[0]: R = np.dot(np.dot(R,Rx),Rz) # cab
      elif a[0]<a[1]<a[2]>a[0]: R = np.dot(R,Ry)            # cba
    elif ndim == 2:
      if a[0]>a[1]: pass                                    # ab
      elif a[0]<a[1]: R = np.dot(R,Rz[:2,:2])               # ba
    return R

  # Calculate the angle between two vectors:
  def angle(a, b):
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

  # Flip x,y,z axes of R2 if they provide a better alignment with R1.
  def flip_axes(R1, R2):
    for i in range(len(R1)):
      if angle(R1[:,i], -R2[:,i]) < angle(R1[:,i], R2[:,i]):
        R2[:,i] *= -1
    return R2
  #-----------------------------FUNCTIONS-----------------------------

  # Set up binning:
  r = np.array(sim['r']) if ndim==3 else np.array(sim['rxy'])
  pos = np.array(sim['pos'])[:,:ndim]
  mass = np.array(sim['mass'])

  if (bins == 'equal'): # Bins contain equal number of particles
      full_bins = sn(np.sort(r[(r>=rmin) & (r<=rmax)]), nbins*2)
      bin_edges = full_bins[0:nbins*2+1:2]
      rbins = full_bins[1:nbins*2+1:2]
  elif (bins == 'log'): # Bins are logarithmically spaced
      bin_edges = np.logspace(np.log10(rmin), np.log10(max), nbins+1)
      rbins = np.sqrt(bin_edges[:-1] * bin_edges[1:])
  elif (bins == 'lin'): # Bins are linearly spaced
      bin_edges = np.linspace(rmin, rmax, nbins+1)
      rbins = 0.5*(bin_edges[:-1] + bin_edges[1:])

  # Initialise the shape arrays:
  rbins = array.SimArray(rbins, sim['pos'].units)
  axial_ratios = array.SimArray(np.zeros([nbins,ndim]), sim['pos'].units)
  N_in_bin = np.zeros(nbins).astype('int')
  rotations = [0]*nbins

  # Loop over all radial bins:
  for i in range(nbins):

    # Initial spherical shell:
    a = np.ones(ndim) * rbins[i]
    a2 = np.zeros(ndim)
    a2[0] = np.inf
    R = np.identity(ndim)

    # Iterate shape estimate until a convergence criterion is met:
    iteration_counter = 0
    while ((np.abs(a[1]/a[0] - np.sort(a2)[-2]/max(a2)) > tol) & \
           (np.abs(a[-1]/a[0] - min(a2)/max(a2)) > tol)) & \
           (iteration_counter < max_iterations):
      a2 = a.copy()
      a,R,N = shell_shape(r,pos,mass, a,R, bin_edges[[i,i+1]], ndim)
      iteration_counter += 1

    # Adjust orientation to match axis ratio order:
    R = realign(R, a, ndim)

    # Ensure consistent coordinate system:
    if np.sign(np.linalg.det(R)) == -1:
      R *= -1

    # Update profile arrays:
    a = np.flip(np.sort(a))
    axial_ratios[i], rotations[i], N_in_bin[i] = a, R, N

  # Ensure the axis vectors point in a consistent direction:
  if justify:
    _, _, _, R_global = shape(sim, nbins=1, rmin=rmin, rmax=rmax, ndim=ndim)
    rotations = np.array([flip_axes(R_global, i) for i in rotations])

  return rbins, np.squeeze(axial_ratios), N_in_bin, np.squeeze(rotations)
