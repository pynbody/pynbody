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
