"""
Calculations relevant to galactic morphology

.. versionadded :: 2.0

    This module was added in version 2.0.

    Previously, only the :func:`decomp` function was available, in its own module within :mod:`pynbody.analysis`.

"""

import logging

import numpy as np

from .. import filt, transformation, util
from . import angmom, profile

logger = logging.getLogger('pynbody.analysis.morphology')

def estimate_jcirc_from_energy(h, particles_per_bin=1000, quantile=0.99):
    """Estimate the circular angular momentum as a function of energy.

    This routine calculates the circular angular momentum as a function of energy
    for the stars in the simulation, using a profile with a fixed number of particles
    per bin. It then estimates a circular angular momentum for each individual particle
    by interpolating the profile.

    Arguments
    ---------

    h : SimSnap
        The simulation snapshot to analyze

    quantile : float
        The circular angular momentum will be estimated as the specified quantile of the scalar angular momentum

    particles_per_bin : int
        The approximate number of particles per bin in the profile. Default is 1000.

    """
    nbins = len(h) // particles_per_bin

    pro_d = profile.QuantileProfile(h, q=(quantile,), nbins=nbins, type='equaln', calc_x = lambda sim : sim['te'])
    pro_d.create_particle_array("j2", particle_name='j_circ2', target_simulation=h)

    h['j_circ'] = np.sqrt(h['j_circ2'])
    del h.ancestor['j_circ2']

    return pro_d

def estimate_jcirc_from_rotation_curve(h, particles_per_bin=1000):
    """Estimate the circular angular momentum as a function of radius in the disk (x-y) plane.

    This routine calculates the circular velocity as a function of radius for the disk, using a profile with a fixed
    number of particles per radial bin. It then estimates a circular angular momentum for each individual particle by
    interpolating the profile.

    .. warning::

        This routine is only valid for simulations where all the stars are anyway in quite a narrow disc. Otherwise
        the interpolation back onto the individual particles carries limited meaning.

        For more general cases, use :func:`estimate_jcirc_from_energy` instead.

    Arguments
    ---------

    h : SimSnap
        The simulation snapshot to analyze

    particles_per_bin : int
        The approximate number of particles per bin in the profile. Default is 1000.

    """
    d = h[filt.Disc('1 Mpc', h['eps'].min() * 3)]

    nbins = len(d) // particles_per_bin

    pro_d = profile.Profile(d, nbins=nbins, type='equaln')

    pro_d.create_particle_array("j_circ", target_simulation=h)


def decomp(h, aligned=False, j_disk_min=0.8, j_disk_max=1.1, E_cut=None, j_circ_from_r=False,
           angmom_size="3 kpc", particles_per_bin = 500):
    """Creates an array 'decomp' for star particles in the simulation, with an integer specifying components.

    The possible values of the components are:

    1. thin disk
    2. halo
    3. bulge
    4. thick disk
    5. pseudo bulge

    First, the simulation is aligned so that the disk is in the xy plane. Then, the maximum angular momentum of
    stars as a function of energy is estimated, by default using the routine :func:`estimate_jcirc_from_energy`.
    The stars are then classified into the components based on their energy and disk-aligned angular momentum component.

    * The thin disk is defined as stars with angular momentum between j_disk_min and j_disk_max.
    * From the remaining stars, a critical angular momentum j_crit is then calculated that separates rotating from
      non-rotating components. By definition, this is chosen such that the mean rotaiton velocity of the non-rotating
      part is zero
    * The most tightly bound rotating component is labelled as the pseudo bulge, while less tightly bound rotating
      stars are labelled as the thick disk, based on E_cut.
    * The non-rotating stars are then separated into bulge and halo components based on their binding energy, again
      based on E_cut.


    This routine is based on an original IDL procedure by Chris Brook.

    .. versionchanged :: 2.0

      The routine now defaults to using a new method to estimate the circular angular momentum as a function of energy;
      see :func:`estimate_jcirc_from_energy`.

      The critical angular momentum for the spheroid is now determined by insisting the mean angular momentum of the
      spheroid should be zero.

      The above changes lead to different, but probably better, classifications compared with pynbody version 1.

      Additionally, this routine is now inside the :mod:`pynbody.analysis.morphology` module.

    Arguments
    ---------

    h : SimSnap
        The simulation snapshot to analyze

    aligned : bool
        If True, the simulation is assumed to be already aligned so that the disk is in the xy plane.
        Otherwise, the simulation is recentered and aligned into the xy plane.

    j_disk_min : float
        The minimum angular momentum as a proportion of the circular angular momentum which a particle must have to be
        part of the 'disk'.

    j_disk_max : float
        The maximum angular momentum as a proportion of the circular angular momentum which a particle can have to be
        part of the 'disk'.

    E_cut : float
        The energy boundary between bulge and halo. If None, this is taken to be the median energy of the stars.  Note
        that the distinction between bulge and halo is somewhat arbitrary and may not be physically meaningful.

    j_circ_from_r : bool
        If True, the maximum angular momentum is determined as a function of disc radius, rather than as a function of
        energy. Default False (determine as function of energy). This option is only valid for simulations where
        all the stars are anyway in quite a narrow disc.

    angmom_size : str
        The size of the disk to use for calculating the angular momentum vector. Default is "3 kpc".

    particles_per_bin : int
        The approximate number of particles per bin in the profile. Default is 500.

    """

    if aligned:
        tx = transformation.NullTransformation(h)
    else:
        tx = angmom.faceon(h, disk_size=angmom_size)

    with tx:

        if j_circ_from_r:
            estimate_jcirc_from_rotation_curve(h, particles_per_bin=particles_per_bin)
        else:
            estimate_jcirc_from_energy(h, particles_per_bin=particles_per_bin)


        h['jz_by_jzcirc'] = h['j'][:, 2] / h['j_circ']
        h_star = h.star

        if 'decomp' not in h_star:
            h_star._create_array('decomp', dtype=int)
        disk = np.where(
            (h_star['jz_by_jzcirc'] > j_disk_min) * (h_star['jz_by_jzcirc'] < j_disk_max))

        h_star['decomp', disk[0]] = 1
        # h_star = h_star[np.where(h_star['decomp']!=1)]

        # Find disk/spheroid angular momentum cut-off to make spheroid
        # angular momentum exactly zero

        JzJcirc = h_star['jz_by_jzcirc']
        te = h_star['te']

        logger.info("Finding spheroid/disk angular momentum boundary...")

        j_crit = util.bisect(-2.0, 2.0, lambda c: np.mean(JzJcirc[JzJcirc < c]))

        logger.info("j_crit = %.2e" % j_crit)

        if j_crit > j_disk_min:
            logger.warning(
                "!! j_crit exceeds j_disk_min. This is usually a sign that something is going wrong (train-wreck galaxy?)")
            logger.warning("!! j_crit will be reset to j_disk_min=%.2e" % j_disk_min)
            j_crit = j_disk_min

        sphere = np.where(h_star['jz_by_jzcirc'] < j_crit)

        if E_cut is None:
            E_cut = np.median(h_star['te'])

        logger.info("E_cut = %.2e" % E_cut)

        halo = np.where((te > E_cut) * (JzJcirc < j_crit))
        bulge = np.where((te <= E_cut) * (JzJcirc < j_crit))
        pbulge = np.where((te <= E_cut) * (JzJcirc > j_crit)
                          * ((JzJcirc < j_disk_min) + (JzJcirc > j_disk_max)))
        thick = np.where((te > E_cut) * (JzJcirc > j_crit)
                         * ((JzJcirc < j_disk_min) + (JzJcirc > j_disk_max)))

        h_star['decomp', halo] = 2
        h_star['decomp', bulge] = 3
        h_star['decomp', thick] = 4
        h_star['decomp', pbulge] = 5
