"""Analysis involving angular momentum.

"""
import logging

import numpy as np

from .. import array, filt, transformation, units, util
from . import halo

logger = logging.getLogger('pynbody.analysis.angmom')


def ang_mom_vec(snap):
    """
    Calculates the angular momentum vector of the specified snapshot.

    Parameters
    ----------

    snap : SimSnap
        The snapshot to analyze

    Returns
    -------

    array.SimArray
        The angular momentum vector of the snapshot

    """

    angmom = (snap['mass'].reshape((len(snap), 1)) *
              np.cross(snap['pos'], snap['vel'])).sum(axis=0).view(np.ndarray)
    result = angmom.view(array.SimArray)
    result.units = snap['mass'].units * snap['pos'].units * snap['vel'].units
    return result


@util.deprecated("ang_mom_vec_units is deprecated. Use ang_mom_vec instead.")
def ang_mom_vec_units(snap):
    """Deprecated alias for ang_mom_vec"""

    return ang_mom_vec(snap)


def spin_parameter(snap):
    """Return the spin parameter lambda' of a centered halo

    The spin parameter is defined as in eq. (5) of Bullock et al. 2001 (2001MNRAS.321..559B).

    Note that the halo has to be aligned such that its center is the coordinate
    origin and the velocity must be zeroed. If you are not sure whether this will
    be true, calculate the spin parameter of h using:

    >>> with pynbody.analysis.angmom.faceon(h):
    >>>     spin = pynbody.analysis.angmom.spin_parameter(h)


    Parameters
    ----------

    snap : SimSnap
        The snapshot to analyze

    Returns
    -------

    float
        The dimensionless spin parameter lambda' of the halo

    """

    m3 = snap['mass'].sum()
    m3 = m3 * m3 * m3
    l = np.sqrt(((ang_mom_vec_units(snap) ** 2).sum()) /
                (2 * units.G * m3 * snap['r'].max()))
    return float(l.in_units('1', **snap.conversion_context()))


def calc_sideon_matrix(angmom_vec, along = [1.0, 0.0, 0.0]):
    """Calculate the rotation matrix to put the specified angular momentum vector side-on.

    The rotation matrix is calculated such that the angular momentum vector will be placed in the y direction
    post-transformation.

    Parameters
    ----------
    angmom_vec : array_like
        The angular momentum vector that will be placed in the y-direction post-transformation.

    along : array_like
        An additional orientation vector. The components of this vector perpendicular to angmom_vec defines the
        direction to transform to 'along', i.e. to the positive x-axis post-transformation.

    Returns
    -------

    array_like
        The rotation matrix
    """
    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in / np.sum(vec_in ** 2).sum() ** 0.5
    vec_p1 = np.cross(along, vec_in)
    vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
    vec_p2 = np.cross(vec_in, vec_p1)

    matr = np.concatenate((vec_p2, vec_in, vec_p1)).reshape((3, 3))

    return matr


def calc_faceon_matrix(angmom_vec, up=[0.0, 1.0, 0.0]):
    """Calculate the rotation matrix to put the specified angular momentum vector face-on.

    The rotation matrix is calculated such that the angular momentum vector will be placed in the z direction
    post-transformation.

    Parameters
    ----------
    angmom_vec : array_like
        The angular momentum vector that will be placed in the z-direction post-transformation.
    up : array_like
        An additional orientation vector. The components of this vector perpendicular to angmom_vec defines the
        direction to transform to 'up', i.e. to the positive y-axis post-transformation.

    Returns
    -------

    array_like
        The rotation matrix

    """
    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in / np.sum(vec_in ** 2).sum() ** 0.5
    vec_p1 = np.cross(up, vec_in)
    vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
    vec_p2 = np.cross(vec_in, vec_p1)

    matr = np.concatenate((vec_p1, vec_p2, vec_in)).reshape((3, 3))

    return matr


def align(h, vec_to_xform, disk_size="5 kpc", move_all=True, already_centered = False,
           center_kwargs = None):
    """Reposition and rotate the ancestor of h to place the angular momentum into a specified orientation.

    The routine first calls the center routine to reposition the halo (unless already_centered is True).
    If there are a sufficient number of gas particles (more than 100), only the gas particles are used for
    centering, since these will also be used for angular momentum calculations; if there is an offset between
    e.g. dark matter and baryons, it is better to centre on the baryons.

    Then, it determines the disk orientation using the angular momentum vector of the gas particles within
    a specified radius of the halo center. If there is no gas within this radius, the routine falls back first
    to stellar particles, and then to all particles.

    Finally, the angular momentum vector is converted into a rotation matrix using the vec_to_xform function,
    and the rotation is applied.

    Parameters
    ----------

    h : SimSnap
      The portion of the simulation from which to extract a centre and orientation. Typically this is a galaxy halo.

    vec_to_xform : function
        The function to use to calculate the rotation matrix from the measured angular momentum vector.

    disk_size : str | float, optional
        The size of the disk to use for calculating the angular momentum vector. Default is "5 kpc".

    move_all : bool, optional
        If True, the ancestor simulation of *h* is transformed. If False, only *h* is moved. Default is True.

    already_centered : bool, optional
        If True, the simulation is assumed to be already centered. Default is False.

    center_kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to the centering routine.

    """


    if center_kwargs is None:
        center_kwargs = {}

    center_kwargs.update({'move_all': move_all})

    if already_centered:
        if move_all:
            top = h.ancestor
        else:
            top = h
        tx = transformation.NullTransformation(top)
    else:
        if len(h.st) > 100:
            h_for_centering = h.st
        else:
            h_for_centering = h
        tx = halo.center(h_for_centering, **center_kwargs)

    try:
        if len(h.gas) > 5:
            cen = h.gas[filt.Sphere(disk_size)]
        elif len(h.st) > 5:
            cen = h.st[filt.Sphere(disk_size)]
        else:
            cen = h[filt.Sphere(disk_size)]

        logger.info("Calculating angular momentum vector...")
        trans = vec_to_xform(ang_mom_vec(cen))

        logger.info("Transforming simulation...")

        tx = tx.rotate(trans)

        logger.info("...done!")

        return tx

    except:
        tx.revert()
        raise


def sideon(h, **kwargs):
    """Reposition and rotate the ancestor of h to place the disk edge-on (i.e. into the x-z plane).

    Since pynbody's imaging routines project along the z direction, one can get a side-on view of a disk or
    other rotationally-supported structure by calling this routine first.

    For details of how the transformation is calculated, see the documentation for the underlying :func:`align` routine.

    Parameters
    ----------

    h : SimSnap
      The portion of the simulation from which to extract a centre and orientation. Typically this is a galaxy halo.

    disk_size : str | float, optional
        The size of the disk to use for calculating the angular momentum vector. Default is "5 kpc".

    move_all : bool, optional
        If True, the ancestor simulation of *h* is transformed. If False, only *h* is moved. Default is True.

    already_centered : bool, optional
        If True, the simulation is assumed to be already centered. Default is False.

    center_kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to the centering routine.

    """

    kwargs.update({'vec_to_xform': calc_sideon_matrix})

    return align(h, **kwargs).set_description('sideon')


def faceon(h, **kwargs):
    """Reposition and rotate the ancestor of h to place the disk face-on (i.e. into the x-y plane).

    Since pynbody's imaging routines project along the z direction, one can get a face-on view of a disk
    or other rotationally-supported structure by calling this routine first.

    For details of how the transformation is calculated, see the documentation for the underlying :func:`align` routine.


    Parameters
    ----------

    h : SimSnap
      The portion of the simulation from which to extract a centre and orientation. Typically this is a galaxy halo.

    disk_size : str | float, optional
        The size of the disk to use for calculating the angular momentum vector. Default is "5 kpc".

    move_all : bool, optional
        If True, the ancestor simulation of *h* is transformed. If False, only *h* is moved. Default is True.

    already_centered : bool, optional
        If True, the simulation is assumed to be already centered. Default is False.

    center_kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to the centering routine.

    """

    kwargs.update({'vec_to_xform': calc_faceon_matrix})

    return align(h, **kwargs).set_description('faceon')
