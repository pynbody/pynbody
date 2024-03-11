"""
snapshot
========

This module implements the  :class:`~pynbody.snapshot.SimSnap` class which manages and stores snapshot data.
It also implements the :class:`~pynbody.snapshot.SubSnap` class (and relatives) which
represent different views of an existing :class:`~pynbody.snapshot.SimSnap`.

"""

import logging
import pathlib

from .. import config, family
from . import util
from .simsnap import SimSnap

logger = logging.getLogger('pynbody.snapshot')


def load(filename, *args, **kwargs) -> SimSnap:
    """Loads a file using the appropriate class, returning a SimSnap
    instance."""

    filename = pathlib.Path(filename)

    priority = kwargs.pop('priority', config['snap-class-priority'])

    for c in SimSnap.iter_subclasses_with_priority(priority):
        if c._can_load(filename):
            logger.info("Loading using backend %s" % str(c))
            return c(filename, *args, **kwargs)

    raise OSError(
        "File %r: format not understood or does not exist" % filename)

def new(n_particles=0, order=None, class_=SimSnap, **families):
    """Create a blank SimSnap, with the specified number of particles.

    Position, velocity and mass arrays are created and filled
    with zeros.

    By default all particles are taken to be dark matter.
    To specify otherwise, pass in keyword arguments specifying
    the number of particles for each family, e.g.

    f = new(dm=50, star=25, gas=25)

    The order in which the different families appear in the snapshot
    is unspecified unless you add an 'order' argument:

    f = new(dm=50, star=25, gas=25, order='star,gas,dm')

    guarantees the stars, then gas, then dark matter particles appear
    in sequence.
    """

    if len(families) == 0:
        families = {'dm': n_particles}

    t_fam = []
    tot_particles = 0

    if order is None:
        for k, v in list(families.items()):

            assert isinstance(v, int)
            t_fam.append((family.get_family(k), v))
            tot_particles += v
    else:
        for k in order.split(","):
            v = families[k]
            assert isinstance(v, int)
            t_fam.append((family.get_family(k), v))
            tot_particles += v

    x = class_()
    x._num_particles = tot_particles
    x._filename = "<created>"

    x._create_arrays(["pos", "vel"], 3)
    x._create_arrays(["mass"], 1)

    rt = 0
    for k, v in t_fam:
        x._family_slice[k] = slice(rt, rt + v)
        rt += v

    x._decorate()
    return x


from . import ascii, gadget, gadgethdf, grafic, nchilada, ramses, subsnap, swift, tipsy
from .subsnap import FamilySubSnap, IndexedSubSnap, SubSnap
