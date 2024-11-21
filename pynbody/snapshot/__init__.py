"""
Implements classes to load and manipulate snapshot data
"""

import logging
import pathlib

from .. import config, family
from . import util
from .simsnap import SimSnap

logger = logging.getLogger('pynbody.snapshot')


def load(filename, *args, **kwargs) -> SimSnap:
    """Loads a file using the appropriate class, returning a SimSnap instance.

    This routine is the main entry point for loading snapshots. It will try to load the file using the appropriate
    class, based on inspection by the candidate subclasses. If no class can load the file, an OSError is raised.

    Parameters
    ----------
    filename : str
        The filename to load

    priority : optional, list[str | type]
        A list of SimSnap subclasses to try, in order. The first class which is capable of loading the file
        is used. If not specified, the ordering is as specified in the configuration files.

    *args, **kwargs :
        Other arguments and keyword arguments are passed to the class constructor that is used to load the file.

    Returns
    -------
    SimSnap
        The loaded snapshot

    """

    filename = pathlib.Path(filename)

    priority = kwargs.pop('priority', config['snap-class-priority'])

    for c in SimSnap.iter_subclasses_with_priority(priority):
        if c._can_load(filename):
            logger.info("Loading using backend %s" % str(c))
            return c(filename, *args, **kwargs)

    raise OSError(
        "File %r: format not understood or does not exist" % filename)

def new(n_particles = 0, order = None, class_ = SimSnap, **families) -> SimSnap:
    """Create a blank SimSnap, with the specified number of particles.

    Position, velocity and mass arrays are created and filled with zeros.

    By default all particles are taken to be dark matter.

    To specify otherwise, pass in keyword arguments specifying the number of particles for each family, e.g.

    >>> f = new(dm=50, star=25, gas=25)

    The order in which the different families appear in the snapshot is unspecified unless you add an 'order' argument:

    >>> f = new(dm=50, star=25, gas=25, order='star,gas,dm')

    guarantees the stars, then gas, then dark matter particles appear in sequence.
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
