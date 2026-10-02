"""
Implements classes to load and manipulate snapshot data
"""

from __future__ import annotations

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

    .. versionchanged:: 2.7.0

      ``take`` accepts a slice as well as an array of indices, and rejects indices which are not strictly
      ascending. Previously a repeated or out-of-order index either produced a snapshot containing the same
      particle twice, or failed with an ``IndexError`` from within the chunking machinery, depending on the
      format.

    Parameters
    ----------
    filename : str
        The filename to load

    priority : optional, list[str | type]
        A list of SimSnap subclasses to try, in order. The first class which is capable of loading the file
        is used. If not specified, the ordering is as specified in the configuration files.

    take : optional, np.ndarray[int] | slice
        If specified, only the particles with the given indices are loaded. This can be used to load a subset of
        the snapshot, e.g. for a halo or a region of interest. The indices refer to the order in which the
        particles lie on disk, and must be strictly ascending; an array which repeats or reorders them, or a
        slice with a negative step, is rejected.

    take_region : optional, pynbody.filt.Filter
        If specified, a sub-region including the particles with the given spatial filter are loaded. Currently
        only Swift snapshots have the required spatial index to support this option.

    *args, **kwargs :
        Other arguments and keyword arguments are passed to the class constructor that is used to load the file.

    Returns
    -------
    SimSnap
        The loaded snapshot

    """

    filename = pathlib.Path(filename)

    priority = kwargs.pop('priority', config['snap-class-priority'])

    c = _identify(filename, priority, kwargs.get('remote_dir'))
    if c is None:
        if kwargs.get('remote_dir') is not None:
            raise OSError("File %r: format not understood or does not exist" % filename)
        raise OSError(f"File {str(filename)!r}: format not understood or does not exist "
                      f"({_describe_unidentified(filename)})")

    logger.info("Loading using backend %s" % str(c))
    return c(filename, *args, **kwargs)

def identify(filename, priority=None) -> type[SimSnap] | None:
    """Return the SimSnap subclass that :func:`load` would use for the specified file, without loading it.

    Parameters
    ----------
    filename : str
        The filename to identify

    priority : optional, list[str | type]
        As for :func:`load`

    Returns
    -------
    type[SimSnap] | None
        The class that would be used, or None if no class can load the file
    """
    if priority is None:
        priority = config['snap-class-priority']
    return _identify(pathlib.Path(filename), priority)

def _identify(filename, priority, remote_dir=None):
    from . import gadgethdf

    # HDF5-based classes share one open of the file while deciding; it is closed before the chosen class loads it
    with gadgethdf._share_files_during_detection() as shared_files:
        for c in SimSnap.iter_subclasses_with_priority(priority):
            if remote_dir is not None:
                if hasattr(c, "_can_load_remote") and c._can_load_remote(filename, remote_dir):
                    return c
                continue
            if not issubclass(c, gadgethdf.GadgetHDFSnap):
                # Other classes may open the file themselves, possibly with different locking flags, which
                # HDF5 refuses while it is still open here
                shared_files.close()
            if c._can_load(filename):
                return c
    return None

def _describe_unidentified(filename: pathlib.Path) -> str:
    """Say what was found at a path that no class could load, to make the error message more helpful"""
    if not filename.exists():
        return "path does not exist"
    if filename.is_dir():
        return "path is a directory"
    from .gadgethdf import h5py
    if h5py is not None and h5py.is_hdf5(filename):
        try:
            with h5py.File(filename, 'r') as f:
                keys = list(f.keys())
        except OSError:
            return "file is HDF5 but could not be opened"
        if len(keys) > 10:
            keys = keys[:10] + ["..."]
        return "file is HDF5 with top-level entries: " + ", ".join(keys)
    return "file is not HDF5, and is not recognised as any other format"

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


from . import (
    ascii,
    gadget,
    gadgethdf,
    grafic,
    nchilada,
    pkdgravhdf,
    ramses,
    subsnap,
    swift,
    tipsy,
)
from .subsnap import FamilySubSnap, IndexedSubSnap, SubSnap
