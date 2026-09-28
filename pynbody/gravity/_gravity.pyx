#cython: embedsignature=True

cimport cython

import numpy as np

from pynbody import array, config, openmp, units

cimport numpy as np

np.import_array()

DTYPE = np.double

# Each argument of _direct gets its own fused type, so that Cython generates a specialisation for
# every combination of single and double precision inputs. A single shared fused type would
# instead force the positions, masses and softenings all to have the same dtype as each other,
# which snapshots are under no obligation to do. See also the same pattern in pynbody/sph/_render.pyx.

ctypedef fused ipos_t:
    np.float32_t
    np.float64_t

ctypedef fused pos_t:
    np.float32_t
    np.float64_t

ctypedef fused mass_t:
    np.float32_t
    np.float64_t

ctypedef fused eps_t:
    np.float32_t
    np.float64_t

cdef extern from "math.h" nogil:
      double sqrt(double)
      float sqrt(float)


def direct(f, ipos, eps=None, int num_threads = 0):
    global config

    if num_threads == 0 :
        num_threads = int(config["number_of_threads"])

    if num_threads < 0:
        num_threads = openmp.get_cpus()

    if num_threads > openmp.get_cpus() :
        num_threads = openmp.get_cpus()

    openmp.set_threads(num_threads)

    if eps is None:
        try:
            eps = f['eps']
        except KeyError:
            eps = f.properties['eps']

    if isinstance(eps, str):
        eps = units.Unit(eps)

    if isinstance(eps, units.UnitBase):
        eps = eps.in_units(f['pos'].units, **f.conversion_context())

    # Duck-typed rather than an isinstance check, so that the softening of a subsnap gets
    # converted too: those are IndexedSimArrays, which are not a subclass of SimArray.
    if units.has_units(eps):
        eps = eps.in_units(f['pos'].units, **f.conversion_context())

    ipos = np.asarray(ipos)

    if np.ndim(eps) == 0:
        eps = np.repeat(np.asarray(eps, dtype=ipos.dtype), len(f))
    else:
        eps = np.asarray(eps)

        if len(eps) != len(f):
            raise ValueError(
                f"The softening array has length {len(eps)}, but the snapshot has {len(f)} particles"
            )

    m_by_r, m_by_r2 = _direct(ipos, np.asarray(f['pos']), np.asarray(f['mass']), np.asarray(eps))

    pot = array.SimArray(-m_by_r,units=f['mass'].units/f['pos'].units * units.G)
    accel = array.SimArray(-m_by_r2,units=f['mass'].units/f['pos'].units**2 * units.G)

    return pot, accel


@cython.cdivision(True)
@cython.boundscheck(False)
def _direct(np.ndarray[ipos_t, ndim=2] ipos, np.ndarray[pos_t, ndim=2] pos,
            np.ndarray[mass_t, ndim=1] mass, np.ndarray[eps_t, ndim=1] eps):
    from cython.parallel cimport prange

    cdef Py_ssize_t nips = len(ipos)
    cdef np.ndarray[ipos_t, ndim=2] m_by_r2 = np.zeros((nips,3), dtype = ipos.dtype)
    cdef np.ndarray[ipos_t, ndim=1] m_by_r = np.zeros(nips, dtype = ipos.dtype)
    cdef Py_ssize_t n = len(mass)

    cdef Py_ssize_t pi, i
    cdef double dx, dy, dz, mass_i, eps_i, drsoft, drsoft3

    for pi in prange(nips, nogil=True, schedule='static'):
        for i in range(n):
            mass_i = mass[i]
            eps_i = eps[i]
            dx = ipos[pi,0] - pos[i,0]
            dy = ipos[pi,1] - pos[i,1]
            dz = ipos[pi,2] - pos[i,2]
            drsoft = 1.0/sqrt(dx*dx + dy*dy + dz*dz + eps_i*eps_i)
            drsoft3 = drsoft*drsoft*drsoft
            m_by_r[pi] += mass_i * drsoft
            m_by_r2[pi,0] += mass_i*dx * drsoft3
            m_by_r2[pi,1] += mass_i*dy * drsoft3
            m_by_r2[pi,2] += mass_i*dz * drsoft3

    return m_by_r, m_by_r2
