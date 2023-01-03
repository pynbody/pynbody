#cython: embedsignature=True

cimport cython

import numpy as np

from pynbody import array, config, openmp, units
from pynbody.util import get_eps

cimport numpy as np

DTYPE = np.double

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

cdef extern from "math.h" nogil:
      double sqrt(double)
      float sqrt(float)


@cython.cdivision(True)
@cython.boundscheck(False)
def direct(f, np.ndarray[DTYPE_t, ndim=2] ipos, eps=None, int num_threads = 0):

    from cython.parallel cimport prange

    global config


    if num_threads == 0 :
        num_threads = int(config["number_of_threads"])

    if num_threads < 0:
        num_threads = openmp.get_cpus()

    if num_threads > openmp.get_cpus() :
        num_threads = openmp.get_cpus()

    openmp.set_threads(num_threads)

    if eps is None:
        eps = get_eps(f)

    cdef unsigned int nips = len(ipos)
    cdef np.ndarray[DTYPE_t, ndim=2] m_by_r2 = np.zeros((nips,3), dtype = ipos.dtype)
    cdef np.ndarray[DTYPE_t, ndim=1] m_by_r = np.zeros(nips, dtype = ipos.dtype)
    cdef np.ndarray[DTYPE_t, ndim=2] pos = f['pos'].view(np.ndarray)
    cdef np.ndarray[DTYPE_t, ndim=1] mass = f['mass'].view(np.ndarray)
    cdef unsigned int n = len(mass)
    cdef np.ndarray[DTYPE_t, ndim=1] epssq = eps * eps

    cdef unsigned int pi, i
    cdef double dx, dy, dz, mass_i, epssq_i, drsoft, drsoft3

    for pi in prange(nips, nogil=True, schedule='static'):
        for i in range(n):
            mass_i = mass[i]
            epssq_i = epssq[i]
            dx = ipos[pi,0] - pos[i,0]
            dy = ipos[pi,1] - pos[i,1]
            dz = ipos[pi,2] - pos[i,2]
            drsoft = 1.0/sqrt(dx*dx + dy*dy + dz*dz + epssq_i)
            drsoft3 = drsoft*drsoft*drsoft
            m_by_r[pi] += mass_i * drsoft
            m_by_r2[pi,0] += mass_i*dx * drsoft3
            m_by_r2[pi,1] += mass_i*dy * drsoft3
            m_by_r2[pi,2] += mass_i*dz * drsoft3


    m_by_r = m_by_r.view(array.SimArray)
    m_by_r2 = m_by_r2.view(array.SimArray)
    m_by_r.units = f['mass'].units/f['pos'].units
    m_by_r2.units = f['mass'].units/f['pos'].units**2

    m_by_r*=units.G
    m_by_r2*=units.G

    return -m_by_r, -m_by_r2
