"""Interface to OpenMP functions"""

cdef extern from "omp.h" :
     void omp_set_num_threads(int)

cdef extern from "omp.h" :
     int omp_get_max_threads()

cdef extern from "omp.h" :
     int omp_get_num_procs()

def get_threads() :
    """Maps to the OpenMP function omp_get_max_threads()"""
    return omp_get_max_threads()

def set_threads(num) :
    """Maps to the OpenMP function omp_set_num_threads()"""
    omp_set_num_threads(num)

def get_cpus() :
    """Maps to the OpenMP function omp_get_num_procs()"""
    return omp_get_num_procs()
