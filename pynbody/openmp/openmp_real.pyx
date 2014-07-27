import sys

cdef extern from "omp.h" :
     void omp_set_num_threads(int)

cdef extern from "omp.h" : 
     int omp_get_max_threads()

cdef extern from "omp.h" : 
     int omp_get_num_procs()

def get_threads() : 
    return omp_get_max_threads()

def set_threads(num) : 
    omp_set_num_threads(num)

def get_cpus() : 
    return omp_get_num_procs()
