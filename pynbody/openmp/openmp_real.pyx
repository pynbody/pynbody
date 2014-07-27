
cimport openmp


def get_threads() :
    return openmp.omp_get_max_threads()

def set_threads(num) :
    openmp.omp_set_num_threads(num)

def get_cpus() :
    return openmp.omp_get_num_procs()
