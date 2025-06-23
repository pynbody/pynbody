import numpy as np

cimport cython
cimport numpy as np

np.import_array()
cimport openmp
from cython.parallel cimport prange
from libc.math cimport atan, pow
from libc.stdlib cimport free, malloc

from .. import config


cdef extern from "geometry_selection.hpp" nogil:
    void wrapfn[T](T & x, T & y, T & z, const T & wrap, const T & wrap_by_two)
    void sphere_selection[T](T* position_array, char* output_array,
                              T x0, T y0, T z0, T max_radius,
                              T wrap,
                              Py_ssize_t num_particles, int num_threads)
    void cube_selection[T](T* position_array, char* output_array,
                            T x0, T y0, T z0, T x1, T y1, T z1,
                            T wrap,
                            Py_ssize_t num_particles, int num_threads)

ctypedef fused fused_float:
    np.float32_t
    np.float64_t

def selection(np.ndarray[fused_float, ndim=2] pos_ar, region, parameters, fused_float wrap):
    """OpenMP selection algorithm for spheres and cubes, optionally wrapped around the box.

    Parameters
    ----------
    pos_ar : array_like
        The positions of the particles, with shape (N,3). Must be C-contiguous.
        The datatype can either be float32 or float64, represented by fused_float.
    region : str
        The region to select particles from. Can be 'sphere' or 'cube'.
    parameters : array_like
        The parameters of the region. For a sphere, it is (x0, y0, z0, r_max), and for a cube,
        it is (x0, y0, z0, x1, y1, z1). Each parameter must be a float (will be cast to fused_float).
    wrap : fused_float
        The size of the box to wrap around. If <0, no wrapping is done.
    """

    cdef Py_ssize_t i
    cdef Py_ssize_t N=len(pos_ar)
    cdef fused_float cx,cy,cz,x,y,z,r2
    cdef fused_float r_max_2
    cdef np.ndarray[np.uint8_t, ndim=1] output = np.empty(len(pos_ar),dtype=np.uint8)

    cdef char* output_data = <char*> np.PyArray_DATA(output)

    if len(pos_ar) == 0:
        # The check for C-contiguous 3-col fails when len is zero, but there is nothing to do anyway, so just return
        return output

    if (np.PyArray_NDIM(pos_ar)!=2 or np.PyArray_DIMS(pos_ar)[1]!=3 or
            np.PyArray_STRIDES(pos_ar)[0] != sizeof(fused_float)*3 or
            np.PyArray_STRIDES(pos_ar)[1] != sizeof(fused_float)):
        raise ValueError("Input array must be C-contiguous and have 3 columns")

    cdef fused_float* pos_ar_data = <fused_float*> np.PyArray_DATA(pos_ar)

    cdef fused_float wrap_by_two = wrap/2
    cdef int num_threads = config['number_of_threads']

    if region == 'sphere':
        if len(parameters) != 4:
            raise ValueError("Sphere selection requires 4 parameters: (x0, y0, z0, r_max)")
        # (ugly syntax below is because starred parameters not allowed)
        sphere_selection(pos_ar_data, output_data,
                         <fused_float> parameters[0],
                         <fused_float> parameters[1],
                         <fused_float> parameters[2],
                         <fused_float> parameters[3], wrap,
                         len(pos_ar), num_threads)
    elif region == 'cube':
        if len(parameters) != 6:
            raise ValueError("Cube selection requires 6 parameters: (x0, y0, z0, x1, y1, z1)")
        cube_selection(pos_ar_data, output_data,
                       <fused_float> parameters[0],
                       <fused_float> parameters[1],
                       <fused_float> parameters[2],
                       <fused_float> parameters[3],
                       <fused_float> parameters[4],
                       <fused_float> parameters[5], wrap, len(pos_ar), num_threads)
    else:
        raise ValueError("Region must be either 'sphere' or 'cube'")

    return output
