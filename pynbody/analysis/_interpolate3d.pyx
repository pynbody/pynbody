cimport cython
cimport numpy as np

import sys

import numpy as np

from cython cimport floating


cdef extern from "math.h":
    int floor(double)nogil

ctypedef fused fused_input_type_1:
    np.float32_t
    np.float64_t

ctypedef fused fused_input_type_2:
    np.float32_t
    np.float64_t

ctypedef fused fused_input_type_3:
    np.float32_t
    np.float64_t


@cython.boundscheck(False)
@cython.wraparound(False)
def interpolate3d(int n,
                  np.ndarray[fused_input_type_1,ndim=1] x,
                  np.ndarray[fused_input_type_2,ndim=1] y,
                  np.ndarray[fused_input_type_3,ndim=1] z,
                  int n_x_vals, np.ndarray[np.float64_t,ndim=1] x_vals,
                  int n_y_vals, np.ndarray[np.float64_t,ndim=1] y_vals,
                  int n_z_vals, np.ndarray[np.float64_t,ndim=1] z_vals,
                  np.ndarray[np.float64_t,ndim=3] vals,
                  np.ndarray[np.float64_t,ndim=1] result_array) :

    from cython.parallel cimport prange

    cdef int x_top_ind, x_bot_ind, y_top_ind, y_bot_ind, z_top_ind, z_bot_ind, mid_ind
    cdef double x_fac, y_fac, z_fac
    cdef double v0, v1, v00, v01, v10, v11, v000, v001, v010, v011, v100, v101, v110, v111
    cdef double xi, yi, zi
    cdef Py_ssize_t i

    for i in prange(n,nogil=True) :
        if n_x_vals > 0 :
            xi = x[i]
        yi = y[i]
        zi = z[i]

        if n_x_vals > 0 :
            # find x indices
            x_top_ind = n_x_vals - 1
            x_bot_ind = 0

            while(x_top_ind > x_bot_ind + 1) :
                mid_ind = floor((x_top_ind-x_bot_ind)/2)+x_bot_ind
                if (xi > x_vals[mid_ind]) :
                    x_bot_ind = mid_ind
                else :
                    x_top_ind = mid_ind

        else :
            x_top_ind = 0
            x_bot_ind = 0

        # find y indices
        y_top_ind = n_y_vals - 1
        y_bot_ind = 0

        while(y_top_ind > y_bot_ind + 1) :
            mid_ind = floor((y_top_ind-y_bot_ind)/2)+y_bot_ind
            if (yi > y_vals[mid_ind]) :
                y_bot_ind = mid_ind
            else :
                y_top_ind = mid_ind

        # find z indices
        z_top_ind = n_z_vals - 1
        z_bot_ind = 0

        while(z_top_ind > z_bot_ind + 1) :
            mid_ind = floor((z_top_ind-z_bot_ind)/2)+z_bot_ind
            if (zi > z_vals[mid_ind]) :
                z_bot_ind = mid_ind
            else :
                z_top_ind = mid_ind

        if n_x_vals > 0 :
            x_fac = (xi - x_vals[x_bot_ind])/(x_vals[x_top_ind] - x_vals[x_bot_ind])

        y_fac = (yi - y_vals[y_bot_ind])/(y_vals[y_top_ind] - y_vals[y_bot_ind])
        z_fac = (zi - z_vals[z_bot_ind])/(z_vals[z_top_ind] - z_vals[z_bot_ind])

        # vertex values
        if n_x_vals > 0 :
            v000 = vals[x_bot_ind,y_bot_ind,z_bot_ind]
            v001 = vals[x_bot_ind,y_bot_ind,z_top_ind]
            v010 = vals[x_bot_ind,y_top_ind,z_bot_ind]
            v011 = vals[x_bot_ind,y_top_ind,z_top_ind]
            v100 = vals[x_top_ind,y_bot_ind,z_bot_ind]
            v101 = vals[x_top_ind,y_bot_ind,z_top_ind]
            v110 = vals[x_top_ind,y_top_ind,z_bot_ind]
            v111 = vals[x_top_ind,y_top_ind,z_top_ind]

            v00 = v000*(1.0-x_fac) + v100*x_fac
            v10 = v010*(1.0-x_fac) + v110*x_fac
            v01 = v001*(1.0-x_fac) + v101*x_fac
            v11 = v011*(1.0-x_fac) + v111*x_fac

        else :
            v00 = vals[0, y_bot_ind, z_bot_ind]
            v01 = vals[0, y_bot_ind, z_top_ind]
            v10 = vals[0, y_top_ind, z_bot_ind]
            v11 = vals[0, y_top_ind, z_top_ind]

        v0 = v00*(1.0-y_fac) + v10*y_fac
        v1 = v01*(1.0-y_fac) + v11*y_fac

        result_array[i] = v0*(1-z_fac) + v1*z_fac


# BISECT DOESNT WORK WITH OPENMP... YIELDS SEGFAULT??
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int bisect(np.float64_t x, int nval, np.float64_t [:] arr) nogil:
    cdef int mid, top, bot
    top = nval - 1
    bot = 0

    while(top > bot + 1) :
        mid = floor((top-bot)/2)+bot
        if (x > arr[mid]) :
            bot = mid
        else :
            top = mid
    return bot
