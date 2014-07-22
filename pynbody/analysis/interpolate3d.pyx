cimport numpy as np
cimport cython
import numpy as np

DTYPE = np.double
ctypedef np.double_t DTYPE_t


cdef extern from "math.h":
    int floor(double)nogil

        
@cython.boundscheck(False)
@cython.wraparound(False)
def interpolate3d(int n, 
                  int n_x_vals, np.ndarray[np.float64_t,ndim=1] x_vals, 
                  int n_y_vals, np.ndarray[np.float64_t,ndim=1] y_vals, 
                  int n_z_vals, np.ndarray[np.float64_t,ndim=1] z_vals, 
                  np.ndarray[np.float64_t,ndim=1] x, 
                  np.ndarray[np.float64_t,ndim=1] y, 
                  np.ndarray[np.float64_t,ndim=1] z,
                  np.ndarray[np.float64_t,ndim=3] vals,
                  np.ndarray[np.float64_t,ndim=1] result_array) :

    from cython.parallel cimport prange 
    
    cdef int x_top_ind, x_bot_ind, y_top_ind, y_bot_ind, z_top_ind, z_bot_ind, mid_ind
    cdef double x_fac, y_fac, z_fac
    cdef double v0, v1, v00, v01, v10, v11, v000, v001, v010, v011, v100, v101, v110, v111    
    cdef double xi, yi, zi
    cdef unsigned int i

    for i in prange(n,nogil=True) : 
        xi = x[i]
        yi = y[i]
        zi = z[i]
        
        # find x indices
        x_top_ind = n_x_vals - 1
        x_bot_ind = 0
        
        while(x_top_ind > x_bot_ind + 1) : 
            mid_ind = floor((x_top_ind-x_bot_ind)/2)+x_bot_ind
            if (xi > x_vals[mid_ind]) : 
                x_bot_ind = mid_ind
            else :
                x_top_ind = mid_ind
	
    
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
                
                
        x_fac = (xi - x_vals[x_bot_ind])/(x_vals[x_top_ind] - x_vals[x_bot_ind])
        y_fac = (yi - y_vals[y_bot_ind])/(y_vals[y_top_ind] - y_vals[y_bot_ind])
        z_fac = (zi - z_vals[z_bot_ind])/(z_vals[z_top_ind] - z_vals[z_bot_ind])        
        
        v111 = vals[x_top_ind,y_top_ind,z_top_ind]
        v110 = vals[x_top_ind,y_top_ind,z_bot_ind]
        v11 = z_fac*(v111 - v110) + v110
        v011 = vals[x_bot_ind,y_top_ind,z_top_ind]
        v010 = vals[x_bot_ind,y_top_ind,z_bot_ind]
        v01 = z_fac*(v011 - v010) + v010
        v101 = vals[x_top_ind,y_bot_ind,z_top_ind]
        v100 = vals[x_top_ind,y_bot_ind,z_bot_ind]
        v10 = z_fac*(v101 - v100) + v100
        v001 = vals[x_bot_ind,y_bot_ind,z_top_ind]
        v000 = vals[x_bot_ind,y_bot_ind,z_bot_ind]
        v00 = z_fac*(v001 - v000) + v000
        v1 = x_fac*(v11 - v10) + v10
        v0 = x_fac*(v01 - v00) + v00
        result_array[i] = y_fac*(v1-v0) + v0
