cimport numpy as np
cimport cython
import numpy as np

import logging
logger = logging.getLogger('pynbody.analysis._com')

        
@cython.boundscheck(False)
@cython.wraparound(False)
def shrink_sphere_center(np.ndarray[np.float64_t, ndim=2] pos,
                         np.ndarray[np.float64_t, ndim=1] mass,
                         int min_particles,
                         float shrink_factor,
                         float starting_rmax,
                         int itermax=1000) :

    cdef int npart = len(pos)
    cdef int npart_all = len(pos)
    cdef float r = 0    
    cdef float tot_mass=0
    cdef np.ndarray[np.float64_t, ndim=1] com = np.zeros(3)
    cdef np.ndarray[np.float64_t, ndim=1] com_x = pos.mean(axis=0) 
    cdef int i
    cdef int iternum=0
    cdef float current_rmax = np.inf

    logger.info("Initial rough COM=%s",com_x)
    
    while npart>min_particles :
        with nogil:
            npart = 0
            for i in range(npart_all) :
                r = (pos[i,0]-com_x[0])**2+(pos[i,1]-com_x[1])**2+(pos[i,2]-com_x[2])**2
                if r<current_rmax :
                    com[0]+=(pos[i,0]-com_x[0])*mass[i]
                    com[1]+=(pos[i,1]-com_x[1])*mass[i]
                    com[2]+=(pos[i,2]-com_x[2])*mass[i]
                    tot_mass+=mass[i]
                    npart+=1
        if npart==0:
            return com_x

        # divide out total mass and shift
        com/=tot_mass
        com+=com_x

        # update for next cycle
        com_x[:]= com
        com[:]=0
        tot_mass=0

        iternum+=1
        if iternum>1 :
            current_rmax*=shrink_factor
        else :
            current_rmax = starting_rmax

        if iternum>itermax:
            raise RuntimeError, "shrink_sphere_center failed to converge after %d iterations"%itermax
            
    return com_x
