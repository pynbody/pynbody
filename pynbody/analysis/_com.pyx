cimport cython
cimport numpy as np

import logging

import numpy as np
from cython.parallel import prange

logger = logging.getLogger('pynbody.analysis._com')


@cython.boundscheck(False)
@cython.wraparound(False)
def shrink_sphere_center(np.ndarray[np.float64_t, ndim=2] pos,
                         np.ndarray[np.float64_t, ndim=1] mass,
                         int min_particles,
                         float shrink_factor,
                         float starting_rmax,
                         int num_threads,
                         int itermax=1000) :

    cdef int npart = len(pos)
    cdef int npart_all = len(pos)
    cdef float r2 = 0
    cdef float tot_mass=0
    cdef np.ndarray[np.float64_t, ndim=1] com = np.zeros(3)
    cdef np.ndarray[np.float64_t, ndim=1] com_x = pos.mean(axis=0)
    cdef int i
    cdef int iternum=0
    cdef float current_rmax = np.inf, current_rmax2
    cdef double cx, cy, cz
    cdef double offset_x, offset_y, offset_z
    cdef double pix, piy, piz, mi

    logger.info("Initial rough COM=%s",com_x)

    while npart>min_particles :
        offset_x=0; offset_y=0; offset_z=0;
        cx=com_x[0]; cy=com_x[1]; cz=com_x[2]
        current_rmax2 = current_rmax*current_rmax
        with nogil:
            npart = 0
            for i in prange(npart_all, schedule='static', num_threads=num_threads):
                pix=pos[i,0]-cx; piy=pos[i,1]-cy; piz=pos[i,2]-cz
                r2 = pix*pix+piy*piy+piz*piz
                if r2<current_rmax2 :
                    mi = mass[i]
                    offset_x+=pix*mi
                    offset_y+=piy*mi
                    offset_z+=piz*mi
                    tot_mass+=mi
                    npart+=1

        if npart==0:
            return com_x


        # divide out total mass and shift
        com[0]=cx+offset_x/tot_mass; com[1]=cy+offset_y/tot_mass; com[2]=cz+offset_z/tot_mass



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

@cython.boundscheck(False)
@cython.wraparound(False)
def move_sphere_center(np.ndarray[np.float64_t, ndim=2] pos,
                       np.ndarray[np.float64_t, ndim=1] mass,
                       int min_particles,
                       float shrink_factor,
                       float starting_rmax,
                       int num_threads,
		       float tol,
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
    cdef double cx, cy, cz
    cdef double nx, ny, nz
    cdef double pix, piy, piz, mi

    logger.info("Initial rough COM=%s",com_x)

    ocen=np.zeros(3)
    while np.sqrt(((com_x-ocen)**2).sum())<=tol: ocen+=1.0

    while np.sqrt(((com_x-ocen)**2).sum())>tol :
        nx=0; ny=0; nz=0;
        cx=com_x[0]; cy=com_x[1]; cz=com_x[2]
        ocen = com_x
        with nogil:
            npart = 0
            for i in prange(npart_all, schedule='static', num_threads=num_threads):
                pix=pos[i,0]-cx; piy=pos[i,1]-cy; piz=pos[i,2]-cz
                r = pix*pix+piy*piy+piz*piz
                if r<current_rmax :
                    mi = mass[i]
                    nx+=pix*mi
                    ny+=piy*mi
                    nz+=piz*mi
                    tot_mass+=mi
                    npart+=1

        if npart==0:
            return com_x

        # divide out total mass and shift
        com[0]=cx+nx/tot_mass; com[1]=cy+ny/tot_mass; com[2]=cz+nz/tot_mass




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
