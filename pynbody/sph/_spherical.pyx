import numpy as np
import healpy as hp

cimport numpy as np
cimport cython
from libc.math cimport atan

        
@cython.boundscheck(False)
@cython.wraparound(False)
def render_spherical_image_core(np.ndarray[np.float64_t, ndim=1] rho, # array of particle densities
								np.ndarray[np.float64_t, ndim=1] mass, # array of particle masses
								np.ndarray[np.float64_t, ndim=1] qtyar, # array of quantity to make image of
                                np.ndarray[np.float64_t, ndim=2] pos, # array of particle positions
                                np.ndarray[np.float64_t, ndim=1] r, # particle radius
								np.ndarray[np.float64_t, ndim=1] h, # particle smoothing length
                                np.ndarray[np.int64_t, ndim=1] ind, # which of the above particles to use
                                np.ndarray[np.float64_t, ndim=1] ds, # what distances to sample at (in units of smoothing)
                                np.ndarray[np.float64_t, ndim=1] weights, # what kernel weighting to use at these samples
                                unsigned int nside) :
                                
    cdef unsigned int i,i0,j,n=len(ind),m=len(ds),num_pix
    cdef long k
    cdef float angle, norm, den
    cdef unsigned int h_power = 2 # to update
    cdef np.ndarray[np.int64_t,ndim=1] healpix_pixels, buff
    cdef np.ndarray[np.float64_t,ndim=1] im, im_norm

    n = len(ind)
    im = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    im_norm = np.zeros_like(im)
    buff = np.empty(len(im), dtype=np.int64)
    
    if not hp.isnsideok(nside):
        raise ValueError('Wrong nside value, must be a power of 2')

	# go through each particle
    for i0 in range(n) :
        i = ind[i0]
		# go through each kernel step
        for j in range(m) :

            angle = atan(h[i]*ds[j]/r[i])
            norm = weights[j]*mass[i]/rho[i]/h[i]**h_power
            den = qtyar[i]*norm

			# find the pixels affected
            healpix_pixels = hp.query_disc(nside, pos[i], angle , inclusive=False, buff=buff)
            num_pix = len(healpix_pixels)

            with nogil :
				# add the required amount to those particles
                for k in range(num_pix) :
                    im[healpix_pixels[k]]+=den
                    im_norm[healpix_pixels[k]]+=norm

                
    return im, im_norm
