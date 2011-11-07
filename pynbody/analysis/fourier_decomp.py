import numpy as np

#
# A module for doing Fourier analysis on a set of particles
#
# Mostly empty now, but will add functions to decompose and reform
# images of fourier components from particle data...
#

"""

Module for fourier decomposition of particles in a disk

Required inputs are particle x,y positions and masses.
All functions assume that the particles have been aligned and centered.

"""

def fourier(px, py, pm):
    """

    Calculate the fourier coefficients for a set of particle positions.

    Assumes that the particles belong to a disk that has been centered
    and aligned.

    """

    phi = np.arctan2(py,px)

    hist, binphi = np.histogram(phi, weights = pm, bins = 100)

    binphi = .5*(binphi[1:]+binphi[:-1])

    c = np.zeros(7,dtype=complex)

    for m in np.arange(0,7):
        c[m] = np.sum(hist*np.exp(1j*m*binphi))


    return c


def fourier_map(p, ncell=100,mmin=0,mmax=7, rmax=10) : 
    """

    Calculate an overdensity map based on the fourier expansion. 
    
    **Input**:

    *p* :  a Profile object

    **Optional Keywords**:
    
    *ncell (default = 100)* : number of cells used to create the map
    
    *mmin (default = 0)* : minimum m-mode to use in the reconstruction

    *mmax (default = 7)* : maximum m-mode to use in the reconstruction

    *rmax (default = 10)* : maximum radius to use in the reconstruction
    
    """

    f_map = np.zeros((ncell,ncell))
    f = p['fourier']
    xbins = np.linspace(-rmax,rmax,ncell+1)
    ybins = np.linspace(-rmax,rmax,ncell+1)

    xmidbins = (xbins[:-1] + xbins[1:])/2
    ymidbins = (ybins[:-1] + ybins[1:])/2

    for i in range(0,ncell) : 
        
        for j in range(0,ncell) : 
            
            rcell = np.sqrt(xmidbins[i]**2 + ymidbins[j]**2)

            if rcell > rmax : f_map[i,j] = 0.0
            else : 
                
                binind = np.nonzero(p['rbins'] > rcell)[0][0]
                binphi = np.arctan2(ymidbins[j],xmidbins[i])
                
                for m in range(mmin,mmax) : 
                    phi = f['phi'][m,binind]

                    f_map[i,j] = f_map[i,j] + f['c'][m,binind]*np.exp(1j*m*(binphi))

    
    
    return f_map,xmidbins,ymidbins
