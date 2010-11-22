import numpy as np
import tipsyio as tio
import profile

#
# A module for doing Fourier analysis on a set of particles
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


def fourier_prof(prof, px, py, pm):
    """

    Produce a profile of fourier decomposition of particle positions.
    Appends the decomposition quantities to the existing instance of
    Profile that is passed as the first argument.

    The function assumes that the particles belong to a disk and they
    have been centered and aligned previously.

    Appends the arrays c, amp, and phi to the Profile object.

    To get the m=2 amplitude, the amp array is accessed as

    m2 = amp[2,:]

    Usage:

    >>> prof = Profile(sqrt(x**2 + y**2), nbins = 50, max = 20)
    >>> fourier_prof(prof, x, y, m)
    >>> plot(prof.midbins, prof.amp[2,:]

    """
    
    
    assert isinstance(prof, profile.Profile)

    # make sure the profile has a mass profile

    if not 'mass' in dir(prof):
        prof.massprof(pm)
    
    prof.phi = np.zeros((7,prof.nbins))
    prof.amp = np.zeros((7,prof.nbins))
    prof.c = np.zeros((7,prof.nbins), dtype=complex)

#    nonzero = prof.ninbin > 100
    
    for i, bin in enumerate(prof.bins[:-1]):
        
        if prof.ninbin[i] > 100:
            prof.c[:,i] = fourier(px[prof.binind[i]],
                                  py[prof.binind[i]],
                                  pm[prof.binind[i]], bins)
     

    im = np.imag(prof.c)
    re = np.real(prof.c)
    
    prof.c /= prof.mass

    prof.amp = np.sqrt(im**2 + re**2)
    prof.phi = np.arctan2(im,re)
        

        
            
            
            
