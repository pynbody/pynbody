"""

util
====

Utility functions for the plotting module

"""

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.signal

def fast_kde(x, y, kern_nx = None, kern_ny = None, gridsize=(100, 100), 
             extents=None, nocorrelation=False, weights=None, norm = False, **kwargs):
    """
    A faster gaussian kernel density estimate (KDE).  Intended for
    computing the KDE on a regular grid (different use case than
    scipy's original scipy.stats.kde.gaussian_kde()).  

    Author: Joe Kington
    License:  MIT License <http://www.opensource.org/licenses/mit-license.php>

    Performs a gaussian kernel density estimate over a regular grid using a
    convolution of the gaussian kernel with a 2D histogram of the data.

    This function is typically several orders of magnitude faster than 
    scipy.stats.kde.gaussian_kde for large (>1e7) numbers of points and 
    produces an essentially identical result.

    **Input**:
    
        *x*: array
            The x-coords of the input data points
        
        *y*: array
            The y-coords of the input data points
        
        *kern_nx*: float 
            size (in units of *x*) of the kernel

        *kern_ny*: float
            size (in units of *y*) of the kernel

        *gridsize*: (Nx , Ny) tuple (default: 200x200) 
            Size of the output grid
                    
        *extents*: (default: extent of input data) A (xmin, xmax, ymin, ymax)
            tuple of the extents of output grid

        *nocorrelation*: (default: False) If True, the correlation between the
            x and y coords will be ignored when preforming the KDE.
        
        *weights*: (default: None) An array of the same shape as x & y that 
            weighs each sample (x_i, y_i) by each value in weights (w_i).
            Defaults to an array of ones the same size as x & y.
            
        *norm*: boolean (default: False) 
            If False, the output is only corrected for the kernel. If True,
            the result is normalized such that the integral over the area 
            yields 1. 

    **Output**:
        A gridded 2D kernel density estimate of the input points. 
    """
   
    #---- Setup --------------------------------------------------------------
    x, y = np.asarray(x), np.asarray(y)
    x, y = np.squeeze(x), np.squeeze(y)
    
    if x.size != y.size:
        raise ValueError('Input x & y arrays must be the same size!')

    nx, ny = gridsize
    n = x.size

    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(n)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size'
                    ' as input x & y arrays!')

    # Default extents are the extent of the data
    if extents is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = map(float, extents)
        
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    #---- Preliminary Calculations -------------------------------------------

    # First convert x & y over to pixel coordinates
    # (Avoiding np.digitize due to excessive memory usage!)
    xyi = np.vstack((x,y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    # Next, make a 2D histogram of x & y
    # Avoiding np.histogram2d due to excessive memory usage with many points
    grid = sp.sparse.coo_matrix((weights, xyi), shape=(nx, ny)).toarray()

    # Calculate the covariance matrix (in pixel coords)
    cov = np.cov(xyi)

    if nocorrelation:
        cov[1,0] = 0
        cov[0,1] = 0

    # Scaling factor for bandwidth
    scotts_factor = np.power(n, -1.0 / 6) # For 2D

    #---- Make the gaussian kernel -------------------------------------------

    # First, determine how big the kernel needs to be
    std_devs = np.diag(np.sqrt(cov))

    if kern_nx is None or kern_ny is None: 
        kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)
    
    else: 
        kern_nx = np.round(kern_nx / dx)
        kern_ny = np.round(kern_ny / dy)

    # Determine the bandwidth to use for the gaussian kernel
    inv_cov = np.linalg.inv(cov * scotts_factor**2) 

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    yy = np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    xx, yy = np.meshgrid(xx, yy)

    # Then evaluate the gaussian function on the kernel grid
    kernel = np.vstack((xx.flatten(), yy.flatten()))
    kernel = np.dot(inv_cov, kernel) * kernel 
    kernel = np.sum(kernel, axis=0) / 2.0 
    kernel = np.exp(-kernel) 
    kernel = kernel.reshape((kern_ny, kern_nx))

    #---- Produce the kernel density estimate --------------------------------

    # Convolve the gaussian kernel with the 2D histogram, producing a gaussian
    # kernel density estimate on a regular grid
    grid = sp.signal.convolve2d(grid, kernel, mode='same', boundary='fill').T

    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.  
    norm_factor = 2 * np.pi * cov * scotts_factor**2
    norm_factor = np.linalg.det(norm_factor)
    #norm_factor = n * dx * dy * np.sqrt(norm_factor)
    norm_factor = np.sqrt(norm_factor)
    
    if norm : norm_factor *= n * dx * dy

    # Normalize the result
    grid /= norm_factor

    return grid


def inv_fourier(p, nmin=1000, mmin=1,mmax=7, nphi = 100) : 
    """

    Invert a profile with fourier coefficients to yield an overdensity
    map.

    **Inputs:** 

    *p* : a :func:`~pynbody.analysis.profile.Profile` object

    **Optional Keywords:**
    
    *nmin* (1000) : minimum number of particles required per bin 

    *mmin* (1)    : lowest multiplicity Fourier component

    *mmax* (7)    : highest multiplicity Fourier component

    *nphi* (100)  : number of azimuthal bins to use for the map

    """

    phi_hist = np.zeros((len(p['rbins']), nphi))
    phi = np.linspace(-np.pi,np.pi,nphi)
    rbins = p['rbins']

    for i in range(len(rbins)) : 
        if p['n'][i] > nmin : 
            for m in range(mmin,mmax) : 
                phi_hist[i,:] = phi_hist[i,:] + p['fourier']['c'][m,i]*np.exp(1j*m*phi)

    return phi, phi_hist
                
