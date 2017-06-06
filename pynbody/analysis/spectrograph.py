"""
spectrographs
=============

Calculate spectrographs of non-axisymmetric perturbations (bars/spirals)
in galaxy disks.

"""

import numpy as np
from . import profile, angmom

def calc_spectrograph(sims, mode, frequency_range=[0, 400], frequency_bins=20,
    radial_range=[0, 20], radial_bins=10, aligned=False, family='stars'):
    """
    
    calc_spectrograph
    =================

    This function calculates spectrograph data for a sequence of simulation outputs (equally
    spaced in time). It is called by pynbody.plot.spectrographs module.

    **Input**:

    *sims* : an ordered list of simulation snapshots that have to be equally spaced in time.
        The number of outputs in the list is ideally not even.

    *mode* : the azimuthal multiplicity of which the spectrograph is to be calculated

    *frequency_range* ([0, 400]) : Range of frequencies to be considered in 1/Gyr. This is
        the frequency at which the pattern recurs, i.e. the mode-fold of the pattern speed.

    *frequency_bins* (20) : Number of bins to split the frequency range into

    *radial_range* ([0, 400]) : Range of radii to be considered in kpc

    *radial_bins* (20) : Number of bins to split the radial range into

    *aligned* (False) : If False, the snapshot will be centered on the central galaxy and
        its disk will be aligned to be in the xy-plane. The snapshots are only modified for this
        analysis. After the routine they will be in their original state. If true, the snapshot
        is assumed to be aligned.

    *family* ('stars') : Consider only the respective family portion of the snapshots.

    **Returns**:

    *spectrum* : 2D array, the first index represents the radial, the second index the
        frequency coordinate.

    """

    times = [sim.properties['time'].in_units('Gyr') for sim in sims]
    dts = np.diff(times)
    test_dts(dts)

    nsims = len(sims)

    W = np.zeros((radial_bins, nsims))*(1. + 1j)
    r_bin_edges = np.linspace(*radial_range, num=radial_bins+1)
    omega_bin_edges = np.linspace(*frequency_range, num=frequency_bins+1)
    omega = np.array([omega_bin_edges[:-1] + .5*np.diff(omega_bin_edges)])
    omega = np.array([omega.repeat(W.shape[0], axis=0)]).repeat(W.shape[1], axis=0).T
    for i, sim in enumerate(sims):
        s = {'stars': sim.s, 'gas': sim.g, 'dark': sim.d}[family]
        if not aligned:
            with angmom.faceon(s, vcen=0):
                W[:,i] = calc_fourier_modes(s, mode, r_bin_edges)
        else:
            W[:,i] = calc_fourier_modes(s, mode, r_bin_edges)
    W *= np.hanning(nsims)
    Wf = np.exp(1j*omega*times)*W
    return (.5*(Wf[:,:,:-1]+Wf[:,:,1:])*dts).sum(axis=2)

def calc_fourier_modes(sim, mode, r_bin_edges):
    """

    Calculate Fourier coefficients of azimuthal mass distribution of multiplicity mode for bins
    according to given bin edges.

    **Input**:

    *sim* : Simulation snapshot

    *mode* : Multiplicity of the perturbation

    *r_bin_edges* : Radial bin edges in kpc

    """

    prof = profile.Profile(sim, bins=r_bin_edges)
    return prof['fourier']['c'][mode]

def test_dts(dts, tol=1e-10):

    """

    test_dts
    ========

    Simple routine to test whether times are equally spaced. Raises an error if not.

    **Input**:

    *dts* : Array of time differences

    *tol* (1e-10) : relative tolerance (std/mean)

    """

    if np.abs(dts.std()/dts.mean()) > tol:
        raise ValueError("The simulation outputs are not equally spaced in time or not ordered.")
