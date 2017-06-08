"""
spectrographs
=============

Calculate spectrographs of non-axisymmetric perturbations (bars/spirals)
in galaxy disks.

"""

import numpy as np
from . import profile, angmom
from .. import units

def calc_spectrograph(halos, mode, frequency_range, radial_range, frequency_bins=20,
    radial_bins=10, aligned=False, family='stars'):
    """
    
    calc_spectrograph
    =================

    This function calculates spectrograph data for a sequence of simulation outputs (equally
    spaced in time). It is called by pynbody.plot.spectrographs module.

    **Input**:

    *halos* : an ordered list of subviews of simulation snapshots that contain the halo in
        question. They have to be equally spaced in time. The number of outputs in the list
        is ideally not even.

    *mode* : the azimuthal multiplicity of which the spectrograph is to be calculated

    *frequency_range* : Range of frequencies to be considered in 1/Gyr. This is
        the frequency at which the pattern recurs, i.e. the mode-fold of the pattern speed.

    *radial_range* : Range of radii to be considered in kpc

    *frequency_bins* (20) : Number of bins to split the frequency range into

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

    times = [halo.properties['time'].in_units('Gyr') for halo in halos]
    dts = np.diff(times)
    test_dts(dts)

    nhalos = len(halos)

    unit_factor = units.kpc.in_units(halos[0]['x'], **halo.conversion_context())

    W = np.zeros((radial_bins, nhalos))*(1. + 1j)
    r_bin_edges = np.linspace(*radial_range, num=radial_bins+1)*unit_factor
    omega_bin_edges = np.linspace(*frequency_range, num=frequency_bins+1)
    omega = np.array([omega_bin_edges[:-1] + .5*np.diff(omega_bin_edges)])
    omega = np.array([omega.repeat(W.shape[0], axis=0)]).repeat(W.shape[1], axis=0).T
    for i, halo in enumerate(halos):
        s = {'stars': halo.s, 'gas': halo.g, 'dark': halo.d}[family]
        if not aligned:
            with angmom.faceon(s, vcen=0):
                W[:,i] = calc_fourier_modes(s, mode, r_bin_edges)
        else:
            W[:,i] = calc_fourier_modes(s, mode, r_bin_edges)
    W *= np.hanning(nhalos)
    Wf = np.exp(1j*omega*times)*W
    return (.5*(Wf[:,:,:-1]+Wf[:,:,1:])*dts).sum(axis=2)

def calc_fourier_modes(halo, mode, r_bin_edges):
    """

    Calculate Fourier coefficients of azimuthal mass distribution of multiplicity mode for bins
    according to given bin edges.

    **Input**:

    *halo* : Simulation snapshot

    *mode* : Multiplicity of the perturbation

    *r_bin_edges* : Radial bin edges in kpc

    """

    prof = profile.Profile(halo, bins=r_bin_edges)
    return prof['fourier']['c'][mode]

def test_dts(dts, tol=1e-6):

    """

    test_dts
    ========

    Simple routine to test whether times are equally spaced. Raises an error if not.

    **Input**:

    *dts* : Array of time differences

    *tol* (1e-6) : relative tolerance (std/mean)

    """

    if np.abs(dts.std()/dts.mean()) > tol:
        raise ValueError("The simulation outputs are not equally spaced in time or not ordered.")
