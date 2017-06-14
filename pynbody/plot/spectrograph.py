"""
spectrographs
=============

Plot spectrographs of non-axisymmetric perturbations (bars/spirals)
in galaxy disks.

"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import warnings

from ..analysis.spectrograph import calc_spectrograph #profile, angmom, halo
#from .. import filt, units, config, array
#from .sph import image
#from .. import units as _units

#import logging
#logger = logging.getLogger('pynbody.plot.stars')

def plot_spectrograph(halos, mode, frequency_range, radial_range, frequency_bins=20,
    radial_bins=20, pattern_speed=True, aligned=False, family='stars', ax=None, fig_kw={},
    ax_kw={}, cmap='viridis', dynamic_range=1):
    """
    
    plot_spectrograph
    =================

    This is a wrapper to plot the results of spectral analysis of non-axisymmetric perturbation
    (bars/spirals) in galaxy disks. It plots the results of the pynbody.analysis.spectrograph
    module.

    **Input**:

    *halos* : an ordered list of subviews of simulation snapshots that contain the halo in
        question. They have to be equally spaced in time. The number of outputs in the list
        is ideally not even.

    *mode* : the azimuthal multiplicity of which the spectrograph is to be calculated

    *frequency_range* : Range of frequencies to be considered in 1/Gyr. This is
        the frequency at which the pattern recurs, i.e. the mode-fold of the pattern speed.

    *frequency_bins* (20) : Number of bins to split the frequency range into

    *radial_range* : Range of radii to be considered in kpc

    *radial_bins* (20) : Number of bins to split the radial range into

    *pattern_speed* (True) : If true, the actual pattern speed is plotted on the y-axis instead of
        frequency at which the pattern itself repeats (the mode-fold of the pattern speed).

    *aligned* (False) : If False, the snapshot will be centered on the central galaxy and
        its disk will be aligned to be in the xy-plane. The snapshots are only modified for this
        analysis. After the routine they will be in their original state. If true, the snapshot
        is assumed to be aligned.

    *family* ('stars') : Consider only the respective family portion of the snapshots.

    *ax* (None) : The axis instance to be used for plotting.

    *fig_kw* ({}) : Keywords for figure instance

    *ax_kw* ({}) : Keywords for axis instance

    *cmap* ('viridis') : Color map to use

    *dynamic_range (1) : Dynamic range to be plotted in log-scale (orders of magnitude)

    **Returns**:

    *fig* : The figure instance used for plotting.

    *ax* : The axis instance used for plotting.

    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw=ax_kw, **fig_kw)
    else:
        fig = ax.get_figure()

    spectrum = calc_spectrograph(halos, mode, frequency_range=frequency_range, frequency_bins=frequency_bins,
        radial_range=radial_range, radial_bins=radial_bins, aligned=aligned, family=family)

    extent = list(radial_range) + list(np.array(frequency_range)/mode)
    aspect = (.0+extent[1]-extent[0])/(extent[3]-extent[2])
    plot_spec = np.abs(spectrum)
    ax.imshow(plot_spec, origin='lower', extent=extent, aspect=aspect,
        norm=matplotlib.colors.LogNorm(), cmap=cmap, vmin=plot_spec.max()*10**(-dynamic_range))
    ax.set_xlabel(r'$R$ [kpc]')
    ax.set_ylabel(r'pattern speed [1/Gyr]')
    ax.text(.95, .95, r'$m={0:d}$'.format(mode), ha='right', va='top', transform=ax.transAxes)

    return fig, ax
