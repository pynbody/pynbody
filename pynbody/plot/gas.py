"""
Functions for plotting gas quantities

.. versionchanged :: 2.0

    ``temp_profile`` has been removed. Use the :mod:`pynbody.analysis.profile` module instead.
    For examples, see the :ref:`profile` tutorial.

"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from ..analysis import angmom, profile
from ..transformation import NullTransformation
from ..units import Unit
from .generic import hist2d

logger = logging.getLogger('pynbody.plot.gas')


def rho_T(sim, rho_units=None, rho_range=None, t_range=None, two_phase='split', **kwargs):
    """
    Plot a 2d histogram of temperature vs density for the gas in the snapshot

    Parameters
    ----------

    sim: pynbody.snapshot.simsnap.SimSnap
         The snapshot or subsnap to plot

    rho_units: str | pynbody.units.Unit | None
        The units to use for the density. If None, the current snapshot units are used.

    rho_range: tuple | None
        The range of densities to plot, in the same units as rho_units. If None, the full range is used.

    t_range: tuple | None
        The range of temperatures to plot. If None, the full range is used.

    two_phase: str
        If two-phase particles are detected, either plot each phase separately ('split'), or merge them ('merge').
        Default is 'split'.

    **kwargs:
        Additional keyword arguments are passed to :func:`~pynbody.plot.generic.hist2d`

    """
    if rho_units is None:
        rho_units = sim.gas['rho'].units

    if t_range is not None:
        kwargs['y_range'] = t_range
        assert len(kwargs['y_range']) == 2

    if rho_range is not None:
        kwargs['x_range'] = rho_range
        assert len(kwargs['x_range']) == 2
    else:
        rho_range = False

    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
        del kwargs['xlabel']
    else:
        xlabel = r'log$_{10}$($\rho$/$' + Unit(rho_units).latex() + '$)'

    if 'ylabel' in kwargs:
        ylabel = kwargs['ylabel']
        del kwargs['ylabel']
    else:
        ylabel = r'log$_{10}$(T/$' + sim.gas['temp'].units.latex() + '$)'

    if 'Tinc' in sim.loadable_keys() and two_phase == 'merge':
        return hist2d(sim.gas['rho'].in_units(rho_units),sim.gas['Tinc'],
                      xlogrange=True,ylogrange=True,xlabel=xlabel,
                      ylabel=ylabel, **kwargs)

    if 'uHot' in sim.loadable_keys() and 'MassHot' in sim.loadable_keys() and two_phase == 'split':
        E = sim.g['uHot']*sim.g['MassHot']+sim.g['u']*(sim.g['mass']-sim.g['MassHot'])
        rho = np.concatenate((np.array(sim.g['rho'].in_units(rho_units)*E/(sim.g['mass']*sim.g['u'])),
            np.array(sim.g['rho'].in_units(rho_units)*E/(sim.g['mass']*sim.g['uHot']))))
        temp = np.concatenate((np.array(sim.g['temp']), np.array(sim.g['temp']/sim.g['u']*sim.g['uHot'])))
        temp = temp[np.where(np.isfinite(rho))]
        rho = rho[np.where(np.isfinite(rho))]
        return hist2d(rho, temp, xlogrange=True,ylogrange=True,xlabel=xlabel,
                ylabel=ylabel, **kwargs)

    return hist2d(sim.gas['rho'].in_units(rho_units),sim.gas['temp'],
                  xlogrange=True,ylogrange=True,xlabel=xlabel,
                  ylabel=ylabel, **kwargs)
