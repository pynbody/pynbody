"""
Flexible plotting functions

.. versionchanged :: 2.0

    The ``qprof`` function has been removed. Use the :mod:`pynbody.analysis.profile` module instead. For
    examples, see the :ref:`profile` tutorial (section on :ref:`prof_deriv_disp`).

    Significant changes to the :func:`~pynbody.plot.generic.hist2d` function have been made. See the
    function documentation for more information.

    The :func:`~pynbody.plot.generic.gauss_kde` function is deprecated. Use :func:`~pynbody.plot.generic.hist2d`
    with the *use_kde* keyword set to ``True`` instead.
"""

import warnings

import numpy as np
import numpy.ma as ma

import pynbody

from ..array import SimArray
from ..units import NoUnit
from ..util import deprecated


def hist2d(x, y, weights=None, values=None, gridsize=(100, 100), nbins = None,
           x_logscale = False, y_logscale = False, x_range = None, y_range = None,
           use_kde = False, kde_kwargs=None, fill_value=None, **kwargs):
    """
    Plot 2D histogram for arbitrary arrays *x* and *y*.

    If *use_kde* is True, instead of a binned histogram, a Gaussian kernel density estimate is used.

    It is also possible to obtain an average of specified *values* which are sampled at the *x* and *y*.
    These can be weighted by *weights* (e.g. mass) if desired. If *weights* are provided without *values*,
    the result is a simple weighted histogram.


    .. versionchanged :: 2.1

        * Added *fill_value* keyword. If provided, empty bins are filled with this value when making a histogram of
          mean values. Default is None, in which case empty bins are filled with NaN which will typically be rendered
          as transparent. Versions prior to 2.1 implicitly had fill_value set to 0.

        * Added *colorbar_label* and *colorbar_format* keywords.


    .. versionchanged :: 2.0

        The *scalemin* and *scalemax* keywords have been deprecated in favor of *vmin* and *vmax*
        for consistency with matplotlib and other pynbody plotting functions.

        The *make_plot* keyword has been deprecated. Use *plot_type* instead.

        The *ret_im* keyword has been deprecated. If you want to use ``imshow`` instead of ``contourf``,
        set *plot_type* to 'image'.

        The *mass* keyword was confusing and has been deprecated. When *mass* was provided previously,
        it was used as the weight, while the weight array was used as the quantity to obtain a
        mass-weighted average of. So, where previously you would pass
        *weights* and *mass* you now pass *values* and *weights* respectively.

        The *draw_contours* keyword has been removed. If you want to overplot mass contours, call
        the function again using the mass array, with *plot_type* set to 'contour'.



    Parameters
    ----------

    x : array-like
        x-coordinates of points

    y : array-like
        y-coordinates of points

    weights : array-like, optional
        weights of points; if not provided, all points are given equal weight. If combined with *values*,
        each pixel is a weighted average of the values in that pixel. If only *weights* is provided, you
        get back a

    values : array-like, optional
        values to assign to each point; if provided, a weighted mean of the value in each pixel is computed

    gridsize : tuple, optional
        number of bins in each dimension. Default is (100,100).

    nbins : int, optional
        An alternative way to specify number of bins for the histogram - if specified,
        gridsize is set to (nbins,nbins).

    plot_type : str, optional
        If 'contour' or 'contourf' use matplotlib to make a contour/filled contour.
        If 'image', use matplotlib imshow (default).
        If 'none' (or, for backward compatibility, False), return the histogram data.

    vmin: float, optional
        Minimum value for the color scale.

    vmax: float, optional
        Maximum value for the color scale.

    x_range: array-like, optional
        Length-2 array specifies the x range. Default is None, in which case the range is set to the min and max of x.

    y_range: array-like, optional
        Length-2 array specifies the y range. Default is None, in which case the range is set to the min and max of y.

    x_logscale: bool, optional
        If True, the histogram will be made in log x space. Default False.

    y_logscale: bool, optional
        If True, the histogram will be made in log y space. Default False.

    nlevels : int | array-like, optional
        Number of levels to use for contours (if plot_type is 'contour'). Default is 20.

    logscale : boolean, optional
        If True, use a log-scaled colorbar and log-spaced contours. Default is True.

    colorbar : boolean, optional
        If True, draw a colorbar. Default is False.

    colorbar_label : str, optional
        If *colorbar* is True, this string will be used as the label for the colorbar.

    colorbar_format : str, optional
        If *colorbar* is True, this string will be used as the format string for the colorbar.

    vmin : float, optional
        Minimum value to use for the color scale.

    vmax : float, optional
        Maximum value to use for the color scale.

    cmap : str, optional
        Colormap to use. Default is None, which uses the default colormap.

    make_plot: str, optional
        Deprecated alias for *plot_type*.

    use_kde: bool, optional
        If True, use a gaussian kernel density estimate instead of a histogram.

    kde_kwargs: dict, optional
        Dictionary of keyword arguments to pass to :func:`~pynbody.plot.util.fast_kde` if
        *use_kde* is True.

    fill_value: float, optional
        If provided, fill empty bins with this value when making a histogram of mean values
        (i.e. only when *values* is provided). Default is None, in which case empty bins are filled with NaN.

    ret_im: bool, optional
        Deprecated. If True, plot_type is set to 'image'.

    scalemin: float, optional
        Deprecated alias for *vmin*.

    scalemax: float, optional
        Deprecated alias for *vmax*.

    xlogrange: bool, optional
        Deprecated alias for *x_logscale*.

    ylogrange: bool, optional
        Deprecated alias for *y_logscale*.


    """
    global config

    # process keywords
    if 'make_plot' in kwargs:
        warnings.warn("The 'make_plot' keyword is deprecated. Use 'plot_type' instead.", DeprecationWarning)
        kwargs['plot_type'] = kwargs.pop('make_plot')

    if 'ret_im' in kwargs:
        warnings.warn("The 'ret_im' keyword is deprecated. Use 'plot_type' instead.", DeprecationWarning)
        if kwargs.pop('ret_im'):
            kwargs['plot_type'] = 'image'

    if 'mass' in kwargs:
        warnings.warn("The 'mass' keyword is deprecated. Where previously you would pass 'weights' and 'mass', you now pass 'values' and 'weights' respectively.", DeprecationWarning)
        values = weights
        weights = kwargs.pop('mass')

    if 'xlogrange' in kwargs:
        warnings.warn("The 'xlogrange' keyword is deprecated. Use 'x_logscale' instead.", DeprecationWarning)
        x_logscale = kwargs.pop('xlogrange')

    if 'ylogrange' in kwargs:
        warnings.warn("The 'ylogrange' keyword is deprecated. Use 'y_logscale' instead.", DeprecationWarning)
        y_logscale = kwargs.pop('ylogrange')

    if y_range is not None:
        if len(y_range) != 2:
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        if y_logscale:
            y_range = [np.log10(np.min(y)), np.log10(np.max(y))]
        else:
            y_range = [np.min(y), np.max(y)]

    if x_range is not None:
        if len(x_range) != 2:
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        if x_logscale:
            x_range = [np.log10(np.min(x)), np.log10(np.max(x))]
        else:
            x_range = [np.min(x), np.max(x)]

    if x_logscale:
        x = np.log10(x)
    else:
        x = x

    if y_logscale:
        y = np.log10(y)
    else:
        y = y

    if nbins is not None:
        gridsize = (nbins, nbins)

    ind = np.where((x > x_range[0]) & (x < x_range[1]) &
                   (y > y_range[0]) & (y < y_range[1]))

    x = x[ind[0]]
    y = y[ind[0]]

    if weights is not None:
        weights = weights[ind[0]]

    if values is not None:
        values = values[ind[0]]
        if weights is None:
            weights = np.ones_like(values)

    if use_kde:
        if kde_kwargs is None:
            kde_kwargs = {}
        def _histogram_generator(weights):
            from .util import fast_kde
            extents = [x_range[0], x_range[1], y_range[0], y_range[1]]
            hist = fast_kde(x, y, weights=weights, gridsize=gridsize, extents=extents, **kde_kwargs)
            xs = np.linspace(x_range[0], x_range[1], gridsize[0] + 1)
            ys = np.linspace(y_range[0], y_range[1], gridsize[1] + 1)
            return hist, ys, xs
    else:
        def _histogram_generator(weights):
            return np.histogram2d(y, x, weights=weights, bins=gridsize, range=[y_range, x_range])

    if values is not None:
        hist, ys, xs = _histogram_generator(weights * values)
        hist_norm, _, _ = _histogram_generator(weights)
        valid = hist_norm > 0
        hist[valid] /= hist_norm[valid]

        hist = hist.view(SimArray)

        if fill_value:
            hist[~valid] = fill_value
        else:
            hist[~valid] = np.nan

        try:
            hist.units = values.units
        except AttributeError:
            hist.units = NoUnit()

    else:
        hist, ys, xs = _histogram_generator(weights)

    try:
        xs = SimArray(.5 * (xs[:-1] + xs[1:]), x.units)
        ys = SimArray(.5 * (ys[:-1] + ys[1:]), y.units)
    except AttributeError:
        xs = .5 * (xs[:-1] + xs[1:])
        ys = .5 * (ys[:-1] + ys[1:])

    plot_type = kwargs.get('plot_type', 'image')
    if plot_type != 'none' and plot_type is not False:
        make_contour_plot(hist, xs, ys,
                          xlabel_display_log = x_logscale, ylabel_display_log = y_logscale,
                          x_range = x_range, y_range = y_range,
                          **kwargs)

    return hist, xs, ys


@deprecated("This function is deprecated. Use pynbody.plot.generic.hist2d instead, with use_kde=True.")
def gauss_kde(*args, **kwargs):
    """
    Deprecated: plot 2D gaussian kernel density estimate (KDE) given values at points (*x*, *y*).

    .. versionchanged :: 2.0

        This function is deprecated. Use :func:`~pynbody.plot.generic.hist2d` with the *use_kde* keyword set to
        ``True`` instead.

    Arguments and keyword arguments are passed to :func:`~pynbody.plot.generic.hist2d`, with
    *use_kde* set to ``True``. Two keywords are given special treatment and passed
    to :func:`~pynbody.plot.util.fast_kde`:

    * *norm*: boolean (default: False)
      If False, the output is only corrected for the kernel. If True,
      the result is normalized such that the integral over the area
      yields 1.

    * *nocorrelation*: (default: False) If True, the correlation
      between the x and y coords will be ignored when preforming
      the KDE.

    """
    kde_kwargs = {}
    if 'norm' in kwargs:
        kde_kwargs['norm'] = kwargs.pop('norm')
    if 'nocorrelation' in kwargs:
        kde_kwargs['nocorrelation'] = kwargs.pop('nocorrelation')

    return hist2d(*args, use_kde=True, kde_kwargs=kde_kwargs,
                  **kwargs)


def make_contour_plot(arr, xs, ys, x_range=None, y_range=None, nlevels=20,
                      logscale=True, xlabel_display_log=False, ylabel_display_log=False,
                      colorbar=False, cmap=None, vmin=None, vmax=None, plot_type='contourf',
                      colorbar_label=None, colorbar_format=None, **kwargs):
    """
    Plot a contour plot of grid *arr* corresponding to bin centers specified by *xs* and *ys*.

    Labels the axes and colobar with units taken from x, if available.

    Called by :func:`~pynbody.plot.generic.hist2d`.

    .. versionchanged :: 2.1

        * Added *colorbar_label* keyword

        * Added *colorbar_format* keyword. If provided, this string will be used as the format string for the colorbar.
          Formerly, the format string was set to '%.2e'.

    .. versionchanged :: 2.0

        To simplify the plot interfaces and make them more coherent, the following changes have been made:

        * It is no longer possible to pass in a *filename*; instead use the matplotlib ``savefig`` function.

        * The *legend* keyword has been removed; instead use matplotlib ``legend``

        * The *subplot* keyword has been removed; instead ensure that the current matplotlib axes are the
          ones you want to plot in.

        * The *clear* keyword has been removed; instead use the matplotlib ``clf`` function before calling
          this function.

        * The *scalemin* and *scalemax* keywords have been deprecated in favor of *vmin* and *vmax*, for
          consistency with matplotlib and with other pynbody plotting functions.

        * The *ret_im* keyword has been deprecated. If you want to use ``imshow`` instead of ``contour``,
          set *plot_type* to 'image'.


    Parameters
    ----------

    arr : array-like
        2D array to plot

    xs : array-like
        x-coordinates of bin centres

    ys : array-like
        y-coordinates of bin centres

    x_range : array-like
        Length-2 array specifies the x range. Default is None, in which case the range is set to the min and max of x.

    y_range : array-like
        Length-2 array specifies the y range. Default is None, in which case the range is set to the min and max of y.

    xlabel_display_log : boolean, optional
        If True, the x-axis label will indicate that the x values are log-scaled. Other than the axis labelling,
        this keyword has no effect on the plot.

    ylabel_display_log : boolean, optional
        If True, the y-axis label will indicate that the y values are log-scaled. Other than the axis labelling,
        this keyword has no effect on the plot.

    nlevels : int, optional
        Number of levels to use for contours. Default is 20.

    logscale : boolean, optional
        If True, use a log-scaled colorbar and log-spaced contours. Default is True.

    colorbar : boolean, optional
        If True, draw a colorbar. Default is False.

    colorbar_label : str, optional
        If *colorbar* is True, this string will be used as the label for the colorbar.

    colorbar_format : str, optional
        If *colorbar* is True, this string will be used as the format string for the colorbar.

    vmin : float, optional
        Minimum value to use for the color scale. Default is arr.min().

    vmax : float, optional
        Maximum value to use for the color scale. Default is arr.max().

    cmap : str, optional
        Colormap to use. Default is None, which uses the default colormap.

    scalemin : float, optional
        Deprecated. Use *vmin* instead.

    scalemax : float, optional
        Deprecated. Use *vmax* instead.

    ret_im : boolean, optional
        Deprecated. If True, plot_type is set to 'image'.

    """

    import matplotlib.pyplot as plt
    from matplotlib import colors

    if kwargs.get('ret_im', None) is not None:
        warnings.warn("The 'ret_im' keyword is deprecated. Use 'plot_type' instead.", DeprecationWarning)
        plot_type = 'image'


    if 'norm' in kwargs:
        del(kwargs['norm'])

    if colorbar_label is None:
        if hasattr(arr,'units') and arr.units != NoUnit() and arr.units != 1:
            colorbar_label = '$' + arr.units.latex() + '$'
        else:
            colorbar_label = '$N$'

    if logscale:
        if vmin is None:
            vmin = np.min(arr[arr > 0])
        if vmax is None:
            vmax = np.max(arr[arr > 0])

        levels = np.logspace(np.log10(vmin), np.log10(vmax), nlevels)
        cont_color = colors.LogNorm(vmin = vmin, vmax = vmax)
    else:
        if vmin is None:
            vmin = np.min(arr[~np.isnan(arr)])
        if vmax is None:
            vmax = np.max(arr[~np.isnan(arr)])
        levels = np.linspace(vmin, vmax, nlevels)
        cont_color = colors.Normalize(vmin = vmin, vmax = vmax)

    arr[arr < vmin] = vmin
    arr[arr > vmax] = vmax

    if plot_type == 'image':
        plot_artist = plt.imshow(arr, origin='lower',
                          aspect='auto', cmap=cmap, norm=cont_color,
                          extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
    elif plot_type == 'contourf':
        plot_artist = plt.contourf(
            xs, ys, arr, levels, norm=cont_color, cmap=cmap, **kwargs)
    elif plot_type == 'contour':
        plot_artist = plt.contour(
            xs, ys, arr, levels, norm=cont_color, cmap=cmap, **kwargs)
    else:
        raise ValueError("Unknown plot_type: %s" % plot_type)

    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
    else:
        try:
            if xlabel_display_log:
                xlabel = r'' + '$log_{10}(' + xs.units.latex() + ')$'
            else:
                xlabel = r'' + '$x/' + xs.units.latex() + '$'
        except AttributeError:
            xlabel = None

    if xlabel:
        try:
            plt.xlabel(xlabel)
        except Exception:
            pass

    if 'ylabel' in kwargs:
        ylabel = kwargs['ylabel']
    else:
        try:
            if ylabel_display_log:
                ylabel = '$log_{10}(' + ys.units.latex() + ')$'
            else:
                ylabel = r'' + '$y/' + ys.units.latex() + '$'
        except AttributeError:
            ylabel = None

    if ylabel:
        try:
            plt.ylabel(ylabel)
        except Exception:
            pass

    if colorbar:
        plt.colorbar(plot_artist, format=colorbar_format).set_label(colorbar_label)

    return plot_artist

def _inv_fourier(p, nmin=1000, mmin=1, mmax=7, nphi=100):
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
    phi = np.linspace(-np.pi, np.pi, nphi)
    rbins = p['rbins']

    for i in range(len(rbins)):
        if p['n'][i] > nmin:
            for m in range(mmin, mmax):
                phi_hist[i, :] = phi_hist[i,:] + p['fourier']['c'][m, i]*np.exp(1j*m*phi)

    return phi, phi_hist


def fourier_map(sim, nbins=100, nmin=1000, nphi=100, mmin=1, mmax=7, rmax=10,
                levels=[.01, .05, .1, .2], return_array=False, **kwargs):
    """Plot an overdensity map generated from a Fourier expansion of the particle distribution.

    A :func:`~pynbody.analysis.profile.Profile` is made and passed to :func:`~pynbody.plot.util.inv_fourier` to
    obtain an overdensity map. The map is plotted using ``matplotlib.contour``.

    .. versionchanged :: 2.0

        The *subplot* keyword has been removed for consistency with other plotting functions. If you want to plot
        on a specific subplot, select that subplot first using the matplotlib interface.

        The *ret* keyword has been renamed to *return_values* for consistency with other plotting functions.

    Parameters
    ----------

    sim : :class:`~pynbody.snapshot.SimSnap`
        The simulation snapshot to analyze.

    nbins : int, optional
        Number of radial bins to use for the profile. Default is 100.

    nmin : int, optional
        Minimum number of particles required per bin. Default is 1000.

    nphi : int, optional
        Number of azimuthal bins to use for the map. Default is 100.

    mmin : int, optional
        Lowest multiplicity Fourier component. Default is 1.

    mmax : int, optional
        Highest multiplicity Fourier component. Default is 7.

    rmax : float, optional
        Maximum radius to use when generating the profile. Default is 10.

    levels : list, optional
        List of levels for plotting contours. Default is [0.01, 0.05, 0.1, 0.2].

    return_array : bool, optional
        If True, return the arrays used to make the plot.

    Returns
    -------

    If *return_array* is True, return the x, y, and value arrays used to make the plot.
    Otherwise, returns None.


    """
    import matplotlib.pylab as plt

    from . import util

    if 'ret' in kwargs:
        warnings.warn("The 'ret' keyword is deprecated. Use 'return_values' instead.", DeprecationWarning)
        return_array = kwargs.pop('ret')

    p = pynbody.analysis.profile.Profile(sim, max=rmax, nbins=nbins)
    phi, phi_inv = _inv_fourier(p, nmin, mmin, mmax, nphi)

    rr, pp = np.meshgrid(p['rbins'], phi)

    xx = (rr * np.cos(pp)).T
    yy = (rr * np.sin(pp)).T

    plt.contour(xx, yy, phi_inv, levels, **kwargs)

    if return_array:
        return xx, yy, phi_inv


def prob_plot(x, y, weight, nbins=(100, 100), extent=None, return_array=False, **kwargs):
    """Make a plot of the probability of y given x, i.e. p(y|x).

    The values are normalized such that the integral along each column is one.

    .. versionchanged :: 2.0

      The axes keyword has been removed for consistency with other functions. If you want to plot on
      specific axes, select those axes first using the matplotlib interface.

      This routine no longer returns the arrays used for plotting unless specifically requested
      using the *return_array* keyword.

    Parameters
    ----------

    x : array-like
        primary binning axis

    y : array-like
        secondary binning axis

    weight : array-like
        weights array

    nbins : tuple of length 2
        number of bins in each direction

    extent : tuple of length 4
        physical extent of the axes (xmin,xmax,ymin,ymax)

    return_array : bool
        If True, return the array used to make the plot.

    **kwargs :
        all optional keywords are passed on to the imshow() command

    Returns
    -------

    If *return_array* is True, return the arrays used to make the plot in the order
    *grid*, *xbinedges*, *ybinedges*. Otherwise, returns None.



    """

    import matplotlib.pylab as plt

    assert(len(nbins) == 2)
    grid = np.zeros(nbins)

    if extent is None:
        extent = (min(x), max(x), min(y), max(y))

    xbinedges = np.linspace(extent[0], extent[1], nbins[0] + 1)
    ybinedges = np.linspace(extent[2], extent[3], nbins[1] + 1)

    for i in range(nbins[0]):

        ind = np.where((x > xbinedges[i]) & (x < xbinedges[i + 1]))[0]
        h, bins = np.histogram(
            y[ind], weights=weight[ind], bins=ybinedges, density=True)
        grid[:, i] = h


    im = plt.imshow(grid, extent=extent, origin='lower', **kwargs)

    cb = plt.colorbar(im, format='%.2f')
    cb.set_label(r'$P(y|x)$')

    if return_array:
        return grid, xbinedges, ybinedges
