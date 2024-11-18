"""
Utility functions for the plotting module

"""

import matplotlib.patches
import matplotlib.quiver
import matplotlib.transforms
import numpy as np
import scipy as sp
import scipy.signal
import scipy.sparse
from matplotlib import pyplot as plt

from ..analysis import cosmology


def add_redshift_axis(sim, labelzs=[0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]):
    """Add a top redshift axis to a plot with time on the x-axis

    Parameters
    ----------

    sim : :class:`pynbody.snapshot.SimSnap`
        The simulation snapshot to which the plot refers

    labelzs : list
        List of redshifts at which to place labels

    """

    old_axis = plt.gca()
    x0, x1 = plt.gca().get_xlim()
    pz = plt.twiny()
    times = cosmology.age(sim, labelzs, unit='Gyr')
    pz.set_xticks(times)
    pz.set_xticklabels([str(x) for x in labelzs])
    pz.set_xlim(x0, x1)
    pz.set_xlabel('$z$')
    plt.sca(old_axis)

def fast_kde(x, y, kern_nx=None, kern_ny=None, gridsize=(100, 100),
             extents=None, nocorrelation=False, weights=None, norm = False, **kwargs):
    """Gaussian kernel density estimation (KDE)

    This function is typically several orders of magnitude faster than
    scipy.stats.kde.gaussian_kde for large (>1e7) numbers of points and
    produces an essentially identical result. Unlike the scipy original, however,
    it is limited to using a regular grid.

    Parameters
    ----------

    x : array
        The x-coords of the input data points

    y : array
        The y-coords of the input data points

    kern_nx : float
        size (in units of x) of the kernel. If None, the size is determined
        automatically based on the Scott's factor.

    kern_ny : float
        size (in units of y) of the kernel. If None, the size is determined
        automatically based on the Scott's factor.

    gridsize : tuple
        Size of the output grid (default 100x100)

    extents : tuple
        Extents of the output grid as (xmin, xmax, ymin, ymax). Default: extent of input data

    nocorrelation : bool
        If True, the correlation between the x and y coords will be ignored when
        preforming the KDE.

    weights : array
        An array of the same shape as x & y that weighs each sample (x_i, y_i) by each
        value in weights (w_i). Defaults to an array of ones the same size as x & y.

    norm : bool
        If False, the output is only corrected for the kernel. If True, the result is
        normalized such that the integral over the area yields 1.

    Returns
    -------

    A gridded 2D kernel density estimate of the input points.


    :Authors: Joe Kington
    :License: MIT License <http://www.opensource.org/licenses/mit-license.php>

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
        xmin, xmax, ymin, ymax = list(map(float, extents))

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    #---- Preliminary Calculations -------------------------------------------

    # First convert x & y over to pixel coordinates
    # (Avoiding np.digitize due to excessive memory usage!)
    xyi = np.vstack((x, y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    # Next, make a 2D histogram of x & y
    # Avoiding np.histogram2d due to excessive memory usage with many points
    grid = sp.sparse.coo_matrix((weights, xyi), shape=(nx, ny)).toarray()

    # Calculate the covariance matrix (in pixel coords)
    cov = np.cov(xyi)

    if nocorrelation:
        cov[1, 0] = 0
        cov[0, 1] = 0

    # Scaling factor for bandwidth
    scotts_factor = np.power(n, -1.0 / 6)  # For 2D

    #---- Make the gaussian kernel -------------------------------------------

    # First, determine how big the kernel needs to be
    std_devs = np.diag(np.sqrt(cov))

    if kern_nx is None or kern_ny is None:
        kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    else:
        kern_nx = np.round(kern_nx / dx)
        kern_ny = np.round(kern_ny / dy)

    # make sure the kernel size is odd, so that there is a center pixel
    kern_nx = int(kern_nx) + 1 if int(kern_nx) % 2 == 0 else int(kern_nx)
    kern_ny = int(kern_ny) + 1 if int(kern_ny) % 2 == 0 else int(kern_ny)

    # Determine the bandwidth to use for the gaussian kernel
    inv_cov = np.linalg.inv(cov * scotts_factor ** 2)

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float64) - kern_nx / 2.0 + 0.5
    yy = np.arange(kern_ny, dtype=np.float64) - kern_ny / 2.0 + 0.5
    xx, yy = np.meshgrid(xx, yy)

    # Then evaluate the gaussian function on the kernel grid
    kernel = np.vstack((xx.flatten(), yy.flatten()))
    kernel = np.dot(inv_cov, kernel) * kernel
    kernel = np.sum(kernel, axis=0) / 2.0
    kernel = np.exp(-kernel)
    kernel = kernel.reshape((int(kern_ny), int(kern_nx)))

    #---- Produce the kernel density estimate --------------------------------

    # Convolve the gaussian kernel with the 2D histogram, producing a gaussian
    # kernel density estimate on a regular grid
    grid = sp.signal.convolve2d(grid, kernel, mode='same', boundary='fill').T

    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.
    norm_factor = 2 * np.pi * cov * scotts_factor ** 2
    norm_factor = np.linalg.det(norm_factor)
    #norm_factor = n * dx * dy * np.sqrt(norm_factor)
    norm_factor = np.sqrt(norm_factor)

    if norm:
        norm_factor *= n * dx * dy

    # Normalize the result
    grid /= norm_factor

    return grid





def _val_or_rc(val, rc_key):
    """Return val if it is not None, otherwise return the value of rc_key in matplotlib.rcParams.

    This is available in some versions of matplotlib, but was added in mid-2023, so we should support
    older versions for now"""

    return val if val is not None else matplotlib.rcParams[rc_key]

class PynbodyQuiverKey(matplotlib.quiver.QuiverKey):
    """An improved version of matplotlib's QuiverKey, allowing a background color to be specified."""

    def __init__(self, *args, **kwargs):
        """An improved quiver key implementation.

        In addition to the arguments of matplotlib.quiver.QuiverKey, additional parameters are below.

        Parameters
        ----------
        boxfacecolor : str or None
            The background color of the key. If None, the default legend face color is used.
        boxedgecolor : str or None
            The edge color of the key. If None, the default legend edge color is used.
        fancybox : bool or None
            If True, the box is drawn with a fancy box style. If None, the default legend fancybox style is used.
            If False, a square-cornered box is drawn.
        *args:
            Additional arguments for matplotlib.quiver.QuiverKey
        **kwargs:
            Additional keyword arguments for matplotlib.quiver.QuiverKey

        """
        self.boxfacecolor = _val_or_rc(kwargs.pop('boxfacecolor', None),
                                                     'legend.facecolor')
        self.boxedgecolor = _val_or_rc(kwargs.pop('boxedgecolor', None),
                                                  'legend.edgecolor')

        if self.boxfacecolor == 'inherit':
            self.boxfacecolor = matplotlib.rcParams['axes.facecolor']

        if self.boxedgecolor == 'inherit':
            self.boxedgecolor = matplotlib.rcParams['axes.edgecolor']

        self.fancybox = _val_or_rc(kwargs.pop("fancybox", None),
                                              'legend.fancybox')

        super().__init__(*args, **kwargs)


    def draw(self, renderer):
        super()._init()

        # the following duplication of bits of super(renderer).draw is necessary to get the
        # text bbox in the right place. Alternative is to actually call super(renderer).draw,
        # but then we end up having to draw twice so that the contents is above the background.
        pos = self.get_transform().transform((self.X, self.Y))
        self.text.set_position(pos + self._text_shift())

        if self.boxfacecolor is not None:
            figure_inverse_trans = self.figure.transFigure.inverted()

            # first find the bbox of the text, in figure coords
            bbox = self.text.get_window_extent(renderer)
            bbox = bbox.transformed(figure_inverse_trans)


            # now find a bbox for the arrow, which is a subtle/annoying thing because the offsets and vertices
            # are transformed differently
            arrow_offsets = self.get_transform().transform(self.vector.get_offsets())
            arrow_vertices =  self.Q.get_transform().transform(self.verts[0])

            arrow_vertices_offset = arrow_offsets + arrow_vertices

            x0y0 = arrow_vertices_offset.min(axis=0)
            x1y1 = arrow_vertices_offset.max(axis=0)

            x0y0 = figure_inverse_trans.transform(x0y0)
            x1y1 = figure_inverse_trans.transform(x1y1)

            # expand bbox to include all arrow_vertices:
            bbox = matplotlib.transforms.Bbox.union([bbox, matplotlib.transforms.Bbox([x0y0, x1y1])])

            # and, at last, we know the coordinates of the background that we need!

            boxstyle = ("round,pad=0.02,rounding_size=0.02" if self.fancybox
                        else "square,pad=0.02")

            background = matplotlib.patches.FancyBboxPatch(
                bbox.min, bbox.width, bbox.height, boxstyle=boxstyle,
                fc=self.boxfacecolor, ec=self.boxedgecolor,
                transform=self.figure.transFigure)

            background.draw(renderer)
        super().draw(renderer)


def _test_quiverkey(scale=10.0, labelpos='E'):
    """A simple test for the PynbodyQuiverKey class."""
    import matplotlib.pyplot as p
    import numpy as np
    X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
    U = np.cos(X)
    V = np.sin(Y)
    fig, ax = p.subplots()
    q = ax.quiver(X, Y, U, V)

    qk = PynbodyQuiverKey(q, .5, .95, scale, "Quiver key", labelpos=labelpos,
                          boxfacecolor='w', boxedgecolor='k', fancybox=True)
    ax.add_artist(qk)
    qk.set_zorder(5)
