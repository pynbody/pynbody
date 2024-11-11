import warnings

import numpy as np
import numpy.testing as npt
import pytest

from pynbody.plot.generic import hist2d


def test_hist2d():
    np.random.seed(1337)
    x = np.random.randn(10000)
    y = np.random.randn(10000)
    h, xedges, yedges = hist2d(x, y, nbins=100, logscale=False)

    npt.assert_allclose(xedges, np.linspace(-3.80291922, 3.84811457, 100), atol=1e-8)
    npt.assert_allclose(yedges, np.linspace(-3.2530673, 3.47677378, 100), atol=1e-8)

    npt.assert_allclose(h[::20,::20], [[0., 0., 0., 0., 0.],
                                   [0., 0., 0., 2., 0.],
                                   [0., 0., 4., 5., 0.],
                                   [0., 1., 7., 2., 0.],
                                   [0., 0., 0., 0., 0.]])

    h, xedges, yedges = hist2d(x, y, nbins=100, logscale=True, use_kde=True)
    npt.assert_allclose(h[::20, ::20], [[1.34990711e-06, 5.91335801e-05, 1.44034986e-02,
           5.72840517e-03, 6.15743557e-03],
          [7.17596299e-04, 1.18115605e-01, 8.85622891e-01,
           1.09601174e+00, 4.05982037e-02],
          [2.29025175e-03, 6.17099734e-01, 5.66523017e+00,
           5.55620067e+00, 4.02563750e-01],
          [6.74389125e-03, 3.96905475e-01, 4.46099383e+00,
           4.04680886e+00, 4.95853275e-01],
          [1.34990711e-06, 9.67328656e-02, 4.62357739e-01,
           5.65621250e-01, 3.53710621e-02]])

def test_hist2d_massweight():
    np.random.seed(1337)
    x = np.random.randn(100000)
    y = np.random.randn(100000)
    mass = np.exp(x**2/2. + y**2/2.) # mass weight to precisely counteract number density

    h, xedges, yedges = hist2d(x, y, nbins=10, weights=mass, logscale=False, x_range=(-1,1), y_range=(-1,1))

    npt.assert_allclose(h, np.ones_like(h)*h.mean(), rtol=0.15)

@pytest.mark.parametrize('kde', [True, False], ids=['kde', 'no_kde'])
def test_hist2d_averaging(kde):
    np.random.seed(1337)
    x = np.random.randn(100000)
    y = np.random.randn(100000)
    fn = np.sin(x+y)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h, xedges, yedges = hist2d(x, y, nbins=20, values=fn, logscale=False,
                                   x_range=(-2, 2), y_range=(-2, 2), use_kde=kde)

    # look only in central region
    xedges = xedges[5:-5]
    yedges = yedges[5:-5]
    h = h[5:-5, 5:-5]

    npt.assert_allclose(h, np.sin(xedges[:,None]+yedges[None,:]), atol=0.02)


def test_hist2d_image_type():
    x = np.random.randn(10000)
    y = np.random.randn(10000)

    import matplotlib.pyplot as plt

    plt.clf()
    hist2d(x, y, nbins=100, plot_type='image')

    assert len(plt.gca().get_images()) > 0

    plt.clf()
    hist2d(x, y, nbins=100, plot_type='contour')

    assert len(plt.gca().collections) > 0
    assert len(plt.gca().get_images()) == 0

    plt.clf()
    hist2d(x, y, nbins=100, plot_type='contourf')

    assert len(plt.gca().collections) > 0
