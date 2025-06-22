import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest

from pynbody.plot.generic import hist2d


def test_hist2d():
    import matplotlib
    matplotlib.use('Agg')  # Use a non-interactive backend for testing
    
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
    print(repr(h[::20, ::20]))
    npt.assert_allclose(h[::20, ::20], [[1.65773493e-06, 1.03443377e-04, 1.98391738e-02,
           7.45085089e-03, 8.65592159e-03],
          [5.42986322e-04, 1.25838171e-01, 9.21976070e-01, 1.13240368e+00, 4.38284746e-02],
          [3.27461479e-03, 6.21391401e-01, 5.57536393e+00, 5.47845734e+00, 4.27261670e-01],
          [9.68454529e-03, 4.16202360e-01, 4.42973922e+00, 3.95274321e+00, 5.27707223e-01],
          [1.21313249e-04, 1.04381327e-01, 4.73558908e-01, 5.71217039e-01, 4.16500118e-02]])

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
        h, xedges, yedges = hist2d(x, y, nbins=100, values=fn, logscale=False,
                                   x_range=(-2, 2), y_range=(-2, 2), use_kde=kde)

    # look only in central region
    xedges = xedges[40:60]
    yedges = yedges[40:60]
    h = h[40:60, 40:60]

    npt.assert_allclose(h, np.sin(xedges[:,None]+yedges[None,:]), atol=0.04)


def test_hist2d_image_type():
    x = np.random.randn(10000)
    y = np.random.randn(10000)



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

def test_hist2d_with_nan():
    np.random.seed(13136)
    x = np.random.normal(size=10000)
    y = np.random.normal(size=10000)
    v = np.random.normal(size=10000)

    plt.clf()
    hist2d(x, y, values=v, nbins=100, x_range=(-3, 3), y_range=(-3, 3), logscale=False, colorbar=True)

    # Now inspect the figure, and check the color range is sensible
    clim = plt.gcf().get_axes()[0].collections[0].get_clim()
    npt.assert_allclose(clim, (-3.8336727348072435, 3.3284910138060635))

def test_hist2d_colorbar_control():
    np.random.seed(13136)
    x = np.random.normal(size=10000)
    y = np.random.normal(size=10000)
    v = np.random.normal(size=10000)

    plt.clf()
    hist2d(x, y, values=v, nbins=100, x_range=(-3, 3), y_range=(-3, 3), logscale=False, colorbar=False)

    assert len(plt.gcf().get_axes()) == 1

    plt.clf()
    hist2d(x, y, values=v, nbins=100, x_range=(-3, 3), y_range=(-3, 3), logscale=False, colorbar=True)

    assert len(plt.gcf().get_axes()) == 2

    plt.clf()
    hist2d(x, y, values=v, nbins=100, x_range=(-3, 3), y_range=(-3, 3), logscale=False, colorbar=True,
            colorbar_label='test_label', colorbar_format="%.1f")

    assert plt.gcf().get_axes()[1].get_ylabel() == 'test_label'
    assert plt.gcf().get_axes()[1].get_yticklabels()[0].get_text() == '-3.8'
