import pynbody
import numpy as np
import numpy.testing as npt
import pylab as p
import pickle


import pytest
from pathlib import Path

import pynbody

test_folder = Path(__file__).parent

def setup_module():
    global f
    f = pynbody.load("testdata/g15784.lr.01024")
    h = f.halos()
    # hard-code the centre so we're not implicitly testing the centering routine too:
    cen = [0.024456279579533, -0.034112552174141, -0.122436359962132]
    #cen = pynbody.analysis.halo.center(h[1],retcen=True)
    #print "[%.15f, %.15f, %.15f]"%tuple(cen)
    f['pos']-=cen

    # derive smoothing lengths direct from file data so we are
    # not testing the kdtree (which is tested elsewhere)

    f.gas['smooth']=(f.gas['mass']/f.gas['rho'])**(1,3)
    f.physical_units()

@pytest.fixture
def compare2d():
    yield np.load(test_folder / "test_im_2d.npy")

@pytest.fixture
def compare3d():
    yield np.load(test_folder / "test_im_3d.npy")

@pytest.fixture
def compare_grid():
    yield np.load(test_folder / "test_im_grid.npy")

@pytest.fixture
def stars_2d():
    yield np.load(test_folder / "test_stars_2d.npy")

def test_images(compare2d, compare3d, compare_grid):

    global f

    im3d = pynbody.plot.sph.image(
        f.gas, width=20.0, units="m_p cm^-3", noplot=True, approximate_fast=False)

    im2d = pynbody.plot.sph.image(
        f.gas, width=20.0, units="m_p cm^-2", noplot=True, approximate_fast=False)

    im_grid = pynbody.sph.to_3d_grid(f.gas,nx=200,x2=20.0)[::50]

    """
    np.save("test_im_2d.npy",im2d)
    np.save("test_im_3d.npy",im3d)
    np.save("test_im_grid.npy",im_grid)
    """

    npt.assert_allclose(im2d,compare2d,rtol=1e-4)
    npt.assert_allclose(im3d,compare3d,rtol=1e-4)
    npt.assert_allclose(im_grid,compare_grid,rtol=1e-4)


    # check rectangular image is OK
    im_rect = pynbody.sph.render_image(f.gas,nx=500,ny=250,x2=10.0,
                                        approximate_fast=False).in_units("m_p cm^-3")
    compare_rect = compare3d[125:-125]
    npt.assert_allclose(im_rect,compare_rect,rtol=1e-4)

def test_approximate_images(compare2d, compare3d):
    global f
    im3d = pynbody.plot.sph.image(
        f.gas, width=20.0, units="m_p cm^-3", noplot=True, approximate_fast=True)
    im2d = pynbody.plot.sph.image(
        f.gas, width=20.0, units="m_p cm^-2", noplot=True, approximate_fast=True )

    # approximate interpolated images are only close in a mean sense
    assert abs(np.log10(im2d/compare2d)).mean()<0.03
    assert abs(np.log10(im3d/compare3d)).mean()<0.03


def test_denoise_projected_image_throws():
    global f
    # this should be fine:
    pynbody.plot.sph.image(f.gas, width=20.0, units="m_p cm^-3", noplot=True, approximate_fast=True, denoise=True)

    with pytest.raises(ValueError):
        # this should not:
        pynbody.plot.sph.image(f.gas, width=20.0, units="m_p cm^-2", noplot=True, approximate_fast=True, denoise=True)


def test_render_stars(stars_2d):
    global f
    im = pynbody.plot.stars.render(f, width=10.0, resolution=100, ret_im=True, plot=False)

    # np.save("test_stars_2d.npy", im[40:60])

    npt.assert_allclose(stars_2d,im[40:60],atol=0.01)


@pynbody.derived_array
def intentional_circular_reference(sim):
    return sim['intentional_circular_reference']

def test_exception_propagation():
    with pytest.raises(RuntimeError):
        pynbody.plot.sph.image(f.gas, qty='intentional_circular_reference')
