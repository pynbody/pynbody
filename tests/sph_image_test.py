from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils
from pynbody.sph import kernels, renderers

test_folder = Path(__file__).parent


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gasoline_ahf")


@pytest.fixture
def snap():
    snap = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    h = snap.halos()
    # hard-code the centre so we're not implicitly testing the centering routine too:
    cen = [0.024456279579533, -0.034112552174141, -0.122436359962132]
    #cen = pynbody.analysis.halo.center(h[1],retcen=True)
    #print "[%.15f, %.15f, %.15f]"%tuple(cen)
    snap['pos']-=cen

    # derive smoothing lengths direct from file data so we are
    # not testing the kdtree (which is tested elsewhere)

    snap.gas['smooth']= (snap.gas['mass'] / snap.gas['rho']) ** (1, 3)
    np.save("result_im_x_pre_phys.npy", snap.gas['x'])
    snap.physical_units()
    np.save("result_im_x_post_phys.npy", snap.gas['x'])

    return snap

@pytest.fixture
def compare2d():
    yield np.load(test_folder / "test_im_2d.npy")

@pytest.fixture
def compare3d():
    yield np.load(test_folder / "test_im_3d.npy")

@pytest.fixture
def compare2d_wendlandC2():
    yield np.load(test_folder / "test_im_2d_wendlandC2.npy")

@pytest.fixture
def compare3d_wendlandC2():
    yield np.load(test_folder / "test_im_3d_wendlandC2.npy")

@pytest.fixture
def compare_grid():
    yield np.load(test_folder / "test_im_grid.npy")

@pytest.fixture
def stars_2d():
    yield np.load(test_folder / "test_stars_2d.npy")

@pytest.fixture
def stars_dust_2d():
    yield np.load(test_folder / "test_stars_dust_2d.npy")

def test_images(compare2d, compare3d, compare_grid, compare2d_wendlandC2, compare3d_wendlandC2, snap):
    im3d = pynbody.plot.sph.image(
        snap.gas, width=20.0, units="m_p cm^-3", noplot=True, approximate_fast=False, resolution=500)

    im2d = pynbody.plot.sph.image(
        snap.gas, width=20.0, units="m_p cm^-2", noplot=True, approximate_fast=False, resolution=500)

    im_grid = pynbody.sph.render_3d_grid(snap.gas, nx=200, x2=20.0, approximate_fast=False)[::50]


    np.save("result_im_2d.npy",im2d)
    np.save("result_im_3d.npy",im3d)
    np.save("result_im_grid.npy",im_grid)


    npt.assert_allclose(im2d,compare2d,rtol=1e-5)
    npt.assert_allclose(im3d,compare3d,rtol=1e-5)
    npt.assert_allclose(im_grid,compare_grid,rtol=1e-5)

    # Make images with a different kernel (Wendland C2)
    im3d_wendlandC2 = pynbody.plot.sph.image(
        snap.gas, width=20.0, units="m_p cm^-3", noplot=True, approximate_fast=False, kernel='wendlandC2',
        resolution=500)
    im2d_wendlandC2 = pynbody.plot.sph.image(
        snap.gas, width=20.0, units="m_p cm^-2", noplot=True, approximate_fast=False, kernel='wendlandC2',
        resolution=500)


    np.save("result_im_2d_wendlandC2.npy",im2d_wendlandC2)
    np.save("result_im_3d_wendlandC2.npy",im3d_wendlandC2)

    # Check that using a different kernel produces a different image
    npt.assert_raises(AssertionError,npt.assert_array_equal,im3d_wendlandC2,im3d)
    npt.assert_raises(AssertionError,npt.assert_array_equal,im2d_wendlandC2,im2d)

    # Check that using a different kernel produces the correct image
    npt.assert_allclose(im2d_wendlandC2,compare2d_wendlandC2,rtol=1e-5)
    npt.assert_allclose(im3d_wendlandC2,compare3d_wendlandC2,rtol=1e-5)

    # check rectangular image is OK
    im_rect = pynbody.sph.render_image(snap.gas,nx=500,ny=250,width=20.0,
                                        approximate_fast=False).in_units("m_p cm^-3")
    np.save("result_im_3d_rectangular.npy",im_rect)

    compare_rect = compare3d[125:-125]
    npt.assert_allclose(im_rect,compare_rect,rtol=1e-4)

def test_approximate_images(compare2d, compare3d, compare_grid, snap):
    im3d = pynbody.plot.sph.image(
        snap.gas, width=20.0, units="m_p cm^-3", noplot=True, approximate_fast=True, resolution=500)
    im2d = pynbody.plot.sph.image(
        snap.gas, width=20.0, units="m_p cm^-2", noplot=True, approximate_fast=True, resolution=500)
    im_grid = pynbody.sph.render_3d_grid(snap.gas, nx=200, x2=20.0, approximate_fast=True)[::50]

    np.save("result_approx_im_2d.npy", im2d)
    np.save("result_approx_im_3d.npy", im3d)
    np.save("result_approx_im_grid.npy", im_grid)

    # approximate interpolated images are only close in a mean sense
    assert abs(np.log10(im2d/compare2d)).mean()<0.02
    assert abs(np.log10(im3d/compare3d)).mean()<0.03
    assert abs(np.log10(im_grid / compare_grid)).mean() < 0.03


def test_denoise_projected_image_throws(snap):
    # this should be fine:
    pipeline = renderers.make_render_pipeline(snap.gas, width=20.0, out_units="m_p cm^-3", denoise=True, resolution=10)
    pipeline.render()

    with pytest.raises(renderers.RenderPipelineLogicError):
        # this should not:
        pipeline = renderers.make_render_pipeline(snap.gas, width=20.0, out_units="m_p cm^-2", denoise=True,
                                                  resolution=10)



@pytest.mark.filterwarnings("ignore:No log file found:UserWarning")
def test_render_stars(stars_2d, stars_dust_2d, snap):

    im = pynbody.plot.stars.render(snap, width=10.0, resolution=100, return_image=True, noplot=True)

    np.save("result_stars_2d.npy", im[40:60])

    npt.assert_allclose(stars_2d,im[40:60],atol=0.01)


@pytest.mark.filterwarnings("ignore:No log file found:UserWarning")
def test_render_stars_with_dust(stars_dust_2d, snap):
    im = pynbody.plot.stars.render(snap, width=10.0, resolution=100, return_image=True, noplot=True, with_dust=True)
    np.save("result_stars_dust_2d.npy", im[40:60])

    npt.assert_allclose(stars_dust_2d, im[40:60], atol=0.01)


@pynbody.derived_array
def intentional_circular_reference(sim):
    return sim['intentional_circular_reference']

# Note: we ignore all warnings here, since pytest will otherwise
# trigger a warning internally because of the exception propagation
@pytest.mark.filterwarnings("ignore:.*")
def test_exception_propagation(snap):
    with pytest.raises(RuntimeError):
        pynbody.plot.sph.image(snap.gas, qty='intentional_circular_reference')

@pytest.fixture
def simple_test_file():
    n_part = 10000
    np.random.seed(1337)
    f = pynbody.new(n_part)
    f['pos'] = np.random.normal(size=(n_part, 3))
    f['pos'].units = 'kpc'
    f['mass'] = np.ones(n_part) / n_part
    f['mass'].units = 'Msol'
    f['temp'] = f['x']
    f['temp'].units = 'K'
    return f

def test_projection_average(simple_test_file):
    f = simple_test_file
    im = pynbody.sph.render_image(f, quantity='temp', weight='rho', width=1)
    im_collapsed = np.mean(im, axis=0)
    answer = np.linspace(-0.5,0.5, len(im_collapsed))
    npt.assert_allclose(im_collapsed, answer, atol=0.01)

    # check it also works to provide custom units
    im = pynbody.sph.render_image(f, quantity='temp', weight='rho', width=1, out_units='0.1 K')
    im_collapsed = np.mean(im, axis=0)
    answer = np.linspace(-5.0, 5.0, len(im_collapsed))
    npt.assert_allclose(im_collapsed, answer, atol=0.1)

    # check it also works with volume weighting
    im = pynbody.sph.render_image(f, quantity='temp', weight=True, width=1, out_units='0.1 K')
    im_collapsed = np.mean(im, axis=0)
    npt.assert_allclose(im_collapsed, answer, atol=0.3)


def test_spherical_render(simple_test_file):
    f = simple_test_file


    im = pynbody.sph.render_spherical_image(f, 'rho', nside=16)

    assert abs(4*np.pi*im.sum() / len(im) - 1.0) < 0.01
    assert im.units == "Msol sr^-1"

    im2 = pynbody.sph.render_spherical_image(f, 'rho', nside=32, out_units="Msol arcsec^-2")


    assert im2.units == "Msol arcsec^-2"
    assert abs((im2.sum() / len(im2))*((60*60*360)**2/np.pi) - 1.0) < 0.01

    im3 = pynbody.sph.render_spherical_image(f, 'temp', weight='rho', nside=16)
    assert im3.units == "K"

    npt.assert_allclose(im3[::100], [0.04377608, -0.45156163, -0.7651039, 0.05943527, -0.63053423, -0.48430285,
                         0.88853973, -1.3249904, 1.3553275, -1.4115001, 0.928761, -0.57776105,
                         0.00862964, 0.67627704, -1.0972726, 1.5024354, -1.6889306, 1.551655,
                        -1.0651828, 0.5519766, 0.01219654, -0.5527712, 1.0270984, -1.2735034,
                         1.3647351, -1.1056445, 0.6303438, 0.70591617, 0.53802025, 0.04858056,
                        -0.46617195], rtol=0.01)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_render_stars_spherical(snap):
    plt.clf()
    res = pynbody.plot.stars.render_mollweide(snap, return_image=True)
    npt.assert_allclose(res[::1000], [[0.2158741, 0.13837814, 0.],
                                 [0.27598915, 0.20095405, 0.],
                                 [0.22899131, 0.16107063, 0.],
                                 [0.26585045, 0.19161034, 0.],
                                 [0.25475198, 0.18265152, 0.],
                                 [0.26186714, 0.19246177, 0.],
                                 [0.2895748, 0.22256851, 0.02087402],
                                 [0.2629215, 0.1937046, 0.],
                                 [0.29285353, 0.22283287, 0.012397],
                                 [0.34632796, 0.29408723, 0.12458153],
                                 [0.46969643, 0.47156182, 0.5308735],
                                 [0.3660347, 0.31326026, 0.14758568],
                                 [0.37119332, 0.3108948, 0.12590408],
                                 [0.35345078, 0.32170448, 0.19717637],
                                 [0.53826296, 0.53094447, 0.56197226],
                                 [0.36239165, 0.2919399, 0.07837562],
                                 [0.37456322, 0.31442222, 0.12232742],
                                 [0.35317573, 0.29436034, 0.11823387],
                                 [0.48886603, 0.4683048, 0.38486785],
                                 [0.59726584, 0.5737755, 0.4664154],
                                 [0.62946856, 0.6194912, 0.53192216],
                                 [0.62166405, 0.599037, 0.59090996],
                                 [0.54687995, 0.53192294, 0.44742966],
                                 [0.5402582, 0.5180721, 0.42749405],
                                 [0.69918764, 0.76421547, 0.888315],
                                 [0.65987283, 0.6458607, 0.5548245],
                                 [0.64007986, 0.6128222, 0.4971203],
                                 [0.5336639, 0.4971096, 0.3965332],
                                 [0.5932957, 0.57914084, 0.5544201],
                                 [0.5504688, 0.52398264, 0.41228446],
                                 [0.54369104, 0.53339136, 0.46228752],
                                 [0.5129448, 0.51816064, 0.47853774],
                                 [0.38571054, 0.3376709, 0.20089912],
                                 [0.35486984, 0.2799797, 0.06326218],
                                 [0.3700363, 0.3130932, 0.12754746],
                                 [0.36907578, 0.31540948, 0.13778381],
                                 [0.43266717, 0.3860607, 0.22645263],
                                 [0.38154602, 0.32660943, 0.16114426],
                                 [0.39782637, 0.3714714, 0.35722312],
                                 [0.417334, 0.38936806, 0.3729435],
                                 [0.38239974, 0.3711525, 0.30398408],
                                 [0.44534835, 0.44895783, 0.41125488],
                                 [0.44165152, 0.45368958, 0.4311821],
                                 [0.29969826, 0.258197, 0.12207337],
                                 [0.35284424, 0.33971292, 0.27164268],
                                 [0.31118774, 0.2614273, 0.16981545],
                                 [0.30190963, 0.26069984, 0.14900322],
                                 [0.34189034, 0.32610703, 0.2380909],
                                 [0.31345788, 0.2856903, 0.18001786],
                                 [0.27264443, 0.22463608, 0.07668419]],
                        atol = 1e-3)
