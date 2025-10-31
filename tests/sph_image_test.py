import platform
import sys
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
    # cen = pynbody.analysis.halo.center(h[1],retcen=True)
    # print "[%.15f, %.15f, %.15f]"%tuple(cen)
    snap['pos'] -= cen

    # derive smoothing lengths direct from file data so we are
    # not testing the kdtree (which is tested elsewhere)

    snap.gas['smooth'] = (snap.gas['mass'] / snap.gas['rho']) ** (1, 3)
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

    np.save("result_im_2d.npy", im2d)
    np.save("result_im_3d.npy", im3d)
    np.save("result_im_grid.npy", im_grid)

    npt.assert_allclose(im2d, compare2d, rtol=1e-5)
    npt.assert_allclose(im3d, compare3d, rtol=1e-5)
    npt.assert_allclose(im_grid, compare_grid, rtol=1e-5)

    # Make images with a different kernel (Wendland C2)
    im3d_wendlandC2 = pynbody.plot.sph.image(
        snap.gas, width=20.0, units="m_p cm^-3", noplot=True, approximate_fast=False, kernel='wendlandC2',
        resolution=500)
    im2d_wendlandC2 = pynbody.plot.sph.image(
        snap.gas, width=20.0, units="m_p cm^-2", noplot=True, approximate_fast=False, kernel='wendlandC2',
        resolution=500)

    np.save("result_im_2d_wendlandC2.npy", im2d_wendlandC2)
    np.save("result_im_3d_wendlandC2.npy", im3d_wendlandC2)

    # Check that using a different kernel produces a different image
    npt.assert_raises(AssertionError, npt.assert_array_equal, im3d_wendlandC2, im3d)
    npt.assert_raises(AssertionError, npt.assert_array_equal, im2d_wendlandC2, im2d)

    # Check that using a different kernel produces the correct image
    npt.assert_allclose(im2d_wendlandC2, compare2d_wendlandC2, rtol=1e-5)
    npt.assert_allclose(im3d_wendlandC2, compare3d_wendlandC2, rtol=1e-5)

    # check rectangular image is OK
    im_rect = pynbody.sph.render_image(snap.gas, nx=500, ny=250, width=20.0,
                                       approximate_fast=False).in_units("m_p cm^-3")
    np.save("result_im_3d_rectangular.npy", im_rect)

    compare_rect = compare3d[125:-125]
    npt.assert_allclose(im_rect, compare_rect, rtol=1e-4)


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
    assert abs(np.log10(im2d / compare2d)).mean() < 0.02
    assert abs(np.log10(im3d / compare3d)).mean() < 0.03
    assert abs(np.log10(im_grid / compare_grid)).mean() < 0.03


def test_denoise_projected_image_throws(snap):
    # this should be fine:
    pipeline = renderers.make_render_pipeline(snap.gas, width=20.0, out_units="m_p cm^-3", denoise=True, resolution=10)
    pipeline.render()

    with pytest.raises(renderers.RenderPipelineLogicError):
        # this should not:
        pipeline = renderers.make_render_pipeline(snap.gas, width=20.0, out_units="m_p cm^-2", denoise=True,
                                                  resolution=10)

def test_no_denoise_weighted_image(snap):
    snap._should_denoise_images = True

    # check auto-denoising is working
    pipeline = renderers.make_render_pipeline(snap.gas, width=20.0, denoise=None, quantity='temp')
    assert isinstance(pipeline, renderers.DenoisedImageRenderer)

    # but there should be NO denoising in this pipeline, since it makes no sense on a weighted image:
    pipeline = renderers.make_render_pipeline(snap.gas, width=20.0, denoise=None, weight='rho', quantity='temp')

    def _assert_not_denoising_step(step):
        if isinstance(step, renderers.DenoisedImageRenderer):
            raise AssertionError("Denoising step found in weighted image pipeline")
    def _traverse_steps(step):
        _assert_not_denoising_step(step)
        if hasattr(step, '_subrenderers'):
            for substep in step._subrenderers:
                _traverse_steps(substep)
    _traverse_steps(pipeline)

@pytest.mark.filterwarnings("ignore:No log file found:UserWarning")
def test_render_stars(stars_2d, stars_dust_2d, snap):
    im = pynbody.plot.stars.render(snap, width=10.0, resolution=100, return_image=True, noplot=True)

    np.save("result_stars_2d.npy", im[40:60])

    npt.assert_allclose(stars_2d, im[40:60], atol=0.01)


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
    answer = np.linspace(-0.5, 0.5, len(im_collapsed))
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

    assert abs(4 * np.pi * im.sum() / len(im) - 1.0) < 0.01
    assert im.units == "Msol sr^-1"

    im2 = pynbody.sph.render_spherical_image(f, 'rho', nside=32, out_units="Msol arcsec^-2")

    assert im2.units == "Msol arcsec^-2"
    assert abs((im2.sum() / len(im2)) * ((60 * 60 * 360) ** 2 / np.pi) - 1.0) < 0.01

    im3 = pynbody.sph.render_spherical_image(f, 'temp', weight='rho', nside=16)
    assert im3.units == "K"

    npt.assert_allclose(im3[::100], [0.04377608, -0.45156163, -0.7651039, 0.05943527, -0.63053423, -0.48430285,
                                     0.88853973, -1.3249904, 1.3553275, -1.4115001, 0.928761, -0.57776105,
                                     0.00862964, 0.67627704, -1.0972726, 1.5024354, -1.6889306, 1.551655,
                                     -1.0651828, 0.5519766, 0.01219654, -0.5527712, 1.0270984, -1.2735034,
                                     1.3647351, -1.1056445, 0.6303438, 0.70591617, 0.53802025, 0.04858056,
                                     -0.46617195], rtol=0.01)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.skipif(platform.system() == "Windows", reason="Healpy not supported on Windows")
@pytest.mark.skipif(sys.version_info >= (3, 14), reason="Healpy binaries seem to be broken on Python 3.14")
def test_render_stars_spherical(snap):
    plt.clf()
    res = pynbody.plot.stars.render_mollweide(snap, return_image=True)
    print(res[::1000].tolist())
    npt.assert_allclose(res[::1000],
                        [[0.2229621410369873, 0.15252339839935303, 0.0], [0.2824336588382721, 0.21390795707702637, 0.0],
                         [0.2376975566148758, 0.1786838173866272, 0.0], [0.27199816703796387, 0.20493198931217194, 0.0],
                         [0.26165881752967834, 0.19684138894081116, 0.0],
                         [0.26918938755989075, 0.2062690407037735, 0.0],
                         [0.2968413829803467, 0.23688426613807678, 0.043666448444128036],
                         [0.27126115560531616, 0.20973926782608032, 0.015258023515343666],
                         [0.3037688732147217, 0.24323955178260803, 0.047282781451940536],
                         [0.351524293422699, 0.30447304248809814, 0.13997876644134521],
                         [0.4625605642795563, 0.4698695242404938, 0.5282549858093262],
                         [0.37370750308036804, 0.32695305347442627, 0.16641995310783386],
                         [0.3791663348674774, 0.3249293565750122, 0.14703862369060516],
                         [0.3593703508377075, 0.33067962527275085, 0.2064322978258133],
                         [0.5411242246627808, 0.5344496369361877, 0.555456817150116],
                         [0.3727896809577942, 0.31097710132598877, 0.11082074791193008],
                         [0.38441041111946106, 0.3319724500179291, 0.150437131524086],
                         [0.36130744218826294, 0.3091670274734497, 0.13937032222747803],
                         [0.49240636825561523, 0.475178062915802, 0.39365532994270325],
                         [0.599542498588562, 0.578336238861084, 0.47194433212280273],
                         [0.6318617463111877, 0.6233565211296082, 0.5370165705680847],
                         [0.6301445960998535, 0.6114792823791504, 0.5874732136726379],
                         [0.5514353513717651, 0.5389474034309387, 0.45507726073265076],
                         [0.5454073548316956, 0.5252165794372559, 0.43494176864624023],
                         [0.706606924533844, 0.770249605178833, 0.8858953714370728],
                         [0.6607868671417236, 0.6481878757476807, 0.5579562187194824],
                         [0.6459441184997559, 0.6218857765197754, 0.5085471272468567],
                         [0.5392252802848816, 0.5053431391716003, 0.40516120195388794],
                         [0.5946093201637268, 0.5831661820411682, 0.5578675270080566],
                         [0.5531724095344543, 0.5293429493904114, 0.4192431569099426],
                         [0.5477307438850403, 0.5403294563293457, 0.47061070799827576],
                         [0.514541506767273, 0.521500289440155, 0.48092564940452576],
                         [0.39495232701301575, 0.3534831404685974, 0.22206301987171173],
                         [0.36129945516586304, 0.29370760917663574, 0.08680113404989243],
                         [0.3780021071434021, 0.3280673325061798, 0.15287357568740845],
                         [0.37590286135673523, 0.32776787877082825, 0.1549510657787323],
                         [0.43925583362579346, 0.3990176320075989, 0.24666476249694824],
                         [0.3886016011238098, 0.3391009569168091, 0.17747341096401215],
                         [0.39992478489875793, 0.3761073350906372, 0.3573252856731415],
                         [0.4210982620716095, 0.3963324725627899, 0.37522652745246887],
                         [0.38443443179130554, 0.37533217668533325, 0.3074904680252075],
                         [0.4462508261203766, 0.45160743594169617, 0.41301605105400085],
                         [0.4433249533176422, 0.45710325241088867, 0.43313515186309814],
                         [0.3065364360809326, 0.2693595290184021, 0.13572002947330475],
                         [0.35695451498031616, 0.3462851941585541, 0.2771396040916443],
                         [0.3158774673938751, 0.26991990208625793, 0.1756984442472458],
                         [0.3081778883934021, 0.27168649435043335, 0.1604747474193573],
                         [0.3461741805076599, 0.3327319324016571, 0.24533382058143616],
                         [0.3185519576072693, 0.29396510124206543, 0.18811184167861938],
                         [0.28043055534362793, 0.23913493752479553, 0.0976463109254837]],
                        atol=1e-3)
