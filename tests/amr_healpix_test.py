"""Tests for rendering AMR cells exactly onto healpix maps (AMRHealpixRenderer)"""

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils
from pynbody.imaging import _render_amr_healpix, renderers


@pytest.fixture(autouse=True)
def use_true_amr_by_default():
    """Most tests here check the behaviour when AMR rendering is switched on in the configuration file"""
    config = pynbody.config_parser
    original = config.get('sph', 'use-true-amr')
    config.set('sph', 'use-true-amr', 'True')
    yield
    config.set('sph', 'use-true-amr', original)

hp = pytest.importorskip("healpy")


def _cube_of_cells(n_sub, size, centre, rotation=np.eye(3)):
    """Positions of n_sub^3 cells exactly tiling a cube of the given size, rotated about its centre"""
    g = (np.arange(n_sub) + 0.5) * size / n_sub - size / 2
    u = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T
    return u @ rotation.T + centre


def _rotation(theta_x, theta_z):
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    return np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])


def _render(pos, size, rotation, nside, qty=1.0, shell_radius=0.0, **kwargs):
    pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    n = len(pos)
    sm = np.broadcast_to(np.asarray(size, dtype=np.float64), (n,)).copy()
    qty = np.broadcast_to(np.asarray(qty, dtype=np.float64), (n,)).copy()
    return _render_amr_healpix.render_healpix_cells(nside, pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy(), sm,
                                                    qty, sm ** 3, np.ones(n), 0.0, rotation, shell_radius, **kwargs)


def _brute_force(pos, size, rotation, nside, pixels, up=128, shell_radius=0.0):
    """Average over each pixel of the exact integrand, along rays through the centres of up^2 sub-pixels"""
    centre = rotation.T @ np.asarray(pos, dtype=np.float64)
    lo, hi = centre - size / 2, centre + size / 2
    result = []
    for pix in pixels:
        nest = hp.ring2nest(nside, pix)
        vec = np.array(hp.pix2vec(nside * up, np.arange(nest * up * up, (nest + 1) * up * up), nest=True)).T
        m = vec @ rotation  # components along the cell axes
        if shell_radius:
            e = shell_radius * m
            result.append(np.all((e >= lo) & (e < hi), axis=1).mean())
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                t1 = np.where(m > 0, lo / m, hi / m)
                t2 = np.where(m > 0, hi / m, lo / m)
            # rays parallel to a pair of faces are unconstrained by them if they start between them, else miss
            parallel = m == 0
            between = (lo <= 0) & (hi >= 0)
            t1 = np.where(parallel, np.where(between, -np.inf, np.inf), t1)
            t2 = np.where(parallel, np.where(between, np.inf, -np.inf), t2)
            t_in = np.maximum(t1.max(axis=1), 0)
            t_out = t2.min(axis=1)
            result.append(np.where(t_out > t_in, (t_out ** 3 - t_in ** 3) / 3, 0.0).mean())
    return np.array(result)


_pixel_area = lambda nside: 4 * np.pi / (12 * nside ** 2)
_pixel_size = lambda nside: np.sqrt(_pixel_area(nside))

# (centre, size) of single cells in different regimes, for nside=8
_cells = {'containing origin': ([0.1, -0.05, 0.2], 1.0),
          'near origin': ([0.9, 0.3, -0.5], 0.8),
          'several pixels': ([2.0, 1.0, 3.0], 5 * _pixel_size(8) * np.sqrt(14)),
          'one pixel': ([2.0, -1.0, 0.5], _pixel_size(8) * np.sqrt(5.25)),
          'subpixel': ([-2.0, -1.0, 0.5], 0.3 * _pixel_size(8) * np.sqrt(5.25)),
          'tiny': ([-2.0, 1.0, 0.5], 0.03 * _pixel_size(8) * np.sqrt(5.25)),
          'at pole': ([1e-3, 2e-3, 3.0], 2 * _pixel_size(8) * 3.0),
          'origin on corner': ([0.25, 0.25, 0.25], 0.5)}


def test_healpix_geometry_matches_healpy():
    rng = np.random.default_rng(1)
    for nside in [1, 2, 8, 64]:
        for face in range(12):
            ix, iy = rng.integers(0, nside, (2, 50))
            nest = hp.xyf2pix(nside, ix, iy, face, nest=True)
            npt.assert_array_equal(_render_amr_healpix._ring_indices(nside, face, ix, iy), hp.nest2ring(nside, nest))
            npt.assert_allclose(_render_amr_healpix._pixel_centre_vectors(nside, face, ix, iy),
                                np.array(hp.pix2vec(nside, nest, nest=True)).T, atol=1e-13)
            for j in range(3):
                corners = _render_amr_healpix._pixel_corners(nside, face, ix[j], iy[j])
                expected = hp.boundaries(nside, nest[j], nest=True).T
                # the same four corners, possibly in a different order
                npt.assert_allclose(np.sort(corners @ expected.T, axis=1)[:, -1], 1.0, atol=1e-13)
        v = rng.normal(size=(1000, 3))
        pix, margin = _render_amr_healpix._locate_pixels(nside, v)
        npt.assert_array_equal(pix, hp.vec2pix(nside, *v.T))
        # directions displaced by less than the margin (see _locate) stay in the same pixel
        v /= np.linalg.norm(v, axis=1)[:, None]
        t = rng.normal(size=v.shape)
        t -= (t * v).sum(axis=1)[:, None] * v
        t /= np.linalg.norm(t, axis=1)[:, None]
        r = 0.999 * margin / (1.5 * nside)
        npt.assert_array_equal(hp.vec2pix(nside, *(v * np.cos(r)[:, None] + t * np.sin(r)[:, None]).T), pix)


def test_cube_section_area():
    npt.assert_allclose(_render_amr_healpix.cube_section_area(0.5, [0, 0, 1], 0.2), 1.0)
    npt.assert_allclose(_render_amr_healpix.cube_section_area(0.5, [0, 0, 1], 0.6), 0.0)
    # a rectangle, whose width is the chord x + y = 0.1 sqrt(2) through the unit square
    npt.assert_allclose(_render_amr_healpix.cube_section_area(0.5, np.array([1, 1, 0]) / np.sqrt(2), 0.1),
                        np.sqrt(2) * (1 - 0.1 * np.sqrt(2)))
    # regular hexagon through the centre, perpendicular to a diagonal
    npt.assert_allclose(_render_amr_healpix.cube_section_area(0.5, np.ones(3) / np.sqrt(3), 0.0), 3 * np.sqrt(3) / 4)


@pytest.mark.parametrize("cell", list(_cells.keys()))
@pytest.mark.parametrize("rotated", [False, True])
def test_projection_mass_conservation(cell, rotated):
    nside = 8
    rotation = _rotation(0.4, 0.7) if rotated else np.eye(3)
    pos, size = _cells[cell]
    im = _render(pos, size, rotation, nside)
    assert im.min() >= 0
    npt.assert_allclose(im.astype(np.float64).sum() * _pixel_area(nside), size ** 3, rtol=1e-6)
    # cells that fall within a pixel deposit their volume directly, which must make no difference
    npt.assert_allclose(_render(pos, size, rotation, nside, deposit=False), im, rtol=1e-5, atol=1e-6 * im.max())


@pytest.mark.parametrize("cell", ['containing origin', 'near origin', 'several pixels', 'one pixel', 'subpixel',
                                  'at pole', 'origin on corner'])
@pytest.mark.parametrize("rotated", [False, True])
def test_projection_against_brute_force(cell, rotated):
    """Pixel values must be the exact average of the column, as estimated with many rays per pixel"""
    nside = 8
    rotation = _rotation(0.4, 0.7) if rotated else np.eye(3)
    pos, size = _cells[cell]
    im = _render(pos, size, rotation, nside)
    pixels = np.argsort(-im)[:3]
    if cell == 'origin on corner':
        pixels = np.concatenate([pixels, hp.query_disc(nside, [-1, -1, 0.1], 0.3)[:3]])  # pixels around the edge
    # cells small compared to the sub-pixels need more rays for an accurate estimate
    reference = _brute_force(pos, size, rotation, nside, pixels, up=256 if size < 0.2 else 64)
    npt.assert_allclose(im[pixels], reference, atol=2e-4 * im.max())


@pytest.mark.parametrize("n_sub", [2, 8])
@pytest.mark.parametrize("size_in_pixels", [0.3, 1.5, 6.0])
@pytest.mark.parametrize("rotated", [False, True])
def test_projection_tiling(n_sub, size_in_pixels, rotated):
    """A cube must give the same map as the cells that tile it"""
    nside = 16
    rotation = _rotation(0.3, 1.1) if rotated else np.eye(3)
    centre = np.array([1.3, -2.1, 0.7])
    size = size_in_pixels * _pixel_size(nside) * np.linalg.norm(centre)
    whole = _render(centre, size, rotation, nside)
    parts = _render(_cube_of_cells(n_sub, size, centre, rotation), size / n_sub, rotation, nside)
    # the small corrections for the shapes of pixels are calculated slightly differently for different sizes of cell
    npt.assert_allclose(parts, whole, atol=1e-4 * whole.max())


def test_projection_tiling_through_origin():
    """A grid of cells around the origin, some with faces or corners at the origin, must give the same map as a
    single cell"""
    nside = 8
    rotation = _rotation(0.3, 1.1)
    whole = _render([0.0, 0.0, 0.0], 2.0, rotation, nside)
    parts = _render(_cube_of_cells(4, 2.0, np.zeros(3), rotation), 0.5, rotation, nside)
    npt.assert_allclose(parts, whole, rtol=2e-5)
    npt.assert_allclose(whole.astype(np.float64).sum() * _pixel_area(nside), 8.0, rtol=1e-6)


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("n_cells, radius", [(16, 1.3), (40, 0.05), (64, 1.7)])
def test_shell_of_uniform_grid(rotated, n_cells, radius):
    """A thin shell through a uniform grid of cells must be uniform"""
    rotation = _rotation(0.4, 0.7) if rotated else np.eye(3)
    pos = _cube_of_cells(n_cells, 4.0, np.array([0.01, 0.02, 0.03]), rotation)
    im = _render(pos, 4.0 / n_cells, rotation, 16, qty=2.5, shell_radius=radius)
    npt.assert_allclose(im, 2.5, rtol=1e-5)


@pytest.mark.parametrize("cell, radius", [('one pixel', None), ('subpixel', None), ('tiny', None),
                                          ('several pixels', 3.4), ('containing origin', 0.2),
                                          ('at pole', 3.05)])
@pytest.mark.parametrize("rotated", [False, True])
def test_shell_against_brute_force(cell, radius, rotated):
    nside = 8
    rotation = _rotation(0.4, 0.7) if rotated else np.eye(3)
    pos, size = _cells[cell]
    if radius is None:
        radius = np.linalg.norm(pos) + 0.1 * size  # a shell straddling the cell
    im = _render(pos, size, rotation, nside, shell_radius=radius)
    assert im.max() > 0
    pixels = np.argsort(-im)[:4]
    # Sampling the step function is not very accurate, especially for cells aligned with the axes, for which the
    # errors can be coherent along an edge (the reference converges to the rendered values only slowly). The rendered
    # values converge rapidly as the sub-pixels are refined.
    pixels = pixels[im[pixels] > 0]
    reference = _brute_force(pos, size, rotation, nside, pixels, up=2048 if size < 0.01 else 256, shell_radius=radius)
    npt.assert_allclose(im[pixels], reference, atol=(5e-3 if rotated else 3e-2) * im.max())
    refined = _render(pos, size, rotation, nside, shell_radius=radius, edge_level=6, cells_across=64.0)
    npt.assert_allclose(im, refined, atol=5e-4 * im.max())


def test_nonfinite_cells_skipped():
    pos = np.array([[1.0, 0.5, 0.2], [np.nan, 0.0, 0.0], [0.5, np.inf, 0.0], [0.5, 0.5, 0.5]])
    size = np.array([0.1, 0.1, 0.1, np.nan])
    im = _render_amr_healpix.render_healpix_cells(8, *(np.ascontiguousarray(pos[:, k]) for k in range(3)), size,
                                                  np.ones(4), np.full(4, 1e-3), np.ones(4), 0.0, np.eye(3))
    npt.assert_allclose(im, _render(pos[0], 0.1, np.eye(3), 8))


@pytest.fixture
def amr_snap():
    """A small synthetic two-level AMR snapshot, with a refined region in one octant"""
    coarse = _cube_of_cells(8, 8.0, np.zeros(3))
    refine = np.all(coarse > 0, axis=1) & np.all(coarse < 2.0, axis=1)
    fine = np.concatenate([_cube_of_cells(2, 1.0, c) for c in coarse[refine]])
    pos = np.concatenate([coarse[~refine], fine]) + np.array([0.1, -0.2, 0.3])
    size = np.concatenate([np.ones((~refine).sum()), np.full(len(fine), 0.5)])
    f = pynbody.new(gas=len(pos))
    f['pos'] = pos
    f['pos'].units = 'kpc'
    f['smooth'] = size
    f['smooth'].units = 'kpc'
    f['rho'] = 1.0 + np.exp(-((pos - 1.0) ** 2).sum(axis=1)) + 0.1 * pos[:, 0]
    f['rho'].units = 'Msol kpc^-3'
    f['mass'] = f['rho'] * f['smooth'] ** 3
    f._amr_cell_families = (pynbody.family.gas,)
    return f


def _core_renderer(pipeline):
    while isinstance(pipeline, renderers.MultipassImageRenderer):
        pipeline = pipeline._subrenderers[0]
    return pipeline


def test_renderer_selection(amr_snap):
    def core_type(**kwargs):
        return type(_core_renderer(renderers.make_render_pipeline(amr_snap.g, target='healpix', nside=4, **kwargs)))

    assert core_type() is renderers.AMRHealpixRenderer
    assert core_type(shell_distance=2.0) is renderers.AMRHealpixRenderer
    assert core_type(weight='rho') is renderers.AMRHealpixRenderer
    assert core_type(use_true_amr=True) is renderers.AMRHealpixRenderer
    assert core_type(use_true_amr=False) is renderers.HealpixRenderer
    assert core_type(kernel='CubicSpline') is renderers.HealpixRenderer
    with pytest.raises(ValueError):
        renderers.make_render_pipeline(amr_snap.g, target='healpix', use_true_amr=True, kernel='CubicSpline')

    with pytest.raises(ValueError, match="without a kernel"):
        renderers.AMRHealpixRenderer(amr_snap.g).set_kernel('CubicSpline')
    with pytest.raises(NotImplementedError):
        renderers.make_render_pipeline(amr_snap.g, target='healpix', nside=4).with_approximate()

    amr_snap._amr_cell_families = ()
    assert core_type() is renderers.HealpixRenderer
    with pytest.raises(ValueError, match="AMR cells"):
        core_type(use_true_amr=True)


@pytest.mark.parametrize("threaded", [False, True])
def test_pipeline_projection(amr_snap, threaded):
    nside = 16
    pipeline = renderers.make_render_pipeline(amr_snap.g, target='healpix', nside=nside, threaded=threaded)
    im = pipeline.render()
    assert im.units == pynbody.units.Unit('Msol sr^-1')
    npt.assert_allclose(im.sum() * _pixel_area(nside), amr_snap.g['mass'].sum(), rtol=1e-6)
    reference = renderers.make_render_pipeline(amr_snap.g, target='healpix', nside=nside, threaded=False).render()
    npt.assert_allclose(im, reference, rtol=1e-5)

    # a volume-weighted projection of a uniform field is uniform
    amr_snap['temp'] = np.full(len(amr_snap), 3.0)
    weighted = renderers.make_render_pipeline(amr_snap.g, 'temp', target='healpix', nside=nside, threaded=threaded,
                                              weight=True).render()
    npt.assert_allclose(weighted, 3.0, rtol=1e-5)


def test_pipeline_shell(amr_snap):
    """A thin shell must show the density of the cells it passes through, averaged over each pixel"""
    nside, up, radius = 8, 16, 1.35
    im = renderers.make_render_pipeline(amr_snap.g, target='healpix', nside=nside, shell_distance=radius).render()
    assert im.units == amr_snap['rho'].units

    # the density at points on the shell through the centres of up^2 sub-pixels of each pixel
    pos, sm, rho = (amr_snap.g[k].view(np.ndarray) for k in ('pos', 'smooth', 'rho'))
    near = np.abs(np.linalg.norm(pos, axis=1) - radius) < sm
    points = radius * np.array(hp.pix2vec(nside * up, np.arange(12 * (nside * up) ** 2), nest=True)).T
    samples = np.zeros(len(points))
    for p, s, r in zip(pos[near], sm[near], rho[near]):
        samples[np.all(np.abs(points - p) < s / 2, axis=1)] = r
    samples = samples.reshape(-1, up * up)[hp.ring2nest(nside, np.arange(12 * nside ** 2))]
    assert samples.min() > 0

    # pixels lying within one cell show its density exactly; others are averages. (To be sure that a pixel lies within
    # one cell, require it and its neighbours to be sampled uniformly.)
    uniform = samples.min(axis=1) == samples.max(axis=1)
    neighbours = hp.get_all_neighbours(nside, np.arange(12 * nside ** 2))
    uniform &= np.all(uniform[neighbours] & (samples[neighbours, 0] == samples[None, :, 0]), axis=0)
    assert 50 < uniform.sum() < len(im)
    npt.assert_allclose(im[uniform], samples[uniform, 0], rtol=1e-5)
    npt.assert_allclose(im, samples.mean(axis=1), atol=0.02 * im.max())


def test_rotated_snapshot(amr_snap):
    """Cells must follow rotations of the snapshot, matching an explicit orientation"""
    nside = 16
    rotation = _rotation(0.4, 0.7)
    with amr_snap.rotate_x(np.degrees(0.4)).rotate_z(np.degrees(0.7)):
        npt.assert_allclose(amr_snap.net_rotation_matrix(), rotation, atol=1e-12)
        rotated = renderers.make_render_pipeline(amr_snap.g, target='healpix', nside=nside).render()
        npt.assert_allclose(rotated.sum() * _pixel_area(nside), amr_snap.g['mass'].sum(), rtol=1e-6)
        rotated_shell = renderers.make_render_pipeline(amr_snap.g, target='healpix', nside=nside,
                                                       shell_distance=3.0).render()

    # rotating the snapshot rotates the map: the value of the rotated map in direction v is that of the unrotated map
    # in direction R^T v. Pixels do not map exactly onto each other, so compare the maps at low resolution.
    unrotated = renderers.make_render_pipeline(amr_snap.g, target='healpix', nside=nside).render()
    pixels_back = hp.vec2pix(nside, *(rotation.T @ np.array(hp.pix2vec(nside, np.arange(12 * nside ** 2)))))
    npt.assert_allclose(hp.ud_grade(np.asarray(rotated), 2), hp.ud_grade(np.asarray(unrotated)[pixels_back], 2),
                        rtol=0.02)

    # with the positions rotated by hand, the orientation must be specified explicitly
    amr_snap['pos'] = amr_snap['pos'] @ rotation.T
    renderer = renderers.AMRHealpixRenderer(amr_snap.g, orientation=rotation)
    renderer.geometry.set_nside(nside)
    renderer.set_quantity('rho')
    renderer.set_projection(True)
    npt.assert_allclose(renderer.render(), rotated, rtol=1e-5)
    npt.assert_allclose(renderer.with_threading(2).render(), rotated, rtol=1e-5)
    renderer.set_shell_distance(3.0)
    npt.assert_allclose(renderer.render(), rotated_shell, rtol=1e-5)


def test_smooth_floor(amr_snap):
    """A smooth floor enlarges the cells; projections still conserve mass, and shells are denoised"""
    nside = 8
    pipeline = renderers.make_render_pipeline(amr_snap.g, target='healpix', nside=nside, smooth_floor=0.8)
    npt.assert_allclose(pipeline.render().sum() * _pixel_area(nside), amr_snap.g['mass'].sum(), rtol=1e-6)
    amr_snap['rho'] = 2.0
    amr_snap['mass'] = amr_snap['rho'] * amr_snap['smooth'] ** 3
    shell = renderers.make_render_pipeline(amr_snap.g, target='healpix', nside=nside, smooth_floor=0.8,
                                           shell_distance=0.5)
    assert isinstance(shell, renderers.DenoisedImageRenderer)
    npt.assert_allclose(shell.render(), 2.0, rtol=1e-5)


@pytest.mark.filterwarnings("ignore:.*namelist:UserWarning")
def test_ramses_healpix():
    pynbody.test_utils.ensure_test_data_available("ramses")
    f = pynbody.load("testdata/ramses/output_00080")
    centre_index = np.argmax(f.g['rho'])
    cen = f.g['pos'][centre_index].view(np.ndarray).copy()
    with f.translate(-cen):
        nside = 8
        pipeline = renderers.make_render_pipeline(f.g, target='healpix', nside=nside, threaded=True)
        assert isinstance(_core_renderer(pipeline), renderers.AMRHealpixRenderer)
        im = pipeline.render()
        npt.assert_allclose(im.sum() * _pixel_area(nside), f.g['mass'].sum(), rtol=1e-5)
        assert im.min() > 0

        # a shell within the densest cell shows only that cell
        shell = renderers.make_render_pipeline(f.g, target='healpix', nside=nside, threaded=True,
                                               shell_distance=f.g['smooth'][centre_index] * 0.3).render()
        npt.assert_allclose(shell, f.g['rho'][centre_index], rtol=1e-5)
