"""Tests for rendering AMR cells as exact cubes (AMRImageRenderer)"""

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils
from pynbody.imaging import _render_amr, renderers


def _cube_of_cells(n_sub, size, centre, rotation=np.eye(3)):
    """Positions of n_sub^3 cells exactly tiling a cube of the given size, rotated about its centre"""
    g = (np.arange(n_sub) + 0.5) * size / n_sub - size / 2
    u = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T
    return u @ rotation.T + centre


def _render_cells(pos, size, qty, rotation, projected, nx=100, width=4.0, z0=0.0, z_lo=-np.inf, z_hi=np.inf,
                  **kwargs):
    pos = np.atleast_2d(pos)
    n = len(pos)
    sm = np.full(n, float(size))
    qty = np.broadcast_to(np.asarray(qty, dtype=np.float64), (n,)).copy()
    mass = sm ** 3
    rho = np.ones(n)
    return _render_amr.render_image_cells(nx, nx, pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy(), sm,
                                      -width / 2, width / 2, -width / 2, width / 2, z0,
                                      qty, mass, rho, 0.0, np.inf, z_lo, z_hi, 0.0, projected, rotation, **kwargs)


def _rotation(theta_x, theta_z):
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    return np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])


@pytest.fixture
def amr_snap():
    """A small synthetic two-level AMR snapshot, with a refined region in one octant"""
    coarse = _cube_of_cells(8, 8.0, np.zeros(3))
    refine = np.all(coarse > 0, axis=1) & np.all(coarse < 2.0, axis=1)
    fine = np.concatenate([_cube_of_cells(2, 1.0, c) for c in coarse[refine]])
    pos = np.concatenate([coarse[~refine], fine])
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


@pytest.mark.parametrize("projected", [True, False])
@pytest.mark.parametrize("rotated", [True, False])
@pytest.mark.parametrize("exact", [True, False])
def test_cells_tile_exactly(projected, rotated, exact):
    """A cube decomposed into subcells must render identically to the cube itself"""
    rotation = _rotation(0.4, 0.7) if rotated else np.eye(3)
    centre = np.array([0.125, -0.0625, 0.25])
    # tables=[] forces exact projections of rotated cells, rather than interpolating tabulated volumes
    kwargs = {'tables': []} if exact else {}
    big = _render_cells(centre, 1.6, 1.0, rotation, projected, z_lo=-0.3, z_hi=0.2, **kwargs)
    small = _render_cells(_cube_of_cells(8, 1.6, centre, rotation), 0.2, 1.0, rotation, projected,
                          z_lo=-0.3, z_hi=0.2, **kwargs)
    assert big.max() > 0.1
    # interpolation in the tables gives errors of order 1e-4 (relative) for the projection of rotated cells
    npt.assert_allclose(small, big, atol=1e-4 if (projected and rotated and not exact) else 1e-5)


@pytest.mark.parametrize("rotation", [_rotation(0.4, 0.7), _rotation(1.2, 2.1),
                                      _rotation(np.radians(30), np.radians(25)),  # one cell axis in image plane
                                      _rotation(1e-6, 1e-6)])
@pytest.mark.parametrize("cell_pixels", [0.3, 1.5, 6.0, 25.0, 300.0])
def test_rotated_projection_against_exact(rotation, cell_pixels):
    """Rotated projections must agree with the exact path, whichever of the faster paths is taken.

    Cells up to 16 pixels interpolate tabulated volumes; larger cells are projected exactly, accumulating along rows
    (see render_image_cells). tables=[] forces the exact projection of all cells."""
    rng = np.random.default_rng(1)
    a = cell_pixels * 4.0 / 100
    n_sub = max(2, min(12, int(6.0 / a)))
    pos = _cube_of_cells(n_sub, n_sub * a, rng.uniform(-0.3, 0.3, 3), rotation)
    qty = rng.lognormal(0.0, 1.0, len(pos))
    fast = _render_cells(pos, a, qty, rotation, True)
    exact = _render_cells(pos, a, qty, rotation, True, tables=[])
    assert fast.min() >= 0
    assert exact.max() > 0
    npt.assert_allclose(fast, exact, atol=1e-3 * exact.max())
    npt.assert_allclose(fast.sum(), exact.sum(), rtol=1e-5)


@pytest.mark.parametrize("rotation", [_rotation(0.4, 0.7), _rotation(np.radians(30), np.radians(25)),
                                      _rotation(np.pi / 2, 0.3)])
@pytest.mark.parametrize("cell_pixels", [0.7, 3.0, 12.0, 40.0])
@pytest.mark.parametrize("z_range", [(-np.inf, np.inf), (-0.1, 0.25)])
@pytest.mark.parametrize("use_tables", [True, False])
def test_rotated_projection_brute_force(rotation, cell_pixels, z_range, use_tables):
    """Projections of single rotated cells must match an independent brute-force calculation, which evaluates the
    exact volume of the cell within x <= X, y <= Y (and the z range) at every pixel corner"""
    nx, width = 48, 4.0
    a = cell_pixels * width / nx
    centre = np.array([0.13, -0.21, 0.07 * a])
    z_lo, z_hi = (z * a for z in z_range)
    kwargs = {} if use_tables else {'tables': []}
    im = _render_cells(centre, a, 1.0, rotation, True, nx=nx, width=width, z_lo=z_lo, z_hi=z_hi, **kwargs)

    corners = np.linspace(-width / 2, width / 2, nx + 1)
    X, Y = np.meshgrid((corners - centre[0]) / a, (corners - centre[1]) / a)
    F = _render_amr.cell_quadrant_volume(rotation, X, Y, (z_lo - centre[2]) / a, (z_hi - centre[2]) / a) * a ** 3
    expected = (F[1:, 1:] - F[1:, :-1] - F[:-1, 1:] + F[:-1, :-1]) / (width / nx) ** 2

    assert expected.max() > 0
    exact = not use_tables or cell_pixels > 16 or np.isfinite(z_lo)
    npt.assert_allclose(im, expected, atol=(1e-5 if exact else 1e-3) * expected.max())
    npt.assert_allclose(im.sum(), expected.sum(), rtol=1e-5)


@pytest.mark.parametrize("rotation", [_rotation(0.4, 0.7), _rotation(np.radians(30), np.radians(25)),
                                      _rotation(np.pi / 2, 0.0), np.eye(3)])
def test_cell_quadrant_volume(rotation):
    """The exact volume of a cube with x <= dx, y <= dy (and optionally restricted in z) must match Monte Carlo"""
    rng = np.random.default_rng(3)
    u = rng.uniform(-0.5, 0.5, (400000, 3)) @ rotation.T
    dx, dy = rng.uniform(-0.8, 0.8, (2, 10))
    for z_lo, z_hi in ((-np.inf, np.inf), (-0.2, 0.3)):
        exact = _render_amr.cell_quadrant_volume(rotation, dx, dy, z_lo, z_hi)
        in_z = (u[:, 2] >= z_lo) & (u[:, 2] <= z_hi)
        mc = [np.mean((u[:, 0] <= x) & (u[:, 1] <= y) & in_z) for x, y in zip(dx, dy)]
        npt.assert_allclose(exact, mc, atol=4e-3)
    npt.assert_allclose(_render_amr.cell_quadrant_volume(rotation, 1.0, 1.0), 1.0, rtol=1e-12)
    npt.assert_allclose(_render_amr.cell_quadrant_volume(rotation, -1.0, 1.0), 0.0, atol=1e-12)


@pytest.mark.parametrize("nx", [10, 12, 15, 20, 24, 25, 30])
def test_uniform_grid_edges(nx):
    """A uniform grid filling the image must give a uniform image. For some resolutions, round-off places cells
    lying just outside the right and top edges of the image at pixel index nx (or ny); these must not be deposited
    (they previously wrapped into other rows, or wrote beyond the image)."""
    a, n = 0.2, 50
    g = (np.arange(n) + 0.5) * a - n * a / 2
    pos = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T
    im = _render_cells(pos, a, 1.0, np.eye(3), True, nx=nx, width=6.0)
    npt.assert_allclose(im, n * a, rtol=1e-5)  # (float32 accumulation round-off)
    im = _render_cells(pos, a, 1.0, np.eye(3), False, nx=nx, width=6.0, z0=0.05)
    npt.assert_allclose(im, 1.0, rtol=1e-5)


def test_cell_beyond_edge_by_roundoff():
    """A cell whose lower edge lies a fraction of a floating-point step inside the right edge of the image"""
    a = 0.2
    centre = np.nextafter(2.0, -np.inf) + a / 2
    for rotation in (np.eye(3), _rotation(0.4, 0.7)):
        im = _render_cells([centre, 0.1, 0.0], a, 1.0, rotation, True, nx=10, width=4.0)
        assert im[:, :-1].sum() == 0  # nothing wrapped into other rows


@pytest.mark.parametrize("bad", ["nan_x", "inf_x", "nan_z", "nan_smooth", "zero_smooth", "negative_smooth"])
@pytest.mark.parametrize("rotated", [True, False])
@pytest.mark.parametrize("projected", [True, False])
def test_nonfinite_cells_skipped(bad, rotated, projected):
    """Cells with undefined positions or sizes are skipped, without affecting other cells"""
    rotation = _rotation(0.4, 0.7) if rotated else np.eye(3)
    pos = np.array([[0.1, 0.0, 0.0], [-0.5, 0.2, 0.0]])
    sizes = np.array([0.1, 0.1])
    if bad == "nan_x": pos[0, 0] = np.nan
    if bad == "inf_x": pos[0, 0] = np.inf
    if bad == "nan_z": pos[0, 2] = np.nan
    if bad == "nan_smooth": sizes[0] = np.nan
    if bad == "zero_smooth": sizes[0] = 0.0
    if bad == "negative_smooth": sizes[0] = -0.1
    n = len(pos)
    im = _render_amr.render_image_cells(100, 100, pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy(), sizes,
                                        -1.0, 1.0, -1.0, 1.0, 0.0, np.ones(n), np.full(n, 0.1 ** 3), np.ones(n),
                                        0.0, np.inf, -np.inf, np.inf, 0.0, projected, rotation)
    reference = _render_cells(pos[1], 0.1, 1.0, rotation, projected, nx=100, width=2.0)
    assert np.all(np.isfinite(im))
    npt.assert_allclose(im, reference, atol=1e-6)


@pytest.mark.parametrize("rotated", [True, False])
def test_cell_projection_conserves_mass(rotated):
    rotation = _rotation(0.4, 0.7) if rotated else np.eye(3)
    for size in (1.3, 0.3, 0.05):
        # (sizes covering the different algorithms for rotated cells)
        im = _render_cells([0.1, -0.2, 0.05], size, 2.0, rotation, True)
        npt.assert_allclose(im.sum() * (4.0 / 100) ** 2, 2.0 * size ** 3, rtol=1e-5)


def test_cell_slice_rotated_area():
    """Slice through a cube rotated about z is a rotated square of known area"""
    rotation = _rotation(0.0, 0.3)
    im = _render_cells([0.1, -0.2, 0.05], 1.3, 2.0, rotation, False)
    npt.assert_allclose(im.sum() * (4.0 / 100) ** 2, 2.0 * 1.3 ** 2, rtol=1e-5)
    assert im.max() == pytest.approx(2.0)

    # tilted about x by angle t, the cross-section through the centre is a 1.3 x 1.3/cos(t) rectangle
    rotation = _rotation(0.3, 0.0)
    im = _render_cells([0.0, 0.0, 0.0], 1.3, 1.0, rotation, False)
    npt.assert_allclose(im.sum() * (4.0 / 100) ** 2, 1.3 ** 2 / np.cos(0.3), rtol=1e-5)


def _core_renderer(pipeline):
    """The renderer at the core of a pipeline, i.e. inside any threading, weighting or denoising wrappers"""
    while isinstance(pipeline, renderers.MultipassImageRenderer):
        pipeline = pipeline._subrenderers[0]
    return pipeline


def test_renderer_selection(amr_snap):
    def core_type(**kwargs):
        return type(_core_renderer(renderers.make_render_pipeline(amr_snap.g, **kwargs)))

    assert core_type() is renderers.AMRImageRenderer
    assert core_type(out_units='Msol kpc^-2') is renderers.AMRImageRenderer
    assert core_type(weight='rho') is renderers.AMRImageRenderer
    assert core_type(amr=False) is renderers.ImageRenderer
    # specifying an SPH kernel, or perspective rendering (not yet supported for AMR) implies SPH rendering
    assert core_type(kernel='CubicSpline') is renderers.ImageRenderer
    with pytest.warns(RuntimeWarning, match="z_camera"):
        assert core_type(z_camera=10.0) is renderers.ImageRenderer
    assert core_type(z_camera=10.0, amr=False) is renderers.ImageRenderer
    assert core_type(target='volume', nx=4) is renderers.Grid3dRenderer

    with pytest.raises(ValueError):
        renderers.make_render_pipeline(amr_snap.g, amr=True, kernel='CubicSpline')
    with pytest.raises(ValueError):
        renderers.make_render_pipeline(amr_snap.g, amr=True, target='volume')
    with pytest.raises(NotImplementedError):
        renderers.make_render_pipeline(amr_snap.g, amr=True, z_camera=10.0)

    with pytest.raises(ValueError, match="AMR"):
        renderers.make_render_pipeline(amr_snap.g, amr=False, exact=True)

    amr_snap._amr_cell_families = ()
    assert core_type() is renderers.ImageRenderer
    assert core_type(amr=True) is renderers.AMRImageRenderer


def test_renderer_selection_multiple_families():
    f = pynbody.new(gas=8, dm=8)
    f._amr_cell_families = (pynbody.family.gas,)
    f['pos'] = np.random.default_rng(1).uniform(-1, 1, (16, 3))
    for k in ('smooth', 'rho', 'mass'):
        f[k] = np.ones(16)
    assert type(_core_renderer(renderers.make_render_pipeline(f.g))) is renderers.AMRImageRenderer
    assert type(_core_renderer(renderers.make_render_pipeline(f, kernel='CubicSpline'))) is renderers.ImageRenderer
    with pytest.raises(ValueError, match="sim.gas"):
        renderers.make_render_pipeline(f, amr=True)


def test_renderer_selection_config(amr_snap):
    config = pynbody.config_parser
    original = config.get('sph', 'amr-images')
    try:
        config.set('sph', 'amr-images', 'False')
        pipeline = renderers.make_render_pipeline(amr_snap.g)
        assert type(_core_renderer(pipeline)) is renderers.ImageRenderer
        pipeline = renderers.make_render_pipeline(amr_snap.g, amr=True)
        assert type(_core_renderer(pipeline)) is renderers.AMRImageRenderer
    finally:
        config.set('sph', 'amr-images', original)
    assert type(_core_renderer(renderers.make_render_pipeline(amr_snap.g))) is renderers.AMRImageRenderer


def test_exact_option(amr_snap):
    with amr_snap.rotate_x(30).rotate_y(20):
        args = dict(width=8.0, resolution=50, out_units='Msol kpc^-2')
        pipeline = renderers.make_render_pipeline(amr_snap.g, exact=True, **args)
        assert _core_renderer(pipeline)._exact
        exact = pipeline.render()
        approx = renderers.make_render_pipeline(amr_snap.g, **args).render()
        assert not np.array_equal(exact, approx)
        npt.assert_allclose(approx, exact, atol=1e-3 * exact.max())

        # compare with the core routine, projecting all cells exactly
        g = {k: amr_snap.g[k].view(np.ndarray) for k in ('x', 'y', 'z', 'smooth', 'rho', 'mass')}
        core = _render_amr.render_image_cells(50, 50, g['x'], g['y'], g['z'], g['smooth'], -4.0, 4.0, -4.0, 4.0, 0.0,
                                              g['rho'], g['mass'], g['rho'], 0.0, np.inf, -np.inf, np.inf,
                                              0.0, True, amr_snap.net_rotation_matrix(), tables=[])
        # rho is in Msol kpc^-3 and positions in kpc, so the native units are already Msol kpc^-2
        npt.assert_allclose(exact, core, rtol=1e-6)


def test_orientation_override(amr_snap):
    """If positions are rotated without pynbody's transformations, the orientation can be supplied explicitly"""
    rotation = _rotation(0.4, 0.7)
    args = dict(width=8.0, resolution=50, out_units='Msol kpc^-2')
    with amr_snap.rotate_x(np.degrees(0.4)).rotate_z(np.degrees(0.7)):
        npt.assert_allclose(amr_snap.net_rotation_matrix(), rotation, atol=1e-12)
        reference = renderers.make_render_pipeline(amr_snap.g, **args).render()

    amr_snap['pos'] = amr_snap['pos'] @ rotation.T
    pipeline = renderers.make_render_pipeline(amr_snap.g, threaded=False, **args)
    npt.assert_allclose(_core_renderer(pipeline)._get_orientation(), np.eye(3))
    assert not np.allclose(pipeline.render(), reference, rtol=1e-3)

    _core_renderer(pipeline).set_orientation(rotation)
    npt.assert_allclose(pipeline.render(), reference, rtol=1e-5, atol=1e-6 * reference.max())

    renderer = renderers.AMRImageRenderer(amr_snap.g, orientation=rotation)
    renderer.set_width(8.0)
    renderer.set_resolution(50)
    renderer.set_quantity('rho')
    renderer.set_output_units('Msol kpc^-2')
    npt.assert_allclose(renderer.render(), reference, rtol=1e-5, atol=1e-6 * reference.max())

    with pytest.raises(ValueError):
        renderer.set_orientation(np.eye(2))
    with pytest.raises(ValueError):
        renderer.set_orientation(2 * np.eye(3))


def test_smooth_floor_denoises_slices(amr_snap):
    amr_snap['rho'] = 2.0
    amr_snap['mass'] = amr_snap['rho'] * amr_snap['smooth'] ** 3
    pipeline = renderers.make_render_pipeline(amr_snap.g, width=8.0, resolution=32, smooth_floor=0.8)
    assert isinstance(pipeline, renderers.DenoisedImageRenderer)
    # enlarging the fine cells into overlapping cubes still gives the right value of a uniform field
    npt.assert_allclose(pipeline.render(), 2.0, rtol=1e-5)

    pipeline = renderers.make_render_pipeline(amr_snap.g, width=8.0, resolution=32, smooth_floor=0.8,
                                              out_units='Msol kpc^-2')
    assert not isinstance(pipeline, renderers.DenoisedImageRenderer)
    npt.assert_allclose(pipeline.render().sum() * 0.25 ** 2, amr_snap['mass'].sum(), rtol=1e-5)


def test_copy_on_access(amr_snap):
    """AMR detection and the orientation of the cells should carry over to a copy-on-access snapshot"""
    args = dict(width=8.0, resolution=50, out_units='Msol kpc^-2')
    with amr_snap.rotate_x(30):
        copy = amr_snap.get_copy_on_access_simsnap()
        reference = renderers.make_render_pipeline(amr_snap.g, **args)
        assert type(_core_renderer(reference)) is renderers.AMRImageRenderer
        npt.assert_allclose(copy.net_rotation_matrix(), amr_snap.net_rotation_matrix())
        pipeline = renderers.make_render_pipeline(copy.g, **args)
        assert type(_core_renderer(pipeline)) is renderers.AMRImageRenderer
        npt.assert_allclose(pipeline.render(), reference.render(), rtol=1e-6)

        with amr_snap.rotate_y(20):
            reference = renderers.make_render_pipeline(amr_snap.g, **args).render()
        with copy.rotate_y(20):
            npt.assert_allclose(renderers.make_render_pipeline(copy.g, **args).render(), reference, rtol=1e-6)


def test_amr_renderer_has_no_kernel(amr_snap):
    pipeline = renderers.make_render_pipeline(amr_snap.g, threaded=True)
    with pytest.raises(ValueError):
        pipeline.set_kernel('CubicSpline')
    pipeline.set_kernel(None)
    assert pipeline._kernel is None
    assert all(r._kernel is None for r in pipeline._subrenderers)


def test_amr_renderer_wrappers(amr_snap):
    """Neither approximation nor (by default) denoising should be applied to the AMR renderer, whether or not it
    is threaded"""
    for threaded in (False, True):
        pipeline = renderers.make_render_pipeline(amr_snap.g, threaded=threaded, approximate_fast=True)
        assert not isinstance(pipeline, (renderers.ApproximateImageRenderer, renderers.DenoisedImageRenderer))
        if threaded:
            assert isinstance(pipeline, renderers.ThreadedImageRenderer)
        # explicitly requesting approximation is an error
        with pytest.raises(NotImplementedError):
            pipeline.with_approximate()

    # but denoising can still be forced, and should make no difference since cells tile space
    reference = renderers.make_render_pipeline(amr_snap.g, width=8.0, resolution=32).render()
    denoised = renderers.make_render_pipeline(amr_snap.g, width=8.0, resolution=32, denoise=True)
    assert isinstance(denoised, renderers.DenoisedImageRenderer)
    npt.assert_allclose(denoised.render(), reference, rtol=1e-5)


def test_amr_slice_exact(amr_snap):
    """Every pixel of a slice must equal the value of the cell containing it"""
    im = renderers.make_render_pipeline(amr_snap.g, width=8.0, resolution=64).render()
    # pixel centres; with 64 pixels across 8 kpc no pixel straddles a cell edge
    px = -4.0 + (np.arange(64) + 0.5) / 8
    xx, yy = np.meshgrid(px, px)
    pts = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    pos, sm, rho = (amr_snap.g[k].view(np.ndarray) for k in ('pos', 'smooth', 'rho'))
    # z=0 lies along cell faces, in which case the cells on the +z side are sampled
    inside = (np.all(np.abs(pts[:, None, :2] - pos[None, :, :2]) < sm[None, :, None] / 2, axis=2)
              & (pos[None, :, 2] - sm[None, :] / 2 <= 0) & (pos[None, :, 2] + sm[None, :] / 2 > 0))
    assert np.all(inside.sum(axis=1) == 1)
    npt.assert_allclose(im.ravel(), rho[np.argmax(inside, axis=1)], rtol=1e-6)


def test_amr_slice_on_face_after_roundoff(amr_snap):
    """After a rotation and its inverse, positions carry round-off; a slice on a face must still be clean"""
    reference = renderers.make_render_pipeline(amr_snap.g, width=8.0, resolution=64).render()
    with amr_snap.rotate_x(37).rotate_z(11):
        pass
    im = renderers.make_render_pipeline(amr_snap.g, width=8.0, resolution=64).render()
    npt.assert_allclose(im, reference, rtol=1e-5)


def test_amr_follows_rotation(amr_snap):
    """Cells must stay aligned with the original simulation axes as the snapshot is rotated"""
    reference = renderers.make_render_pipeline(amr_snap.g, width=8.0, resolution=64, out_units='Msol kpc^-2').render()
    with amr_snap.rotate_z(90):
        im = renderers.make_render_pipeline(amr_snap.g, width=8.0, resolution=64, out_units='Msol kpc^-2').render()
    npt.assert_allclose(im, np.rot90(reference, k=-1), rtol=1e-5)

    # under a general rotation, the projected mass is conserved
    with amr_snap.rotate_x(30).rotate_y(20):
        im = renderers.make_render_pipeline(amr_snap.g, width=20.0, resolution=100,
                                            out_units='Msol kpc^-2').render()
    npt.assert_allclose(im.sum() * 0.2 ** 2, amr_snap.g['mass'].sum(), rtol=1e-5)


def test_net_rotation_matrix():
    f = pynbody.new(dm=2)
    f['pos'] = [[0.0, 0, 0], [1.0, 0, 0]]
    npt.assert_allclose(f.net_rotation_matrix(), np.eye(3))
    with f.rotate_x(30).translate([1, 2, 3]).rotate_z(40):
        with f.dm.rotate_y(10):
            # f['pos'][1] - f['pos'][0] is the transformed x unit vector
            npt.assert_allclose(f.net_rotation_matrix()[:, 0], f['pos'][1] - f['pos'][0], atol=1e-12)
            npt.assert_allclose(f.dm.net_rotation_matrix(), f.net_rotation_matrix())
    npt.assert_allclose(f.net_rotation_matrix(), np.eye(3))


@pytest.mark.filterwarnings("ignore:.*namelist:UserWarning")
def test_ramses_image():
    pynbody.test_utils.ensure_test_data_available("ramses")
    f = pynbody.load("testdata/ramses/output_00080")
    f.physical_units()
    cen = f.g['pos'][np.argmax(f.g['rho'])].view(np.ndarray).copy()
    with f.translate(-cen):
        pipeline = renderers.make_render_pipeline(f.g, width=60.0, resolution=100, out_units='Msol kpc^-2')
        assert isinstance(_core_renderer(pipeline), renderers.AMRImageRenderer)
        im = pipeline.render()
        # all cells straddling the image edge are larger than a pixel, so compare to the mass in fully-enclosed
        # cells plus fractions of the straddling ones
        x, y, sm, mass = (f.g[k].view(np.ndarray) for k in ('x', 'y', 'smooth', 'mass'))
        fx = np.clip((np.minimum(x + sm / 2, 30) - np.maximum(x - sm / 2, -30)) / sm, 0, 1)
        fy = np.clip((np.minimum(y + sm / 2, 30) - np.maximum(y - sm / 2, -30)) / sm, 0, 1)
        npt.assert_allclose(im.sum() * 0.6 ** 2, (mass * fx * fy).sum(), rtol=1e-5)

        slice_im = renderers.make_render_pipeline(f.g, width=60.0, resolution=101).render()
        # the central pixel lies entirely within the densest cell
        assert slice_im[50, 50] == pytest.approx(f.g['rho'].max(), rel=1e-5)


@pytest.mark.parametrize("amr", [None, False])
def test_velocity_image_consistent(amr_snap, monkeypatch, amr):
    """The velocity vectors must be rendered in the same way as the underlying image"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import pynbody.plot.imaging

    amr_snap['vel'] = np.ones((len(amr_snap), 3))
    amr_snap['vel'].units = 'km s^-1'
    renderer_types = []
    make_render_pipeline = renderers.make_render_pipeline

    def recording_make_render_pipeline(*args, **kwargs):
        pipeline = make_render_pipeline(*args, **kwargs)
        renderer_types.append(type(_core_renderer(pipeline)))
        return pipeline

    monkeypatch.setattr(renderers, 'make_render_pipeline', recording_make_render_pipeline)
    pynbody.plot.imaging.velocity_image(amr_snap.g, width=8.0, resolution=16, vector_resolution=4, amr=amr)
    plt.close('all')
    expected = renderers.ImageRenderer if amr is False else renderers.AMRImageRenderer
    assert renderer_types == [expected, expected]
