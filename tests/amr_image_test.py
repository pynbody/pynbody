"""Tests for rendering AMR cells as exact cubes (CellKernel)"""

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils
from pynbody.sph import _render, kernels, renderers


def _cube_of_cells(n_sub, size, centre, rotation=np.eye(3)):
    """Positions of n_sub^3 cells exactly tiling a cube of the given size, rotated about its centre"""
    g = (np.arange(n_sub) + 0.5) * size / n_sub - size / 2
    u = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T
    return u @ rotation.T + centre


def _render_cells(pos, size, qty, rotation, projected, nx=100, width=4.0, z0=0.0, z_lo=-np.inf, z_hi=np.inf):
    pos = np.atleast_2d(pos)
    n = len(pos)
    sm = np.full(n, float(size))
    qty = np.broadcast_to(np.asarray(qty, dtype=np.float64), (n,)).copy()
    mass = sm ** 3
    rho = np.ones(n)
    return _render.render_image_cells(nx, nx, pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy(), sm,
                                      -width / 2, width / 2, -width / 2, width / 2, z0,
                                      qty, mass, rho, 0.0, np.inf, z_lo, z_hi, 0.0, projected, rotation)


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
    f._gas_particles_are_amr_cells = True
    return f


@pytest.mark.parametrize("projected", [True, False])
@pytest.mark.parametrize("rotated", [True, False])
def test_cells_tile_exactly(projected, rotated):
    """A cube decomposed into subcells must render identically to the cube itself"""
    rotation = _rotation(0.4, 0.7) if rotated else np.eye(3)
    centre = np.array([0.125, -0.0625, 0.25])
    big = _render_cells(centre, 1.6, 1.0, rotation, projected, z_lo=-0.3, z_hi=0.2)
    small = _render_cells(_cube_of_cells(8, 1.6, centre, rotation), 0.2, 1.0, rotation, projected,
                          z_lo=-0.3, z_hi=0.2)
    assert big.max() > 0.1
    npt.assert_allclose(small, big, atol=1e-5)


@pytest.mark.parametrize("rotated", [True, False])
def test_cell_projection_conserves_mass(rotated):
    rotation = _rotation(0.4, 0.7) if rotated else np.eye(3)
    im = _render_cells([0.1, -0.2, 0.05], 1.3, 2.0, rotation, True)
    # rotated cells are supersampled where pixels straddle their projected edges, so are not quite exact
    npt.assert_allclose(im.sum() * (4.0 / 100) ** 2, 2.0 * 1.3 ** 3, rtol=1e-4 if rotated else 1e-6)


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


def test_default_kernel_selection(amr_snap):
    assert isinstance(renderers.make_render_pipeline(amr_snap.g)._kernel, kernels.CellKernel)
    assert isinstance(renderers.make_render_pipeline(amr_snap.g, kernel='CubicSpline')._kernel,
                      kernels.CubicSplineKernel)
    assert isinstance(renderers.make_render_pipeline(amr_snap.g, target='volume', nx=4)._kernel,
                      kernels.CubicSplineKernel)
    # perspective images are not yet supported, so fall back to SPH
    assert isinstance(renderers.make_render_pipeline(amr_snap.g, z_camera=10.0)._kernel, kernels.CubicSplineKernel)
    with pytest.raises(ValueError):
        renderers.make_render_pipeline(amr_snap.g, target='volume', kernel='cell')

    del amr_snap._gas_particles_are_amr_cells
    assert isinstance(renderers.make_render_pipeline(amr_snap.g)._kernel, kernels.CubicSplineKernel)


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
        assert isinstance(pipeline._subrenderers[0]._kernel, kernels.CellKernel)
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
