"""Tests of perspective rendering (z_camera) of SPH particles"""

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
from pynbody.imaging import renderers


def _single_particle(pos, smooth=0.2):
    f = pynbody.new(gas=1)
    f['pos'] = np.array([pos], dtype=np.float64)
    f['pos'].units = 'kpc'
    f['smooth'] = smooth
    f['smooth'].units = 'kpc'
    f['mass'] = 1.0
    f['mass'].units = 'Msol'
    f['rho'] = 1.0
    f['rho'].units = 'Msol kpc^-3'
    return f


def _render(f, **kwargs):
    args = dict(quantity='rho', width=20.0, resolution=400, z_camera=5.0, threaded=False, approximate_fast=False,
                denoise=False)
    args.update(kwargs)
    return renderers.make_render_pipeline(f, **args).render()


def _moments(image, width=20.0):
    n = image.shape[0]
    g = (np.arange(n) + 0.5) * width / n - width / 2
    X, Y = np.meshgrid(g, g)
    total = image.sum()
    cx, cy = (image * X).sum() / total, (image * Y).sum() / total
    return total * (width / n) ** 2, (cx, cy), X - cx, Y - cy


def test_perspective_projection_along_ray():
    """The projection of a particle seen obliquely is its integral along the rays, which are longer than the
    depth range by sec(theta), and its image is correspondingly elongated away from the centre of the image.

    Before version 2.9.0, the projected kernel was evaluated at the offset in the plane of the particle, rather than
    perpendicular to the ray, so that neither effect was included."""
    f = _single_particle((3.0, 2.0, 0.0))
    image = _render(f, out_units='Msol kpc^-2')
    sec = np.sqrt(1.0 + (3.0 ** 2 + 2.0 ** 2) / 5.0 ** 2)

    # the particle lies in the plane z=0, where the image coordinates are physical; the total is compared with the
    # orthographic projection to remove the effects of sampling the kernel
    total, (cx, cy), dX, dY = _moments(image)
    orthographic_total, _, _, _ = _moments(_render(f, out_units='Msol kpc^-2', z_camera=None))
    assert total / orthographic_total == pytest.approx(sec, rel=2e-3)
    # (under perspective, the image of a sphere is displaced outwards from the projection of its centre, by of order
    # r (h / distance)^2)
    assert (cx, cy) == pytest.approx((3.0, 2.0), abs=1e-2)

    # second moments along and perpendicular to the direction away from the centre of the image
    radial = np.array([3.0, 2.0]) / np.hypot(3.0, 2.0)
    u = dX * radial[0] + dY * radial[1]
    v = -dX * radial[1] + dY * radial[0]
    assert (image * u ** 2).sum() / (image * v ** 2).sum() == pytest.approx(sec ** 2, rel=1e-2)


def test_perspective_projection_on_axis():
    """On the axis of the camera, a perspective projection agrees with the orthographic one, scaled"""
    f = _single_particle((0.0, 0.0, 1.0))
    perspective = _render(f, out_units='Msol kpc^-2')
    # at z=1, the field of view is 4/5 of its width at z=0
    orthographic = _render(f, out_units='Msol kpc^-2', width=16.0, z_camera=None)
    npt.assert_allclose(perspective, orthographic, atol=2e-3 * orthographic.max())


def test_perspective_slice():
    """A perspective slice shows the plane z = z0 as seen from the camera, with pixels at the depth of the plane.

    Before version 2.9.0, the pixels were placed at the depth of each particle instead."""
    f = _single_particle((3.0, 2.0, 1.1), smooth=0.4)
    pipeline = renderers.make_render_pipeline(f, quantity='rho', width=20.0, resolution=400, z_camera=5.0,
                                              threaded=False, approximate_fast=False, denoise=False)
    pipeline.geometry.z_plane = 1.0
    image = pipeline.render()
    _, (cx, cy), _, _ = _moments(image)
    # the plane z=1 is at 4/5 of the distance from the camera of the plane z=0
    assert (cx, cy) == pytest.approx((3.0 / 0.8, 2.0 / 0.8), abs=5e-3)

    # and is equivalent to an orthographic slice through that plane, rescaled
    orthographic = renderers.make_render_pipeline(f, quantity='rho', width=16.0, resolution=400, threaded=False,
                                                  approximate_fast=False, denoise=False)
    orthographic.geometry.z_plane = 1.0
    npt.assert_allclose(image, orthographic.render(), rtol=1e-5, atol=1e-6 * image.max())


def test_perspective_slice_behind_camera():
    f = _single_particle((0.0, 0.0, 6.0))
    pipeline = renderers.make_render_pipeline(f, quantity='rho', width=20.0, resolution=50, z_camera=5.0,
                                              threaded=False, approximate_fast=False, denoise=False)
    pipeline.geometry.z_plane = 6.0
    assert np.all(pipeline.render() == 0)
