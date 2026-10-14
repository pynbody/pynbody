"""Tests that the locations from which imaging moved (in version 2.9.0) continue to work"""

import importlib
import warnings

import pytest

import pynbody
import pynbody.imaging
import pynbody.imaging.renderers


@pytest.mark.parametrize("old_name, new_name", [("render_image", "render_image"),
                                                ("render_3d_grid", "render_3d_grid"),
                                                ("render_spherical_image", "render_spherical_image")])
def test_sph_functions_deprecated(old_name, new_name):
    with pytest.warns(DeprecationWarning, match=f"pynbody.imaging.{new_name}"):
        old = getattr(pynbody.sph, old_name)
    assert old is getattr(pynbody.imaging, new_name)


def test_sph_from_import_deprecated():
    with pytest.warns(DeprecationWarning):
        from pynbody.sph import render_image
    assert render_image is pynbody.imaging.render_image


def test_sph_renderers_module_deprecated():
    # importing the old module is silent; using anything from it gives a warning, but returns the same objects
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        old_module = importlib.import_module("pynbody.sph.renderers")
        assert pynbody.sph.renderers is old_module

    with pytest.warns(DeprecationWarning, match="pynbody.imaging.renderers"):
        assert old_module.make_render_pipeline is pynbody.imaging.renderers.make_render_pipeline

    with pytest.warns(DeprecationWarning):
        from pynbody.sph.renderers import AMRImageRenderer, ImageRenderer
    assert AMRImageRenderer is pynbody.imaging.renderers.AMRImageRenderer
    assert ImageRenderer is pynbody.imaging.renderers.ImageRenderer

    with pytest.raises(AttributeError):
        old_module.no_such_name


def test_sph_kernels_unchanged():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from pynbody.sph import kernels
        assert pynbody.imaging.renderers.kernels is kernels
    with pytest.raises(AttributeError):
        pynbody.sph.no_such_name


def test_plot_sph_alias():
    # the old plotting module is (for now) a silent alias
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        import pynbody.plot.sph
        from pynbody.plot.sph import image
        assert image is pynbody.plot.imaging.image is pynbody.plot.image
        assert pynbody.plot.sph.contour is pynbody.plot.imaging.contour
        assert pynbody.plot.sph.velocity_image is pynbody.plot.imaging.velocity_image
        assert "image" in pynbody.plot.sph.__all__
    with pytest.raises(AttributeError):
        pynbody.plot.sph.no_such_name
