def test_import_plot_module():
    import pynbody
    pl_dir = dir(pynbody.plot)
    im = pynbody.plot.sph.image
    assert callable(im)
