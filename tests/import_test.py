def test_import_plot_module():
    import pynbody
    pl_dir = dir(pynbody.plot)
    im = pynbody.plot.imaging.image
    assert callable(im)
