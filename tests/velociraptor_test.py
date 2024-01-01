import pynbody


def test_swift_velociraptor():
    f = pynbody.load("testdata/SWIFT/snap_0150.hdf5")
    assert pynbody.halo.velociraptor.VelociraptorCatalogue._can_load(f)
    h = pynbody.halo.velociraptor.VelociraptorCatalogue(f)
    assert len(h) == 209

    assert len(h[1]) == 443
    assert len(h[1].dm) == 246
    assert len(h[1].gas) == 197



    h_unbound = pynbody.halo.velociraptor.VelociraptorCatalogue(f, include_unbound=True)
    assert len(h_unbound[1]) == 444

    testvals = [395183, 411313, 394929, 386993, 419631, 402993, 411571, 395313,
          386865, 419505, 411439, 419503, 395059, 395187, 419507, 402995,
          411437, 395057, 395315, 411565, 419635, 403121, 403377, 419633,
          403253, 411443, 411441, 395185, 403123, 403251, 403371, 403507,
          395181, 403509, 419761, 411183, 403117, 411185, 419629, 411697,
          411445, 395053, 411573, 419763, 411317, 403373, 395055, 403255,
          386995, 411435, 411311, 411569, 411309, 403247, 411563, 403125,
          419759, 403245, 395061, 411567, 419500, 394922, 386608, 395050,
          394932, 403124, 411444, 403254, 395182, 411186, 402988, 411566,
          419630, 411312, 386862, 403376, 411442, 419632, 419502, 403244,
          411316, 394928, 386990, 402992, 395056, 394924, 403246, 386994,
          403122, 402990, 395060, 395186, 395054, 386864, 394800, 403374,
          386866, 395184, 403250, 395058, 403248, 386992, 403252, 411564,
          386738, 411436, 411562, 419504, 395310, 403116, 403372, 403370,
          411438, 411568, 411314, 403120, 411310, 419628, 411182, 411570,
          394930, 395314, 411440, 411308, 386736, 402994, 419634, 411306,
          386860, 411434, 395188, 403118, 395180, 394798, 419506, 411184,
          403378, 402986]

    assert (h[20]['iord'] == testvals).all()


def test_swift_velociraptor_parents_and_children():
    f = pynbody.load("testdata/SWIFT/snap_0150.hdf5")
    h = pynbody.halo.velociraptor.VelociraptorCatalogue(f)

    assert h[2].properties['parent'] == -1
    assert (h[2].properties['children'] == [208]).all()
    assert h[208].properties['parent'] == 2
    assert (h[3].properties['children'] == [209]).all()
    assert h[209].properties['parent'] == 3

    # the above were from the genuine velociraptor output. But since these were the only subhalos, the test was
    # a bit too trivial. The following are some faked subhalos to test a more complex scenario

    assert (h[1].properties['children'] == [203, 204, 206]).all()
    assert (h[4].properties['children'] == [205, 207]).all()
    assert (h[203].properties['parent'] == 1)
