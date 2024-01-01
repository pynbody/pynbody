import pynbody

def test_swift_velociraptor():
    # TODO: replace with a small enough file to include in test data
    f = pynbody.load("...")
    assert pynbody.halo.velociraptor.VelociraptorCatalogue._can_load(f)
    h = pynbody.halo.velociraptor.VelociraptorCatalogue(f)
    assert len(h) == 100419

    assert len(h[0]) == 152690
    assert len(h[0].dm) == 81012
    assert len(h[0].gas) == 61568
    assert len(h[0].st) == 9963
    assert len(h[0].bh) == 147

    h_unbound = pynbody.halo.velociraptor.VelociraptorCatalogue(f, include_unbound=True)
    assert len(h_unbound[0]) == 156836


def test_swift_velociraptor_parents_and_children():
    # TODO: replace with a small enough file to include in test data
    f = pynbody.load("...")
    h = pynbody.halo.velociraptor.VelociraptorCatalogue(f)

    assert h[100].properties['parent'] == -1
    assert h[100].properties['children']

