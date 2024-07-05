import numpy as np
import numpy.testing as npt
import pytest

import pynbody


@pytest.fixture
def snap():
    f = pynbody.new(1000)
    np.random.seed(1337)
    f['pos'] = np.random.normal(scale=1.0, size=f['pos'].shape)
    f['vel'] = np.random.normal(scale=1.0, size=f['vel'].shape)
    f['mass'] = np.random.uniform(1.0, 10.0, size=f['mass'].shape)
    f['pos'].units = 'kpc'
    f['vel'].units = 'km s^-1'
    f['mass'].units = 'Msol'
    return f

@pytest.fixture(params=[-0.5, 0.0], ids=["centred", "zero-origin"])
def wrapping_snap(request):
    f = pynbody.new(5000)
    min = request.param
    np.random.seed(1337)
    f['pos'] = np.random.uniform(0, 1.0, size=f['pos'].shape) + min
    f['vel'] = np.random.normal(scale=1.0, size=f['vel'].shape)
    f['mass'] = np.random.uniform(1.0, 10.0, size=f['mass'].shape)
    f['pos'].units = 'kpc'
    f['vel'].units = 'km s^-1'
    f['mass'].units = 'Msol'
    f.properties['boxsize'] = 1.0
    return f, min




def test_sphere(snap):
    f = snap
    sp = f[pynbody.filt.Sphere(0.5)]
    assert (sp.get_index_list(f) == np.where(f['r'] < 0.5)[0]).all()
    assert sp['x'].max() < 0.5
    assert sp['x'].min() > -0.5
    assert sp['r'].max() < 0.5

    sp_units = f[pynbody.filt.Sphere('500 pc')]
    assert len(sp_units.intersect(sp)) == len(sp)


def test_empty_sphere():
    snap = pynbody.new(0)
    # This would fail due to the C code seeing an array it couldn't handle (due to zero length)
    sp_empty = snap[pynbody.filt.Sphere(1.0)]
    assert len(sp_empty) == 0

def test_wrapping_sphere(wrapping_snap):
    f, min = wrapping_snap
    # put a sphere at the lower left corner of the box
    sp = f[pynbody.filt.Sphere(0.5, (min, min, min))]

    f['pos']-=np.array([min, min, min]) # put our sphere at the origin and re-wrap
    f.wrap()
    f.wrap()

    sphere_unwrapped = f[pynbody.filt.Sphere(0.5)]
    assert (sphere_unwrapped.get_index_list(f) == sp.get_index_list(f)).all()



def test_passfilters(snap):
    f = snap

    hp = f[pynbody.filt.HighPass('mass', 5)]
    lp = f[pynbody.filt.LowPass('mass', 5)]
    bp = f[pynbody.filt.BandPass('mass', 2, 7)]

    assert len(hp) > 0
    assert len(lp) > 0
    assert len(bp) > 0

    assert len(hp.intersect(lp)) == 0

    assert (hp.get_index_list(f) == np.where(f['mass'] > 5)[0]).all()
    assert (lp.get_index_list(f) == np.where(f['mass'] < 5)[0]).all()
    assert (bp.get_index_list(f) == np.where(
        (f['mass'] > 2) * (f['mass'] < 7))[0]).all()

@pytest.mark.parametrize("with_kdtree", [True, False])
def test_cuboid(snap, with_kdtree):
    f = snap
    x1s = np.random.uniform(-0.5, 0.0, 10)
    x2s = np.random.uniform(0.0, 0.5, 10)
    y1s = np.random.uniform(-0.5, 0.0, 10)
    y2s = np.random.uniform(0.0, 0.5, 10)
    z1s = np.random.uniform(-0.5, 0.0, 10)
    z2s = np.random.uniform(0.0, 0.5, 10)

    if with_kdtree:
        f.build_tree()

    for x1, x2, y1, y2, z1, z2 in zip(x1s, x2s, y1s, y2s, z1s, z2s):
        c = f[pynbody.filt.Cuboid(x1, y1, z1, x2, y2, z2)]
        assert (c.get_index_list(f) == np.where(
            (f['x'] > x1) * (f['x'] < x2) * (f['y'] > y1) * (f['y'] < y2) * (f['z'] > z1) * (f['z'] < z2))[0]).all()

def test_wrapping_cuboid(wrapping_snap):
    snap, origin = wrapping_snap

    # NB because 'wrapping' in pynbody only takes place once, if any boundaries get more than one wrap distance
    # away from the origin, they will not be included in the wrapped box. This is a limitation of the current
    # implementation that has been chosen for efficiency. It is likely to be completely fine for all realistic
    # use cases. But it also means that choosing any combination of box_corner and box_size can result in a failure. The test avoids such cases.
    for box_corner in ((0.5,0.5,0.5), (0.3, 0.5, 0.5), (0.2,-0.2,0.0)):
        for box_size in ((0.7, 0.7, 0.5), (0.9, 0.3, 0.1), (0.2, 0.8, 0.2), (0.2, -0.2, 0.2)):
            # nb the last box_size being 'negative' (-0.2) should be equivalent to +0.8, tested via the %1.0 in the
            # 'unwrapped' box
            subbox = snap[pynbody.filt.Cuboid(origin+box_corner[0], origin+box_corner[1], origin+box_corner[2],
                                              *(origin+c+s for c,s in zip(box_corner, box_size)))]

            snap['pos']-=origin+np.array(box_corner)+0.5 # put our cuboid at the bottom left
            snap.wrap()
            snap.wrap() # wrap twice in case any particles have ended up more than one wrap distance away
            # (the limitation mentioned above in not being aable to move particles more than one wrap distance
            # within the filter evaluation is a separate issue from this)

            subbox_unwrapped = snap[pynbody.filt.Cuboid(-0.5, -0.5, -0.5,
                                                        box_size[0]%1.0-0.5, box_size[1]%1.0-0.5, box_size[2]%1.0-0.5)]

            assert (subbox_unwrapped.get_index_list(snap) == subbox.get_index_list(snap)).all()


def test_logic(snap):
    f = snap

    comp = f[pynbody.filt.BandPass('mass', 2, 7)]

    and_test = f[
        pynbody.filt.HighPass('mass', 2) & pynbody.filt.LowPass('mass', 7)]

    assert and_test == comp

    comp = f[pynbody.filt.LowPass('mass', 2)]
    or_test = f[
        (pynbody.filt.LowPass('mass', 1) | pynbody.filt.BandPass('mass', 1, 2))]

    assert or_test == comp

    comp = f[pynbody.filt.BandPass('mass', 2, 7)]
    not_test = f[~(pynbody.filt.BandPass('mass', 2, 7))]

    assert comp.union(not_test) == f
    assert len(comp.intersect(not_test)) == 0

    assert len(comp)+len(not_test)==len(f)
    assert len(comp)!=0
    assert len(not_test)!=0

def test_family_filter():
    f = pynbody.new(dm=100,gas=100)
    f_dm = f.dm
    f_dm_filter = f[pynbody.filt.FamilyFilter(pynbody.family.dm)]
    f_gas = f.gas
    f_gas_filter = f[pynbody.filt.FamilyFilter(pynbody.family.gas)]
    assert (f_dm.get_index_list(f) == f_dm_filter.get_index_list(f)).all()
    assert (f_gas.get_index_list(f) == f_gas_filter.get_index_list(f)).all()

def test_annulus(snap):
    a = snap[pynbody.filt.Annulus(0.5, 1.0)]
    assert len(a)>100 and len(a)<200
    assert (a.get_index_list(snap) == np.where(
        (snap['r'] > 0.5) * (snap['r'] < 1.0))[0]).all()

def test_disc(snap):
    d = snap[pynbody.filt.Disc(0.8, 0.3)]
    assert len(d)>50 and len(d)<100
    assert (d.get_index_list(snap) == np.where(
        (snap['rxy'] < 0.8) * (abs(snap['z']) < 0.3))[0]).all()

def test_solar_neighbourhood(snap):
    filt = pynbody.filt.SolarNeighborhood(0.5, 1.0, 0.3)
    sn = snap[filt]
    assert len(sn) > 50 and len(sn) < 100
    assert (sn.get_index_list(snap) == np.where(
        (snap['rxy'] > 0.5) * (snap['rxy'] < 1.0) * (abs(snap['z']) < 0.3))[0]).all()

def test_hashing():
    X = {}
    X[pynbody.filt.Sphere('100 kpc')] = 5

    X[pynbody.filt.FamilyFilter(pynbody.family.gas)] = 10
    assert X.get(pynbody.filt.Sphere('100 kpc'), None) == 5
    assert X.get(pynbody.filt.FamilyFilter(pynbody.family.gas),None)==10
    with pytest.raises(KeyError):
        X[pynbody.filt.FamilyFilter(pynbody.family.dm)]
