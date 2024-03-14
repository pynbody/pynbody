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
    f = pynbody.new(1000)
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

def test_wrapping_sphere(wrapping_snap):
    f, min = wrapping_snap
    # put a sphere at the lower left corner of the box
    sp = f[pynbody.filt.Sphere(0.5, (min, min, min))]

    # auto-wrapping should pick out a fractional volume (4/3 pi) (1/2)^3 ~= 0.52
    # of the unit cube. With the particular random seed here, we get 513 particles
    # of the 1000 in the box, which makes sense

    assert len(sp) == 513

    # now let's switch the wrapping off
    del f.properties['boxsize']
    sp = f[pynbody.filt.Sphere(0.5, (min, min, min))]

    # should be around 1/8th of the particles we had previously, and we actually
    # get 58 which again seems fine
    assert len(sp) == 58


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
    import warnings

    import pylab as p

    subbox = snap[pynbody.filt.Cuboid(origin+0.5, origin+0.5, origin+0.5, origin+1.0, origin+1.0, origin+1.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.clf()
        p.plot(snap['x'], snap['y'], 'k.', alpha=0.1)
        p.plot(subbox['x'], subbox['y'], 'r.')

        p.savefig("snap.pdf")

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


def test_hashing():
    X = {}
    X[pynbody.filt.Sphere('100 kpc')] = 5

    X[pynbody.filt.FamilyFilter(pynbody.family.gas)] = 10
    assert X.get(pynbody.filt.Sphere('100 kpc'), None) == 5
    assert X.get(pynbody.filt.FamilyFilter(pynbody.family.gas),None)==10
    with pytest.raises(KeyError):
        X[pynbody.filt.FamilyFilter(pynbody.family.dm)]
