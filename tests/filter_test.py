import numpy as np
import numpy.testing as npt
import pytest

import pynbody


def setup_module():
    global f
    f = pynbody.new(1000)
    f['pos'] = np.random.normal(scale=1.0, size=f['pos'].shape)
    f['vel'] = np.random.normal(scale=1.0, size=f['vel'].shape)
    f['mass'] = np.random.uniform(1.0, 10.0, size=f['mass'].shape)
    f['pos'].units = 'kpc'
    f['vel'].units = 'km s^-1'
    f['mass'].units = 'Msol'


def teardown_module():
    global f
    del f


def test_sphere():
    global f
    sp = f[pynbody.filt.Sphere(0.5)]
    assert (sp.get_index_list(f) == np.where(f['r'] < 0.5)[0]).all()
    assert sp['x'].max() < 0.5
    assert sp['x'].min() > -0.5
    assert sp['r'].max() < 0.5

    sp_units = f[pynbody.filt.Sphere('500 pc')]
    assert len(sp_units.intersect(sp)) == len(sp)


def test_passfilters():
    global f

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


def test_logic():
    global f

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
