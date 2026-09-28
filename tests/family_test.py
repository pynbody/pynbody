import numpy as np
import pytest

import pynbody
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gasoline_ahf")


def test_pickle():
    # Regression test for issue 80
    import pickle
    assert pickle.loads(pickle.dumps(pynbody.family.gas)) is pynbody.family.gas

def test_cannot_assign_family_name_or_alias():
    f = pynbody.new(dm=10, gas=10)
    for name in ('dm', 'd', 'dark', 'gas', 'g'):
        with pytest.raises(AttributeError, match="Cannot assign family name or alias"):
            setattr(f, name, 1)
    with pytest.raises(AttributeError):
        f.gas.d = 1

    # the protection should not get in the way of assigning other attributes
    f.some_other_attribute = 1
    assert f.some_other_attribute == 1

    # f.d should still refer to the dark matter
    assert len(f.d) == 10

def test_cannot_assign_new_family_alias():
    pynbody.family.Family("test_family_for_aliases", aliases=["tffa"])
    f = pynbody.new(dm=10)
    with pytest.raises(AttributeError):
        f.tffa = 1

def test_unpickled_family_aliases_are_protected():
    fam = pynbody.family.get_family("test_family_for_pickling", create=True)
    fam.__setstate__({"aliases": ["tffp"]}) # as happens when unpickling creates the family
    assert pynbody.family.is_family_name("tffp")
    with pytest.raises(AttributeError):
        pynbody.new(dm=10).tffp = 1

def test_family_array_dtype() :
    # test for issue #186
    f = pynbody.load('testdata/gasoline_ahf/g15784.lr.01024.gz')
    f.g['rho'] = np.zeros(len(f.g), dtype=np.float32)
    f.s['rho']

def test_family_array_null_slice():
    """Regression test for issue where getting a family array for an IndexedSubSnap containing no members of that family
    - would erroneously return the entire family array"""

    test = pynbody.new(dm=10, star=10, order='dm,star')
    test.star['TestFamilyArray'] = 1.0
    assert len(test[[1,3,5,7]].star)==0 # this always succeeded
    assert len(test[[1,3,5,7]].star['mass'])==0 # this always succeeded
    assert len(test[1:9:2].star['TestFamilyArray'])==0 # this always succeeded
    assert len(test[[1, 3, 5, 11,13]].star['TestFamilyArray']) == 2  # this always succeeded
    assert len(test[[1,3,5,7]].star['TestFamilyArray'])==0 # this would fail

def test_family_array_sim():
    """Test that the simulation of a family array is a family slice"""

    test = pynbody.new(dm=10, star=10)
    test.dm._create_array('mass')
    assert test.dm['mass'].sim == test.dm
