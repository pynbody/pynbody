import numpy as np
import numpy.testing as npt
import pytest

import pynbody


def setup_module():
    # create a dummy gadget file
    global f, h
    f = pynbody.new(dm=2097152)
    f['iord'] = np.arange(2097152)
    f.properties['z']=1.6591479493605812
    f._filename = "testdata/rockstar/snapshot_015"
    h = f.halos()


def test_load_rockstar():
    global f, h
    assert len(h)==5851
    assert isinstance(h, pynbody.halo.RockstarCatalogue)

def test_rockstar_properties():
    global h
    h_properties = h[4977].properties
    assert h_properties['num_p']==40
    npt.assert_allclose(h_properties['pos'], [43.892704, 0.197397, 40.751919], rtol=1e-6)

def test_rockstar_particles():
    global h
    assert (h[4977]['iord']==[1801964, 1802346, 1818475, 1818729, 1818730, 1818857, 1818858, 1818859, 1818986,
                             1834860, 1834986, 1834987, 1835113, 1835114, 1835115, 1835116, 1835242, 1835243,
                             1835244, 1835369, 1835370, 1835371, 1835498, 1835499, 1851372, 1851625, 1851626,
                             1851627, 1851628, 1851754, 1851755, 1851756, 1851884, 1868010, 1868011, 1868012,
                             1884394, 1884395, 1900651, 1933291]).all()

def test_reject_unsuitable_rockstar_files():
    fwrong = pynbody.new(dm=2097152)
    fwrong.properties['z']=0
    with pytest.raises(RuntimeError):
        hwrong = fwrong.halos()
