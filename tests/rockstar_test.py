import numpy as np
import numpy.testing as npt
import pytest

import pynbody


@pytest.fixture
def dummy_file():
    # create a dummy gadget file
    f = pynbody.new(dm=2097152)
    f['iord'] = np.arange(2097152)
    f.properties['z']=1.6591479493605812
    f._filename = "testdata/rockstar/snapshot_015"

    return f

@pytest.fixture
def rockstar_halos(dummy_file):
    h = pynbody.halo.rockstar.RockstarCatalogue(dummy_file)
    return h

def test_rockstar_single_cpu():
    test = pynbody.halo.rockstar._RockstarCatalogueOneCpu("testdata/rockstar/halos_15.1.bin")
    assert test.halo_min_inclusive == 668
    assert test.halo_max_exclusive == 1426
    assert len(test) == 758
    assert (test.read_iords_for_halo(669) ==
        np.array([729699, 746082, 729443, 745954, 745827, 746083, 778725, 778981,
           729444, 762467, 762212, 778852, 729827, 778979, 746212, 746210,
           729572, 778726, 762595, 762211, 746085, 746086, 762469, 746213,
           745955, 762339, 729701, 778982, 762597, 762468, 778980, 746084,
           762341, 729571, 762596, 729829, 729700, 746211, 745956, 762342,
           729828, 762470, 762340, 745958, 778854, 762598, 745957, 778853,
           746214, 729698, 729570, 778724, 762594, 729573, 762338, 762466,
           778723, 778851])).all()

    props = test.read_properties_for_halo(669)

    assert props['id'] == 669
    assert np.allclose(props['Xoff'], 26.437117)
    assert np.allclose(props['vel'], np.array([  77.035904, -119.406364,  -27.567175]))

def test_load_rockstar(dummy_file):
    h = pynbody.halo.rockstar.RockstarCatalogue(dummy_file)
    assert len(h)==5851
    assert isinstance(h, pynbody.halo.rockstar.RockstarCatalogue)

def test_autodetect_rockstar_from_filename(dummy_file):
    dummy_file._filename = ""
    h = dummy_file.halos(filename="testdata/rockstar/halos_15.0.bin")
    assert isinstance(h, pynbody.halo.rockstar.RockstarCatalogue)

def test_rockstar_properties(rockstar_halos):
    h_properties = rockstar_halos[4977].properties
    assert h_properties['num_p']==40
    npt.assert_allclose(h_properties['pos'], [43.892704, 0.197397, 40.751919], rtol=1e-6)

def test_rockstar_all_properties(rockstar_halos):
    # rockstar reader doesn't generate units
    all_properties = rockstar_halos.get_properties_all_halos(with_units=False)
    assert len(all_properties['num_p'])==5851
    halo_index = rockstar_halos.number_mapper.number_to_index(4977)
    # halo_index will probably be 4977 but we shouldn't/won't take that for granted
    assert all_properties['num_p'][halo_index]==40

    npt.assert_allclose(all_properties['pos'][halo_index], [43.892704, 0.197397, 40.751919], rtol=1e-6)

@pytest.mark.parametrize('load_all', [True, False])
def test_rockstar_particles(rockstar_halos, load_all):
    if load_all:
        rockstar_halos.load_all()
    assert (np.sort(rockstar_halos[4977]['iord'])==[1801964, 1802346, 1818475, 1818729, 1818730, 1818857, 1818858, 1818859, 1818986,
                             1834860, 1834986, 1834987, 1835113, 1835114, 1835115, 1835116, 1835242, 1835243,
                             1835244, 1835369, 1835370, 1835371, 1835498, 1835499, 1851372, 1851625, 1851626,
                             1851627, 1851628, 1851754, 1851755, 1851756, 1851884, 1868010, 1868011, 1868012,
                             1884394, 1884395, 1900651, 1933291]).all()

def test_reject_unsuitable_rockstar_files():
    fwrong = pynbody.new(dm=2097152)
    fwrong.properties['z']=0
    with pytest.raises(RuntimeError):
        hwrong = fwrong.halos()
