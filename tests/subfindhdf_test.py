import warnings

import numpy as np
import numpy.testing as npt
import pytest

import pynbody

# load Alan Duffy's module from https://bitbucket.org/astroduff/pyreadgadget
from pynbody.test_utils.pyread_gadget_hdf5 import pyread_gadget_hdf5


@pytest.fixture
def snap():
    return pynbody.load('testdata/gadget3/data/subhalos_103/subhalo_103')

@pytest.mark.parametrize('load_all', (True, False))
def test_halo_loading(snap, load_all) :
    """ Check that halo loading works """
    h = pynbody.halo.subfindhdf.SubFindHDFHaloCatalogue(snap)

    if load_all:
        h.load_all()


    assert len(h) == 4226
    assert len(h[0]) == 15682
    assert len(h[10]) == 2052
    assert (h[0]['iord'][::100] == [1992767, 1910973, 1977535, 1878091, 1829826, 1796672, 1994317,
          1977404, 1846089, 1993277,   26437,   26444, 2091972, 1961924,
          2026691, 1879228,   43338, 1994823,    9929, 1878477, 1975888,
          2025924, 2073669, 1811915,   59852, 1993403, 1895483, 2010435,
          1779272,   43725, 1796425, 2024514, 2043206, 1960125, 1959244,
          1977162, 1926859, 1943632, 2057932, 2026705,   10062, 2058055,
          1975238, 2023881, 1893316, 4074692, 3991240, 4057412, 4040134,
          4156105, 3976261, 4156366, 4041918, 4008000, 4090186, 4122316,
          3959240, 4139596, 4058693, 4008253, 4188876, 4090959, 4154698,
          4106436, 3975619, 4058439, 4188104, 4107217, 4170954, 3943750,
          4089420, 3926590, 4188108, 3976772, 3925836, 4090449, 4073937,
          3992650, 4140492, 2107470, 4171336, 4058952, 3958859, 4058043,
          2124365, 4089929, 4073282, 3893191, 4107334, 4172356, 3991626,
          4074172, 4140357, 4009019, 4139841, 4042173, 4072779, 4090556,
          4139079, 3958478, 4107455, 4123204, 3860548, 4009276, 4042560,
          3925315, 4057148, 3975119, 4188097, 4123582, 4057149, 4189386,
          4072391, 4009790, 3894209, 4022986, 2140232, 3976894, 4139466,
          4139720, 4172109, 4137416, 3992521, 1895749, 1961287, 1943873,
          2009540, 1895362, 1878718,   10951, 1894084, 1942726, 1878852,
          1861185, 2025542, 2010179, 2090444, 1961408, 1910854, 1861443,
          1878342, 1912000, 1862595, 2059211, 1942987, 1976655, 1994058,
          2042696, 1992905, 2092106, 2074698, 2026952, 1944013,   10185,
          1960656, 1912012, 2090949]).all()
    assert (h[10]['iord'][::100] == [ 782722,  865539,  865923,  784260,  849150,  750470,  898950,
           866689,  882560, 2913154, 2865029, 2880647, 2896644, 2995974,
          2913929, 2881289, 2896517, 2995969,  832898,  865671,  832639]).all()

    assert h[0].properties['NsubPerHalo'] == 6
    assert np.all(h[0].properties['children'] == [0, 1, 2, 3, 4, 5])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Accessing multiple halos")
        for i, halo in enumerate(h[0:10]):
            halo['mass'].sum()
            for fam in [halo.g, halo.d, halo.s]:
                assert (len(fam['iord']) ==
                        snap._hdf_files[0][snap._family_to_group_map[fam.families()[0]][0]]['Length'][i])

def test_subhalos(snap):
    """Test that subhalos can be accessed from each parent FOF group, as well as directly through a subs object"""
    h = pynbody.halo.subfindhdf.SubFindHDFHaloCatalogue(snap)
    subs = pynbody.halo.subfindhdf.SubFindHDFHaloCatalogue(snap, subhalos=True)
    h.load_all()

    assert (h[0].subhalos[0]['iord'] == subs[0]['iord']).all()
    assert (h[1].subhalos[1]['iord'] == subs[7]['iord']).all()

    with pytest.warns(DeprecationWarning):
        assert (h[1].sub[1]['iord'] == subs[7]['iord']).all()

def test_deprecated_subs_keyword(snap):
    with pytest.warns(DeprecationWarning):
        new_halos = snap.halos(subs=True)
        assert isinstance(new_halos, pynbody.halo.subfindhdf.SubFindHDFHaloCatalogue)
        assert len(new_halos) == 3294

def test_finds_correct_halo(snap):
    h = snap.halos()
    assert isinstance(h, pynbody.halo.subfindhdf.SubFindHDFHaloCatalogue)
    h = snap.halos(subhalos=True)
    assert isinstance(h, pynbody.halo.subfindhdf.SubFindHDFHaloCatalogue)


def test_grp_array(snap):
    h = snap.halos()
    grp = h.get_group_array()
    for i in range(0,100,10):
        assert len(snap['iord'][grp==i]) == len(h[i])
        assert (h[i]['iord'] == snap['iord'][grp==i]).all()

def test_fof_vs_sub_assignment(snap):
    h = snap.halos()
    file_mass_unit = snap.infer_original_units("g")
    npt.assert_allclose(h.get_dummy_halo(0).properties['Mass'].in_units(file_mass_unit),
                        28.604694074339932)
    npt.assert_allclose(h.get_dummy_halo(0).properties['Halo_M_Crit200'].in_units(file_mass_unit),
                        29.796955896599684)
    npt.assert_allclose(h.get_dummy_halo(1).properties['Mass'].in_units(file_mass_unit),
                        8.880245794949587)
    npt.assert_allclose(h.get_dummy_halo(1).properties['Halo_M_Crit200'].in_units(file_mass_unit),
                        8.116568749712314)

@pytest.mark.parametrize('subhalos', (True, False))
@pytest.mark.parametrize('with_units', (True, False))
def test_properties_all_halos(snap, subhalos, with_units):
    h = snap.halos(subhalos=subhalos)
    properties = h.get_properties_all_halos(with_units=with_units)

    filesub = 'testdata/gadget3/data/subhalos_103/subhalo_103'
    dir = 'subfind' if subhalos else 'fof'
    FoF_Mass = pyread_gadget_hdf5(filesub + '.0.hdf5', 10, 'Mass', sub_dir=dir, nopanda=True, silent=True)
    FoF_CoM = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'CenterOfMass', sub_dir=dir, nopanda=True, silent=True)

    npt.assert_allclose(FoF_Mass.ravel(), properties['Mass'])
    npt.assert_allclose(FoF_CoM, properties['CenterOfMass'])

    if not subhalos:
        for i in range(5):
            assert (h[i].properties['children'] == properties['children'][i]).all()

    if with_units:
        npt.assert_allclose(properties['Mass'].units.ratio(snap.infer_original_units("g")), 1.0)
        npt.assert_allclose(properties['CenterOfMass'].units.ratio(snap.infer_original_units("cm")), 1.0)
    else:
        assert not hasattr(properties['Mass'], 'units')

@pytest.mark.filterwarnings("ignore:Accessing multiple halos")
@pytest.mark.parametrize('load_all', (True, False))
def test_halo_values(snap, load_all) :
    """ Check that halo values (and sizes) agree with pyread_gadget_hdf5 """

    filesub = 'testdata/gadget3/data/subhalos_103/subhalo_103'



    FoF_Mass = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'Mass', sub_dir='fof', nopanda=True, silent=True)
    FoF_MassType = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'MassType', sub_dir='fof', nopanda=True, silent=True)
    Sub_Mass = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'Mass', sub_dir='subfind', nopanda=True, silent=True)
    Sub_MassType = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'MassType', sub_dir='subfind', nopanda=True, silent=True)
    NsubPerHalo = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'NsubPerHalo', sub_dir='subfind', nopanda=True, silent=True)
    OffsetHalo = np.roll(NsubPerHalo.cumsum(), 1)
    OffsetHalo[0]=0 ## To start counter

    h = snap.halos()
    if load_all:
        h.load_all()

    FoF_CoM = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'CenterOfMass', sub_dir='fof', nopanda=True, silent=True)
    Sub_CoM = pyread_gadget_hdf5(filesub+'.0.hdf5', 10, 'CenterOfMass', sub_dir='subfind', nopanda=True, silent=True)

    # Check the Halo Array values
    for i,halo in enumerate(h[0:10]) :
        assert(np.allclose(halo.properties['CenterOfMass'], FoF_CoM[i], rtol=1e-3))

        for j, s in enumerate(halo.subhalos) :
            assert(np.allclose(s.properties['CenterOfMass'], Sub_CoM[OffsetHalo[i]+j], rtol=1e-3))

    ###
    # Test the Halo particle information
    ###

    # Mass of each component for FOF halos
    for i,halo in enumerate(h[0:10]) :
        assert(np.allclose(halo.g['mass'].sum(), FoF_MassType[i,0], rtol=1e-3))
        assert(np.allclose(halo.dm['mass'].sum(), FoF_MassType[i,1], rtol=1e-3))
        assert(np.allclose(halo.s['mass'].sum(), FoF_MassType[i,4], rtol=1e-3))
        assert(np.allclose(halo['mass'].sum(), FoF_Mass[i], rtol=1e-3))

    # Mass of each component for Subhalos
    for i,halo in enumerate(h[0:10]) :
        for j, s in enumerate(halo.subhalos) :
            assert(np.allclose(s.g['mass'].sum(), Sub_MassType[OffsetHalo[i]+j,0], rtol=1e-3))
            assert(np.allclose(s.dm['mass'].sum(), Sub_MassType[OffsetHalo[i]+j,1], rtol=1e-3))
            assert(np.allclose(s.s['mass'].sum(), Sub_MassType[OffsetHalo[i]+j,4], rtol=1e-3))
            assert(np.allclose(s['mass'].sum(), Sub_Mass[OffsetHalo[i]+j], rtol=1e-3))

    FoF_Temp = pyread_gadget_hdf5(filesub+'.0.hdf5', 0, 'Temperature', sub_dir='fof', nopanda=True, silent=True, physunits=True)[:, 0]
    FoF_Length = pyread_gadget_hdf5(filesub+'.0.hdf5', 0, 'Length', sub_dir='fof', nopanda=True, silent=True, physunits=True).astype(int)[:, 0]
    FoF_Offset = pyread_gadget_hdf5(filesub+'.0.hdf5', 0, 'Offset', sub_dir='fof', nopanda=True, silent=True, physunits=True).astype(int)[:, 0]

    # Test the Particle Temperature and implicitly the particle ordering
    for i,halo in enumerate(h[0:10]) :
        npt.assert_allclose(
            halo.g['temp'],
            FoF_Temp[FoF_Offset[i]:FoF_Offset[i]+FoF_Length[i]],
        )

@pytest.mark.parametrize('load_all', (True, False))
def test_halo_properties_physical_units(snap, load_all):
    h = snap.halos()
    if load_all:
        h.load_all()
    h.physical_units()
    npt.assert_allclose(h[0].properties['CenterOfMass'], [1242.674894, 1571.460534, 2232.622292])
