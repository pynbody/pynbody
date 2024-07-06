import numpy as np
import pytest
from scipy.io import FortranFile as FF

import pynbody
import pynbody.test_utils
from pynbody.halo.adaptahop import (
    AdaptaHOPCatalogue,
    BaseAdaptaHOPCatalogue,
    NewAdaptaHOPCatalogue,
    NewAdaptaHOPCatalogueFullyLongInts,
)


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("ramses", "adaptahop_longint")

# Note: we do not use a module-wide fixture here to prevent caching of units
@pytest.fixture
def f():
    yield pynbody.load("testdata/ramses/output_00080")


@pytest.fixture
def halos(f):
    yield AdaptaHOPCatalogue(f)


def test_load_adaptahop_catalogue(halos):
    assert len(halos) == halos._headers["nhalos"] + halos._headers["nsubs"]


@pytest.mark.parametrize(
    ("path", "nhalos", "halo1_len", "halo2_len", "halo1_iord", "halo2_iord"),
    (("testdata/ramses/output_00080", 170, 235, 1201,
      [   48,  7468, 33923],
      [     91,    1203,    2703,    4151,    5539,    6907,    8535,   10227,   11355,
       12739,   14303,   15547,  992896,  993104  ,  993284,  993464,  993608,  993784,
      993932,  994152,  994288,  994452,  994596  ,  994720,  994896,  995060,  995216,
      995352,  995512,  995620,  995772,  995924  ,  996068,  996208,  996336,  996512,
      996624,  996784,  996908,  997080,  997224  ,  997408,  997552,  997692,  997852,
      998008,  998152,  998296,  998428,  998556  ,  998692,  998880,  999036,  999152,
      999300,  999412,  999556,  999700,  999912  , 1000056, 1000200, 1000356, 1000508,
     1000668, 1000796, 1000924, 1001084, 1001264  , 1001420, 1001584, 1001712, 1001892,
     1002064, 1002212, 1002392, 1002552, 1002700  , 1002876, 1003048, 1003204, 1003344,
     1003548, 1003700, 1003828, 1004020, 1004204  , 1004360, 1004536, 1004676, 1004824,
     1005032, 1005188, 1005308, 1005420, 1005560  , 1005716, 1005864, 1006064, 1006188,
     1006308, 1006436, 1006580, 1006712, 1006880  , 1007060, 1007160, 1007308, 1007460,
     1007620, 1007736, 1007872, 1008060, 1008160  , 1008276, 1008404, 1008548, 1008688,
     1010244, 1012248, 1031365, 1033027,]
      ),
     ("testdata/ramses/new_adaptahop_output_00080", 2, 22, 26,
      [852173],
      [762435,  21188,  29809])
     )
)
@pytest.mark.parametrize('load_all', [True, False])
def test_load_halo(path, nhalos, halo1_len, halo2_len, halo1_iord, halo2_iord, load_all):
    f = pynbody.load(path)
    halos = f.halos()
    assert isinstance(halos, BaseAdaptaHOPCatalogue)

    if load_all:
        halos.load_all()

    assert len(halos) == nhalos

    assert len(halos[2]) == halo2_len
    assert (halos[2].dm['iord'][::10] == halo2_iord).all()

    assert len(halos[1]) == halo1_len
    assert (halos[1].dm['iord'][::100] == halo1_iord).all()


def test_properties_are_simarrays(f, halos):
    halo = halos[1]

    # Test these properties exist and have the right dimensions
    # NOTE: we don't check that the units are the same, only
    #       that we can convert to the units below.
    properties = [
        ("m", "Msol"),
        ("pos_x", "kpc"),
        ("pos_y", "kpc"),
        ("pos_z", "kpc"),
        ("vel_x", "km s**-1"),
        ("vel_y", "km s**-1"),
        ("vel_z", "km s**-1"),
        ("angular_momentum_x", "Msol km s**-1 kpc"),
        ("angular_momentum_y", "Msol km s**-1 kpc"),
        ("angular_momentum_z", "Msol km s**-1 kpc"),
        ("max_distance", "kpc"),
        ("shape_a", "kpc"),
        ("shape_b", "kpc"),
        ("shape_c", "kpc"),
        ("kinetic_energy", "km**2 Msol s**-2"),
        ("potential_energy", "km**2 Msol s**-2"),
        ("total_energy", "km**2 Msol s**-2"),
        ("virial_radius", "kpc"),
        ("virial_mass", "Msol"),
        ("virial_temperature", "K"),
        ("virial_velocity", "km s**-1"),
        ("nfw_rho0", "Msol kpc**-3"),
        ("nfw_R_c", "kpc"),
        ("pos", "kpc"),
        ("vel", "km s**-1"),
    ]

    for prop, unit in properties:
        v = halo.properties[prop]
        assert isinstance(v, pynbody.array.SimArray)
        assert v.sim is f
        # Verify that they are convertible
        v.in_units(unit)


def test_physical_conversion_from_halo(f, halos):
    halo1 = halos[1]
    halo2 = halos[2]

    # Make sure the conversion is propagated to the parent
    fields = (
        halo1.properties["m"],
        halo1["mass"],
        f.dm["mass"],
        halo2.properties["m"]
    )
    for field in fields:
        assert field.units != "Msol"
    halo1.physical_units()
    for field in fields:
        assert field.units == "Msol"

    # Get another halo and make sure it is also in physical units
    halo3 = halos[3]
    assert halo3.properties["m"].units == "Msol"


def test_physical_conversion_from_halo_catalogue(f, halos):
    assert f.dm["mass"].units != "Msol"
    halos.physical_units()
    assert f.dm["mass"].units == "Msol"
    assert halos[1].properties["m"].units == "Msol"
    assert halos[1].dm["mass"].units == "Msol"


def test_physical_conversion_from_snapshot(f):
    # Convert then load
    assert f.dm["mass"].units != "Msol"
    f.physical_units()
    halos = f.halos()

    assert f.dm["mass"].units == "Msol"
    assert halos[1].properties["m"].units == "Msol"
    assert halos[1].dm["mass"].units == "Msol"

    # Load then convert
    f = pynbody.load("testdata/ramses/output_00080")
    assert f.dm["mass"].units != "Msol"

    halos = f.halos()
    f.physical_units()

    assert f.dm["mass"].units == "Msol"
    assert halos[1].properties["m"].units == "Msol"
    assert halos[1].dm["mass"].units == "Msol"

def test_get_group(f, halos):
    group_array = halos.get_group_array(family='dm')
    iord = f.dm["iord"]

    for halo_id in range(1, len(halos) + 1):
        mask = group_array == halo_id

        # Check that the indices
        # - read from halo (halos[halo_id]['iord'])
        # - obtained from get_group_array masking
        # are the same (in term of sets)
        iord_1 = np.sort(iord[mask])
        iord_2 = np.sort(halos[halo_id].dm["iord"])

        np.testing.assert_equal(iord_1, iord_2)


def test_halo_particle_ids(halos):
    halos.load_all()
    with FF(halos._fname, mode="r") as f:
        for halo_id in range(1, len(halos) + 1):
            # Manually read the particle ids and make sure pynbody is reading them as it should
            f._fp.seek(halos[halo_id].properties["file_offset"])

            f.read_ints()  # number of particles
            expected_members = f.read_ints("i")  # halo members

            np.testing.assert_equal(
                expected_members, halos[halo_id].dm["iord"]
            )


@pytest.mark.parametrize(
    ("fname", "Halo_T", "ans"),
    (
        (
            "testdata/ramses/output_00080/Halos/tree_bricks080",
            AdaptaHOPCatalogue,
            dict(_longint=False, _read_contamination=False),
        ),
        (
            "testdata/ramses/new_adaptahop_output_00080/Halos/tree_bricks080",
            NewAdaptaHOPCatalogue,
            dict(_longint=False, _read_contamination=True),
        ),
        (
            "testdata/adaptahop_longint/tree_bricks047_contam",
            NewAdaptaHOPCatalogue,
            dict(_longint=True, _read_contamination=True),
        ),
        (
            "testdata/adaptahop_longint/tree_bricks047_nocontam",
            NewAdaptaHOPCatalogue,
            dict(_longint=True, _read_contamination=False),
        ),
        (
            "testdata/adaptahop_longint/tree_bricks100_full_long_ints",
            NewAdaptaHOPCatalogueFullyLongInts,
            dict(_longint=True, _read_contamination=True),
        ),
    ),
)
def test_longint_contamination_autodetection(f, fname, Halo_T, ans):
    # Note: we hack things a little bit here because
    # we just want to make sure the longint/contamination
    # flags are properly detected.

    halos = Halo_T(f, filename=fname)
    assert halos._longint == ans["_longint"]
    assert halos._read_contamination == ans["_read_contamination"]

def test_halo_iteration(halos):
    h = list(halos)

    assert len(h) == len(halos)
    assert h[0] is halos[1]
    assert h[-1] is halos[len(halos)]

def test_dm_not_first_family(f):
    # AdaptaHOP files only refer to DM particles, but we can't assume that DM is the first family
    # e.g. tracer particles come first

    f = pynbody.load("testdata/ramses/new_adaptahop_output_00080")
    f_with_tracers = pynbody.new(gas_tracer=100, dm=len(f.dm), star=len(f.star), gas=len(f.gas))
    f_with_tracers.dm['iord'] = f.dm['iord']
    f_with_tracers.properties.update(f.properties)

    halos = f.halos()

    halos2 = pynbody.halo.adaptahop.NewAdaptaHOPCatalogue(f_with_tracers,
                                                          filename="testdata/ramses/new_adaptahop_output_00080/Halos/tree_bricks080")

    assert (halos2[1].dm['iord'] == halos[1].dm['iord']).all()
    assert (halos2[2].dm['iord'] == halos[2].dm['iord']).all()
