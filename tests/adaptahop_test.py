import numpy as np
import pytest
from scipy.io import FortranFile as FF

import pynbody
from pynbody.halo.adaptahop import AdaptaHOPCatalogue, NewAdaptaHOPCatalogue


# Note: we do not use a module-wide fixture here to prevent caching of units
@pytest.fixture
def f():
    yield pynbody.load("testdata/output_00080")


@pytest.fixture
def halos(f):
    yield f.halos()


def test_load_adaptahop_catalogue(halos):
    assert len(halos) == halos._headers["nhalos"] + halos._headers["nsubs"]


@pytest.mark.parametrize(
    ("path", "nhalos"),
    (("testdata/output_00080", 170), ("testdata/new_adaptahop_output_00080", 2)),
)
def test_load_one_halo(path, nhalos):
    f = pynbody.load(path)
    halos = f.halos()
    np.testing.assert_allclose(halos[1].properties["members"], halos[1]["iord"])
    assert len(halos) == nhalos


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
    assert halos[1]["mass"].units == "Msol"


def test_physical_conversion_from_snapshot(f):
    # Convert then load
    assert f.dm["mass"].units != "Msol"
    f.physical_units()
    halos = f.halos()

    assert f.dm["mass"].units == "Msol"
    assert halos[1].properties["m"].units == "Msol"
    assert halos[1]["mass"].units == "Msol"

    # Load then convert
    f = pynbody.load("testdata/output_00080")
    assert f.dm["mass"].units != "Msol"

    halos = f.halos()
    f.physical_units()

    assert f.dm["mass"].units == "Msol"
    assert halos[1].properties["m"].units == "Msol"
    assert halos[1]["mass"].units == "Msol"

def test_get_group(f, halos):
    group_array = halos.get_group_array()
    iord = f.dm["iord"]

    for halo_id in range(1, len(halos) + 1):
        mask = group_array == halo_id

        # Check that the indices
        # - read from halo (halos[halo_id]['iord'])
        # - obtained from get_group_array masking
        # are the same (in term of sets)
        iord_1 = np.sort(iord[mask])
        iord_2 = np.sort(halos[halo_id]["iord"])

        np.testing.assert_equal(iord_1, iord_2)


def test_halo_particle_ids(halos):
    with FF(halos._fname, mode="r") as f:
        for halo_id in range(1, len(halos) + 1):
            # Manually read the particle ids and make sure pynbody is reading them as it should
            f._fp.seek(halos[halo_id].properties["file_offset"])

            f.read_ints()  # number of particles
            expected_members = f.read_ints("i")  # halo members

            np.testing.assert_equal(
                expected_members, halos[halo_id].properties["members"]
            )


@pytest.mark.parametrize(
    ("fname", "Halo_T", "ans"),
    (
        (
            "testdata/output_00080/Halos/tree_bricks080",
            AdaptaHOPCatalogue,
            dict(_longint=False, _read_contamination=False),
        ),
        (
            "testdata/new_adaptahop_output_00080/Halos/tree_bricks080",
            NewAdaptaHOPCatalogue,
            dict(_longint=False, _read_contamination=True),
        ),
        (
            "testdata/EDGE_adaptahop_output/tree_bricks047_contam",
            NewAdaptaHOPCatalogue,
            dict(_longint=True, _read_contamination=True),
        ),
        (
            "testdata/EDGE_adaptahop_output/tree_bricks047_nocontam",
            NewAdaptaHOPCatalogue,
            dict(_longint=True, _read_contamination=False),
        ),
    ),
)
def test_longint_contamination_autodetection(f, fname, Halo_T, ans):
    # Note: we hack things a little bit here because
    # we just want to make sure the longint/contamination
    # flags are properly detected.

    halos = Halo_T(f, fname=fname)
    assert halos._longint == ans["_longint"]
    assert halos._read_contamination == ans["_read_contamination"]

def test_halo_iteration(halos):
    h = list(halos)

    assert len(h) == len(halos)
    assert h[0] is halos[1]
    assert h[-1] is halos[len(halos)]
