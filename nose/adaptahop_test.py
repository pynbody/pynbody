import numpy as np
from scipy.io import FortranFile as FF

import pynbody
from pynbody.halo.adaptahop import AdaptaHOPCatalogue, NewAdaptaHOPCatalogue
from pynbody.array import SimArray


def test_load_adaptahop_catalogue():
    f = pynbody.load("testdata/output_00080")
    h = f.halos()
    assert len(h) == h._headers["nhalos"] + h._headers["nsubs"]


def test_load_one_halo():
    def helper(path, nhalos):
        f = pynbody.load(path)
        h = f.halos()
        np.testing.assert_allclose(h[1].properties["members"], h[1]["iord"])
        assert len(h) == nhalos

    for path, nhalos in (
        ("testdata/output_00080", 170),
        ("testdata/new_adaptahop_output_00080", 2),
    ):
        yield helper, path, nhalos

def test_properties_are_simarrays():
    f = pynbody.load("testdata/output_00080")
    halo = f.halos()[1]

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
        assert isinstance(v, SimArray)
        assert v.sim is f
        # Verify that they are convertible
        v.in_units(unit)


def test_physical_conversion():
    f = pynbody.load("testdata/output_00080")
    halo = f.halos()[1]

    assert halo.properties["m"].units != "Msol"
    halo.physical_units()
    assert halo.properties["m"].units == "Msol"


def test_get_group():
    f = pynbody.load("testdata/output_00080")
    h = f.halos()

    group_array = h.get_group_array()
    iord = f.dm["iord"]

    for halo_id in range(1, len(h) + 1):
        mask = group_array == halo_id

        # Check that the indices
        # - read from halo (h[halo_id]['iord'])
        # - obtained from get_group_array masking
        # are the same (in term of sets)
        iord_1 = np.sort(iord[mask])
        iord_2 = np.sort(h[halo_id]["iord"])

        np.testing.assert_equal(iord_1, iord_2)


def test_halo_particle_ids():
    f = pynbody.load("testdata/output_00080")
    h = f.halos()

    with FF(h._fname, mode="r") as f:
        for halo_id in range(1, len(h) + 1):
            # Manually read the particle ids and make sure pynbody is reading them as it should
            f._fp.seek(h[halo_id].properties["file_offset"])

            f.read_ints()  # number of particles
            expected_members = f.read_ints("i")  # halo members

            np.testing.assert_equal(expected_members, h[halo_id].properties["members"])


def test_longint_contamination_autodetection():
    # Note: we hack things a little bit here because
    # we just want to make sure the longint/contamination
    # flags are properly detected.
    f = pynbody.load("testdata/output_00080")

    answers = {
        ("testdata/output_00080/Halos/tree_bricks080", AdaptaHOPCatalogue): dict(
            _longint=False,
            _read_contamination=False,
        ),
        ("testdata/new_adaptahop_output_00080/Halos/tree_bricks080", NewAdaptaHOPCatalogue): dict(
            _longint=False,
            _read_contamination=True,
        ),
        ("testdata/EDGE_adaptahop_output/tree_bricks047_contam", NewAdaptaHOPCatalogue): dict(
            _longint=True,
            _read_contamination=True,
        ),
        ("testdata/EDGE_adaptahop_output/tree_bricks047_nocontam", NewAdaptaHOPCatalogue): dict(
            _longint=True,
            _read_contamination=False,
        ),
    }

    def tester(fname, Halo_T, ans):
        h = Halo_T(f, fname=fname)
        assert h._longint == ans["_longint"]
        assert h._read_contamination == ans["_read_contamination"]

    for (fname, Halo_T), ans in answers.items():
        yield tester, fname, Halo_T, ans
