import pynbody
import numpy as np
import numpy.testing as npt


def setup():
    global halo_boundary, rs, rhos, mass, c, NFW1, NFW2

    halo_boundary = 347.
    rs = 10
    rhos = 1e8
    mass = 26161.289818949765
    c = halo_boundary/rs

    NFW1 = pynbody.analysis.theoretical_profiles.NFWprofile(halo_boundary, central_density=rhos, scale_radius=rs)

    NFW2 = pynbody.analysis.theoretical_profiles.NFWprofile(halo_boundary, halo_mass=mass, concentration=c)


def test_assignement_nfw():

    assert(NFW1['scale_radius'] == rs)
    assert(NFW1['central_density'] == rhos)
    assert(NFW1['concentration'] == c)
    assert(NFW1.get_enclosed_mass(halo_boundary) == mass)

    assert(NFW2['scale_radius'] == rs)
    assert(NFW2['central_density'] == rhos)
    assert(NFW2['concentration'] == c)
    assert(NFW2.get_enclosed_mass(halo_boundary) == mass)


def test_functionals_nfw():

    r = np.linspace(10, 500, 100)

    rho_from_static = NFW1.profile_functional_static(r, rhos, rs)
    rho_from_instance = NFW1.profile_functional(r)

    truth = np.array([])

    npt.assert_allclose(rho_from_instance, truth, rtol=1e-5)
    npt.assert_allclose(rho_from_instance, rho_from_static, rtol=1e-5)

    npt.assert_allclose(NFW1.get_dlogrho_dlogr_static(r, rs), truth, rtol=1e-5)


def test_fit_nfw():
    pass