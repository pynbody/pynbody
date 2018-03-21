import pynbody
import numpy as np
import numpy.testing as npt


def test_nfw():

    halo_boundary = 347.
    rs = 10
    rhos = 1e8

    NFW = pynbody.analysis.theoretical_profiles.NFWprofile(halo_boundary, central_density=rhos, scale_radius=rs)

    assert(NFW['scale_radius'] == rs)
    assert(NFW['central_density'] == rhos)
    assert(NFW['concentration'] == halo_boundary/rs)

    r = np.linspace(10, 500, 1000)

    rho_from_static = NFW.profile_functional_static(r, rhos, rs)
    rho_from_instance = NFW.profile_functional(r)

    npt.assert_allclose(rho_from_instance, rho_from_static, rtol=1e-5)

    assert(NFW.get_enclosed_mass(halo_boundary) == 26161.289818949765)

    truth = np.array([])
    npt.assert_allclose(NFW.get_dlogrho_dlogr_static(r, rs), truth, rtol=1e-5)


def test_fit_nfw():
    pass