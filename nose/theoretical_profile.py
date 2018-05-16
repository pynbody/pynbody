import pynbody
from pynbody.array import SimArray
import numpy as np
import numpy.testing as npt


def setup():
    global halo_boundary, rs, rhos, mass, c, NFW1, NFW2

    halo_boundary = SimArray(347., units="kpc")
    rs = SimArray(10, units="kpc")
    rhos = SimArray(1e8, units="Msol kpc**-3")
    mass = SimArray(3271229711997.4863, units="Msol")
    c = halo_boundary/rs

    NFW1 = pynbody.analysis.theoretical_profiles.NFWprofile(halo_boundary, density_scale_radius=rhos, scale_radius=rs)

    NFW2 = pynbody.analysis.theoretical_profiles.NFWprofile(halo_boundary, halo_mass=mass, concentration=c)


def test_assignement_nfw():
    """ There are two ways to initialise an NFW profile. Make sure they are equivalent."""

    assert(NFW1['scale_radius'] == rs)
    assert(NFW1['density_scale_radius'] == rhos)
    assert(NFW1['concentration'] == c)
    assert(NFW1.get_enclosed_mass(halo_boundary) == mass)

    assert(NFW2['scale_radius'] == rs)
    assert(NFW2['density_scale_radius'] == rhos)
    assert(NFW2['concentration'] == c)
    assert(NFW2.get_enclosed_mass(halo_boundary) == mass)


def test_functionals_nfw():

    r = SimArray(np.linspace(10, 500, 100), units="kpc")

    rho_from_static = NFW1.profile_functional_static(r, rhos, rs)
    rho_from_instance = NFW1.profile_functional(r)

    truth = SimArray([  2.50000000e+07,   1.07460773e+07,   5.62154816e+06,
            3.31384573e+06,   2.11880566e+06,   1.43727440e+06,
            1.01995927e+06,   7.50047528e+05,   5.67701535e+05,
            4.40058190e+05,   3.48027380e+05,   2.79994777e+05,
            2.28614491e+05,   1.89084016e+05,   1.58172652e+05,
            1.33652241e+05,   1.13951988e+05,   9.79427606e+04,
            8.47986748e+04,   7.39061323e+04,   6.48027682e+04,
            5.71356745e+04,   5.06323136e+04,   4.50799429e+04,
            4.03108479e+04,   3.61916013e+04,   3.26151535e+04,
            2.94949429e+04,   2.67604620e+04,   2.43538877e+04,
            2.22274964e+04,   2.03416640e+04,   1.86633064e+04,
            1.71646535e+04,   1.58222797e+04,   1.46163307e+04,
            1.35299042e+04,   1.25485503e+04,   1.16598657e+04,
            1.08531640e+04,   1.01192041e+04,   9.44996741e+03,
            8.83847317e+03,   8.27862511e+03,   7.76508333e+03,
            7.29315708e+03,   6.85871449e+03,   6.45810642e+03,
            6.08810188e+03,   5.74583322e+03,   5.42874931e+03,
            5.13457550e+03,   4.86127933e+03,   4.60704094e+03,
            4.37022755e+03,   4.14937145e+03,   3.94315081e+03,
            3.75037305e+03,   3.56996035e+03,   3.40093694e+03,
            3.24241803e+03,   3.09360004e+03,   2.95375207e+03,
            2.82220830e+03,   2.69836135e+03,   2.58165640e+03,
            2.47158592e+03,   2.36768506e+03,   2.26952753e+03,
            2.17672188e+03,   2.08890827e+03,   2.00575548e+03,
            1.92695835e+03,   1.85223535e+03,   1.78132654e+03,
            1.71399162e+03,   1.65000822e+03,   1.58917038e+03,
            1.53128714e+03,   1.47618129e+03,   1.42368822e+03,
            1.37365490e+03,   1.32593890e+03,   1.28040761e+03,
            1.23693741e+03,   1.19541297e+03,   1.15572662e+03,
            1.11777779e+03,   1.08147240e+03,   1.04672245e+03,
            1.01344552e+03,   9.81564374e+02,   9.51006589e+02,
            9.21704207e+02,   8.93593413e+02,   8.66614246e+02,
            8.40710329e+02,   8.15828624e+02,   7.91919200e+02,
            7.68935025e+02], 'Msol kpc**-3')

    npt.assert_allclose(rho_from_instance, truth, rtol=1e-5)
    npt.assert_allclose(rho_from_instance, rho_from_static, rtol=1e-5)


def test_fit_nfw():
    r = SimArray(np.linspace(1, 500, 100), units="kpc")

    truth = NFW1.profile_functional_static(r, rhos, rs)
    noise = np.sqrt(truth) * np.random.normal(0, 1, truth.shape)
    noise.units = "Msol kpc**-3"

    easy_fit = truth + noise

    param, cov = pynbody.analysis.theoretical_profiles.NFWprofile.fit(radial_data=r, profile_data=easy_fit,
                                                                      profile_err=np.sqrt(truth),
                                                                      use_analytical_jac=True)

    npt.assert_allclose(param, [rhos, rs], rtol=1e-3)


def test_log_slope_nfw():
    r = SimArray(np.linspace(0.01, 1000, 1000), units="kpc")
    slope_from_instance = NFW1.get_dlogrho_dlogr(r)
    slope_from_static = NFW1.get_dlogrho_dlogr_static(r, rs)

    npt.assert_allclose(slope_from_instance, slope_from_static, rtol=1e-5)

    assert(np.isclose(slope_from_instance[0], -1.0, rtol=1e-2))
    assert(np.isclose(slope_from_instance[-1], -3.0, rtol=1e-2))
    assert(NFW1.get_dlogrho_dlogr_static(rs, rs) == -2.0)