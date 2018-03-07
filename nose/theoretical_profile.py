import pynbody


def test_nfw():

    NFW = pynbody.analysis.theoretical_profiles.NFWprofile(347, central_density=1e8, scale_radius=10)

    NFW.scale_radius