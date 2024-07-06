import numpy as np
import numpy.testing as npt
import pytest

import pynbody


def test_2D_shape():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024.gz")
    h = f.halos()
    pynbody.analysis.halo.center(h[0], mode='hyb')

    rbins, axis_lengths, N, rotation_matrices = pynbody.analysis.halo.shape(h[0].d, nbins=3, rmin=0, rmax=0.004, ndim=2)

    assert rbins.units == f['pos'].units
    assert axis_lengths.units == f['pos'].units


    npt.assert_allclose(rbins, [0.00023014, 0.00094025, 0.00234817], atol=1e-5)
    npt.assert_allclose(axis_lengths, [[0.00023933, 0.0002213],
                                       [0.0009826, 0.00089972],
                                       [0.00255131, 0.0021612]], atol=1e-5)
    assert np.all(N == [51987, 51953, 52368])
    npt.assert_allclose(np.abs(rotation_matrices), [[[0.79139587, 0.61130399],
                                                     [0.61130399, 0.79139587]],
                                                    [[0.81009301, 0.58630139],
                                                     [0.58630139, 0.81009301]],
                                                    [[0.86194519, 0.50700147],
                                                     [0.50700147, 0.86194519]]], atol=1e05)


def test_3D_shape():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024.gz")
    h = f.halos()
    pynbody.analysis.halo.center(h[0], mode='hyb')

    rbins, axis_lengths, N, rotation_matrices = pynbody.analysis.halo.shape(h[0].d, nbins=3, rmin=0, rmax=0.004, ndim=3)

    assert rbins.units == f['pos'].units
    assert axis_lengths.units == f['pos'].units

    npt.assert_allclose(rbins, [0.00031606, 0.00120896, 0.00275828], atol=1e-5)
    npt.assert_allclose(axis_lengths, [[0.00034493, 0.00031589, 0.00028975],
                                       [0.00136191, 0.0012902, 0.00100561],
                                       [0.00306224, 0.00283095, 0.00242072]], atol=1e-5)
    assert np.all(N == [50179, 50298, 50153])

    npt.assert_allclose(abs(rotation_matrices), np.abs( [[[0.57289184, -0.79477302, -0.20032668],
                                             [0.75671193, 0.4189613, 0.50185505],
                                             [-0.31493173, -0.43909825, 0.84143374]],
                                            [[0.14331639, -0.94978421, 0.2781553],
                                             [-0.84688607, -0.2631251, -0.46211381],
                                             [0.51209804, -0.16933736, -0.84206915]],
                                            [[-0.79726644, -0.52493542, 0.29800844],
                                             [-0.55662529, 0.44834656, -0.6993952],
                                             [0.23352626, -0.72348336, -0.649644]]]), atol=1e-5)



def test_halo_shape_wrapper():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024.gz")
    h = f.halos()
    pynbody.analysis.halo.center(h[0], mode='hyb')

    with pytest.warns(DeprecationWarning):
        r, ba, ca, angle, Es = pynbody.analysis.halo.halo_shape(h[0], 3, rin=0, rout=0.004)

    expected_r = [0.00031609, 0.00120895, 0.00275823]
    expected_ba = [0.915934, 0.9474405, 0.92449305]
    expected_ca = [0.83998546, 0.73845282, 0.79051978]
    expected_angle = [0.96076658, 1.42698473, 0.6480433]
    expected_E =  np.array([[[0.57289184, -0.79477302, -0.20032668],
                          [0.75671193, 0.4189613, 0.50185505],
                          [-0.31493173, -0.43909825, 0.84143374]],

                         [[0.14331639, -0.94978421, 0.2781553],
                          [-0.84688607, -0.2631251, -0.46211381],
                          [0.51209804, -0.16933736, -0.84206915]],

                         [[-0.79726644, -0.52493542, 0.29800844],
                          [-0.55662529, 0.44834656, -0.6993952],
                          [0.23352626, -0.72348336, -0.649644]]])


    np.testing.assert_allclose(r, expected_r, atol=1e-5)
    np.testing.assert_allclose(ba, expected_ba, atol=1e-5)
    np.testing.assert_allclose(ca, expected_ca, atol=1e-5)
    np.testing.assert_allclose(angle, expected_angle, atol=1e-5)
    np.testing.assert_allclose(abs(Es), abs(expected_E), atol=1e-5)
