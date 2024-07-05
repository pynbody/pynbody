import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("grafic")

@pytest.fixture
def snap():
    return pynbody.load("testdata/grafic_test/")

def test_vel(snap):
    npt.assert_allclose(snap['vel'][::10], [[-14.57081223, 25.25094223, -4.56839371],
                                            [ -7.05367327, -32.75279617,  22.61590385],
                                            [-19.63732147, -24.00996208, -46.54066467],
                                            [-14.16232014,   1.76954174,  -0.32404357],
                                            [ 10.31383133,  21.45753098, -19.05607605],
                                            [ -1.8790369 ,  11.05182552,  28.62895203],
                                            [ 11.11509323,  32.4705925 ,  21.7310524 ]])

def test_pos(snap):
    npt.assert_allclose(snap['pos'][::10], [[8.87648277, 8.98403011, 8.90349648],
                                            [44.56012215, 44.49071605,  8.97691362],
                                            [ 8.86279955, 26.68265913, 26.62181005],
                                            [44.5409237 , 62.41562002, 26.74662813],
                                            [ 8.94368918, 44.63712288, 44.52770704],
                                            [44.57409738,  8.9456823 , 62.48815973],
                                            [ 8.94585317, 62.49853492, 62.46953045]])


def test_iord(snap):
    assert (snap['iord'][::10] == [0, 40, 5, 45, 10, 35, 15]).all()

def test_mass(snap):
    npt.assert_allclose(snap['mass'], 2.157418e+14)

def test_deltab(snap):
    npt.assert_allclose(snap['deltab'][::10], [0.0058372 , -0.00301547, -0.00595138, -0.00416407, -0.01490865,
                                               -0.01760859, -0.00807793], atol=1e-8)

def test_partial_loading(snap):
    np.random.seed(1)  # so that results are reproducible
    f1 = snap
    for test_len in [16, 32, 48]:
        for i in range(5):
            subindex = np.random.permutation(np.arange(0, len(f1)))[:test_len]
            subindex.sort()
            f2 = pynbody.load("testdata/grafic_test/", take=subindex)

            assert (f2['x'] == f1[subindex]['x']).all()
            assert (f2['iord'] == f1[subindex]['iord']).all()
