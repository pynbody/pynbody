import numpy as np
import numpy.testing as npt
import pytest

import pynbody


def setup_module():
    global f
    f = pynbody.load("testdata/grafic_test/")

def test_vel():
    npt.assert_allclose(f['vel'][::10], [[-14.57081223,  25.25094223,  -4.56839371],
          [ -7.05367327, -32.75279617,  22.61590385],
          [-19.63732147, -24.00996208, -46.54066467],
          [-14.16232014,   1.76954174,  -0.32404357],
          [ 10.31383133,  21.45753098, -19.05607605],
          [ -1.8790369 ,  11.05182552,  28.62895203],
          [ 11.11509323,  32.4705925 ,  21.7310524 ]])

def test_pos():
    npt.assert_allclose(f['pos'][::10], [[ 8.87648277,  8.98403011,  8.90349648],
          [44.56012215, 44.49071605,  8.97691362],
          [ 8.86279955, 26.68265913, 26.62181005],
          [44.5409237 , 62.41562002, 26.74662813],
          [ 8.94368918, 44.63712288, 44.52770704],
          [44.57409738,  8.9456823 , 62.48815973],
          [ 8.94585317, 62.49853492, 62.46953045]])


def test_iord():
    assert (f['iord'][::10]==[ 0, 40,  5, 45, 10, 35, 15]).all()

def test_mass():
    npt.assert_allclose(f['mass'], 2.15729774e+14)

def test_deltab():
    npt.assert_allclose(f['deltab'][::10], [0.0058372 , -0.00301547, -0.00595138, -0.00416407, -0.01490865,
          -0.01760859, -0.00807793], atol=1e-8)

def test_partial_loading():
    np.random.seed(1)  # so that results are reproducible
    f1 = f
    for test_len in [16, 32, 48]:
        for i in range(5):
            subindex = np.random.permutation(np.arange(0, len(f1)))[:test_len]
            subindex.sort()
            f2 = pynbody.load("testdata/grafic_test/", take=subindex)

            assert (f2['x'] == f1[subindex]['x']).all()
            assert (f2['iord'] == f1[subindex]['iord']).all()
