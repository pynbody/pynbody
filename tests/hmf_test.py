import pynbody
import pynbody.analysis.hmf as hmf

import numpy as np
import pytest

def test_powspec_static():
    f = pynbody.new()
    with pytest.warns(RuntimeWarning, match="Planck 2018"):
        ps = hmf.PowerSpectrumCAMB(f)

    assert np.allclose(ps.k.min(), 1.0e-4)
    assert np.allclose(ps.k.max(), 1.0e4)
    assert np.argmax(ps(ps.k)) == 369
    assert np.allclose(ps(ps.k[369]), 25452.30251039246)

def test_powspec_live():
    f = pynbody.new()
    with pytest.warns(RuntimeWarning, match="Assuming default value"):
        ps = hmf.PowerSpectrumCAMBLive(f)

    assert np.allclose(ps.k.min(), 1.0e-4)
    assert np.allclose(ps.k.max(), 1.0e4)
    assert np.argmax(ps(ps.k)) == 368
    assert np.allclose(ps(ps.k[368]), 25986.02572910632)

