import numpy as np
import pytest

import pynbody
import pynbody.analysis.hmf as hmf


def test_powspec_static(recwarn):
    f = pynbody.new()
    ps = hmf.PowerSpectrumCAMB(f)

    assert any(["Assuming default value" in str(w.message) for w in recwarn.list])
    assert any(["assumes Planck 2018" in str(w.message) for w in recwarn.list])

    assert np.allclose(ps.k.min(), 1.0e-4)
    assert np.allclose(ps.k.max(), 1.0e4)
    assert np.argmax(ps(ps.k)) == 369
    assert np.allclose(ps(ps.k[369]), 25452.63366178445)

def test_powspec_live(recwarn):
    f = pynbody.new()

    ps = hmf.PowerSpectrumCAMBLive(f)

    assert any(["Assuming default value" in str(w.message) for w in recwarn.list])

    assert np.allclose(ps.k.min(), 1.0e-4)
    assert np.allclose(ps.k.max(), 1.0e4)
    assert np.argmax(ps(ps.k)) == 368
    assert np.allclose(ps(ps.k[368]), 25985.50574103266)
