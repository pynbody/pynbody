import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.analysis.hmf as hmf


def test_powspec_static(recwarn):
    f = pynbody.new()
    ps = hmf.PowerSpectrum(f)

    assert any(["Assuming default value" in str(w.message) for w in recwarn.list])
    assert any(["assumes Planck 2018" in str(w.message) for w in recwarn.list])

    assert np.allclose(ps.k.min(), 1.0e-4)
    assert np.allclose(ps.k.max(), 1.0e4)
    assert np.argmax(ps(ps.k)) == 369
    assert np.allclose(ps(ps.k[369]), 24405., rtol=1e-4)

@pytest.mark.parametrize("omegaM, peak_bin, peak_power", [[0.3111, 369, 24405.], [0.25, 355, 39609.]])
def test_powspec_live(recwarn, omegaM, peak_power, peak_bin):
    f = pynbody.new()
    f.properties['omegaM0'] = omegaM

    ps = hmf.PowerSpectrumCAMB(f)

    assert any(["Assuming default value" in str(w.message) for w in recwarn.list])

    assert np.allclose(ps.k.min(), 1.0e-4)
    assert np.allclose(ps.k.max(), 1.0e4)
    assert np.argmax(ps(ps.k)) == peak_bin
    assert np.allclose(ps(ps.k[peak_bin]), peak_power, rtol=1e-4)

def test_get_hmf(recwarn):

    f = pynbody.new()
    m, sig, dn_dlogm = pynbody.analysis.hmf.halo_mass_function(f, log_M_min=10, log_M_max=15, delta_log_M=0.1,
                                                               kern="REEDU", pspec=hmf.PowerSpectrum)

    npt.assert_allclose(m[::10], [1.12201845e+10, 1.12201845e+11, 1.12201845e+12,
                                  1.12201845e+13, 1.12201845e+14], rtol=1e-4)

    npt.assert_allclose(sig[::10], [3.70061772, 2.84212658, 2.08112932, 1.43170708, 0.90729266],
                        rtol=1e-4)

    npt.assert_allclose(dn_dlogm[::10], [4.75228357e-01, 6.23530663e-02, 8.17930180e-03, 9.94758260e-04,
                        8.09997113e-05], rtol=1e-4)

def test_get_empirical_hmf(recwarn):

    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    bin_center, bin_counts, err = pynbody.analysis.hmf.simulation_halo_mass_function(f,
                                                                                     log_M_min=10, log_M_max=15,
                                                                                     delta_log_M=1.0)
    npt.assert_allclose(bin_center, [5.5e+10, 5.5e+11, 5.5e+12, 5.5e+13, 5.5e+14])
    npt.assert_allclose(bin_counts, [2.74270242e-03, 9.11568735e-04, 3.83818415e-04, 2.47882726e-04,
          3.99810849e-05])
    npt.assert_allclose(err, [1.48092011e-04, 8.53762344e-05, 5.53994163e-05, 4.45210519e-05,
          1.78800847e-05])
