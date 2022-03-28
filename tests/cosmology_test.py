import numpy as np
import numpy.testing as npt

import pynbody


def test_a_to_t():
    """Test scalefactor -> time conversion for accuracy. See also issue #479"""
    f = pynbody.new() # for default cosmology
    ipoints = pynbody.analysis.cosmology._interp_points
    interp_a = np.linspace(0.005, 1.0, ipoints+1) # enough values to force interpolation rather than direct calculation
    interp_z = 1./interp_a - 1.
    interp_t = pynbody.analysis.cosmology.age(f, z=interp_z)
    direct_aform = interp_a[::100]
    z = 1./direct_aform-1.
    direct_tform = pynbody.analysis.cosmology.age(f,z)
    npt.assert_almost_equal(direct_tform, interp_t[::100], decimal=4)

def test_aform_saturation():
    """Test that NaN is returned when tform cannot be calculated from aform"""
    ipoints = pynbody.analysis.cosmology._interp_points
    f = pynbody.new(ipoints + 1)
    f['aform'] = np.linspace(0.0,1.1,ipoints+1)-0.05
    tf = f['tform'][::100]
    assert tf[0]!=tf[0] # nan outside range
    assert tf[-1]!=tf[-1] # nan outside range
    assert (tf[1:-1]==tf[1:-1]).all() # no nans inside range
