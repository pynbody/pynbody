import pynbody
import numpy as np
import numpy.testing as npt
import pylab as p
import pickle


def setup():
    global f
    f = pynbody.load("testdata/g15784.lr.01024")
    h = pynbody.halo.AHFCatalogue(f)

    h = h[::10]
    assert len(h)==141

def test_binning_hmf():

    center, means, err = pynbody.analysis.hmf.simulation_halo_mass_function(f, log_M_min=8, log_M_max=14, delta_log_M=0.5)

    assert(len(means) == len(center) == len(err) == 13)
