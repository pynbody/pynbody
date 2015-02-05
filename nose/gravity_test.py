import pynbody
import numpy as np
import numpy.testing as npt


def test_gravity():
    f = pynbody.load("testdata/g15784.lr.01024")
    h = f.halos()
    pynbody.analysis.angmom.faceon(h[1])
    pro = pynbody.analysis.profile.Profile(
        h[1], type='equaln', nbins=50, min='100 pc', max='50 kpc')

    v_circ_correct = [57.10235902,  102.47010057,  131.4695253,  155.23337252,
                      175.28099575,  193.16085934,  209.50653916,  224.36660666,
                      237.98273323,  250.67698223,  262.59859863,  273.63670486,
                      283.730395,  292.97096577,  301.18648746,  308.45552038,
                      314.7448155,  319.99299234,  324.19927048,  327.4048513,
                      329.67439265,  330.97009665,  331.2823001,  330.57049506,
                      328.95650172,  326.53389029,  323.57413842,  321.05794597,
                      318.71532239,  316.26507825,  313.02278807,  308.60242346,
                      303.82744776,  299.81889019,  296.74381703,  290.89202583,
                      283.90655109,  277.86591701,  271.35836067,  267.88987852,
                      263.84927022,  259.83027431,  256.14563506,  251.44650286,
                      247.06627248,  244.91720688,  240.34087759,  238.29066475,
                      233.92235749,  229.69618534]

    v_circ = pro['v_circ'].in_units('km s^-1')

    npt.assert_allclose(v_circ, v_circ_correct,atol=1e-5)