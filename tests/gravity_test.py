import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gasoline_ahf")
    pynbody.test_utils.ensure_test_data_available("gizmo")


def test_gravity():
    f = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    h = f.halos()
    pynbody.analysis.angmom.faceon(h[0])
    pro = pynbody.analysis.profile.Profile(
        h[0], type='equaln', nbins=50, rmin='100 pc', rmax='50 kpc')

    v_circ_correct = np.array([ 57.19445406, 103.07628964, 132.11223368, 155.58932808,
          175.48202525, 193.19929044, 209.50654738, 224.36527055,
          237.93659247, 250.68636441, 262.72949256, 273.84865161,
          283.90032809, 293.10420183, 301.32998659, 308.540641  ,
          314.80948167, 320.0281338 , 324.2028914 , 327.40124838,
          329.64625769, 330.92733193, 331.25336332, 330.57151552,
          328.97485235, 326.54886075, 323.60357994, 321.03602331,
          318.69193219, 316.16834582, 312.98912943, 308.66044926,
          303.84896218, 299.75583821, 296.77125812, 290.91251525,
          283.84275867, 277.87033036, 271.31481248, 267.90817792,
          263.89271744, 259.83899922, 256.07340296, 251.42403094,
          247.00347206, 244.89404984, 240.3383157 , 238.17681295,
          233.90977942, 229.66882597])

    v_circ = pro['v_circ'].in_units('km s^-1')

    npt.assert_allclose(v_circ, v_circ_correct, rtol=1e-2, atol=0)

def test_gravity_float():
    f = pynbody.new(100)
    np.random.seed(0)
    coords = np.random.normal(size=(100,3))
    del f['pos']
    del f['mass']
    f['pos'] = np.array(coords,dtype=np.float32)
    f['eps'] = np.ones(100,dtype=np.float32)
    f['mass'] = np.ones(100,dtype=np.float32)
    pynbody.gravity.all_direct(f)

def test_eps_retrieval_str():
    f = pynbody.load("testdata/gadget2/test_g2_snap.0")
    f.properties['eps'] = "0.3 kpc"
    pynbody.gravity.all_direct(f)
    true_phi_10 = np.array([-0.050296, -0.054202, -0.053822, -0.05072 , -0.050815, -0.052855,
                            -0.05137 , -0.051314, -0.05611 , -0.055222])
    npt.assert_allclose(f['phi'][:10], true_phi_10, rtol=1e-5)


def test_eps_retrieval_unit():
    f = pynbody.load("testdata/gadget2/test_g2_snap.0")
    f.properties['eps'] = 0.3 * pynbody.units.kpc
    pynbody.gravity.all_direct(f)
    true_phi_10 = np.array([-0.050296, -0.054202, -0.053822, -0.05072 , -0.050815, -0.052855,
                            -0.05137 , -0.051314, -0.05611 , -0.055222])
    npt.assert_allclose(f['phi'][:10], true_phi_10, rtol=1e-5)


def test_eps_retrieval_number():
    f = pynbody.load("testdata/gadget2/test_g2_snap.0")
    f.properties['eps'] = 0.3
    pynbody.gravity.all_direct(f)
    true_phi_10 = np.array([-0.06696571, -0.07087147, -0.07049192,
                            -0.06739005, -0.06748439, -0.0695245,
                            -0.06803885, -0.0679833,  -0.07277965, -0.07189107])
    npt.assert_allclose(f['phi'][:10], true_phi_10)

def test_varying_eps():
    f = pynbody.new(dm = 2)
    f['pos'] = np.array([[0, 0, 0], [1, 1, 1]])
    f['mass'] = np.array([1, 1])
    f['eps'] = np.array([0.0, 1000.0])
    pot, accel = pynbody.gravity.direct(f, np.array([[0.5, 0.0, 0.0]]))

    # test that the second particle doesn't contribute, due to its large softening length
    npt.assert_allclose(accel, np.array([[-4.0, 0.0, 0.0]]), atol=1e-8, rtol=0)


@pytest.mark.filterwarnings("ignore:Unable to find cosmological")
def test_direct_gravity_large_snapshot_no_segfault():
    # Load data with more than a few million particles
    snapshot = pynbody.load("testdata/gizmo/snapshot_000.hdf5")

    # Set softening length
    snapshot.properties["eps"] = "0.05 kpc"

    # Calculate rotation curve
    pos = np.linspace(4, 10, 20)
    result = pynbody.gravity.midplane_rot_curve(snapshot.dm, pos)
    assert result is not None


def _dtype_test_snapshot(npart=50, pos_dtype=np.float64, mass_dtype=np.float64,
                         eps_dtype=None, eps_value=0.1, eps_units='kpc'):
    """Build a small snapshot with independently specified dtypes for pos, mass and eps.

    If eps_dtype is None, no eps array is created at all.
    """
    f = pynbody.new(dm=npart)

    del f['pos']
    del f['mass']

    np.random.seed(0)
    f['pos'] = np.random.normal(size=(npart, 3)).astype(pos_dtype)
    f['pos'].units = 'kpc'
    f['mass'] = np.ones(npart, dtype=mass_dtype)
    f['mass'].units = 'Msol'

    if eps_dtype is not None:
        f['eps'] = pynbody.array.SimArray(np.full(npart, eps_value, dtype=eps_dtype), eps_units)

    return f


_IPOS = np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [-2.0, 0.0, 0.0]])


@pytest.mark.parametrize("ipos_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("eps_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mass_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("pos_dtype", [np.float32, np.float64])
def test_mixed_dtypes(pos_dtype, mass_dtype, eps_dtype, ipos_dtype):
    """Positions, masses, softenings and evaluation points may each be single or double precision.

    The kernel is compiled for every combination, so no input needs converting and none of these
    16 cases should differ from the all-double answer by more than single-precision round-off.
    """
    reference = pynbody.gravity.direct(_dtype_test_snapshot(eps_dtype=np.float64), _IPOS)

    f = _dtype_test_snapshot(pos_dtype=pos_dtype, mass_dtype=mass_dtype, eps_dtype=eps_dtype)
    pot, accel = pynbody.gravity.direct(f, _IPOS.astype(ipos_dtype))

    # the result takes its precision from the evaluation points
    assert pot.dtype == ipos_dtype
    assert accel.dtype == ipos_dtype

    npt.assert_allclose(pot, reference[0], rtol=1e-5)
    npt.assert_allclose(accel, reference[1], rtol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scalar_softening_adopts_snapshot_dtype(dtype):
    """A scalar or unit softening has no dtype of its own, and must work in either precision"""
    f = _dtype_test_snapshot(pos_dtype=dtype, mass_dtype=dtype)
    ipos = np.array([[0.5, 0.0, 0.0]], dtype=dtype)

    for eps in (0.1, '100 pc', 0.1 * pynbody.units.kpc):
        f.properties['eps'] = eps
        pot, _ = pynbody.gravity.direct(f, ipos)
        assert pot.dtype == dtype


def test_non_float_softening_raises():
    """Only single and double precision have specialisations; anything else must fail, not be cast"""
    f = _dtype_test_snapshot()
    f['eps'] = pynbody.array.SimArray(np.ones(len(f), dtype=np.int32), 'kpc')
    assert f['eps'].dtype == np.int32

    with pytest.raises(TypeError):
        pynbody.gravity.direct(f, _IPOS)


def test_softening_without_units_is_taken_as_position_units():
    """An array softening carrying no units is assumed to be in the position units.

    This matches how a bare number in f.properties['eps'] is treated.
    """
    npart = 50
    with_units = _dtype_test_snapshot(npart=npart, eps_dtype=np.float64, eps_value=0.1,
                                      eps_units='kpc')

    without_units = _dtype_test_snapshot(npart=npart)
    without_units['eps'] = np.full(npart, 0.1)
    assert not pynbody.units.has_units(without_units['eps'])

    npt.assert_allclose(pynbody.gravity.direct(without_units, _IPOS)[0],
                        pynbody.gravity.direct(with_units, _IPOS)[0], rtol=1e-10)


def test_softening_array_of_wrong_length_raises():
    f = _dtype_test_snapshot(npart=50)

    with pytest.raises(ValueError, match="length"):
        pynbody.gravity.direct(f, _IPOS, eps=np.ones(3))


def test_softening_units_respected_for_subsnap():
    """The softening of a subsnap must be converted into the position units.

    IndexedSimArray is not a subclass of SimArray, so a subsnap softening used to bypass the
    unit conversion entirely and be interpreted as though it were already in the position units.
    """
    npart = 50

    in_kpc = _dtype_test_snapshot(npart=npart, eps_dtype=np.float64, eps_value=0.1, eps_units='kpc')
    in_pc = _dtype_test_snapshot(npart=npart, eps_dtype=np.float64, eps_value=100.0, eps_units='pc')

    # a sphere large enough to contain everything, so that the two calculations must agree exactly
    subsnap = in_pc[pynbody.filt.Sphere('1000 kpc')]
    assert len(subsnap) == npart
    assert isinstance(subsnap['eps'], pynbody.array.IndexedSimArray)

    npt.assert_allclose(pynbody.gravity.direct(subsnap, _IPOS)[0],
                        pynbody.gravity.direct(in_kpc, _IPOS)[0], rtol=1e-10)


def test_midplane_rot_curve_with_mixed_dtypes():
    """The rotation curve is the route by which issue #1029 hit the dtype mismatch"""
    f = _dtype_test_snapshot(pos_dtype=np.float64, eps_dtype=np.float32)
    rxy = np.linspace(0.5, 2.0, 4)

    v = pynbody.gravity.midplane_rot_curve(f, rxy)

    assert len(v) == len(rxy)
    assert np.all(np.isfinite(v))


def test_all_direct_with_mixed_dtypes():
    f = _dtype_test_snapshot(pos_dtype=np.float64, eps_dtype=np.float32)

    pynbody.gravity.all_direct(f)

    assert np.all(np.isfinite(f['phi']))
    assert np.all(np.isfinite(f['acc']))
