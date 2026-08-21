import warnings

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils
from pynbody.test_utils import make_blob

np.random.seed(1)


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gasoline_ahf")

def make_fake_bar(npart=100000, max=1, min=-1, barlength=.8, barwidth=0.05, phi=0, fraction=0.2):

    x = np.random.sample(int(npart*fraction))*(max-min) + min
    y = np.random.sample(int(npart*fraction))*(max-min) + min

    xbar = np.random.sample(npart, )*(barlength/2+barlength/2) - barlength/2
    ybar = np.random.sample(npart)*(barwidth/2+barwidth/2) - barwidth/2

    x = np.concatenate([x,xbar])
    y = np.concatenate([y,ybar])

    good = np.where(x**2 + y**2 < 1)[0]

    s = pynbody.snapshot.new(len(good))
    s['x'] = x[good]
    s['y'] = y[good]
    s['pos'].units = 'kpc'
    s['mass'] = 1.0
    s['mass'].units = 'Msol'
    s['vel'] = 1.0
    s['vel'].units = 'km s^-1'
    s['eps'] = (max-min)/np.sqrt(npart)
    s['eps'].units = 'kpc'
    s.rotate_z(phi)
    return s

def test_fourier_profile():
    bar = make_fake_bar(phi=45)

    p = pynbody.analysis.profile.Profile(bar, nbins=50)

    assert(np.all(p['fourier']['amp'][2,4:20] > 0.1))
    assert(np.allclose(np.abs(p['fourier']['phi'][2,4:20]/2), np.pi/4.0, rtol=0.05))

def test_create_particle_array():
    np.random.seed(1337)
    Npart = 10000
    f = pynbody.new(Npart)
    f['pos'] = np.random.normal(size=(Npart,3))
    f['mass'] = np.ones(Npart)/Npart
    p = pynbody.analysis.profile.Profile(f, nbins=50, ndim=3)

    p.create_particle_array('density', 'pt_density')

    npt.assert_allclose(f['pt_density'], np.exp(-f['r']**2/2)/np.sqrt(2*np.pi)**3, atol=1e-2, rtol=0)

    # test on a different simulation
    f2 = pynbody.new(Npart)
    np.random.seed(1338)
    f2['pos'] = np.random.normal(size=(Npart,3))
    f2['mass'] = np.ones(Npart)/Npart

    p.create_particle_array('density', 'pt_density', target_simulation=f2)
    npt.assert_allclose(f2['pt_density'], np.exp(-f2['r'] ** 2 / 2) / np.sqrt(2 * np.pi) ** 3, atol=1e-2, rtol=0)

@pytest.mark.parametrize("perform_align", [True, False])
@pytest.mark.parametrize("profile_quantity", ['v_circ', 'pot'])
def test_plane_warnings(perform_align, profile_quantity):
    bar = make_fake_bar(phi=45)

    if perform_align:
        pynbody.analysis.angmom.faceon(bar)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = pynbody.analysis.profile.Profile(bar, nbins=50)
        p[profile_quantity] # noqa - just want to access it

    if perform_align:
        assert len(w) == 0
    else:
        assert len(w) == 1
        assert "this routine assumes the disk is in the x-y plane" in str(w[0])



def test_potential_profile_fp64():
    f = pynbody.new(100)
    coords = np.random.normal(size=(100,3))
    del f['pos']
    del f['mass']
    f['pos'] = np.array(coords,dtype=np.float64)
    f['eps'] = np.ones(100,dtype=np.float64)
    f['mass'] = np.ones(100,dtype=np.float64)
    p = pynbody.analysis.profile.Profile(f, nbins=50)
    with pytest.warns(UserWarning):
        p['pot']


def test_potential_profile_fp32():
    f = pynbody.new(100)
    coords = np.random.normal(size=(100,3))
    del f['pos']
    del f['mass']
    f['pos'] = np.array(coords,dtype=np.float32)
    f['eps'] = np.ones(100,dtype=np.float32)
    f['mass'] = np.ones(100,dtype=np.float32)
    p = pynbody.analysis.profile.Profile(f, nbins=50)
    with pytest.warns(UserWarning):
        p['pot']

@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_angmom_profile():
    f = pynbody.new(100)
    coords = np.random.normal(size=(100,3))
    f['pos'] = np.array(coords, dtype=np.float64)
    f['mass'] = np.ones(100)
    rand_j = np.random.normal(size=(100,3))
    rand_j[:,1]*=0.001
    f['j'] = np.array(rand_j, dtype=np.float64)
    p = pynbody.analysis.profile.Profile(f, nbins=50)
    assert(np.nanmin(p['j_phi'])<np.pi/2)
    assert(np.nanmax(p['j_phi'])>np.pi/2)



def test_unique_hash_generation():
    f1 = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    p1 = pynbody.analysis.profile.Profile(f1, nbins=50)
    p2 = pynbody.analysis.profile.Profile(f1[:1000], nbins=50)

    hash1 = p1._generate_hash_filename_from_particles()
    hash2 = p2._generate_hash_filename_from_particles()

    assert(hash1 != hash2)
    assert(type(hash1) is str)
    assert(type(hash2) is str)


def test_kappa_profile():
    f1 = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")
    p = pynbody.analysis.profile.Profile(f1[:100], nbins=5)
    with pytest.warns(UserWarning):
        p['kappa'].in_units('km s**-1 kpc**-1')

def test_write_profile():
    f1 = pynbody.load("testdata/gasoline_ahf/g15784.lr.01024")

    p = pynbody.analysis.profile.Profile(f1[:1000], nbins=50)
    p['rbins'], p['density']

    # Write profile and read again
    p.write()
    read_profile = pynbody.analysis.profile.Profile(f1[:1000], load_from_file=True)

    npt.assert_allclose(read_profile.min, p.min)
    npt.assert_allclose(read_profile.max, p.max)
    npt.assert_allclose(read_profile.nbins, p.nbins)
    npt.assert_allclose(read_profile['rbins'], p['rbins'])
    npt.assert_allclose(read_profile['density'], p['density'])


def test_plot_density_profile():
    # very minimal test to check if the plot function runs without errors
    f = make_fake_bar()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pynbody.plot.profile.density_profile(f)
        pynbody.plot.profile.rotation_curve(f, center=False)

@pytest.mark.parametrize("weight", [False, True])
def test_quantile_profile(weight):
    Npart = 500000
    f = make_blob.make_uniform_blob(Npart)
    np.random.seed(1337)
    f['testquantity'] = np.random.normal(size=Npart)*0.2 + f['r']
    if weight:
        # intentionally bias things to have fatter tails
        weights = (f['testquantity'] - f['r'])**2
    else:
        weights = None

    pro = pynbody.analysis.profile.QuantileProfile(f, q=(0.16,0.84), weights=weights, nbins=50, type='equaln')

    # +/- 1 sigma should average to the rbin value.

    npt.assert_allclose(np.mean(pro['testquantity'], axis=1)[1:], pro['rbins'][1:], atol=2e-2)

    # the width, if normally distributed, should be 0.4
    expected_width = 0.4
    if weight:
        # this correction comes from getting the cdf for the fat-tailed distribution x^2 e^(-x^2/2), then solving
        # for where the cdf is 0.16 and 0.84
        expected_width *= 1.8724
    npt.assert_allclose(np.diff(pro['testquantity'], axis=1), expected_width, atol=2.5e-2)


def _make_lazy_loading_snapshot(npart=1000):
    np.random.seed(1337)
    base = pynbody.new(star=npart)
    base['pos'] = pynbody.array.SimArray(np.random.normal(size=(npart, 3)), 'kpc')
    base['vel'] = pynbody.array.SimArray(np.random.normal(size=(npart, 3)), 'km s^-1')
    base['mass'] = pynbody.array.SimArray(np.ones(npart), 'Msol')
    return base.get_copy_on_access_simsnap()


@pytest.fixture
def lazy_loading_snapshot():
    """A snapshot which lazily copies its arrays from another, as a file-backed one would.

    Like a real loader, it advertises ``vel`` rather than ``vz`` in loadable_keys(), and nothing
    is in memory until it is asked for."""
    return _make_lazy_loading_snapshot()


@pytest.fixture
def preloaded_snapshot():
    """The same snapshot, but with the velocities already pulled into memory."""
    f = _make_lazy_loading_snapshot()
    f['vel']
    return f


@pytest.mark.parametrize("name", ['vz', 'vz_disp', 'vz_rms', 'vz_med', 'd_vz'])
def test_profile_of_1d_slice_without_preloading(name, lazy_loading_snapshot, preloaded_snapshot):
    """Profiles of 1D slices must not depend on whether the ND array happens to be loaded already.

    See issue #1018: 'vz' is only in keys() once 'vel' is in memory, and loaders advertise 'vel'
    rather than 'vz' in loadable_keys(), so asking for e.g. 'vz_disp' first used to raise KeyError.
    """
    assert 'vel' not in lazy_loading_snapshot.keys()

    p_lazy = pynbody.analysis.profile.Profile(lazy_loading_snapshot, rmin=0, rmax=3, nbins=5)
    p_preloaded = pynbody.analysis.profile.Profile(preloaded_snapshot, rmin=0, rmax=3, nbins=5)

    npt.assert_allclose(p_lazy[name], p_preloaded[name])


def test_quantile_profile_of_1d_slice_without_preloading(lazy_loading_snapshot, preloaded_snapshot):
    p_lazy = pynbody.analysis.profile.QuantileProfile(lazy_loading_snapshot, rmin=0, rmax=3, nbins=5)
    p_preloaded = pynbody.analysis.profile.QuantileProfile(preloaded_snapshot, rmin=0, rmax=3, nbins=5)

    npt.assert_allclose(p_lazy['vz'], p_preloaded['vz'])


def test_profile_still_rejects_unknown_arrays(lazy_loading_snapshot):
    p = pynbody.analysis.profile.Profile(lazy_loading_snapshot, rmin=0, rmax=3, nbins=5)
    for name in ['nonsense', 'nonsense_disp', 'nonsense_rms', 'nonsense_med', 'nonsense_x',
                 'd_nonsense']:
        with pytest.raises(KeyError):
            p[name]


def _make_bin_centred_snapshot(nbins=10, rmax=10.0, per_bin=20):
    """A snapshot whose particles sit exactly at the centre of each radial bin.

    Placing them this way makes the average of a quantity within a bin an exact function of that
    bin's centre, so a profile built from a linear quantity is itself exactly linear and its
    radial derivative is known analytically."""
    centres = (np.arange(nbins) + 0.5) * (rmax / nbins)
    r = np.repeat(centres, per_bin)

    pos = np.zeros((len(r), 3))
    pos[:, 0] = r

    f = pynbody.new(star=len(r))
    f['pos'] = pynbody.array.SimArray(pos, 'kpc')
    f['mass'] = pynbody.array.SimArray(np.ones(len(r)), 'Msol')
    f['myquantity'] = pynbody.array.SimArray(3.0 * r + 7.0, 'km s^-1')
    return f, centres


def test_derivative_of_array_profile():
    """``d_<name>`` differentiates the ``<name>`` profile with respect to radius."""
    nbins, rmax = 10, 10.0
    f, centres = _make_bin_centred_snapshot(nbins=nbins, rmax=rmax)

    p = pynbody.analysis.profile.Profile(f, rmin=0, rmax=rmax, nbins=nbins, ndim=3)

    # the profile of 3r + 7 is exactly 3r + 7 at the bin centres, so its gradient is exactly 3
    npt.assert_allclose(p['myquantity'], 3.0 * centres + 7.0)
    npt.assert_allclose(p['d_myquantity'], 3.0)
    assert p['d_myquantity'].units == p['myquantity'].units / p['dr'].units


def test_derivative_of_registered_profile():
    """A derivative may also be taken of a profile that is not backed by a snapshot array."""
    nbins, rmax, per_bin = 10, 10.0, 20
    f, _ = _make_bin_centred_snapshot(nbins=nbins, rmax=rmax, per_bin=per_bin)

    p = pynbody.analysis.profile.Profile(f, rmin=0, rmax=rmax, nbins=nbins, ndim=3)

    # every bin holds the same mass, so the enclosed mass rises linearly and its gradient is that
    # mass divided by the bin width
    dr = rmax / nbins
    npt.assert_allclose(p['mass_enc'], per_bin * (np.arange(nbins) + 1.0))
    npt.assert_allclose(p['d_mass_enc'], per_bin / dr)
    assert p['d_mass_enc'].units == p['mass_enc'].units / p['dr'].units


class _BrokenDerivationSnap(pynbody.snapshot.simsnap.SimSnap):
    pass


@_BrokenDerivationSnap.derived_array
def broken(sim):
    return sim['no_such_array'] * 2


def test_profile_reports_why_a_registered_derivation_failed():
    """A name can be registered as derivable and still fail when the derivation is actually run.

    Availability is therefore established by asking for the array; when that fails the underlying
    reason must not be swallowed by the generic 'not a valid profile' message.
    """
    npart = 100
    f = pynbody.new(star=npart, class_=_BrokenDerivationSnap)
    f['pos'] = pynbody.array.SimArray(np.random.normal(size=(npart, 3)), 'kpc')
    f['mass'] = pynbody.array.SimArray(np.ones(npart), 'Msol')

    assert 'broken' in f.all_keys()  # registered, so a name-based check would think it available

    p = pynbody.analysis.profile.Profile(f, rmin=0, rmax=3, nbins=5)
    for name in ['broken', 'broken_disp']:
        with pytest.raises(KeyError) as excinfo:
            p[name]
        assert 'no_such_array' in str(excinfo.value) + str(excinfo.value.__cause__)
