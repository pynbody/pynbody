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
