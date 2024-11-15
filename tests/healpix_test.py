import healpy as hp
import numpy as np
import pytest

import pynbody


@pytest.mark.parametrize('nside', [8, 32, 64, 256, 512])
def test_query_disc(nside):
    """pynbody has its own implementation of query_disc for efficiency reasons
    this test checks that it returns the same result as the healpy implementation.

    To balance between efficiency and catching weird cases, we generate random trials
    with a decreasing number of trials as nside increases.
    """

    Ntrials = 40960 // nside

    np.random.seed(1337)

    trial_positions = np.random.randn(Ntrials, 3)
    trial_positions /= np.linalg.norm(trial_positions, axis=1)[:, None]
    trial_radii = np.random.uniform(0.01, 1.0, Ntrials)**3 * np.pi

    npix = hp.nside2npix(nside)

    for vec, radius in zip(trial_positions, trial_radii):

        fake_map = np.zeros(npix)

        pixels = hp.query_disc(nside, vec, radius)

        fake_map[pixels] = 1.0

        pixels2 = pynbody.util._util.query_healpix_disc(nside, vec, radius)
        fake_map[pixels2] -=1.0



        try:
            assert np.all(fake_map == 0)
        except:
            import pylab as p
            hp.mollview(fake_map)
            p.savefig('healpix_test.png')
            raise
