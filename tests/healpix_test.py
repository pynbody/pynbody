import healpy as hp
import numpy as np
import pynbody

def test_query_disc():
    # pynbody has its own implementation of query_disc for efficiency reasons
    # this test checks that it returns the same result as the healpy implementation

    Ntrials = 1000

    np.random.seed(1337)

    trial_positions = np.random.randn(Ntrials, 3)
    trial_positions /= np.linalg.norm(trial_positions, axis=1)[:, None]
    trial_radii = np.random.uniform(0.01, 6.0, Ntrials)

    nside = 8
    npix = hp.nside2npix(nside)

    for vec, radius in zip(trial_positions, trial_radii):
        #if abs(vec[2])<0.97:
        #    continue
        print("TRIAL:",vec,radius)

        # create a fake map
        fake_map = np.zeros(npix)

        pixels = hp.query_disc(nside, vec, radius)

        fake_map[pixels] = 1.0

        pixels = pynbody.util._util.query_healpix_disc(nside, vec, radius)
        print(len(pixels),repr(pixels))
        fake_map[pixels] -=1.0


        try:
            assert np.all(fake_map == 0.0)
        except:
            import pylab as p
            hp.mollview(fake_map)
            p.savefig('healpix_test.png')
            raise