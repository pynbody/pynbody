import pynbody
import numpy as np


def setup():
    global f, h
    f = pynbody.new(dm=1000, star=500, gas=500, order='gas,dm,star')
    f['pos'] = pynbody.array.SimArray(np.random.normal(scale=1.0, size=f['pos'].shape), units='kpc')
    f['vel'] = pynbody.array.SimArray(np.random.normal(scale=1.0, size=f['vel'].shape), units='km s**-1')
    f['mass'] = pynbody.array.SimArray(np.random.uniform(1.0, 10.0, size=f['mass'].shape), units='Msol')
    f.gas['rho'] = pynbody.array.SimArray(np.ones(500, dtype=float), units='Msol kpc**-3')


def test_spherical_coordinates_arrays():

    # Spherical coordinates are derivable
    assert 'r' in f.derivable_keys()
    assert 'az' in f.derivable_keys()
    assert 'theta' in f.derivable_keys()

    # And their associated velocities
    assert 'vr' in f.derivable_keys()
    assert 'vphi' in f.derivable_keys()
    assert 'vtheta' in f.derivable_keys()

    # Derive the arrays
    f['r'], f['az'], f['theta']
    f['vr'], f['vphi'], f['vtheta']

    # Test the range of values make sense for spherical coordinates

    # Azimuth is chosen between -pi and pi
    np.testing.assert_array_less(f['az'], np.pi * np.ones(f['az'].shape))
    np.testing.assert_array_less(- np.pi * np.ones(f['az'].shape), f['az'])

    # Polar angle is between 0 and pi
    np.testing.assert_array_less(f['theta'], np.pi * np.ones(f['theta'].shape))
    np.testing.assert_array_less(np.zeros(f['theta'].shape), f['theta'])

    # Pynbody uses a different approach than projecting to compute spherical coordinates
    # Check the two methods are producing the same results

    # Should find that x = r cos(azimuth) cos(polar declination)
    #                  y = r sin(azimuth) cos(polar declination)
    #                  z = r cos(polar declination)
    np.testing.assert_allclose(f['x'], f['r'] * np.cos(f['az']) * np.sin(f['theta']))
    np.testing.assert_allclose(f['y'], f['r'] * np.sin(f['az']) * np.sin(f['theta']))
    np.testing.assert_allclose(f['z'], f['r'] * np.cos(f['theta']))

    # Should find vr = vx * cos(azimuth) sin(polar declination) + vy sin(azimuth) sin(polar declination)
    # + vz cos (polar declination)
    vr = np.sin(f['theta']) * np.cos(f['az']) * f['vx'] + np.sin(f['theta']) * np.sin(f['az']) * f['vy'] + \
         np.cos(f['theta']) * f['vz']
    np.testing.assert_allclose(vr, f['vr'])

    # Should find vphi = - vx * sin(azimuth) + vy cos(azimuth) + 0
    vphi = - np.sin(f['az']) * f['vx'] + np.cos(f['az']) * f['vy']
    np.testing.assert_allclose(vphi, f['vphi'])

    # Should find vtheta = vx * cos(azimuth) cos(polar declination) + vy sin(azimuth) cos(polar declination)
    # - vz sin (polar declination)
    vtheta = np.cos(f['theta']) * np.cos(f['az']) * f['vx'] + np.cos(f['theta']) * np.sin(f['az']) * f['vy'] - \
             np.sin(f['theta']) * f['vz']
    np.testing.assert_allclose(vtheta, f['vtheta'])


def test_azimuth_depencies():
    ## FFS, the azimuth doesn't get upddated
    # Try to understand why dependencies
    azi = f['az']
    print(azi)
    print(f._dependency_tracker.get_dependents('az'))
    print(f._dependency_tracker.get_dependents('x'))
    print(f['az'].derived)
    print(f.auto_propagate_off)
    with pynbody.analysis.angmom.faceon(f):
        print(f['az'])
    print(f._dependency_tracker.get_dependents('az'))
    print(f._dependency_tracker.get_dependents('x'))
    raise Exception



def test_spherical_coordinates_after_coordinate_transform():

    # test_spherical_coordinates_arrays()
    # np.testing.assert_allclose(f['x'], f['r'] * np.cos(f['az']) * np.sin(f['theta']))

    r = f['r']
    azimuth = f['az']
    polar = f['theta']
    with pynbody.analysis.angmom.faceon(f):
        # r has been successfully updated
        # np.testing.assert_allclose(r, f['r'])
        np.testing.assert_allclose(azimuth, f['az'])
        # Polar angle gets updated
        # np.testing.assert_allclose(polar, f['theta'])
        np.testing.assert_allclose(f['x'], f['r'] * np.cos(f['az']) * np.sin(f['theta']))

    with f.rotate_z(90):
        test_spherical_coordinates_arrays()

    with pynbody.transformation.translate(f, pynbody.array.SimArray([-100, 140, 200], units='kpc')):
        with f.rotate_x(173):
            test_spherical_coordinates_arrays()

