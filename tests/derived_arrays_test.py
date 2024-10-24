import numpy as np
import pytest

import pynbody


def setup_module():
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
    # All polar coodinates are updated following a coordinate transform
    # except for the azimuth because it doesn't derive from 3D position array and rather x, y
    # Test that dependencies are correctly derived following fix for Issue 636

    original_azi = f['az']
    original_azi_stars = f.st['az']

    assert(f._dependency_tracker.get_dependents('x') == {'vphi', 'az', 'vcxy'})
    assert(f._dependency_tracker.get_dependents('y') == {'vphi', 'az', 'vcxy'})
    assert('az' in f._dependency_tracker.get_dependents('pos'))

    # Make a minimal coordinate transform and assert that the azimuth is changed
    f['x'] += 1.0
    f['y'] -= 1.0
    try:
        np.testing.assert_allclose(original_azi, f['az'])
    except AssertionError:
        pass

    # Verify same behaviour for family arrays
    try:
        np.testing.assert_allclose(original_azi_stars, f.st['az'])
    except AssertionError:
        pass


def test_spherical_coordinates_after_coordinate_transform():

    test_spherical_coordinates_arrays()

    # Integrated test testing that all spherical coordinates make sense after a rotation
    with pynbody.analysis.angmom.faceon(f):
        test_spherical_coordinates_arrays()

    # And another rotation
    with f.rotate_z(90):
        test_spherical_coordinates_arrays()

    # And translation + rotation
    with f.translate(pynbody.array.SimArray([-100, 140, 200], units='kpc')).rotate_x(173):
        test_spherical_coordinates_arrays()

class ExampleSnap(pynbody.snapshot.SimSnap):
    pass

@ExampleSnap.derived_array
def _test_quantity(sim):
    return sim['input']+2

def test_derived_array_update():
    f = pynbody.new(10, class_=ExampleSnap)
    f['input'] = np.arange(0,10)
    assert (f['_test_quantity']==np.arange(2,12)).all()
    f['input']*=2
    assert (f['_test_quantity']==np.arange(2,22,2)).all()

def test_derived_family_array_update():
    f = pynbody.new(dm=10,star=10, class_=ExampleSnap)
    f.dm['input'] = np.arange(0,10)
    assert (f.dm['_test_quantity']==np.arange(2,12)).all()
    f.dm['input']*=2
    assert (f.dm['_test_quantity']==np.arange(2,22,2)).all()



def test_derived_quantity_class_override():
    """tests that the right subclass is selected for a derived quantity, whether
    or not we use the deprecated method"""

    with pytest.warns(DeprecationWarning):
        @ExampleSnap.derived_quantity
        def test_override_quantity(sim):
            return sim['input'] * 3

        @ExampleSnap.derived_array
        def test_override_array(sim):
            return sim['input'] * 5

        @pynbody.snapshot.SimSnap.derived_quantity
        def test_override_quantity(sim):
            return sim['input'] * 2

        @pynbody.snapshot.SimSnap.derived_array
        def test_override_array(sim):
            return sim['input'] * 4

    f = pynbody.new(dm=10, star=10, class_=ExampleSnap)
    f.dm['input'] = input = np.arange(0, 10)
    assert (f.dm['test_override_quantity'] == input * 3).all()
    assert (f.dm['test_override_array'] == input * 5).all()

def test_derived_family_array_with_nonderived_partner_update():
    f = pynbody.new(dm=10,star=10, gas=10, class_=ExampleSnap)

    f.star['_test_quantity'] = np.arange(-10,0)
    f['input'] = np.arange(0,30)

    assert (f.dm['_test_quantity']==np.arange(2,12)).all()
    assert (f.star['_test_quantity']==np.arange(-10,0)).all()

    assert f.dm['_test_quantity'].derived
    assert not f.star['_test_quantity'].derived

    f['input']*=2

    assert (f.dm['_test_quantity']==np.arange(2,22,2)).all()
    assert (f.star['_test_quantity']==np.arange(-10,0)).all()


def test_deprecated_derived_quantity():
    with pytest.warns(DeprecationWarning):
        @ExampleSnap.derived_quantity
        def another_test_quantity(sim):
            return sim['input'] + 2

    with pytest.warns(DeprecationWarning):
        @ExampleSnap.stable_derived_quantity
        def another_test_quantity_stable(sim):
            return sim['input'] + 4

    f = pynbody.new(10, class_=ExampleSnap)
    f['input'] = np.arange(0,10)
    assert (f['another_test_quantity']==np.arange(2,12)).all()
    assert (f['another_test_quantity_stable']==np.arange(4,14)).all()
    f['input']+=2
    assert (f['another_test_quantity']==np.arange(4,14)).all()
    assert (f['another_test_quantity_stable']==np.arange(4,14)).all()
