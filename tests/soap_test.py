import pathlib

import h5py
import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.snapshot.soap
import pynbody.snapshot.swift
import pynbody.test_utils

_soap_path = "testdata/SOAP/soap_example.hdf5"


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("soap", "swift")


@pytest.fixture(scope='module')
def soap_hdf():
    with h5py.File(_soap_path, "r") as f:
        yield f


def test_load_identifies_soap(load_kwargs):
    f = pynbody.load(_soap_path, **load_kwargs)
    assert isinstance(f, pynbody.snapshot.soap.SOAPSnap)
    assert pynbody.snapshot.identify(_soap_path, **load_kwargs) is pynbody.snapshot.soap.SOAPSnap


def test_soap_not_claimed_by_swift():
    # SOAPSnap is tried before SwiftSnap, so must not claim SWIFT snapshots; nor must SwiftSnap claim SOAP files
    assert pynbody.snapshot.identify("testdata/SWIFT/snap_0150.hdf5") is pynbody.snapshot.swift.SwiftSnap
    assert not pynbody.snapshot.swift.SwiftSnap._can_load(pathlib.Path(_soap_path))


def test_soap_families(load_kwargs):
    f = pynbody.load(_soap_path, **load_kwargs)
    assert f.families() == [pynbody.family.halo]
    assert len(f) == len(f.halo) == 952


def test_soap_properties(load_kwargs):
    f = pynbody.load(_soap_path, **load_kwargs)
    assert np.allclose(f.properties['a'], 1.0)
    assert np.allclose(f.properties['z'], 0.0)
    assert np.allclose(f.properties['h'], 0.673)
    assert np.allclose(f.properties['omegaM0'], 0.3145)
    assert np.allclose(f.properties['omegaL0'], 0.6855)
    assert np.allclose(f.properties['boxsize'].in_units("Mpc a"), 103.97944183)
    # the time is taken from the copy of the SWIFT header
    assert np.allclose(f.properties['time'].in_units("Gyr"), 13.8229, rtol=1e-4)


def test_soap_loadable_keys(load_kwargs):
    f = pynbody.load(_soap_path, **load_kwargs)
    keys = f.loadable_keys()
    for k in ['pos', 'InputHalos/HaloCentre', 'BoundSubhalo/TotalMass', 'BoundSubhalo/CentreOfMassVelocity',
              'InputHalos/HBTplus/TrackId', 'SO/200_crit/TotalMass', 'SO/500_crit/SORadius',
              'InputHalos/HBTplus/HostFOFId', 'InputHalos/FOF/Centres', 'SOAP/HostHaloIndex']:
        assert k in keys
    assert len(keys) == len(set(keys))
    # there is no single natural choice for these, so they are not mapped
    for k in ['mass', 'vel', 'iord', 'eps']:
        assert k not in keys
    with pytest.raises(KeyError):
        f['mass']
    # nothing from the metadata groups
    assert not any(k.split("/")[0] in ('Cells', 'Cosmology', 'Header', 'Parameters', 'PhysicalConstants', 'SWIFT',
                                       'Units', 'Code') for k in keys)


def test_soap_arrays(load_kwargs, soap_hdf):
    f = pynbody.load(_soap_path, **load_kwargs)
    npt.assert_equal(f['pos'], soap_hdf['InputHalos/HaloCentre'][:])
    npt.assert_equal(f['BoundSubhalo/TotalMass'], soap_hdf['BoundSubhalo/TotalMass'][:])
    npt.assert_equal(f['InputHalos/HBTplus/TrackId'], soap_hdf['InputHalos/HBTplus/TrackId'][:])
    npt.assert_equal(f['SO/200_crit/SORadius'], soap_hdf['SO/200_crit/SORadius'][:])
    npt.assert_equal(f['SOAP/HostHaloIndex'], soap_hdf['SOAP/HostHaloIndex'][:])
    assert f['SO/200_crit/CentreOfMass'].shape == (952, 3)


def test_soap_mapped_array_by_path(load_kwargs, soap_hdf):
    f = pynbody.load(_soap_path, **load_kwargs)
    npt.assert_equal(f['InputHalos/HaloCentre'], soap_hdf['InputHalos/HaloCentre'][:])
    assert f['InputHalos/HaloCentre'].units == f['pos'].units
    # the two are separate arrays, so that a transformation moves only pos
    with f.translate([1.0, 0.0, 0.0]):
        npt.assert_allclose(f['pos'][:, 0], soap_hdf['InputHalos/HaloCentre'][:, 0] + 1.0, rtol=1e-6)
        npt.assert_equal(f['InputHalos/HaloCentre'], soap_hdf['InputHalos/HaloCentre'][:])


def test_soap_units(load_kwargs):
    f = pynbody.load(_soap_path, **load_kwargs)
    assert np.allclose(f['pos'].units.ratio("Mpc a"), 1.0)
    assert np.allclose(f['SO/200_crit/SORadius'].units.ratio("Mpc a"), 1.0)
    assert np.allclose(f['BoundSubhalo/TotalMass'].units.ratio("1e10 Msol"), 1.0)
    assert np.allclose(f['BoundSubhalo/CentreOfMassVelocity'].units.ratio("km s^-1 a"), 1.0)
    # the maximum circular velocity is stored as physical, so has no scalefactor
    assert np.allclose(f['BoundSubhalo/MaximumCircularVelocity'].units.ratio("km s^-1"), 1.0)

    npt.assert_allclose(f['BoundSubhalo/TotalMass'][:3].in_units("Msol"), [6.27e12, 3.39e12, 4.24e12], rtol=1e-3)


def test_soap_take(load_kwargs, soap_hdf):
    take = np.array([3, 10, 500])
    f = pynbody.load(_soap_path, take=take, **load_kwargs)
    assert len(f) == 3
    npt.assert_equal(f['BoundSubhalo/TotalMass'], soap_hdf['BoundSubhalo/TotalMass'][:][take])
    npt.assert_equal(f['SO/200_mean/TotalMass'], soap_hdf['SO/200_mean/TotalMass'][:][take])


@pytest.mark.parametrize("region", [pynbody.filt.Sphere(10.0, (50, 50, 50)),
                                    pynbody.filt.Cuboid(0, 0, 0, 20, 20, 20)])
def test_soap_take_region(load_kwargs, region):
    full = pynbody.load(_soap_path)
    f = pynbody.load(_soap_path, take_region=region, **load_kwargs)
    # whole cells are loaded, so there are more halos than in the region; but none in the region can be missing
    assert len(full[region]) < len(f) < len(full)
    npt.assert_equal(np.sort(f[region]['InputHalos/HBTplus/TrackId']),
                     np.sort(full[region]['InputHalos/HBTplus/TrackId']))


def test_soap_user_mapping(load_kwargs, soap_hdf):
    pynbody.config_parser.set('soap-name-mapping', 'SO/200_crit/TotalMass', 'mass')
    try:
        f = pynbody.load(_soap_path, **load_kwargs)
    finally:
        pynbody.config_parser.remove_option('soap-name-mapping', 'SO/200_crit/TotalMass')
    assert 'mass' in f.loadable_keys()
    npt.assert_equal(f['mass'], soap_hdf['SO/200_crit/TotalMass'][:])
    npt.assert_equal(f['SO/200_crit/TotalMass'], soap_hdf['SO/200_crit/TotalMass'][:])
