import shutil
from pathlib import Path

import h5py
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises

import pynbody
import pynbody.snapshot.remote_swift
import pynbody.test_utils

try:
    import hdfstream
except ImportError:
    hdfstream = None

#
# This is a copy of most of the tests from
# pynbody.snapshot.swift.SwiftSnap.  We're going to run these twice:
# first accessing a local file with h5py then accessing a remote file
# with hdfstream. In both cases we use the multi-file manager from
# pynbody/snapshot/remote_swift.py.
#
server_url = "https://dataweb.cosma.dur.ac.uk:8443/hdfstream"
testdata_dir = "Tests/pynbody/"
test_args = [
    {
        "server" : None,
        "dir" : "",
        "class" : pynbody.snapshot.remote_swift.LocalSwiftSnap,
    },
    {
        "server" : server_url,
        "dir" : testdata_dir,
        "class" : pynbody.snapshot.remote_swift.RemoteSwiftSnap,
    },
]

@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("swift")

def pynbody_load(params, path, **kwargs):
    snap_class = params["class"]
    filename = params["dir"]+path
    if params["server"] is None:
        return snap_class(filename, **kwargs)
    else:
        return snap_class(filename, server=params["server"], **kwargs)

@pytest.mark.skipif(hdfstream is None, reason="hdfstream module is not available")
@pytest.mark.parametrize("test_args", test_args)
def test_swift_properties(test_args):
    f = pynbody_load(test_args, "testdata/SWIFT/snap_0150.hdf5")
    assert np.allclose(f.properties['a'], 0.38234515)
    assert np.allclose(f.properties['z'], 1.61543791)
    assert np.allclose(f.properties['h'], 0.703)
    assert np.allclose(f.properties['boxsize'].in_units("Mpc a h^-1", **f.conversion_context()), 100.)
    assert np.allclose(f.properties['omegaM0'], 0.276)
    assert np.allclose(f.properties['omegaL0'], 0.724)
    assert np.allclose(f.properties['omegaNu0'], 0.0)

@pytest.mark.skipif(hdfstream is None, reason="hdfstream module is not available")
@pytest.mark.parametrize("test_args", test_args)
def test_swift_arrays(test_args):
    f = pynbody_load(test_args, "testdata/SWIFT/snap_0150.hdf5")
    assert np.allclose(f.dm['pos'].units.ratio("Mpc a", **f.conversion_context()), 1.0)
    assert np.allclose(f.dm['vel'].units.ratio("km s^-1", **f.conversion_context()), 1.0)
    # the reason the following isn't exactly 1.0 is because our solar mass is slightly different to swift's
    # (the pynbody value is outdated but it will need some work to think about how to fix this without
    # breaking backwards compatibility)
    assert np.allclose(f.dm['mass'].units.ratio("1e10 Msol", **f.conversion_context()), 1.0)
    assert np.allclose(f.dm['vel'][::50000], np.array([[-249.5395 ,   122.65865 , -144.79892 ],
                                             [  75.57313 ,  -51.598354 , 250.10258 ],
                                             [-139.62218 , -132.5298   , 479.02545 ],
                                             [ 147.22443 , -168.17662  ,-249.17387 ],
                                             [  27.643984,  161.06497  ,  21.430338],
                                             [  79.65777 ,   25.674492 , -45.813534]]))
    assert np.allclose(f.gas['pos'][::50000], np.array([[  2.54333146,   0.56501471,   3.08184457],
                                                        [ 27.33643304,  78.82288643,  62.55081956],
                                                        [ 50.74337741, 134.5097336 ,  95.28821923],
                                                        [ 86.53860045,  83.61342416, 129.95370508],
                                                        [111.7177823 ,  24.85736548,  55.55540164],
                                                        [128.76603044,  73.44601203, 139.65444299]]))

    assert np.allclose(float(f.dm['mass'].sum()+f.gas['mass'].sum()), 10895511.25)

def _assert_multifile_contents_is_sensible(f):
    assert len(f.dm) == 262144
    assert np.allclose(f.dm['pos'][::50000], np.array([[0.90574413, 1.23148826, 1.08457044],
                                                       [63.52396676, 5.42349734, 27.65864702],
                                                       [114.45492871, 9.92260142, 54.59853019],
                                                       [32.11092237, 32.28962438, 81.32341474],
                                                       [83.46050257, 20.91647755, 116.546989],
                                                       [125.83232028, 49.9732396, 72.2264199]]))

@pytest.mark.skipif(hdfstream is None, reason="hdfstream module is not available")
@pytest.mark.parametrize("test_args", test_args)
def test_swift_multifile_with_vds(test_args):
    f = pynbody_load(test_args, "testdata/SWIFT/multifile_with_vds/snap_0000.hdf5")
    assert isinstance(f, test_args["class"])
    assert len(f._hdf_files) == 1
    _assert_multifile_contents_is_sensible(f)

@pytest.mark.skipif(hdfstream is None, reason="hdfstream module is not available")
@pytest.mark.parametrize("test_args", test_args)
def test_swift_multifile_without_vds(test_args):
    f = pynbody_load(test_args, "testdata/SWIFT/multifile_without_vds/snap_0000")
    assert isinstance(f, test_args["class"])
    assert len(f._hdf_files) == 10
    _assert_multifile_contents_is_sensible(f)

@pytest.mark.skipif(hdfstream is None, reason="hdfstream module is not available")
@pytest.mark.parametrize("test_args", test_args)
def test_swift_singlefile_partial_loading(test_args):
    f = pynbody_load(test_args,
                     "testdata/SWIFT/snap_0150.hdf5",
                     take_swift_cells=[5,15,20,25])
    assert len(f.dm) == 1849
    assert len(f.gas) == 1882
    assert (f.dm['iord'][::100] == [ 16468,   9172,  41176,  49874,   9342,  10234,  33908,  25852,
                                     42628,  34566,  26566,  10818, 502992,  67776,  34896,  68286,
                                     28052,  35988,  69524]).all()
    assert (f.gas['iord'][::100] == [16471,  8667, 49109, 57813, 17913, 26115, 10115, 50429, 50437,
                                     26367,  2503, 26825, 10827, 59463, 26703, 51909, 27927, 12057,
                                     36761]).all()

@pytest.mark.skipif(hdfstream is None, reason="hdfstream module is not available")
@pytest.mark.parametrize("test_args", test_args)
def test_swift_multifile_partial_loading(test_args):
    f = pynbody_load(test_args,
                     "testdata/SWIFT/multifile_without_vds/snap_0000",
                     take_swift_cells=[0,5,20,200])

    assert len(f) == 2048

    assert np.allclose(f['pos'][::100],
                       [[  0.90574413,   1.23148826,   1.08457044],
                          [  5.38928105,   3.39531932,  12.16627829],
                          [  0.94682741,   9.99268563,  14.42636433],
                          [ 16.65709673,   3.38461118,   5.47390269],
                          [  9.87028024,  14.62473673,   0.98727947],
                          [ 14.47500443,  16.773776  ,   9.74889636],
                          [ 54.2541276 ,  25.47568787,  14.10461116],
                          [ 58.85629125,  34.2701821 ,   7.44578035],
                          [ 67.83300647,  18.60943356,   0.78988041],
                          [ 63.32954201,  27.78617349,   0.87468019],
                          [ 67.95890776,  27.66693028,  16.39712484],
                          [  3.26246761,   1.20717194, 103.37377035],
                          [  5.47939785,  14.4774205 ,  92.18117726],
                          [  9.93556488,   5.69315011,  92.2609549 ],
                          [ 16.79507021,   7.98814361, 103.17866188],
                          [ 10.07660354,  16.88916246, 105.41116598],
                          [  1.3004053 ,  36.6884804 ,  81.22794421],
                          [  7.71374451,  45.49463307,  74.65383169],
                          [ 12.14706372,  38.8390906 ,  76.86117397],
                          [ 16.59894837,  36.67247821,  85.82433427],
                          [  9.79404938,  52.28051827,  81.30710868]])

@pytest.mark.skipif(hdfstream is None, reason="hdfstream module is not available")
@pytest.mark.parametrize("test_args", test_args)
def test_swift_multifile_partial_loading_order_insensitive(test_args):
    f = pynbody_load(test_args,
                     "testdata/SWIFT/multifile_without_vds/snap_0000",
                     take_swift_cells=[0, 5, 20, 200])
    f2 = pynbody_load(test_args,
                      "testdata/SWIFT/multifile_without_vds/snap_0000",
                     take_swift_cells=[0, 5, 20, 200][::-1])
    assert (f['iord'] == f2['iord']).all()

@pytest.mark.skipif(hdfstream is None, reason="hdfstream module is not available")
@pytest.mark.parametrize("test_args", test_args)
def test_swift_vds_partial_loading(test_args):
    f = pynbody_load(test_args,
                     "testdata/SWIFT/multifile_with_vds/snap_0000.hdf5", # <-- VDS file
                     take_swift_cells=[0,5,20,200])
    f2 = pynbody_load(test_args,
                      "testdata/SWIFT/multifile_without_vds/snap_0000",
                      take_swift_cells=[0, 5, 20, 200][::-1])
    assert (f['iord'] == f2['iord']).all()

@pytest.mark.skipif(hdfstream is None, reason="hdfstream module is not available")
@pytest.mark.parametrize("test_args", test_args)
def test_swift_fof_groups(test_args):
    f = pynbody_load(test_args, "testdata/SWIFT/snap_0150.hdf5")
    h = f.halos(priority = ['HaloNumberCatalogue'])

    with raises(KeyError):
        _ = h[0]

    assert len(h[1]) == 444
    assert len(h[1].dm) == 247
    assert len(h[1].gas) == 197

    assert len(h) < 1000

@pytest.mark.skipif(hdfstream is None, reason="hdfstream module is not available")
@pytest.mark.parametrize("test_args", test_args)
def test_swift_dtypes(test_args):
    f = pynbody_load(test_args, "testdata/SWIFT/snap_0150.hdf5")
    assert np.issubdtype(f['iord'].dtype, np.integer)
    assert np.issubdtype(f['pos'].dtype, np.floating)

@pytest.mark.skipif(hdfstream is None, reason="hdfstream module is not available")
@pytest.mark.parametrize("test_args", test_args)
@pytest.mark.parametrize('test_region',
                         [pynbody.filt.Sphere(50., (50., 50., 50.)),
                         pynbody.filt.Cuboid(-20.0)]) # note the cuboid test region wraps around the box
def test_swift_take_geometric_region(test_args, test_region):
    f = pynbody_load(test_args, "testdata/SWIFT/snap_0150.hdf5",
                     take_region = test_region)

    f_full = pynbody_load(test_args, "testdata/SWIFT/snap_0150.hdf5")

    assert len(f) < len(f_full)

    assert np.all(f[test_region]['iord'] == f_full[test_region]['iord'])

@pytest.mark.skipif(hdfstream is None, reason="hdfstream module is not available")
@pytest.mark.parametrize("test_args", test_args)
def test_swift_scalefactor_in_units(test_args):
    f = pynbody_load(test_args, "testdata/SWIFT/snap_0150.hdf5")

    # naively, one would assume cm^2 s^-2, but actual units header says 1e10 a^2 cm^2 s^-2
    # So this tests pynbody respects that. Note that we don't give the scalefactor context,
    # so if scalefactor exponents are wrong, this will raise an exception
    npt.assert_allclose((f.gas['u'].units).in_units("km^2 s^-2 a^-2"), 1.0)

    npt.assert_allclose(f.gas['pos'].units.in_units("Mpc a"), 1.0)
