try:
    import hdfstream
except ImportError:
    hdfstream = None
import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--hdfstream-server", type=str, default=None, help="hdfstream server URL for the test"
    )
    parser.addoption(
        "--no-verify-cert", action="store_true", default=False, help="Don't verify SSL certificates if set"
    )
    parser.addoption(
        "--testdata-prefix", type=str, default="/", help="Location of pynbody testdata dir on the server"
    )


def _open_remote_dir(request):
    """
    Open the remote directory with the test data
    """
    # Get the server URL.
    server = request.config.getoption("--hdfstream-server")
    if server is None:
        pytest.skip("hdfstream server URL not specified")
    # Check we have the client module. If a server is specified but the
    # client module is not present, tests should fail rather than skip.
    if hdfstream is None:
        raise RuntimeError("Server URL was specified but the hdfstream module could not be imported")
    # We might not have a valid certificate in development builds of the server
    hdfstream.verify_cert(not request.config.getoption("--no-verify-cert"))
    # Pynbody test data might be in a subdirectory on the server
    prefix = request.config.getoption("--testdata-prefix")
    # Open and return the remote directory
    return hdfstream.open(server, prefix)


@pytest.fixture(scope="module")
def remote_kwargs(request):
    """
    Returns the keyword args for load() to open a remote file.
    """
    return {"remote_dir" : _open_remote_dir(request)}


@pytest.fixture(scope="module", params=[False, True])
def load_kwargs(request):
    """
    This fixture can be used to repeat tests on local and remote files.
    """
    if request.param:
        # This is a remote file test
        return {"remote_dir" : _open_remote_dir(request)}
    else:
        # This is a local file test, so no extra args are needed
        return {}


class _StandInRemoteDataset:
    """Stands in for a remote dataset (such as hdfstream's RemoteDataset), by wrapping an h5py dataset in an object
    that is not one, and that reads as hdfstream's does: read_direct only into C-contiguous arrays, converting only
    where that is safe; and indexing by an array of rows, recording every such array it is given"""

    def __init__(self, dataset):
        self._dataset = dataset
        self.shape, self.dtype, self.ndim, self.attrs = dataset.shape, dataset.dtype, dataset.ndim, dataset.attrs
        self.index_arrays = []

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        rows = key[0] if isinstance(key, tuple) else key
        if isinstance(rows, (np.ndarray, list)):
            self.index_arrays.append(np.array(rows))
            return self._dataset[np.asarray(rows)]
        return self._dataset[key]

    def read_direct(self, array, source_sel=None, dest_sel=None):
        if not array.flags.c_contiguous:
            raise RuntimeError("Destination for read_direct() must be C contiguous")
        if array.dtype != self.dtype and not np.can_cast(self.dtype, array.dtype, casting='safe'):
            raise RuntimeError(f"Cannot safely cast {self.dtype} to {array.dtype}")
        self._dataset.read_direct(array, source_sel=source_sel)


@pytest.fixture
def stand_in_remote_dataset():
    """The class _StandInRemoteDataset, for tests of reading datasets other than h5py's"""
    return _StandInRemoteDataset
