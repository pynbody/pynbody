try:
    import hdfstream
except ImportError:
    hdfstream = None
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


@pytest.fixture(scope="module", params=[False, True])
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
