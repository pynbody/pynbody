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

@pytest.fixture(scope="module", params=[False, True])
def load_kwargs(request):
    if request.param:
        # This is a remote file test. Get the server URL.
        server = request.config.getoption("--hdfstream-server")
        if server is None:
            pytest.skip("hdfstream server URL not specified")
        # Check we have the client module
        if hdfstream is None:
            pytest.skip("hdfstream client module not available")
        # We might not have a valid certificate in development builds of the server
        hdfstream.verify_cert(not request.config.getoption("--no-verify-cert"))
        # Pynbody test data might be in a subdirectory on the server
        prefix = request.config.getoption("--testdata-prefix")
        # Open and return the remote directory
        return {"remote_dir" : hdfstream.open(server, prefix)}
    else:
        # This is a local file test, so no extra args are needed
        return {}
