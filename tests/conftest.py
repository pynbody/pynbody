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

@pytest.fixture
def hdfstream_server_url(request):
    if hdfstream is not None:
        hdfstream.verify_cert(not request.config.getoption("--no-verify-cert"))
    return request.config.getoption("--hdfstream-server")
