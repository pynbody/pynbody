import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--hdfstream-server", type=str, default=None, help="hdfstream server URL for the test"
    )

@pytest.fixture
def hdfstream_server_url(request):
    return request.config.getoption("--hdfstream-server")
