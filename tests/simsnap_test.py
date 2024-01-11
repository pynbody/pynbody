"""Note that most simsnap tests are in more specific test files."""

import pathlib
import pynbody
import pytest

def test_load_empty_folder():

    pathlib.Path("testdata").joinpath('empty_folder_0001').mkdir(exist_ok=True)

    with pytest.raises(IOError) as excinfo:
        f = pynbody.load("testdata/empty_folder_0001")
    assert "Unable to load" in str(excinfo.value)
