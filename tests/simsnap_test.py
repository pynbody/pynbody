"""Note that most simsnap tests are in more specific test files."""

import pathlib

import pytest

import pynbody


def test_load_empty_folder():

    pathlib.Path("testdata").joinpath('empty_folder_0001').mkdir(exist_ok=True)

    with pytest.raises(IOError) as excinfo:
        f = pynbody.load("testdata/empty_folder_0001")
    assert "Is a directory" in str(excinfo.value) or "Permission denied" in str(excinfo.value) # latter is the windows error
