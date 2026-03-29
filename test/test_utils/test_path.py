import os
from pathlib import Path
from unittest.mock import patch

from tgfm.utils.path import get_curr_parent, get_cwd, get_root_dir, get_scratch


def test_get_root_dir():
    """Verify root dir is exactly three levels above the file."""
    result = get_root_dir()
    assert result.exists()
    assert result.is_dir()
    assert (result / "tgfm").exists()

def test_get_curr_parent():
    """Verify parent dir is exactly two levels above the file."""
    result = get_curr_parent()
    assert result.name == "tgfm"
    assert result.is_dir()

def test_get_scratch():
    """Verify scratch directory construction logic."""
    mock_home = Path("/home/mockuser")
    expected_scratch = Path("/home/mockuser/scratch")

    with patch("pathlib.Path.home", return_value=mock_home):
        result = get_scratch()
        assert result == expected_scratch
        assert result.name == "scratch"
        assert result.parent == mock_home

def test_get_cwd():
    """Verify it returns the current working directory."""
    assert get_cwd() == Path(os.getcwd())
