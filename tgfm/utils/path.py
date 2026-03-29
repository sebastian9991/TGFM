from pathlib import Path


def get_root_dir() -> Path:
    """Return the project root directory.

    Defined as three levels above this file.
    """
    return Path(__file__).parent.parent.parent


def get_curr_parent() -> Path:
    """Return the parent directory two levels above this file."""
    return Path(__file__).parent.parent


def get_scratch() -> Path:
    """Return the user's scratch directory."""
    home = Path.home()
    scratch_dir = home.parent / home.name / 'scratch'
    return scratch_dir


def get_cwd() -> Path:
    """Return the current working directory."""
    return Path.cwd()
