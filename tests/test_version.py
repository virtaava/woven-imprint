"""Test that woven_imprint.__version__ equals pyproject.toml version."""

import tomllib
from pathlib import Path

import woven_imprint


def test_version_consistency():
    """Assert that woven_imprint.__version__ equals the version in pyproject.toml."""
    # Resolve pyproject.toml relative to this test file's location
    test_dir = Path(__file__).parent
    pyproject_path = test_dir.parent / "pyproject.toml"
    
    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)
    
    pyproject_version = pyproject["project"]["version"]
    package_version = woven_imprint.__version__
    
    assert (
        package_version == pyproject_version
    ), f"Version mismatch: package has {package_version!r}, pyproject.toml has {pyproject_version!r}"
