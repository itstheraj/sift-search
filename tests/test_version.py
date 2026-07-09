import importlib.metadata as md
import tomllib
from pathlib import Path

import sift

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_installed_distribution():
    assert sift.__version__ == md.version("sift-search-kde")


def test_version_matches_pyproject():
    """__version__ is derived from metadata, so it must track pyproject."""
    declared = tomllib.loads(_PYPROJECT.read_text())["project"]["version"]
    assert sift.__version__ == declared


def test_cli_reports_the_same_version(capsys):
    import pytest

    from sift import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"sift {sift.__version__}"
