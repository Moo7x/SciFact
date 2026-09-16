"""Smoke tests.

These prove the package is installed and importable, and nothing more. They exist so that
CI is red for a real reason on day one rather than green for no reason — a pipeline that has
never failed is a pipeline nobody has tested.
"""

from __future__ import annotations

import subprocess
import sys

import scifact
from scifact.cli import main


def test_package_imports() -> None:
    assert scifact.__version__ == "0.0.0"


def test_cli_info_runs() -> None:
    assert main(["info"]) == 0


def test_cli_is_installed_as_a_console_script() -> None:
    """The entry point must work as an installed command, not only as an import.

    This is the check that `src/` layout buys: it fails if the package was never actually
    installed and the tests were only passing because the repo root happened to be on
    ``sys.path``.
    """
    result = subprocess.run(
        [sys.executable, "-m", "scifact.cli", "info"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "scifact" in result.stdout
