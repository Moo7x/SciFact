"""Command-line entry point.

Every number that ends up in a write-up must come from a command in here. That constraint
is the point of the module: a result you cannot re-run is a result you cannot defend, and
the gap between "I remember getting 0.62" and "here is the command that produces 0.62" is
the whole difference between a portfolio project and a screenshot.
"""

from __future__ import annotations

import argparse

from scifact import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scifact", description=__doc__)
    parser.add_argument("--version", action="version", version=f"scifact {__version__}")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("info", help="report environment and data status")

    args = parser.parse_args(argv)

    if args.command == "info":
        return _info()

    parser.print_help()
    return 0


def _info() -> int:
    import platform
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent
    manifest = repo_root / "data" / "MANIFEST.json"

    print(f"scifact      {__version__}")
    print(f"python       {sys.version.split()[0]}")
    print(f"platform     {platform.platform()}")
    print(f"data         {'present' if manifest.exists() else 'not downloaded'}")
    if not manifest.exists():
        print("             run: python scripts/download_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
