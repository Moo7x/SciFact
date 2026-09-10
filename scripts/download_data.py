"""Download the SciFact release into ``data/``.

Design notes, because this is the first thing that runs in the repository:

* **Standard library only.** The first script anyone executes should not be able to fail
  because of dependency resolution. It should fail only for reasons about the data.
* **A manifest is written, not just the files.** ``data/MANIFEST.json`` records the source
  URL, a SHA-256 of the archive, and a SHA-256 plus record count for every extracted file.
  The data is gitignored, so the manifest is the only committed proof of *which* data
  produced a given result. Without it, "re-run the script" is not reproduction, it is hope.
* **Extraction is filtered.** ``tarfile`` will happily write outside the destination if the
  archive contains ``../`` members or absolute paths (CVE-2007-4559). ``filter="data"``
  refuses those. The archive here is trusted, but the habit of unpacking trusted archives
  unsafely is what makes the untrusted case go wrong.

Usage::

    python scripts/download_data.py
    python scripts/download_data.py --verify     # re-check checksums, download nothing
    python scripts/download_data.py --force      # re-download over an existing copy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

# Source: https://github.com/allenai/scifact -- script/download-data.sh
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "MANIFEST.json"

CHUNK = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    """Streamed SHA-256, so a large file never has to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl_records(path: Path) -> int | None:
    """Count records in a JSONL file, or ``None`` if it is not JSONL.

    Every line is parsed rather than merely counted. A file that counts as N lines but
    fails to parse on line N-1 is a corrupt download that would otherwise be discovered
    much later, inside a training loop.
    """
    if path.suffix != ".jsonl":
        return None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}: malformed JSON on line {lineno}: {exc}") from exc
            count += 1
    return count


def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``, reporting progress."""
    print(f"  fetching {url}")
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed, non-user URL
        total = int(response.headers.get("Content-Length", 0))
        seen = 0
        with dest.open("wb") as handle:
            while chunk := response.read(CHUNK):
                handle.write(chunk)
                seen += len(chunk)
                if total:
                    pct = 100.0 * seen / total
                    print(f"\r  {seen / 1e6:.1f} / {total / 1e6:.1f} MB  ({pct:.0f}%)", end="")
    print()


def _prune_apple_double(root: Path) -> list[str]:
    """Delete macOS AppleDouble metadata (``._name``, ``__MACOSX/``). Returns what was removed.

    The SciFact archive was built on macOS, so alongside ``data/`` it carries ``._data`` --
    a 220-byte resource fork. These are not part of the dataset.

    They caused a real bug worth remembering. The flattening step below originally read
    "if there is exactly one top-level entry and it is a directory, descend into it".
    ``._data`` made the count two, the condition went false, and everything was extracted
    one level too deep as ``data/data/``. It failed *silently and plausibly*: a valid tree,
    just the wrong one. A heuristic about an artifact's shape has to be checked against the
    actual artifact, not against the shape such artifacts usually have.
    """
    removed: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.name.startswith("._") or path.name == "__MACOSX":
            removed.append(path.name)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return removed


def extract(archive: Path, dest: Path) -> None:
    """Extract ``archive`` into ``dest``, flattening any single top-level directory."""
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        with tarfile.open(archive, "r:gz") as tar:
            # filter="data" rejects absolute paths, parent-directory escapes, device
            # nodes and symlinks pointing outside the destination.
            tar.extractall(staging, filter="data")

        junk = _prune_apple_double(staging)
        if junk:
            print(f"  pruned {len(junk)} macOS metadata file(s): {', '.join(sorted(set(junk)))}")

        entries = list(staging.iterdir())
        # The AllenAI archive wraps everything in a single directory. Flatten it so the
        # layout does not depend on how they happened to build the tarball.
        root = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
        if root is staging:
            print(f"  note: {len(entries)} top-level entries, not flattening")

        for item in root.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))


def build_manifest(archive_sha: str, archive_bytes: int) -> dict[str, object]:
    files: dict[str, object] = {}
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.json", ".gitkeep"}:
            continue
        entry: dict[str, object] = {
            "sha256": sha256_of(path),
            "bytes": path.stat().st_size,
        }
        records = count_jsonl_records(path)
        if records is not None:
            entry["records"] = records
        files[str(path.relative_to(DATA_DIR)).replace("\\", "/")] = entry

    return {
        "source_url": SCIFACT_URL,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "files": files,
    }


def verify() -> int:
    """Re-hash every file and compare against the manifest. Returns a process exit code."""
    if not MANIFEST_PATH.exists():
        print("No MANIFEST.json. Run without --verify to download.", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    recorded: dict[str, dict[str, object]] = manifest["files"]
    problems = 0

    for name, entry in recorded.items():
        path = DATA_DIR / name
        if not path.exists():
            print(f"  MISSING  {name}")
            problems += 1
            continue
        actual = sha256_of(path)
        if actual != entry["sha256"]:
            print(f"  CHANGED  {name}")
            print(f"           expected {entry['sha256']}")
            print(f"           actual   {actual}")
            problems += 1
        else:
            records = entry.get("records")
            suffix = f"  ({records} records)" if records is not None else ""
            print(f"  ok       {name}{suffix}")

    if problems:
        print(f"\n{problems} problem(s). The data on disk is not the data that was recorded.")
        return 1
    print(f"\nAll {len(recorded)} file(s) match the manifest.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="re-check checksums against the manifest"
    )
    parser.add_argument("--force", action="store_true", help="re-download over an existing copy")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        return verify()

    if MANIFEST_PATH.exists() and not args.force:
        print("Data already present. Use --verify to check it, or --force to re-download.")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "scifact.tar.gz"
        download(SCIFACT_URL, archive)
        archive_sha = sha256_of(archive)
        archive_bytes = archive.stat().st_size
        print(f"  sha256 {archive_sha}")
        print("  extracting")
        extract(archive, DATA_DIR)

    print("  building manifest")
    manifest = build_manifest(archive_sha, archive_bytes)
    # newline="\n" explicitly: this file is committed and is the reproducibility record.
    # Python's default text mode would write CRLF on Windows and LF on Linux, so the same
    # data would produce a different file depending on who ran the script.
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")

    print(f"\nData in {DATA_DIR}")
    files: dict[str, dict[str, object]] = manifest["files"]  # type: ignore[assignment]
    width = max(len(n) for n in files) if files else 0
    for name, entry in files.items():
        size_mb = int(entry["bytes"]) / 1e6  # type: ignore[call-overload]
        records = entry.get("records")
        tail = f"{records:>7,} records" if records is not None else " " * 15
        print(f"  {name:<{width}}  {size_mb:>7.2f} MB  {tail}")

    print(f"\nManifest: {MANIFEST_PATH.relative_to(REPO_ROOT)} (committed; the data is not)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
