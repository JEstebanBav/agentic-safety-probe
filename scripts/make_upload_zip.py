"""
Build the archive to upload to a GPU pod.

WHY NOT PowerShell's Compress-Archive
-------------------------------------
It writes entry names with backslash separators. The ZIP specification mandates
forward slashes, so extracting such an archive on Linux yields a flat directory
of files literally named `src\\model_loader.py` — one file whose name contains a
backslash, not a `src` directory containing `model_loader.py`. The symptom is

    ModuleNotFoundError: No module named 'src'

while `ls` appears to show everything present, which is a genuinely confusing
half hour on a rented pod. Python's zipfile always writes forward slashes.

What goes in: the source, the dataset, the entry-point scripts. What stays out:
the virtualenv (Windows-specific, useless on Linux and hundreds of MB), any
results, and every __pycache__ (compiled for a different Python than the pod's).

Run:
    python make_upload_zip.py
    python make_upload_zip.py --output /tmp/probe.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

# Directories copied whole, minus the exclusions below.
INCLUDE_DIRS = ["src", "data"]

# Individual files at the project root.
INCLUDE_FILES = [
    "run_experiment.py",
    "bootstrap_delta_p.py",
    "validate_corrections.py",
    "requirements_runpod.txt",
    "tool_definitions.json",
    "resume_pod.sh",
]

# Skipped anywhere in the tree. `.venv` and `results` are not walked at all
# since only INCLUDE_DIRS are traversed, but the guard is kept explicit so the
# intent survives someone adding a directory to the list.
EXCLUDE_PARTS = {"__pycache__", ".venv", ".git", "results", "figures"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pdf"}


def _should_skip(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return True
    return path.suffix.lower() in EXCLUDE_SUFFIXES


def build(root: Path, output: Path) -> None:
    root = root.resolve()
    written: list[str] = []
    missing: list[str] = []

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for dirname in INCLUDE_DIRS:
            directory = root / dirname
            if not directory.is_dir():
                missing.append(dirname + "/")
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(root)
                if _should_skip(relative):
                    continue
                # as_posix() is the guarantee that matters here.
                archive.write(path, relative.as_posix())
                written.append(relative.as_posix())

        for filename in INCLUDE_FILES:
            path = root / filename
            if not path.is_file():
                missing.append(filename)
                continue
            archive.write(path, filename)
            written.append(filename)

    size_mb = output.stat().st_size / 1e6
    print(f"{output}  ({size_mb:.1f} MB, {len(written)} files)")

    # Confirm no entry carries a backslash — the whole point of this script.
    with zipfile.ZipFile(output) as archive:
        bad = [n for n in archive.namelist() if "\\" in n]
    if bad:
        raise SystemExit(f"ERROR: {len(bad)} entries contain backslashes: {bad[:3]}")
    print("separators: OK (forward slashes)")

    if missing:
        print(f"\nWARNING: not found, omitted: {', '.join(missing)}")

    print("\nUpload it, then on the pod:")
    print("  cd /workspace")
    print(f"  python -c \"import zipfile; "
          f"zipfile.ZipFile('{output.name}').extractall('agentic-safety-probe')\"")
    print("  cd agentic-safety-probe && bash resume_pod.sh")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="project root (default: .)")
    parser.add_argument("--output", default="probe.zip", help="archive path")
    args = parser.parse_args()
    build(Path(args.root), Path(args.output))


if __name__ == "__main__":
    main()
