#!/usr/bin/env python3
"""
release.py — bump version, commit, tag, and optionally push.

Usage:
    python release.py <new-version>          # e.g. 1.2.3
    python release.py --dry-run <new-version>
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
CARGO_TOML = ROOT / "Cargo.toml"
CARGO_LOCK = ROOT / "Cargo.lock"
PYPROJECT_TOML = ROOT / "pyproject.toml"


# ── helpers ───────────────────────────────────────────────────────────────────


def run(*args: str, check=True, capture=False) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=check,
        capture_output=capture,
        text=True,
    )


def git(*args: str, check=True, capture=False) -> subprocess.CompletedProcess:
    return run("git", *args, check=check, capture=capture)


def current_version() -> str:
    text = CARGO_TOML.read_text()
    m = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        sys.exit("Could not determine current version from Cargo.toml")
    return m.group(1)


def validate_semver(version: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        sys.exit(f"Invalid version '{version}': must be MAJOR.MINOR.PATCH (e.g. 1.2.3)")


def bump_cargo(new: str) -> None:
    text = CARGO_TOML.read_text()
    # Only replace the first occurrence (the [package] version, not dependency versions).
    updated, n = re.subn(
        r'^(\s*version\s*=\s*)"[^"]+"',
        rf'\g<1>"{new}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n == 0:
        sys.exit("Could not find version field in Cargo.toml")
    CARGO_TOML.write_text(updated)


def bump_pyproject(new: str) -> None:
    text = PYPROJECT_TOML.read_text()
    updated, n = re.subn(
        r'^(\s*version\s*=\s*)"[^"]+"',
        rf'\g<1>"{new}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n == 0:
        sys.exit("Could not find version field in pyproject.toml")
    PYPROJECT_TOML.write_text(updated)


def working_tree_clean() -> bool:
    result = git("status", "--porcelain", capture=True)
    return result.stdout.strip() == ""


def reset(files: list[Path]) -> None:
    git("checkout", "--", *[str(f) for f in files], check=False)
    print("Changes reset.")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump version, commit, tag, and push.")
    parser.add_argument("version", help="New version (MAJOR.MINOR.PATCH)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making any changes",
    )
    args = parser.parse_args()

    new_version = args.version.lstrip("v")
    validate_semver(new_version)
    tag = f"v{new_version}"

    old_version = current_version()
    if old_version == new_version:
        sys.exit(f"Version is already {new_version}.")

    # Guard: uncommitted changes could be swept up in the release commit.
    if not working_tree_clean():
        sys.exit(
            "Working tree has uncommitted changes.\n"
            "Please commit or stash them before releasing."
        )

    print(f"  Current version : {old_version}")
    print(f"  New version     : {new_version}")
    print(f"  Tag             : {tag}")
    print()

    if args.dry_run:
        print("[dry-run] Would update Cargo.toml and pyproject.toml,")
        print(
            f"[dry-run] commit as 'chore: release {tag}', tag '{tag}', then ask to push."
        )
        return

    # ── Apply changes ──────────────────────────────────────────────────────
    changed_files = [CARGO_TOML, PYPROJECT_TOML]
    bump_cargo(new_version)
    bump_pyproject(new_version)
    print(f"Updated Cargo.toml and pyproject.toml to {new_version}")

    # Regenerate Cargo.lock so it reflects the new version.
    run("cargo", "generate-lockfile")
    print("Regenerated Cargo.lock")

    # ── Commit ─────────────────────────────────────────────────────────────
    git("add", str(CARGO_TOML), str(CARGO_LOCK), str(PYPROJECT_TOML))
    git("commit", "-m", f"chore: release {tag}")
    print(f"Committed: chore: release {tag}")

    # ── Tag ────────────────────────────────────────────────────────────────
    git("tag", "-a", tag, "-m", f"Release {tag}")
    print(f"Created tag: {tag}")

    # ── Ask before pushing ─────────────────────────────────────────────────
    print()
    try:
        answer = input(f"Push commit and tag '{tag}' to origin? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""

    if answer == "y":
        git("push", "origin", "HEAD")
        git("push", "origin", tag)
        print(f"Pushed. Release workflow will trigger for {tag}.")
    else:
        print("Push cancelled — resetting commit and tag.")
        git("tag", "-d", tag)
        git("reset", "--soft", "HEAD~1")
        reset(changed_files + [CARGO_LOCK])


if __name__ == "__main__":
    main()
