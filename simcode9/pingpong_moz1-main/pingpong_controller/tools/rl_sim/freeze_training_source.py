#!/usr/bin/env python3
"""Freeze the complete simulator/RMP source used by a training launcher."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


EXCLUDED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})
SOURCE_ROOTS = (
    Path("pingpong_controller/tools/rl_sim"),
    Path("ReactiveMotionPlanner/rmp-recovery/recovered_rmp"),
)
TOP_LEVEL_SOURCES = (Path("AGENTS.md"), Path("TRAINING_CURRICULA.md"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout


def iter_source_files(repo_root: Path) -> tuple[Path, ...]:
    """Return every regular source/model asset needed by the simulator."""

    paths: list[Path] = []
    for relative in TOP_LEVEL_SOURCES:
        path = repo_root / relative
        if path.is_file():
            paths.append(path)
    for relative_root in SOURCE_ROOTS:
        root = repo_root / relative_root
        if not root.is_dir():
            raise FileNotFoundError(f"source root is missing: {root}")
        for path in root.rglob("*"):
            if path.is_file() and not EXCLUDED_PARTS.intersection(
                path.relative_to(root).parts
            ):
                paths.append(path)
    return tuple(sorted(set(paths)))


def freeze_source(
    *, repo_root: Path, run_dir: Path, launcher: Path
) -> Path:
    """Create an immutable-by-convention source snapshot and hash manifest."""

    repo_root = repo_root.resolve()
    run_dir = run_dir.resolve()
    launcher = launcher.resolve()
    try:
        launcher.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("launcher must be inside repo_root") from exc
    target = run_dir / "source_snapshot"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite source snapshot: {target}")
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = run_dir / ".source_snapshot_incomplete"
    if temporary.exists():
        raise FileExistsError(f"incomplete source snapshot exists: {temporary}")
    temporary.mkdir()

    files = iter_source_files(repo_root)
    if launcher not in files:
        files = tuple(sorted((*files, launcher)))
    hashes: dict[str, dict[str, object]] = {}
    try:
        for source in files:
            relative = source.relative_to(repo_root)
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            hashes[relative.as_posix()] = {
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        metadata = temporary / "_snapshot_metadata"
        metadata.mkdir()
        (metadata / "git_status.txt").write_text(
            _git(repo_root, "status", "--short", "--branch", "--untracked-files=all"),
            encoding="utf-8",
        )
        (metadata / "git_diff.patch").write_text(
            _git(repo_root, "diff", "--binary"), encoding="utf-8"
        )
        (metadata / "git_diff_cached.patch").write_text(
            _git(repo_root, "diff", "--cached", "--binary"), encoding="utf-8"
        )
        manifest = {
            "schema": "pingpong-training-source-snapshot-v1",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "repo_root": str(repo_root),
            "git_head": _git(repo_root, "rev-parse", "HEAD").strip(),
            "launcher": launcher.relative_to(repo_root).as_posix(),
            "file_count": len(hashes),
            "files": hashes,
        }
        (temporary / "source_snapshot_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target / "source_snapshot_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = freeze_source(
        repo_root=args.repo_root,
        run_dir=args.run_dir,
        launcher=args.launcher,
    )
    print(f"[source-freeze] manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()
