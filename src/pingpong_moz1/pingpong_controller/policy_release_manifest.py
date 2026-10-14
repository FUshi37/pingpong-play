"""Fail-closed policy release identity and asset verification.

This module deliberately uses only the Python standard library.  It is safe to
run before ROS, camera, MuJoCo, JAX, or policy initialization and is shared by
the shell config-only path and the Python live entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ReleaseContractError(RuntimeError):
    """Raised when a release cannot be identified exactly."""


_REQUIRED_RELEASE_FIELDS = {
    "release_id",
    "policy_profile",
    "checkpoint_path",
    "checkpoint_sha256",
    "policy_xml_path",
    "policy_xml_sha256",
    "observation_dim",
    "control_rate_hz",
    "deployment_ball_observation_rate_hz",
    "training_ball_observation_rate_hz",
    "action_semantics",
    "reset_mode",
    "policy_dq_mode",
}
_SHA256_HEX_LENGTH = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_asset(repo_root: Path, raw_path: object, label: str) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise ReleaseContractError(f"release {label} is empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _finite_positive(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReleaseContractError(
            f"release {field} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ReleaseContractError(
            f"release {field} must be a finite positive number")
    return number


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ReleaseContractError(
                f"cannot read git identity for {repo_root}: {exc}") from exc
        return result.stdout.rstrip("\n")

    head = run("rev-parse", "HEAD").strip()
    status = run("status", "--short").splitlines()
    return {
        "repo_head": head,
        "repo_dirty": bool(status),
        "repo_status_short": status,
    }


@dataclass(frozen=True)
class ValidatedPolicyRelease:
    manifest_path: Path
    manifest_sha256: str
    release: Mapping[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    policy_xml_path: Path
    policy_xml_sha256: str
    metadata: Mapping[str, Any]

    @property
    def release_id(self) -> str:
        return str(self.release["release_id"])

    @property
    def policy_profile(self) -> str:
        return str(self.release["policy_profile"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "validated": True,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "manifest_release": dict(self.release),
            "release_id": self.release_id,
            "policy_profile": self.policy_profile,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "policy_xml_path": str(self.policy_xml_path),
            "policy_xml_sha256": self.policy_xml_sha256,
            **dict(self.metadata),
        }


def load_release_manifest(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ReleaseContractError(
            f"policy release manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseContractError(
            f"invalid policy release manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseContractError("policy release manifest root must be an object")
    if int(payload.get("schema_version", 0) or 0) != 1:
        raise ReleaseContractError(
            "policy release manifest schema_version must be 1")
    releases = payload.get("releases")
    if not isinstance(releases, list) or not releases:
        raise ReleaseContractError(
            "policy release manifest releases must be a non-empty array")
    return manifest_path, payload, sha256_file(manifest_path)


def validate_policy_release(
    manifest_path: str | Path,
    release_id: str,
    repo_root: str | Path,
    *,
    policy_profile: str | None = None,
    checkpoint_path: str | Path | None = None,
    policy_xml_path: str | Path | None = None,
    observation_dim: int | None = None,
    control_rate_hz: float | None = None,
    deployment_ball_observation_rate_hz: float | None = None,
    policy_dq_mode: str | None = None,
    explicit_overrides: Mapping[str, Any] | None = None,
    include_git: bool = True,
) -> ValidatedPolicyRelease:
    """Resolve one release and verify all identity-bearing fields and bytes."""
    repo = Path(repo_root).expanduser().resolve()
    resolved_manifest, payload, manifest_hash = load_release_manifest(
        manifest_path)
    matches = [
        item for item in payload["releases"]
        if isinstance(item, dict)
        and str(item.get("release_id", "")) == str(release_id)
    ]
    if len(matches) != 1:
        available = sorted(
            str(item.get("release_id")) for item in payload["releases"]
            if isinstance(item, dict) and item.get("release_id") is not None)
        raise ReleaseContractError(
            f"release_id={release_id!r} resolved {len(matches)} entries; "
            f"available={available}")
    release = dict(matches[0])
    missing = sorted(_REQUIRED_RELEASE_FIELDS - set(release))
    if missing:
        raise ReleaseContractError(
            f"release_id={release_id!r} missing fields: {missing}")

    expected_profile = str(release["policy_profile"])
    actual_profile = expected_profile if policy_profile is None else str(policy_profile)
    if actual_profile != expected_profile:
        raise ReleaseContractError(
            f"profile mismatch for release {release_id}: manifest="
            f"{expected_profile!r}, requested={actual_profile!r}")
    expected_obs_dim = int(release["observation_dim"])
    actual_obs_dim = expected_obs_dim if observation_dim is None else int(observation_dim)
    if actual_obs_dim != expected_obs_dim:
        raise ReleaseContractError(
            f"observation_dim mismatch for release {release_id}: "
            f"manifest={expected_obs_dim}, requested={actual_obs_dim}")
    expected_control_hz = _finite_positive(
        release["control_rate_hz"], "control_rate_hz")
    actual_control_hz = (
        expected_control_hz if control_rate_hz is None
        else _finite_positive(control_rate_hz, "requested control_rate_hz"))
    if not math.isclose(actual_control_hz, expected_control_hz, abs_tol=1e-12):
        raise ReleaseContractError(
            f"control_rate_hz mismatch for release {release_id}: "
            f"manifest={expected_control_hz}, requested={actual_control_hz}")
    expected_deploy_hz = _finite_positive(
        release["deployment_ball_observation_rate_hz"],
        "deployment_ball_observation_rate_hz",
    )
    actual_deploy_hz = (
        expected_deploy_hz
        if deployment_ball_observation_rate_hz is None
        else _finite_positive(
            deployment_ball_observation_rate_hz,
            "requested deployment_ball_observation_rate_hz",
        ))
    if not math.isclose(actual_deploy_hz, expected_deploy_hz, abs_tol=1e-12):
        raise ReleaseContractError(
            f"deployment ball rate mismatch for release {release_id}: "
            f"manifest={expected_deploy_hz}, requested={actual_deploy_hz}")
    # This difference is intentional and recorded, never treated as mismatch.
    _finite_positive(
        release["training_ball_observation_rate_hz"],
        "training_ball_observation_rate_hz",
    )
    expected_dq_mode = str(release["policy_dq_mode"])
    actual_dq_mode = expected_dq_mode if policy_dq_mode is None else str(policy_dq_mode)
    if actual_dq_mode != expected_dq_mode:
        raise ReleaseContractError(
            f"POLICY_DQ_MODE mismatch for release {release_id}: "
            f"manifest={expected_dq_mode!r}, requested={actual_dq_mode!r}")

    manifest_checkpoint = _resolved_asset(
        repo, release["checkpoint_path"], "checkpoint_path")
    manifest_xml = _resolved_asset(repo, release["policy_xml_path"], "policy_xml_path")
    requested_checkpoint = (
        manifest_checkpoint if checkpoint_path is None
        else Path(checkpoint_path).expanduser().resolve())
    requested_xml = (
        manifest_xml if policy_xml_path is None
        else Path(policy_xml_path).expanduser().resolve())
    if requested_checkpoint != manifest_checkpoint:
        raise ReleaseContractError(
            f"checkpoint path mismatch for release {release_id}: manifest="
            f"{manifest_checkpoint}, requested={requested_checkpoint}")
    if requested_xml != manifest_xml:
        raise ReleaseContractError(
            f"policy XML path mismatch for release {release_id}: manifest="
            f"{manifest_xml}, requested={requested_xml}")

    for label, asset in (("checkpoint", requested_checkpoint), ("policy XML", requested_xml)):
        if not asset.is_file():
            raise ReleaseContractError(
                f"release {release_id} {label} asset missing: {asset}")
    expected_checkpoint_hash = str(release["checkpoint_sha256"]).lower()
    expected_xml_hash = str(release["policy_xml_sha256"]).lower()
    for label, digest in (
        ("checkpoint_sha256", expected_checkpoint_hash),
        ("policy_xml_sha256", expected_xml_hash),
    ):
        if len(digest) != _SHA256_HEX_LENGTH or any(
                char not in "0123456789abcdef" for char in digest):
            raise ReleaseContractError(
                f"release {release_id} has invalid {label}: {digest!r}")
    actual_checkpoint_hash = sha256_file(requested_checkpoint)
    actual_xml_hash = sha256_file(requested_xml)
    if actual_checkpoint_hash != expected_checkpoint_hash:
        raise ReleaseContractError(
            f"release {release_id} checkpoint SHA256 mismatch: expected="
            f"{expected_checkpoint_hash}, actual={actual_checkpoint_hash}, "
            f"path={requested_checkpoint}")
    if actual_xml_hash != expected_xml_hash:
        raise ReleaseContractError(
            f"release {release_id} policy XML SHA256 mismatch: expected="
            f"{expected_xml_hash}, actual={actual_xml_hash}, path={requested_xml}")

    if str(release_id).startswith("fplc_"):
        contract_path = _resolved_asset(repo, release.get("model_contract_path"), "model_contract_path")
        if (not contract_path.is_file()
                or sha256_file(contract_path) != release.get("model_contract_sha256")):
            raise ReleaseContractError("FPLC model configuration SHA256 mismatch")
        try:
            model_contract = json.loads(contract_path.read_text(encoding="utf-8"))[release_id]
        except (OSError, ValueError, KeyError) as exc:
            raise ReleaseContractError("FPLC model configuration missing or invalid") from exc
        if (release["policy_profile"] != release_id
                or release["observation_dim"] != model_contract["actor_dim"]
                or release.get("actor_schema") != model_contract["saved_nominal_domain"]["fixed_contact_actor_observation_schema"]
                or release["checkpoint_sha256"] != model_contract["sha256"]
                or release["policy_xml_sha256"] != model_contract["policy_xml_sha256"]
                or release.get("reset_preset") != model_contract["native_reset_contract"]["preset"]
                or release.get("reset_preset") != "simcode9_native"
                or release["action_semantics"] != "normalized_velocity_persistent_qref_cartesian_projection_to_existing_robot_rmp"
                or expected_control_hz != 200.0 or expected_deploy_hz != 90.0
                or expected_dq_mode != "raw_coherent"):
            raise ReleaseContractError("FPLC release identity/schema/control contract mismatch")

    metadata: dict[str, Any] = {
        "manifest_content_sha256": canonical_json_sha256(payload),
        "explicit_overrides": dict(explicit_overrides or {}),
        "training_ball_observation_rate_hz": float(
            release["training_ball_observation_rate_hz"]),
        "deployment_ball_observation_rate_hz": expected_deploy_hz,
        "ball_rate_override_status": "verified_deployment_override",
    }
    if include_git:
        metadata.update(_git_metadata(repo))
    return ValidatedPolicyRelease(
        manifest_path=resolved_manifest,
        manifest_sha256=manifest_hash,
        release=release,
        checkpoint_path=requested_checkpoint,
        checkpoint_sha256=actual_checkpoint_hash,
        policy_xml_path=requested_xml,
        policy_xml_sha256=actual_xml_hash,
        metadata=metadata,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--policy-profile")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--policy-xml-path", type=Path)
    parser.add_argument("--observation-dim", type=int)
    parser.add_argument("--control-rate-hz", type=float)
    parser.add_argument("--deployment-ball-rate-hz", type=float)
    parser.add_argument("--policy-dq-mode")
    parser.add_argument("--explicit-override-key", action="append", default=[])
    parser.add_argument(
        "--format", choices=("json", "shell"), default="json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overrides = {
        key: os.environ.get(key)
        for key in args.explicit_override_key
        if key in os.environ
    }
    try:
        validated = validate_policy_release(
            args.manifest,
            args.release_id,
            args.repo_root,
            policy_profile=args.policy_profile,
            checkpoint_path=args.checkpoint_path,
            policy_xml_path=args.policy_xml_path,
            observation_dim=args.observation_dim,
            control_rate_hz=args.control_rate_hz,
            deployment_ball_observation_rate_hz=args.deployment_ball_rate_hz,
            policy_dq_mode=args.policy_dq_mode,
            explicit_overrides=overrides,
        )
    except ReleaseContractError as exc:
        print(f"ERROR: policy release contract failed: {exc}", file=os.sys.stderr)
        return 2
    payload = validated.as_dict()
    if args.format == "json":
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        rows = {
            "POLICY_RELEASE_ID": validated.release_id,
            "POLICY_PROFILE": validated.policy_profile,
            "POLICY_RELEASE_CHECKPOINT_PATH": str(validated.checkpoint_path),
            "POLICY_RELEASE_XML_PATH": str(validated.policy_xml_path),
            "POLICY_RELEASE_MANIFEST_SHA256": validated.manifest_sha256,
            "POLICY_RELEASE_CHECKPOINT_SHA256": validated.checkpoint_sha256,
            "POLICY_RELEASE_XML_SHA256": validated.policy_xml_sha256,
            "POLICY_RELEASE_REPO_HEAD": str(payload.get("repo_head", "")),
            "POLICY_RELEASE_REPO_DIRTY": str(
                bool(payload.get("repo_dirty", False))).lower(),
            "POLICY_RELEASE_CONTRACT_JSON": json.dumps(
                payload, sort_keys=True, separators=(",", ":")),
        }
        for key, value in rows.items():
            print(f"{key}\t{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
