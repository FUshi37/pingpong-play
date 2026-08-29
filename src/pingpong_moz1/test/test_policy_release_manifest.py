#!/usr/bin/env python3

import hashlib
import json
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.policy_release_manifest import (  # noqa: E402
    ReleaseContractError,
    validate_policy_release,
)


def _write_release(tmp_path: Path, *, checkpoint_hash: str | None = None):
    checkpoint = tmp_path / "policy.pkl"
    xml = tmp_path / "robot.xml"
    checkpoint.write_bytes(b"frozen actor fixture\n")
    xml.write_text("<mujoco/>\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    release = {
        "release_id": "fixture_release",
        "policy_profile": "simcode5_pure_actuator67",
        "checkpoint_path": checkpoint.name,
        "checkpoint_sha256": checkpoint_hash or digest(checkpoint),
        "policy_xml_path": xml.name,
        "policy_xml_sha256": digest(xml),
        "observation_dim": 67,
        "control_rate_hz": 200.0,
        "deployment_ball_observation_rate_hz": 90.0,
        "training_ball_observation_rate_hz": 60.0,
        "action_semantics": "acceleration_integrated_once_to_dq_once_to_q",
        "reset_mode": "racket_launch",
        "policy_dq_mode": "raw_coherent",
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "releases": [release],
    }), encoding="utf-8")
    return manifest, checkpoint, xml


def test_release_hash_mismatch_fails_closed(tmp_path):
    manifest, _checkpoint, _xml = _write_release(
        tmp_path, checkpoint_hash="0" * 64)
    with pytest.raises(ReleaseContractError, match="checkpoint SHA256 mismatch"):
        validate_policy_release(
            manifest, "fixture_release", tmp_path, include_git=False)


def test_release_missing_asset_is_clear(tmp_path):
    manifest, checkpoint, _xml = _write_release(tmp_path)
    checkpoint.unlink()
    with pytest.raises(ReleaseContractError, match="checkpoint asset missing"):
        validate_policy_release(
            manifest, "fixture_release", tmp_path, include_git=False)


def test_release_profile_mismatch_fails_closed(tmp_path):
    manifest, _checkpoint, _xml = _write_release(tmp_path)
    with pytest.raises(ReleaseContractError, match="profile mismatch"):
        validate_policy_release(
            manifest,
            "fixture_release",
            tmp_path,
            policy_profile="simcode6_qvel67",
            include_git=False,
        )


def test_dirty_repo_metadata_and_verified_rate_override(tmp_path, monkeypatch):
    manifest, _checkpoint, _xml = _write_release(tmp_path)
    monkeypatch.setattr(
        "pingpong_controller.policy_release_manifest._git_metadata",
        lambda _root: {
            "repo_head": "abc123",
            "repo_dirty": True,
            "repo_status_short": [" M user_file.py"],
        },
    )
    validated = validate_policy_release(
        manifest,
        "fixture_release",
        tmp_path,
        explicit_overrides={"POLICY_PROFILE": "simcode5_pure_actuator67"},
    )
    payload = validated.as_dict()
    assert payload["repo_dirty"] is True
    assert payload["repo_status_short"] == [" M user_file.py"]
    assert payload["training_ball_observation_rate_hz"] == 60.0
    assert payload["deployment_ball_observation_rate_hz"] == 90.0
    assert payload["ball_rate_override_status"] == "verified_deployment_override"
    assert payload["explicit_overrides"]["POLICY_PROFILE"] == (
        "simcode5_pure_actuator67")
