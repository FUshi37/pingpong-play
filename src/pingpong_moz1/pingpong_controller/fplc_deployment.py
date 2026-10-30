"""Versioned simcode9/10 asset and real-workspace contracts (standard library)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "config" / "fplc_model_contracts.json").is_file())
CONTRACT_PATH = REPO_ROOT / "config" / "fplc_model_contracts.json"
FPLC_CONTRACTS = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
SIMCODE10_CONTRACT_PATH = REPO_ROOT / "config" / "fplc_simcode10_model_contracts.json"
SIMCODE10_CONTRACTS = json.loads(SIMCODE10_CONTRACT_PATH.read_text(encoding="utf-8"))
if FPLC_CONTRACTS.keys() & SIMCODE10_CONTRACTS.keys():
    raise ValueError("simcode10 must not replace an existing FPLC model identity")
FPLC_CONTRACTS.update(SIMCODE10_CONTRACTS)
SIMCODE10_PROFILES = tuple(SIMCODE10_CONTRACTS)
FPLC_PROFILES = tuple(FPLC_CONTRACTS)
# Authenticated training bounds; the actor's projection keeps these unchanged.
FPLC_LOWER_M = (-0.348, -0.700, 1.000)
FPLC_UPPER_M = (0.348, -0.280, 1.150)
# Operator-requested real envelope, independent of the training projection.
# Explicit physical limits may tighten it; reference orientation is unchanged.
FPLC_REAL_LOWER_M = (-0.150, -0.600, 0.950)
FPLC_REAL_UPPER_M = (0.150, -0.350, 1.200)
FPLC_REAL_ROLL_PITCH_LIMIT_DEG = (4.0, 4.0)


# Running feedback tolerates a bounded receiver delay without waiting away a
# control period. Arming/reset/stop-hold retain their separate 10 ms checks.
def joint_feedback_max_age_s(value: str | float = "15") -> float:
    try:
        milliseconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("FPLC_JOINT_FEEDBACK_MAX_AGE_MS must be between 10 and 20") from exc
    if not math.isfinite(milliseconds) or not 10.0 <= milliseconds <= 20.0:
        raise ValueError("FPLC_JOINT_FEEDBACK_MAX_AGE_MS must be between 10 and 20")
    return milliseconds / 1000.0


FPLC_JOINT_FEEDBACK_MAX_AGE_S = joint_feedback_max_age_s(
    os.environ.get("FPLC_JOINT_FEEDBACK_MAX_AGE_MS", "15"))
FPLC_JOINT_FEEDBACK_WAIT_S = 0.005
FPLC_JOINT_FEEDBACK_MAX_GAP_S = (
    FPLC_JOINT_FEEDBACK_MAX_AGE_S + FPLC_JOINT_FEEDBACK_WAIT_S)


def fplc_assets(profile: str, repo_root: Path = REPO_ROOT) -> tuple[Path, Path]:
    contract = FPLC_CONTRACTS[profile]
    return (repo_root / contract["checkpoint_path"],
            repo_root / contract["policy_xml_path"])


def fplc_workspace_intersection(limits):
    """Intersect physical limits with the operator-requested real envelope."""
    import math
    if len(limits) != 3:
        raise ValueError("FPLC requires exactly three real workspace limits")
    output = []
    for axis, limit in enumerate(limits):
        if limit is None or len(limit) != 2:
            raise ValueError("FPLC requires all three real workspace limits")
        low, high = map(float, limit)
        if not all(math.isfinite(v) for v in (low, high)) or low >= high:
            raise ValueError("FPLC workspace limits must be finite increasing pairs")
        low = max(low, FPLC_REAL_LOWER_M[axis])
        high = min(high, FPLC_REAL_UPPER_M[axis])
        if low >= high:
            raise ValueError("real workspace has no intersection with the FPLC workspace")
        output.append((low, high))
    return tuple(output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=FPLC_PROFILES)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    for axis in "xyz":
        parser.add_argument(f"--{axis}-limits")
    args = parser.parse_args()
    model, xml = fplc_assets(args.profile, args.repo_root)
    print(f"FPLC_MODEL_PATH\t{model}")
    print(f"FPLC_POLICY_XML_PATH\t{xml}")
    print(f"FPLC_OBS_DIM\t{FPLC_CONTRACTS[args.profile]['actor_dim']}")
    print(f"FPLC_REAL_ROLL_PITCH_LIMITS_DEG\t{','.join(map(str, FPLC_REAL_ROLL_PITCH_LIMIT_DEG))}")
    if args.x_limits is not None:
        pairs = [tuple(map(float, value.split(','))) for value in (args.x_limits, args.y_limits, args.z_limits)]
        for axis, pair in zip("XYZ", fplc_workspace_intersection(pairs)):
            print(f"DONE_EE_{axis}_LIMITS_M\t{pair[0]},{pair[1]}")
