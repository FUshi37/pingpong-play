#!/usr/bin/env python3
"""Verify the frozen GPU1 V153 training-source release without running it."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
EXPECTED_SHA256 = {
    "train_juggle_mjx_curriculum.py": (
        "390d76daf5a8b1bc37237a3389d945ac29e10372a25d4f5da1a80eaf23b8a3d3"
    ),
    "train_juggle_mjx_ppo.py": (
        "8a0bebb21c8613676095e52c26e33a7c04d90528eb60dae54edf8c18901380d3"
    ),
    "mjx_juggle_env.py": (
        "254eb1a94a1286d7c41fa5e3d9548dfdef41f47bd2d3fe5ea639483f41bfb7c4"
    ),
    "mjx_smoke.py": (
        "fd0c4748039ebb53e022857f3cbf673ef50f0ce2ee910655742bb57ad5ce8f68"
    ),
    "ball_mass_measurement.py": (
        "c298615d811666e1d1a8702ede56bef49482b8bc9f5eab408c083e3e45de7b88"
    ),
    "run_with_host_memory_guard.sh": (
        "82b38cf6845501e58b0d37a6c98e34b6dc04f99eaedbc1c6c6e226054e3b8470"
    ),
    "moz1_pd.xml": (
        "7d98f2adfdbad6082be0defcec2dbd0cbbcaf1f0fc06ce45ba424b5b3257cc92"
    ),
    "V153_FRONTIER_ACCEPTANCE.json": (
        "c34d208da2ec4c5323a04e7c87d4ed6926a66647c1b94e2bfccbd00f72c490f7"
    ),
    "V153_BOUNDED_ACCEPTANCE.json": (
        "3af3d86c354eb1fdcb32afb28077a4e12cafc4628b2f6fa2d8d90a6e1c4881cc"
    ),
    "ball_mass_contract_manifest.json": (
        "d68dec984413dae93b9d88820b7ed674e15111a7edbd4051d4dafd6871d1883f"
    ),
    "run_v153_initial_from_v152_stage51.sh": (
        "d63634cbdcaf91f9a7658f183592560572dea36a832658efff861d632f7e0712"
    ),
    "run_v153_resume1_selected_source.sh": (
        "3d905863c68fe44fa20836f078cf4bde305d214a1ecd1be3f843be8473ebc96c"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for name, expected in EXPECTED_SHA256.items():
        path = ROOT / name
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"SHA-256 mismatch for {name}: {actual}")

    trainer = (ROOT / "train_juggle_mjx_curriculum.py").read_text(
        encoding="utf-8"
    )
    required_fragments = (
        "dual_domain_homotopy_v153",
        'record_new3_sim2real_v153_b2_b3_energy_{label}_60hz',
        '(0.25, "p025")',
        "V153 expected 66 stages",
    )
    for fragment in required_fragments:
        if fragment not in trainer:
            raise SystemExit(f"missing V153 trainer fragment: {fragment}")

    for launcher in (
        "run_v153_initial_from_v152_stage51.sh",
        "run_v153_resume1_selected_source.sh",
    ):
        subprocess.run(["bash", "-n", str(ROOT / launcher)], check=True)

    print("GPU1 V153 frozen training release verified")


if __name__ == "__main__":
    main()
