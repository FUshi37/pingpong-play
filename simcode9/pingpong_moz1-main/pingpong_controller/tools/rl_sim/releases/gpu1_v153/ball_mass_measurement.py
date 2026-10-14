"""Validate auditable mass evidence for the opt-in 4.0 g ball path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MEASUREMENT_SCHEMA_VERSION = "gpu1_ball_mass_measurements_v1"
USER_ATTESTATION_SCHEMA_VERSION = "gpu1_ball_mass_user_attestation_v1"
NOMINAL_MASS_KG = 0.004
MIN_MEASUREMENT_COUNT = 3
MAX_SMALL_DR_HALF_WIDTH_KG = 0.0002


@dataclass(frozen=True)
class BallMassMeasurementContract:
    """Centered mass support and its explicitly identified evidence source."""

    manifest_path: Path
    manifest_sha256: str
    schema_version: str
    evidence_kind: str
    ball_id: str
    scale_id: str | None
    scale_resolution_kg: float | None
    confirmation_source: str | None
    nominal_mass_kg: float
    measurements_kg: tuple[float, ...]
    half_width_kg: float
    dr_range_kg: tuple[float, float]
    preserve_legacy_ball_parameters: bool

    @property
    def measurement_count(self) -> int:
        return len(self.measurements_kg)


def _require_nonempty_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def load_ball_mass_measurement_contract(
    manifest_path: str | Path,
) -> BallMassMeasurementContract:
    """Load a measured or explicit user-attested centered DR contract.

    Repeated-scale evidence recomputes its half-width from raw readings. A
    user-attested contract records the user's nominal mass and authorized
    robustness half-width without fabricating scale metadata or readings.
    Both paths remain centered on exactly 4.0 g and capped at +/-0.2 g.
    """

    path = Path(manifest_path).expanduser().resolve()
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid ball-mass JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("ball-mass manifest must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version not in {
        MEASUREMENT_SCHEMA_VERSION,
        USER_ATTESTATION_SCHEMA_VERSION,
    }:
        raise ValueError(
            "schema_version must identify repeated measurements or explicit "
            f"user attestation, got {schema_version!r}"
        )

    ball_id = _require_nonempty_string(payload, "ball_id")
    try:
        nominal_mass_kg = float(payload["nominal_mass_kg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("nominal_mass_kg must be a finite number") from exc
    if not math.isclose(
        nominal_mass_kg,
        NOMINAL_MASS_KG,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("nominal_mass_kg must be exactly 0.004 kg")
    if schema_version == MEASUREMENT_SCHEMA_VERSION:
        evidence_kind = "repeated_measurements"
        scale_id = _require_nonempty_string(payload, "scale_id")
        confirmation_source = None
        try:
            scale_resolution_kg = float(payload["scale_resolution_kg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "scale_resolution_kg must be a finite number"
            ) from exc
        if not math.isfinite(scale_resolution_kg) or scale_resolution_kg <= 0.0:
            raise ValueError("scale_resolution_kg must be positive and finite")

        raw_measurements = payload.get("measurements_kg")
        if not isinstance(raw_measurements, list):
            raise ValueError("measurements_kg must be a JSON list")
        try:
            measurements_kg = tuple(float(value) for value in raw_measurements)
        except (TypeError, ValueError) as exc:
            raise ValueError("measurements_kg must contain only numbers") from exc
        if len(measurements_kg) < MIN_MEASUREMENT_COUNT:
            raise ValueError(
                "measurements_kg requires at least three repeated readings"
            )
        if not all(
            math.isfinite(value) and value > 0.0 for value in measurements_kg
        ):
            raise ValueError(
                "measurements_kg must contain positive finite readings"
            )

        half_width_kg = max(
            abs(value - NOMINAL_MASS_KG) for value in measurements_kg
        ) + 0.5 * scale_resolution_kg
        if half_width_kg > MAX_SMALL_DR_HALF_WIDTH_KG + 1.0e-12:
            raise ValueError(
                "repeated readings do not support a small DR interval centered "
                "on 0.004 kg: recomputed half-width "
                f"{half_width_kg:.9f} kg exceeds "
                f"{MAX_SMALL_DR_HALF_WIDTH_KG:.9f} kg"
            )
        preserve_legacy_ball_parameters = True
    else:
        evidence_kind = "user_attestation"
        scale_id = None
        scale_resolution_kg = None
        measurements_kg = ()
        confirmation_source = _require_nonempty_string(
            payload, "confirmation_source"
        )
        if payload.get("preserve_legacy_ball_parameters") is not True:
            raise ValueError(
                "preserve_legacy_ball_parameters must be true for user attestation"
            )
        preserve_legacy_ball_parameters = True
        try:
            half_width_kg = float(payload["dr_half_width_kg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("dr_half_width_kg must be a finite number") from exc
        if (
            not math.isfinite(half_width_kg)
            or half_width_kg <= 0.0
            or half_width_kg > MAX_SMALL_DR_HALF_WIDTH_KG + 1.0e-12
        ):
            raise ValueError(
                "user-attested dr_half_width_kg must be positive and no larger "
                f"than {MAX_SMALL_DR_HALF_WIDTH_KG:.9f} kg"
            )
        forbidden_measurement_fields = {
            "scale_id",
            "scale_resolution_kg",
            "measurements_kg",
        }.intersection(payload)
        if forbidden_measurement_fields:
            raise ValueError(
                "user attestation must not fabricate measurement fields: "
                + ", ".join(sorted(forbidden_measurement_fields))
            )

    dr_range_kg = (
        NOMINAL_MASS_KG - half_width_kg,
        NOMINAL_MASS_KG + half_width_kg,
    )
    return BallMassMeasurementContract(
        manifest_path=path,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        schema_version=str(schema_version),
        evidence_kind=evidence_kind,
        ball_id=ball_id,
        scale_id=scale_id,
        scale_resolution_kg=scale_resolution_kg,
        confirmation_source=confirmation_source,
        nominal_mass_kg=NOMINAL_MASS_KG,
        measurements_kg=measurements_kg,
        half_width_kg=half_width_kg,
        dr_range_kg=dr_range_kg,
        preserve_legacy_ball_parameters=preserve_legacy_ball_parameters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute the centered 4.0 g DR range from raw readings."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--format",
        choices=("json", "range"),
        default="json",
    )
    args = parser.parse_args()
    contract = load_ball_mass_measurement_contract(args.manifest)
    if args.format == "range":
        print(f"{contract.dr_range_kg[0]:.12g} {contract.dr_range_kg[1]:.12g}")
        return
    print(
        json.dumps(
            {
                "schema_version": contract.schema_version,
                "manifest_path": str(contract.manifest_path),
                "manifest_sha256": contract.manifest_sha256,
                "evidence_kind": contract.evidence_kind,
                "ball_id": contract.ball_id,
                "scale_id": contract.scale_id,
                "measurement_count": contract.measurement_count,
                "scale_resolution_kg": contract.scale_resolution_kg,
                "confirmation_source": contract.confirmation_source,
                "nominal_mass_kg": contract.nominal_mass_kg,
                "half_width_kg": contract.half_width_kg,
                "dr_range_kg": list(contract.dr_range_kg),
                "preserve_legacy_ball_parameters": (
                    contract.preserve_legacy_ball_parameters
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
