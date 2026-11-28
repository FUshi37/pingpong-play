"""MOZ1 mobile base dynamics identification and racket-jitter calibration toolchain.

This package provides:
- ``contract``: Fail-closed interface contract loading, validation, and hashing.
- ``inspect_base_interface``: Read-only repository and ROS 2 interface auditor.
- ``waveforms``: Pure-function, geofence-verified excitation waveform library.
- ``safety``: Fail-closed live gates, real-time geofence, staleness watchdog, and brake bursts.
- ``run_base_id_experiment``: Dry-run-by-default ROS 2 data collector and concurrency tester.
- ``fit_moz1_base_dynamics``: Offline system identification, audit, report, and figure generator.
"""

from .contract import (
    CONTRACT_SCHEMA_VERSION,
    UNRESOLVED_SENTINEL,
    compute_contract_sha256,
    load_contract,
    validate_contract,
)

__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "UNRESOLVED_SENTINEL",
    "compute_contract_sha256",
    "load_contract",
    "validate_contract",
]
