"""Authenticated GPU0 V97 model release on the existing REAL-RMP runtime.

V97 keeps the GPU0-QVEL/REAL-RMP V85 actor observation and command boundary.
Only checkpoint identity and training-only fixed-base/4 g ball metadata are
new here.  The runtime implementation is deliberately reused instead of
forked.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pingpong_controller.tools.rl_2real.gpu0_qvel_real_rmp_reference import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    CheckpointContractError,
    GPU0QvelRealRmpReference,
    validate_gpu0_v85_checkpoint,
    sha256_file,
)


CHECKPOINT_PATH = (
    "pingpong_controller/outputs/rl_sim/"
    "selected_best_models_and_normal_reset_videos_20260901/gpu0_v97/"
    "gpu0_v97_best_step5836242944.pkl"
)
CHECKPOINT_SHA256 = (
    "f166a4b8adb6b5b6a846b14407baa5da32ba3535492d23cb0ad5978cece8568d"
)
PROFILE_NAME = "goal_d455_measured_qvel_rmp_vertical_v97"
CHECKPOINT_STAGE_NAME = "rmp97_rmp_internal_commit_12p5"
CHECKPOINT_STAGE_INDEX = 32
CHECKPOINT_STAGE_UPDATE = 164
CHECKPOINT_GLOBAL_UPDATE = 5_329
CHECKPOINT_STEP = 5_836_242_944
CHECKPOINT_SEED = 20_261_018
BALL_MASS_RANGE_KG = np.asarray([0.0039, 0.0041], dtype=np.float64)
ZERO_OBSERVATION_ACTOR_MEAN = np.asarray(
    [
        -0.013275788,
        0.009566929,
        0.019368205,
        0.067597359,
        0.120253950,
        -0.124855369,
        0.022645233,
    ],
    dtype=np.float32,
)


def resolve_checkpoint_path(
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> Path:
    """Resolve the repository-relative V97 checkpoint path."""

    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        repository_root = Path(__file__).resolve().parents[3]
        path = repository_root / path
    return path.resolve()


def _require_equal(
    values: Mapping[str, object], name: str, expected: object
) -> None:
    actual = values.get(name)
    if actual != expected:
        raise CheckpointContractError(
            f"checkpoint {name}={actual!r}, expected {expected!r}"
        )


def validate_gpu0_v97_checkpoint(payload: Mapping[str, object]) -> None:
    """Fail closed unless ``payload`` is the selected V97 checkpoint."""

    _require_equal(payload, "obs_dim", ACTOR_OBS_DIM)
    _require_equal(payload, "critic_obs_dim", CRITIC_OBS_DIM)
    _require_equal(payload, "act_dim", ACTION_DIM)
    _require_equal(payload, "stage_name", CHECKPOINT_STAGE_NAME)
    _require_equal(payload, "stage_index", CHECKPOINT_STAGE_INDEX)
    _require_equal(payload, "stage_update", CHECKPOINT_STAGE_UPDATE)
    _require_equal(payload, "global_update", CHECKPOINT_GLOBAL_UPDATE)
    _require_equal(payload, "step", CHECKPOINT_STEP)

    args = payload.get("args")
    if not isinstance(args, Mapping):
        raise CheckpointContractError("checkpoint args are missing")
    _require_equal(args, "curriculum_profile", PROFILE_NAME)
    _require_equal(args, "seed", CHECKPOINT_SEED)

    env_cfg = payload.get("env_cfg")
    if not isinstance(env_cfg, Mapping):
        raise CheckpointContractError("checkpoint env_cfg is missing")
    _require_equal(env_cfg, "simulation_base_mode", "aligned_fixed")
    mass_range_kg = np.asarray(
        env_cfg.get("dr_ball_mass_range", ()), dtype=np.float64
    )
    if mass_range_kg.shape != (2,) or not np.allclose(
        mass_range_kg, BALL_MASS_RANGE_KG, rtol=0.0, atol=1e-12
    ):
        raise CheckpointContractError(
            "V97 ball-mass range is not the measured 3.9--4.1 g contract"
        )
    _require_equal(env_cfg, "dr_ball_normalized_inertia_range", (0.4, 0.4))

    # Reuse the already reviewed V85 validator for every actor/runtime field.
    # Only identity fields are substituted in this private validation copy.
    compatibility_payload = copy.deepcopy(dict(payload))
    compatibility_payload.update(
        stage_name="rmp85_complete_nonexecution_full_episode_polish",
        stage_index=24,
        stage_update=174,
        global_update=430,
        step=2_922_905_600,
    )
    compatibility_args = dict(args)
    compatibility_args.update(
        curriculum_profile="goal_d455_measured_qvel_rmp_vertical_v85",
        seed=20_261_004,
    )
    compatibility_payload["args"] = compatibility_args
    validate_gpu0_v85_checkpoint(compatibility_payload)


def load_released_actor(
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> Any:
    """Authenticate, validate, and load the deterministic V97 actor mean."""

    path = resolve_checkpoint_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"released GPU0 V97 checkpoint not found: {path}")
    actual_digest = sha256_file(path)
    if actual_digest != CHECKPOINT_SHA256:
        raise CheckpointContractError(
            "released GPU0 V97 checkpoint SHA-256 mismatch: "
            f"{actual_digest} != {CHECKPOINT_SHA256}"
        )
    from pingpong_controller.tools.rl_2real.mjx_policy_controller import (
        NumpyMJXActor,
        load_mjx_checkpoint,
    )

    payload = load_mjx_checkpoint(path)
    validate_gpu0_v97_checkpoint(payload)
    actor = NumpyMJXActor(payload["params"])
    if (actor.obs_dim, actor.act_dim) != (ACTOR_OBS_DIM, ACTION_DIM):
        raise CheckpointContractError(
            "released GPU0 V97 actor dimensions changed after loading"
        )
    zero_mean = actor.mean_action(
        np.zeros(ACTOR_OBS_DIM, dtype=np.float32)
    )
    if not np.allclose(
        zero_mean,
        ZERO_OBSERVATION_ACTOR_MEAN,
        rtol=0.0,
        atol=2e-7,
    ):
        raise CheckpointContractError(
            "released GPU0 V97 actor failed the golden-vector check"
        )
    return actor


__all__ = [
    "CHECKPOINT_PATH",
    "CHECKPOINT_SHA256",
    "GPU0QvelRealRmpReference",
    "load_released_actor",
    "validate_gpu0_v97_checkpoint",
]
