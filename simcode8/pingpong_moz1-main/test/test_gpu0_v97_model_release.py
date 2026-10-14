"""Regression tests for the authenticated GPU0 V97 model release."""

import copy

import numpy as np
import pytest

from pingpong_controller.tools.rl_2real.mjx_policy_controller import (
    load_mjx_checkpoint,
)
from pingpong_controller.tools.rl_2real.gpu0_v97_model_release import (
    CHECKPOINT_PATH,
    CheckpointContractError,
    load_released_actor,
    resolve_checkpoint_path,
    validate_gpu0_v97_checkpoint,
)


def _load_payload() -> dict[str, object]:
    return load_mjx_checkpoint(resolve_checkpoint_path(CHECKPOINT_PATH))


def test_gpu0_v97_checkpoint_and_actor_are_authenticated() -> None:
    payload = _load_payload()
    validate_gpu0_v97_checkpoint(payload)
    actor = load_released_actor()
    assert (actor.obs_dim, actor.act_dim) == (57, 7)
    action = actor.mean_action(np.zeros(57, dtype=np.float32))
    assert action.shape == (7,)
    assert np.all(np.isfinite(action))


def test_gpu0_v97_validator_rejects_runtime_contract_drift() -> None:
    payload = _load_payload()
    invalid = copy.deepcopy(payload)
    invalid["env_cfg"]["recovered_rmp_bounded_qvel_reference"] = False
    with pytest.raises(CheckpointContractError):
        validate_gpu0_v97_checkpoint(invalid)


def test_gpu0_v97_validator_rejects_ball_mass_drift() -> None:
    payload = _load_payload()
    invalid = copy.deepcopy(payload)
    invalid["env_cfg"]["dr_ball_mass_range"] = (0.0038, 0.0042)
    with pytest.raises(CheckpointContractError, match="ball-mass"):
        validate_gpu0_v97_checkpoint(invalid)
