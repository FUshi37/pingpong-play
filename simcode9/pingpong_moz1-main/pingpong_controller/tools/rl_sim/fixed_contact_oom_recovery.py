"""Explicitly pinned recovery of safely persisted fixed-contact checkpoints."""
from __future__ import annotations

import hashlib
from pathlib import Path

CONTRACTS = {
    'fixed_contact_plane_vision_robust_qvel_rmp_v38': {
        'sha256': 'ffb0fcb4c546d35fe709291bcde548807bcdf63c15dcaffd6c5493cf8a05430f',
        'csv_sha256': 'c390be410e3c0164a71833eedbb2e9a9f95725b0e7f505433f031ad9accd1066',
        'stage': 'v38_s0_center_acquire', 'step': 221642752,
        'stage_index': 4, 'stage_update': 1550, 'global_update': 1691,
    },
    'fixed_contact_plane_vision_robust_qvel_rmp_v39': {
        'sha256': 'cfb7647c3880daf4388f44f2ea752d61c6d4a448737c816ed3e9206d1c4a42a6',
        'csv_sha256': 'df3f20abd7f31dd6056c0eb624a16471bda9edb7e79e49f4a441f82718a54a60',
        'stage': 'v39_s1_single_outcome', 'step': 303300608,
        'stage_index': 1, 'stage_update': 1900, 'global_update': 1900,
    },
    'fixed_contact_plane_vision_robust_qvel_rmp_v40': {
        'sha256': '3f0f70a2be9f4b5a68eba901775378e0ba7c7ac038ebd5506bb610d452f96658',
        'csv_sha256': 'b25f1f39ff7b429d604c076132eafc7220d3b7a0767d86691b08365511f432bd',
        'stage': 'v40_s0_center_hold_at_target', 'step': 231473152,
        'stage_index': 5, 'stage_update': 100, 'global_update': 100,
    },
}


def validate_recovery_args(args) -> None:
    """Keep this exception separate from original initialization routes."""
    contract = CONTRACTS.get(args.curriculum_profile)
    if contract is None:
        raise ValueError('OOM recovery is restricted to pinned fixed-contact runs')
    if (args.resume_from is None or not args.resume_curriculum_state
            or args.resume_start_stage != contract['stage']
            or args.reset_optimizer_on_resume or args.reset_critic_on_resume
            or args.allow_obs_dim_migration or not args.clear_jax_caches_between_stages
            or args.wandb_id):
        raise ValueError('OOM recovery requires exact stage, full actor/critic/Adam/history, '
                         'cache cleanup and a new W&B identity; no migration/reset')


def verify_recovery_files(profile: str, source: Path) -> None:
    """Hash before unpickling and retain the causal CSV filter in the trainer."""
    contract = CONTRACTS[profile]
    for path, key in [(source, 'sha256'),
                      (source.parent / 'curriculum_progress.csv', 'csv_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != contract[key]:
            raise ValueError(f'OOM recovery source hash mismatch: {path}')


def validate_recovery_payload(profile: str, payload: dict) -> None:
    """Reject metadata/layout drift even for the pinned checkpoint."""
    contract = CONTRACTS[profile]
    expected = {k: contract[k] for k in ('step', 'stage_index', 'stage_update', 'global_update')}
    expected.update(stage_name=contract['stage'], obs_dim=80, critic_obs_dim=401, act_dim=7)
    if any(payload.get(k) != value for k, value in expected.items()):
        raise ValueError('OOM recovery metadata mismatch')
    if payload['args']['curriculum_profile'] != profile or 'opt' not in payload:
        raise ValueError('OOM recovery must retain same profile and Adam state')
