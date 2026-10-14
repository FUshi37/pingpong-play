"""Authenticate completed-stage continuation without changing course numerics."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def validate_completed_stage(payload: dict[str, Any], *, next_stage: str) -> None:
    """Require the immediate registered successor and complete saved gates."""
    from train_juggle_mjx_curriculum import build_curriculum
    course = build_curriculum(
        curriculum_profile=payload['args']['curriculum_profile'],
        measured_ball_mass_range_kg=(.0039, .0041), asymmetric_critic=True,
        critic_command_history_steps=12, right_arm_pd_profile='recovered_rmp_rmpmd_v2')
    index = int(payload.get('stage_index', 0))
    if not 1 <= index < len(course):
        raise ValueError('Completed resume requires a registered nonfinal stage')
    if payload.get('stage_name') != course[index-1].name or next_stage != course[index].name:
        raise ValueError('Completed resume must use the immediate registered successor')
    row = payload.get('last_row')
    if not isinstance(row, dict):
        raise ValueError('Completed resume requires its saved validation row')
    for field, value in dict(stage_name=payload['stage_name'], stage_index=index,
                             stage_update=payload['stage_update'],
                             global_step=payload['step']).items():
        if row.get(field) != value:
            raise ValueError(f'Completed resume row disagrees with checkpoint: {field}')
    required = ['convergence/stage_converged', 'advance_eval/required',
                'advance_eval/ran', 'advance_eval/passed']
    if row.get('retention_eval/required', 0.):
        required.extend(('retention_eval/ran', 'retention_eval/passed'))
    for field in required:
        value = float(row.get(field, 0.))
        if not math.isfinite(value) or value != 1.:
            raise ValueError(f'Completed resume lacks a passing {field}')


def verify_gpu0_new_updates(csv_path: Path, *, base_step: int, minimum_updates: int) -> int:
    """Count distinct causal 1024x128 updates after this resume's source step."""
    if base_step < 0 or minimum_updates < 1:
        raise ValueError('Invalid GPU0 ordering contract')
    with csv_path.open() as stream:
        steps = [int(row['global_step']) for row in csv.DictReader(stream)
                 if row['stage_name'].startswith('fplc_v154_gpu0_')
                 and int(row['global_step']) > base_step]
    expected = [base_step + (i+1)*1024*128 for i in range(len(steps))]
    if steps != expected or len(steps) < minimum_updates:
        raise ValueError(f'GPU0 needs {minimum_updates} new consecutive updates; found {len(steps)}')
    return len(steps)


def verify_resume_only_source(path: Path, expected_sha: str, proof_path: Path) -> None:
    """Check hashes and exact AST equality outside the tested resume validator."""
    if path.name not in ('fixed_plane_v154.py', 'fixed_plane_v155.py',
                         'fixed_plane_v171_gpu0.py', 'fixed_plane_v172_gpu1.py'):
        raise ValueError('Resume-source proof only applies to explicitly audited validators')
    proof = json.loads(proof_path.read_text())[path.name]
    original = Path(proof['original_path'])
    digest = lambda file: hashlib.sha256(file.read_bytes()).hexdigest()
    if digest(original) != expected_sha or digest(path) != proof['current_sha256']:
        raise ValueError('Resume-source hash mismatch')
    def numerical_ast(file):
        tree = ast.parse(file.read_text())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.FunctionDef) and node.name == 'validate_resume_payload')]
        return ast.dump(tree, include_attributes=False)
    if numerical_ast(path) != numerical_ast(original):
        raise ValueError('A resume repair changed course construction or other numerical code')
