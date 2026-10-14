"""Recompute V154 admission from hashed evidence before a manual launch."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle

from fixed_plane_v154 import SOURCE_SHA256, validate_resume_payload
from train_juggle_mjx_curriculum import build_curriculum


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_source(path: Path) -> str:
    """Authenticate frozen evaluation across two explicitly tested entry fixes."""
    tree = ast.parse(path.read_text())
    from fixed_plane_refinement_source import RefinementRegistration
    tree = RefinementRegistration().visit(tree)
    # V155 is an isolated opt-in profile registration. Its addition must not
    # invalidate preserved V154 evidence or change a V154 numerical function.
    profile_name = 'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU1_V155_PROFILE'
    profile_group = 'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V155_PROFILES'
    class V155Registration(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t,ast.Name) and t.id in (profile_name,profile_group)
                   for t in node.targets):
                return None
            return self.generic_visit(node)
        def visit_FunctionDef(self, node):
            if node.name == '_fixed_plane_limit_cycle_qvel_rmp_gpu1_v155_stages':
                return None
            return self.generic_visit(node)
        def visit_If(self, node):
            expected = ast.parse(f'if args.curriculum_profile == {profile_name}:\n    pass').body[0].test
            if ast.dump(node.test) == ast.dump(expected):
                return None
            return self.generic_visit(node)
        def visit_Dict(self, node):
            pairs = [(k,v) for k,v in zip(node.keys,node.values)
                     if not (isinstance(k,ast.Name) and k.id == profile_name)]
            node.keys=[k for k,v in pairs]
            node.values=[v for k,v in pairs]
            return self.generic_visit(node)
        def visit_Tuple(self, node):
            node.elts=[x for x in node.elts if not (
                isinstance(x,ast.Starred) and isinstance(x.value,ast.Name)
                and x.value.id == profile_group) and not (
                isinstance(x,ast.Constant) and x.value == 'fplc_v155_gpu1_')]
            return self.generic_visit(node)
        visit_List = visit_Tuple
        visit_Set = visit_Tuple
    tree = V155Registration().visit(tree)
    tree.body = [n for n in tree.body if not (
        isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and
        t.id=='FIXED_PLANE_LIMIT_CYCLE_PROFILE_LABELS' for t in n.targets))]
    allowed = ast.parse('if getattr(stage.cfg, "fixed_contact_quality_pose_uses_normal_tilt", False):\n    recent = history[-window:]\n').body[0]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='convergence_status':
            kept=[]
            for statement in node.body:
                if isinstance(statement,ast.If) and any(
                    isinstance(x,ast.Constant) and x.value=='fixed_contact_quality_pose_uses_normal_tilt'
                    for x in ast.walk(statement.test)):
                    assert ast.dump(statement,include_attributes=False)==ast.dump(allowed,include_attributes=False)
                else:
                    kept.append(statement)
            node.body=kept
    return ast.dump(tree,include_attributes=False)


def paired_refinement_summary(source: list[dict], candidate: list[dict]) -> dict[str, float]:
    """Recompute motion gain and survival/height regression from paired blocks."""
    if len(source) != len(candidate) or len(source) < 2:
        raise ValueError('GPU1 refinement requires two paired seed blocks')
    names = ('rapid_steady_racket_vxy_rms_m_s', 'episode_truncation_rate',
             'rapid_steady_band_rate')
    values = [[tuple(float(row[f'advance_eval/{name}']) for name in names)
               for row in population] for population in (source, candidate)]
    if not all(math.isfinite(v) for population in values for row in population for v in row):
        raise ValueError('GPU1 refinement metrics must be finite')
    old, new = values
    baseline_rms = sum(row[0] for row in old)
    if baseline_rms <= 0:
        raise ValueError('GPU1 source RMS must be positive')
    return dict(relative_xy_rms_reduction=1-sum(row[0] for row in new)/baseline_rms,
                min_full_survival=min(row[1] for row in new),
                min_height_band=min(row[2] for row in new),
                max_survival_regression=max(0., *(a[1]-b[1] for a,b in zip(old,new))),
                max_band_regression=max(0., *(a[2]-b[2] for a,b in zip(old,new))))


def verify_gpu1_refinement(evidence: Path, checkpoint: Path, stage) -> dict[str, float]:
    """Authenticate raw paired blocks; never trust a declared improvement flag."""
    summary = json.loads((evidence/'gpu1_refinement.json').read_text())
    assert summary['checkpoint_sha256'] == sha(checkpoint)
    directory = evidence/summary['refined_directory']
    assert directory.parent == evidence and directory.name.startswith('gpu1_refined_s5_')
    manifest = json.loads((directory/'admission.json').read_text())
    assert manifest['source_sha256'] == sha(checkpoint)
    assert manifest['stage'] == stage.name
    assert manifest['cfg_sha256'] == hashlib.sha256(repr(vars(stage.cfg)).encode()).hexdigest()
    assert manifest['seeds'] == [20265411,20265412]
    assert manifest['deterministic'] is False
    for name,digest in manifest['artifacts'].items():
        assert Path(name).name == name and sha(directory/name) == digest
    for path,digest in manifest['source_hashes'].items():
        assert sha(Path(path)) == digest
    baseline=[]
    refined=[]
    for seed in manifest['seeds']:
        baseline.append(json.loads((evidence/'gpu1_s5'/f'block_{seed}.json').read_text()))
        row = json.loads((directory/f'block_{seed}.json').read_text())
        assert row['advance_eval/passed'] == 1 and row['advance_eval/episodes'] >= 144
        assert row['advance_eval/mean_hits'] >= 12 and row['advance_eval/hit12_rate'] >= .97
        assert row['advance_eval/fixed_contact_quality_ok'] == 1
        refined.append(row)
    actual = paired_refinement_summary(baseline,refined)
    for name,value in actual.items():
        assert math.isclose(value,float(summary[name]),rel_tol=1e-10,abs_tol=1e-12)
    assert actual['relative_xy_rms_reduction'] >= .10
    assert actual['min_full_survival'] >= .99 and actual['min_height_band'] >= .98
    assert actual['max_survival_regression'] <= .005 and actual['max_band_regression'] <= .005
    return actual


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    cli.add_argument('--checkpoint', type=Path, required=True)
    cli.add_argument('--csv', default='')
    cli.add_argument('--csv-sha', default='')
    cli.add_argument('--evidence', type=Path, required=True)
    cli.add_argument('--start-stage', required=True)
    cli.add_argument('--kind', choices=('formal', 'probe'), required=True)
    cli.add_argument('--gpu0-run-dir', default='')
    cli.add_argument('--completed-stage-resume', action='store_true')
    cli.add_argument('--resume-source-proof', type=Path)
    args = cli.parse_args()
    p = pickle.loads(args.checkpoint.read_bytes())
    validate_resume_payload(p, gpu=args.gpu, requested_start_stage=args.start_stage,
        resume_curriculum_state=bool(args.csv) and not args.completed_stage_resume,
        reset_optimizer=False, reset_critic=False)
    assert not args.completed_stage_resume or args.csv
    if args.csv:
        csv_path = Path(args.csv)
        assert args.csv_sha and sha(csv_path) == args.csv_sha
        rows = list(csv.DictReader(csv_path.open()))
        assert rows[-1]['stage_name'] == p['stage_name']
        assert int(rows[-1]['stage_update']) == p['stage_update']
        assert int(rows[-1]['global_step']) == p['step']
    else:
        assert sha(args.checkpoint) == SOURCE_SHA256[args.gpu]
    course = build_curriculum(
        curriculum_profile=f'fixed_plane_limit_cycle_qvel_rmp_gpu{args.gpu}_v154',
        measured_ball_mass_range_kg=(.0039, .0041), asymmetric_critic=True,
        critic_command_history_steps=12, right_arm_pd_profile='recovered_rmp_rmpmd_v2')
    admission_stages = (5,) if args.gpu == 0 else (4, 5)
    for index in admission_stages:
        directory = args.evidence / f'gpu{args.gpu}_s{index}'
        manifest = json.loads((directory/'admission.json').read_text())
        assert manifest['source_sha256'] == SOURCE_SHA256[args.gpu]
        assert manifest['stage'] == course[index - 1].name
        assert manifest['cfg_sha256'] == hashlib.sha256(repr(vars(course[index - 1].cfg)).encode()).hexdigest()
        assert manifest['seeds'] == [20265411, 20265412]
        for name, digest in manifest['artifacts'].items():
            assert Path(name).name == name and sha(directory/name) == digest
        for path, digest in manifest['source_hashes'].items():
            source_file = Path(path)
            if sha(source_file) != digest:
                if source_file.name == 'fixed_plane_v154.py':
                    from fixed_plane_completed_resume import verify_resume_only_source
                    assert args.resume_source_proof
                    verify_resume_only_source(source_file, digest, args.resume_source_proof)
                    continue
                # The first formal entry exposed a label-registry KeyError.
                # Verify exact AST equivalence outside that metadata dictionary.
                fix = json.loads((args.evidence/'verified_entry_fixes.json').read_text())
                file_fix = fix.get('files',{}).get(source_file.name,fix)
                originals=[Path(x['path']) for x in file_fix['originals'] if x['sha256']==digest]
                assert len(originals)==1
                original = originals[0]
                assert source_file.name in ('train_juggle_mjx_curriculum.py','mjx_juggle_env.py')
                assert sha(original)==digest
                if sha(source_file)!=file_fix['current_sha256']:
                    from fixed_plane_refinement_source import verify_legacy_source
                    verify_legacy_source(source_file,file_fix['current_sha256'])
                assert numerical_source(source_file)==numerical_source(original)
        for seed in manifest['seeds']:
            result = json.loads((directory/f'block_{seed}.json').read_text())
            assert result['advance_eval/probe_stage_index'] == index
            assert result['advance_eval/deterministic'] == 0
            assert result['advance_eval/passed'] == 1
            assert result['advance_eval/episodes'] >= 144
            assert result['advance_eval/mean_hits'] >= 12
            assert result['advance_eval/episode_truncation_rate'] >= .99
            assert result['advance_eval/hit12_rate'] >= .97
            assert result['advance_eval/rapid_steady_band_rate'] >= .98
            assert result['advance_eval/fixed_contact_quality_ok'] == 1
    smoke = json.loads((args.evidence/f'gpu{args.gpu}_dr_smoke_r2/smoke.json').read_text())
    assert smoke['cfg_sha256'] == hashlib.sha256(repr(vars(course[-1].cfg)).encode()).hexdigest()
    assert smoke['held_lane_ticks'] > 0 and smoke['obs_dim'] == (156 if args.gpu == 0 else 72)
    assert smoke['reward_channels'] >= 190
    assert all(x['std'] > 0 for x in smoke['sampled'].values())
    if args.gpu == 1:
        assert args.gpu0_run_dir, 'GPU1 requires evidence of 50 completed GPU0 updates'
        gpu0_dir = Path(args.gpu0_run_dir)
        rows = list(csv.DictReader((gpu0_dir/'curriculum_progress.csv').open()))
        completed = [r for r in rows if r.get('stage_name','').startswith('fplc_v154_gpu0_')]
        assert len(completed)>=50, 'GPU0 has not completed 50 new PPO updates'
        assert int(completed[49]['global_step'])>=2112880640+50*1024*128
        p0 = pickle.loads((gpu0_dir/'mjx_curriculum_last.pkl').read_bytes())
        assert p0['args']['curriculum_profile']=='fixed_plane_limit_cycle_qvel_rmp_gpu0_v154'
        assert p0['args']['n_envs']==1024
        rank = json.loads((args.evidence/'gpu1_reward_audit/acceptance.json').read_text())
        for name, digest in rank['artifacts'].items():
            assert Path(name).name == name and sha(args.evidence/'gpu1_reward_audit'/name) == digest
        assert rank['source_sha256'] == SOURCE_SHA256[1]
        assert all(r['source_beats_bad'] >= .9 and r['source_beats_zero'] >= .9 for r in rank['results'])
        if args.kind == 'formal':
            verify_gpu1_refinement(args.evidence,args.checkpoint,course[4])
    print(f'V154 GPU{args.gpu}: exact full-state checkpoint, all block gates and small-DR evidence verified')


if __name__ == '__main__':
    main()
