"""Admit the opt-in continuous mean-error curriculum from its working source."""
import argparse
from dataclasses import asdict
import hashlib,json,pickle
from pathlib import Path
import numpy as np
from fixed_plane_v161_gpu0 import PROFILE,PARENT_SHA256,REWARD_FIELDS,validate_resume_payload

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha',required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--gpu1-run-dir',type=Path,required=True)
    parser.add_argument('--kind',choices=['formal'],required=True)
    args=parser.parse_args()
    from fixed_plane_ordered_updates import verify_new_updates
    verify_new_updates(args.gpu1_run_dir,stage_prefix='fplc_v159_gpu1_',base_step=1023016960,minimum_new_updates=15)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(args.checkpoint)==args.checkpoint_sha
    p=pickle.loads(args.checkpoint.read_bytes());own=p['args']['curriculum_profile']==PROFILE
    validate_resume_payload(p,requested_start_stage=p['stage_name'] if own else 'fplc_v161_gpu0_s7_clean_stability',resume_curriculum_state=own,reset_optimizer=False,reset_critic=False)
    if not own:assert args.checkpoint_sha==PARENT_SHA256
    from train_juggle_mjx_curriculum import build_curriculum
    kw=dict(measured_ball_mass_range_kg=(.0039,.0041),asymmetric_critic=True,critic_command_history_steps=12,right_arm_pd_profile='recovered_rmp_rmpmd_v2')
    old=build_curriculum(curriculum_profile='fixed_plane_limit_cycle_qvel_rmp_gpu0_v154',**kw);new=build_curriculum(curriculum_profile=PROFILE,**kw)
    for before,after in zip(old,new):
        assert {k for k,v in asdict(after.cfg).items() if asdict(before.cfg).get(k)!=v}<={*REWARD_FIELDS,'fixed_contact_training_phase'}
    baseline=args.evidence.parent/'v160_evidence_20261009_r1/source_blocks';manifest=json.loads((baseline/'manifest.json').read_text());assert manifest['checkpoint_sha256']==PARENT_SHA256
    assert manifest['n_envs']==288 and manifest['steps']==1200 and len(manifest['seeds'])==2
    for name,digest in manifest['artifacts'].items():assert Path(name).name==name and sha(baseline/name)==digest
    audit=json.loads((args.evidence/'reward_analysis.json').read_text())
    for path,digest in audit['reviewed_source_hashes'].items():
        if sha(Path(path))!=digest:
            from fixed_plane_refinement_source import verify_v161_successor_source
            verify_v161_successor_source(Path(path),digest)
    for seed in manifest['seeds']:
        b=json.loads((baseline/f'block_{seed}.json').read_text())
        assert b['advance_eval/episode_truncation_rate']>=.995 and b['advance_eval/hit12_rate']>=.99 and b['advance_eval/mean_hits']>=12.8
        ledger=np.load(baseline/f'ledger_{seed}.npz')
        assert np.max(np.abs(sum(ledger[k] for k in audit['additive_terms'])-ledger['reward/total']))<.0002
    assert len(new)==len(old)==14
    from dataclasses import replace
    reward_reference=build_curriculum(curriculum_profile='fixed_plane_limit_cycle_qvel_rmp_gpu0_v160',**kw)
    for before,after in zip(reward_reference,new):
        assert asdict(replace(after.cfg,fixed_contact_training_phase=before.cfg.fixed_contact_training_phase))==asdict(before.cfg)
    for stage in new[6:]:
        assert stage.target_mean_hits==12.9 and stage.target_mean_len_frac==.995
        assert stage.target_episode_truncation_rate==.995 and stage.target_hit12_rate==.99
    assert audit['equal_height_population_counterexample_fixed']
    print('V161 full-state,79-leaf closure,working source and GPU1-first15 verified',flush=True)

if __name__=='__main__':main()
