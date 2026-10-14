"""Authenticate a gates-only72-D continuation and preserved policy quality."""
import argparse
from dataclasses import asdict
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from fixed_plane_v172_gpu1 import PROFILE, PARENT_PROFILE, PARENT_SHA256, GATE_FIELDS, validate_resume_payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha',required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--kind',choices=['formal'],required=True)
    p.add_argument('--resume-start-stage')
    p.add_argument('--resume-completed-stage',action='store_true')
    a=p.parse_args()
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(a.checkpoint)==a.checkpoint_sha
    payload=pickle.loads(a.checkpoint.read_bytes())
    validate_resume_payload(payload,
        requested_start_stage=a.resume_start_stage or payload['stage_name'],
        resume_curriculum_state=not a.resume_completed_stage,
        reset_optimizer=False,reset_critic=False)
    if payload['args']['curriculum_profile']==PARENT_PROFILE:
        assert a.checkpoint_sha==PARENT_SHA256
    from train_juggle_mjx_curriculum import build_curriculum,load_resumed_curriculum_history
    kw=dict(measured_ball_mass_range_kg=(.0039,.0041),asymmetric_critic=True,
        critic_command_history_steps=12,right_arm_pd_profile='recovered_rmp_rmpmd_v2')
    old=build_curriculum(curriculum_profile=PARENT_PROFILE,**kw)
    new=build_curriculum(curriculum_profile=PROFILE,**kw)
    assert len(old)==len(new)==17
    for x,y in zip(old,new):
        assert asdict(x.cfg)==asdict(y.cfg) and x.name==y.name
        assert {k for k,v in asdict(y).items() if asdict(x)[k]!=v}<={*GATE_FIELDS,'notes'}
    h,u,_=load_resumed_curriculum_history(a.checkpoint,payload,
        stage_index=payload['stage_index'],stage_name=payload['stage_name'],max_rows=96)
    if payload['args']['curriculum_profile']==PARENT_PROFILE:
        assert len(h)>=32 and u>=120
    audit=json.loads((a.evidence/'gate_analysis.json').read_text())
    assert audit['source_checkpoint_sha256']==PARENT_SHA256 and audit['source_qualifies']
    for path,digest in audit['reviewed_source_hashes'].items():
        if sha(Path(path))!=digest:
            from fixed_plane_completed_resume import verify_resume_only_source
            verify_resume_only_source(Path(path),digest,
                a.evidence.parent/'v171_v172_resume_repair_20261009_r1/validator_source_proof.json')
    retention=json.loads((a.evidence/'paired_retention_acceptance.json').read_text())
    assert retention['accepted'] and retention['candidate_sha256']==PARENT_SHA256
    for path,digest in retention['artifact_hashes'].items():
        assert sha(Path(path))==digest
    d=Path(retention['paired_directory'])
    for seed in retention['seeds']:
        for prefix in ('','reference_'):
            ledger=np.load(d/f'{prefix}ledger_{seed}.npz')
            assert np.max(np.abs(sum(ledger[k] for k in audit['additive_terms'])-ledger['reward/total']))<.0002
    print('V172 gates only: unchanged72-D actor/rewards/PPO, full causal state and paired retention verified',flush=True)


if __name__=='__main__':main()
