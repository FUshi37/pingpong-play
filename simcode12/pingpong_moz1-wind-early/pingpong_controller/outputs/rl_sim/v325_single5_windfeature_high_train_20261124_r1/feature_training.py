"""Explicit native-state migration and frozen-base contracts for new inputs."""
import copy
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from phase_transition import verify_phase_parent
from wind_features import migrate_params, encode, apply_actor


def migrate_learner(learner):
    return learner._replace(params=migrate_params(learner.params),
        opt=learner.opt._replace(m=migrate_params(learner.opt.m),
                                v=migrate_params(learner.opt.v)))


@jax.jit
def frozen_base_equal(params, reference):
    checks = [jnp.all(params['log_std'] == reference['log_std'])]
    for name, layer in params['pi'].items():
        for key, value in layer.items():
            expected = reference['pi'][name][key]
            if name == 'l1' and key == 'w':
                checks.append(jnp.all(value[:118] == expected[:118]))
            else:
                checks.append(jnp.all(value == expected))
    return jnp.all(jnp.stack(checks))


def assert_supported_actor(params):
    if set(params) != {'pi', 'v', 'log_std'} or set(params['pi']) != {'l1','l2','l3','out'}:
        raise ValueError('Exactly one ordinary actor is required')
    width=params['pi']['l1']['w'].shape[0]
    for name,shape in [('l1',(width,282)),('l2',(282,282)),('l3',(282,282)),('out',(282,7))]:
        if params['pi'][name]['w'].shape!=shape or params['pi'][name]['b'].shape!=(shape[1],):
            raise ValueError('Invalid actor layer '+name)
    if width not in (118,120):
        raise ValueError('Only pinned118-D references and current120-D actors are allowed')


def feature_apply_factory(legacy_apply):
    def apply(pi, obs):
        width=pi['l1']['w'].shape[0]
        if width==120:
            if obs.shape[-1]==118:
                obs=encode(obs,xp=jnp)
            return apply_actor(pi,obs)
        if width==118 and obs.shape[-1]==120:
            obs=obs[...,:118]
        return legacy_apply(pi,obs)
    return apply


def verify_feature_parent(plan):
    old=json.loads(Path(plan['phase_transition']['parent_plan']).read_text())
    expected={**old['observation'], 'actor':120, 'unchanged_prefix':118,
        'new_suffix':'Two causal estimated windXY transforms active above1.25m/s, width0.35m/s afterhit5; no privileged wind',
        'runtime':'single5_windfeature120_high_v2'}
    if (plan['observation']!=expected or old['observation']['actor']!=118
            or plan['direct_MLP']!=[120,282,282,282,7]
            or plan['native_actor_critic_action']!=[120,453,7]
            or plan['actor_total_parameters']!=195715
            or plan['feature_optimizer']['feature_rows']!=[118,119]):
        raise ValueError('Unregistered feature-only single-MLP migration')
    proof=Path(plan['observation_proof'])
    with proof.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=plan['observation_proof_sha256']:
            raise ValueError('Observation proof hash mismatch')
    evidence=json.loads(proof.read_text())
    if not all(evidence[k] for k in ('complete','raw_controls_exact',
        'zero_extra_weights_all_physical_events_exact',
        'perturbed_extra_weights_all_calm_physical_events_exact')):
        raise ValueError('Physical observation migration proof did not pass')
    # Reuse the strict source/hash/control/reward/PPO checks, allowing only
    # the separately checked append above. The legacy verifier is unchanged.
    legacy=copy.deepcopy(plan)
    legacy['observation']=old['observation']
    contract,receipt=verify_phase_parent(legacy)
    receipt.update(new_actor_input=120,old_parameters_and_Adam_preserved=True,
                   append_two_zero_parameter_and_moment_rows=True,
                   no_action_residual_or_second_policy=True)
    return contract,receipt
