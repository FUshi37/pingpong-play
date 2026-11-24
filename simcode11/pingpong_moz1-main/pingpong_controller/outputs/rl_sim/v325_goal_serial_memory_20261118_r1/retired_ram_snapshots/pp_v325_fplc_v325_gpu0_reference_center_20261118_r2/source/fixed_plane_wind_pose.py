"""Opt-in V325/V326 gradual pose authority and bounded correction-angle rewards."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, make_dataclass, replace
from functools import lru_cache
import inspect
import json
import math
import textwrap


REVISION = 'wind4_pose20_causal_reference_20261113'
SOURCE_SHA256 = '12b963df7721179585b3b768344fdc97d912797053db27e3911a6433294b35d2'


@dataclass(frozen=True)
class PoseSchedule:
    """Policy-step clock, retained through episode resets; angles in degrees."""

    start_deg: float = 2.
    target_deg: float = 2.
    ramp_updates: int = 0
    origin_steps: int = 0
    elapsed_steps_offset: int = 0
    flatness_start_deg: float = .25
    flatness_target_deg: float = .35
    quality_start_scale: float = 0.
    quality_target_scale: float = .5
    objective_version: int = 325
    quality_gated: bool = False
    strong_wind_reward: bool = False

    def __post_init__(self):
        if (not all(math.isfinite(x) for x in (self.start_deg, self.target_deg))
                or not 2. <= self.start_deg <= self.target_deg <= 20.
                or any(type(x) is not int or x < 0 for x in (
                    self.ramp_updates, self.origin_steps, self.elapsed_steps_offset))
                or (self.start_deg != self.target_deg and self.ramp_updates == 0)
                or not .25<=self.flatness_start_deg<=self.flatness_target_deg<=.35
                or not 0.<=self.quality_start_scale<=self.quality_target_scale<=1.
                or self.objective_version not in (325,326)):
            raise ValueError('Invalid bounded pose schedule')


def pose_contract() -> dict:
    from fixed_plane_wind_strong import strong_contract
    return strong_contract()


def archived_pose_first_course(version: int, old: tuple) -> tuple:
    from fixed_plane_v325_wind import WindLesson

    @dataclass(frozen=True)
    class PoseLesson(WindLesson):
        pose_start_deg: float = 2.
        pose_target_deg: float = 3.
        pose_ramp_updates: int = 0
        pose_flatness_start_deg: float = .35
        pose_flatness_target_deg: float = .35
        pose_quality_start_scale: float = .5
        pose_quality_target_scale: float = .5

    pose=PoseLesson(f'fplc_v{version}_pose_expand_020',120.,.20,(.3,.7,0.,0.),
        pose_start_deg=2.,pose_target_deg=3.,pose_ramp_updates=800,
        pose_flatness_start_deg=.25,pose_quality_start_scale=0.,pose_quality_target_scale=.5)
    rows=(('wind030_constant',old[9],.5,.75,100),
          ('wind040_mixed',old[13],.75,1.,100),
          ('final1200_mixed',old[-1],1.,1.,0))
    tail=tuple(PoseLesson(**{**asdict(s),'name':f'fplc_v{version}_{name}'},
        pose_start_deg=3.,pose_target_deg=3.,pose_ramp_updates=ramp,
        pose_quality_start_scale=start,pose_quality_target_scale=target)
        for name,s,start,target,ramp in rows)
    return (*old[:9],pose,*tail)


def pose_course(version: int, old: tuple) -> tuple:
    from fixed_plane_v325_wind import WindLesson
    from fixed_plane_wind_strong import FORCE_LIMIT_N

    @dataclass(frozen=True)
    class StrongLesson(WindLesson):
        pose_start_deg: float = 20.
        pose_target_deg: float = 20.
        pose_ramp_updates: int = 0
        pose_flatness_start_deg: float = .25
        pose_flatness_target_deg: float = .35
        pose_quality_start_scale: float = 1.
        pose_quality_target_scale: float = 1.
        strong_wind_reward: bool = True
        peak_wind_fraction: float = .25
        history_enabled: bool = True
        fast_pose_growth: bool = True
        pose_time_driven: bool = False
        continue_on_quality_regression: bool = True

        def domain(self):
            from fixed_plane_wind_strong import strong_wind_config
            return strong_wind_config(replace(super().domain(),force_limit_n=FORCE_LIMIT_N),
                                      self.peak_wind_fraction)

    constant = (.3, .7, 0., 0.)
    slow = (.3, .2, .5, 0.)
    mixed = (.3, .2, .2, .3)
    from fixed_plane_wind_progression import POSE_UPDATES, progress_segment_seconds
    pose = StrongLesson(f'fplc_v{version}_pose20_expand_020',60.,.20,constant,
        pose_start_deg=2., pose_target_deg=20., pose_ramp_updates=POSE_UPDATES,
        pose_time_driven=True,
        pose_quality_start_scale=.5, pose_quality_target_scale=1.,peak_wind_fraction=0.)
    constants = tuple((f'wind{round(speed*100):03d}_constant', speed, constant, 2., (.5,1.5))
        for speed in (.35,.6,.9,1.2,1.6,2.,2.5,3.,3.5,4.))
    speeds = (.6,1.2,2.,2.8,3.4,4.)
    taus = (2.,1.5,1.25,1.,.75,.75)
    slow_rows = tuple((f'wind{round(speed*100):03d}_slow', speed, slow, tau, (.5,1.5))
                      for speed,tau in zip(speeds,taus))
    gusts = ((1.,2.),(.75,1.5),(.5,1.25),(.5,1.),(.35,.9),(.25,.75))
    mixed_rows = tuple((f'wind{round(speed*100):03d}_mixed', speed, mixed, tau, gust)
                       for speed,tau,gust in zip(speeds,taus,gusts))
    tail = tuple(StrongLesson(f'fplc_v{version}_{name}',progress_segment_seconds(
                     name.rsplit('_',1)[1],speed,final=name=='wind400_mixed'),speed,modes,tau,gust)
                 for name,speed,modes,tau,gust in (*constants,*slow_rows,*mixed_rows))
    return (*old[:9], pose, *tail)


@lru_cache(maxsize=4)
def _config_type(parent_type: type) -> type:
    return make_dataclass('WindPoseConfig',
        [('wind_pose', PoseSchedule, field(default_factory=PoseSchedule))],
        bases=(parent_type,), frozen=True)


def with_pose_curriculum(cfg, lesson, version: int = 325):
    if not hasattr(lesson, 'pose_target_deg'):
        return cfg
    schedule = PoseSchedule(start_deg=lesson.pose_start_deg,
        target_deg=lesson.pose_target_deg, ramp_updates=lesson.pose_ramp_updates,
        flatness_start_deg=lesson.pose_flatness_start_deg,
        flatness_target_deg=lesson.pose_flatness_target_deg,
        quality_start_scale=lesson.pose_quality_start_scale,
        quality_target_scale=lesson.pose_quality_target_scale,objective_version=version,
        quality_gated=bool(lesson.pose_ramp_updates) and not getattr(lesson,'pose_time_driven',False),
        strong_wind_reward=getattr(lesson,'strong_wind_reward',False))
    cfg = replace(cfg, cartesian_pose_roll_limit_deg=schedule.target_deg,
                  cartesian_pose_pitch_limit_deg=schedule.target_deg)
    return _config_type(type(cfg))(**vars(cfg), wind_pose=schedule)


def with_pose_clock(cfg, *, origin_steps: int = 0, elapsed_steps: int = 0):
    if not hasattr(cfg, 'wind_pose'):
        return cfg
    return replace(cfg, wind_pose=replace(cfg.wind_pose, origin_steps=0 if cfg.wind_pose.quality_gated else origin_steps,
                                         elapsed_steps_offset=elapsed_steps))


def schedule_fraction(total_steps, cfg: PoseSchedule, *, xp):
    elapsed = xp.maximum(total_steps - cfg.origin_steps + cfg.elapsed_steps_offset, 0)
    fraction = xp.clip(elapsed / max(1, cfg.ramp_updates * 128), 0., 1.)
    if cfg.ramp_updates == 0:
        fraction = xp.ones_like(fraction)
    return fraction


def scheduled_degrees(total_steps, cfg: PoseSchedule, *, xp):
    return cfg.start_deg+(cfg.target_deg-cfg.start_deg)*schedule_fraction(total_steps,cfg,xp=xp)


def scheduled_quality_scale(total_steps, cfg: PoseSchedule, *, xp):
    return cfg.quality_start_scale+(cfg.quality_target_scale-cfg.quality_start_scale)*schedule_fraction(total_steps,cfg,xp=xp)


def shelter_gain(state, dt_s: float, *, xp):
    elapsed = state.step_count * dt_s - state.wind_startup_started_s
    fraction = xp.clip(elapsed, 0., 1.)
    gain = xp.sin(.5 * math.pi * fraction)**2
    return xp.where((state.wind_startup_hit_count >= 5)
                    & (state.wind_startup_started_s >= 0.), gain, 0.)


def pose_clock_steps(state, cfg: PoseSchedule):
    return (state.wind_pose_ramp_steps if cfg.quality_gated else state.total_env_steps)


def exposure_elapsed_steps(state, cfg: PoseSchedule, *, xp):
    """Actual authority progress, independent of optimizer age and graduation."""
    if not cfg.ramp_updates:
        return xp.zeros_like(state.wind_pose_ramp_steps)
    elapsed=pose_clock_steps(state,cfg)-cfg.origin_steps+cfg.elapsed_steps_offset
    return xp.clip(elapsed,0,cfg.ramp_updates*128)


def effective_degrees(state, dt_s: float, cfg: PoseSchedule, *, xp):
    target = scheduled_degrees(pose_clock_steps(state, cfg), cfg, xp=xp)
    return 2. + shelter_gain(state, dt_s, xp=xp) * (target - 2.)


def flatness_deadband_rad(state, dt_s: float, old_rad: float, *, xp,
                          schedule: PoseSchedule | None = None):
    gain = shelter_gain(state, dt_s, xp=xp)
    if schedule is None:
        return xp.zeros_like(gain)+old_rad
    target=schedule.flatness_start_deg+(schedule.flatness_target_deg-schedule.flatness_start_deg)*schedule_fraction(pose_clock_steps(state,schedule),schedule,xp=xp)
    return old_rad+gain*(target*math.pi/180.-old_rad)


def ramp_complete(stage_updates: int, lesson) -> bool:
    return stage_updates >= getattr(lesson, 'pose_ramp_updates', 0)


def pose_config_matches(stored: dict, current: dict) -> bool:
    """Authenticate static settings; runtime clocks are separately validated."""
    from fixed_plane_v323_ball_obs_noise import canonical
    stored, current = json.loads(canonical(stored)), json.loads(canonical(current))
    if 'wind_pose' not in current:
        return canonical(stored) == canonical(current)
    for cfg in (stored, current):
        clock = cfg.get('wind_pose')
        if not isinstance(clock, dict):
            return False
        for name in ('origin_steps', 'elapsed_steps_offset'):
            value = clock.get(name)
            if type(value) is not int or value < 0:
                return False
        cfg['wind_pose'] = {k: v for k, v in clock.items()
                            if k not in ('origin_steps', 'elapsed_steps_offset')}
    return canonical(stored) == canonical(current)


def _batched_projector(original):
    source = textwrap.dedent(inspect.getsource(original))
    if source.count('rp_limit[None, :]') != 2:
        raise ValueError('Frozen pose projection broadcasting changed')
    namespace = dict(original.__globals__)
    exec(compile(source.replace('rp_limit[None, :]', 'rp_limit'),
                 '<wind_pose_batched_projector>', 'exec'), namespace)
    return namespace[original.__name__]


def build_pose_hooks(native: type, wind_step_source: str):
    """Private hooks only; preserve every legacy scalar-bound implementation."""
    projection = native._project_cartesian_pose_reference
    source = textwrap.dedent(inspect.getsource(projection))
    if source.count('outer_rp_limit[None, :]') != 2:
        raise ValueError('Frozen pose nonlinear safety check changed')
    source = source.replace('self.cartesian_pose_roll_pitch_limit', 'self.wind_pose_limits(state)[0]')
    source = source.replace('self.cartesian_pose_outer_roll_pitch_limit', 'self.wind_pose_limits(state)[1]')
    source = source.replace('outer_rp_limit[None, :]', 'outer_rp_limit')
    namespace = {**projection.__globals__, 'project_cartesian_pose_qvel_jax':
                 _batched_projector(projection.__globals__['project_cartesian_pose_qvel_jax'])}
    exec(compile(source, '<wind_pose_reference>', 'exec'), namespace)
    projection_fn = namespace[projection.__name__]

    old = 'self.cartesian_pose_outer_roll_pitch_limit[None, :]'
    if wind_step_source.count(old) != 1:
        raise ValueError('Frozen actual orientation diagnostic changed')
    source = wind_step_source.replace(old, 'self.wind_pose_limits(state)[1]')
    call = '    reward, reward_terms = self._reward(\n'
    if source.count(call) != 1:
        raise ValueError('Frozen reward dispatch anchor changed')
    step_source = source.replace(call, call+'        wind_pose_state=state,\n')
    reward = native._reward
    source = textwrap.dedent(inspect.getsource(reward))
    signature = 'def _reward(\n    self,\n'
    if source.count(signature) != 1:
        raise ValueError('Frozen reward signature changed')
    source = source.replace(signature, signature+'    wind_pose_state,\n    wind_reference_tilt_deg,\n')
    # Keep the exact native penalty form, replacing only its angle deadband.
    start = '    racket_flatness_pen = racket_flatness_penalty_cost(\n'
    end = '    term_racket_flatness_penalty = (\n'
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('Frozen dense flatness reward anchors changed')
    left, rest = source.split(start, 1)
    _, right = rest.split(end, 1)
    source = left + '    racket_flatness_pen = self.wind_pose_flatness_cost(wind_pose_state, racket_up_cos, wind_reference_tilt_deg)\n' + end + right
    target = 'float(self.cfg.hit_flatness_target_cos) - hit_racket_up_cos'
    if source.count(target) != 1:
        raise ValueError('Frozen hit flatness reward anchor changed')
    source = source.replace(target, 'self.wind_pose_hit_cos(wind_pose_state, wind_reference_tilt_deg) - hit_racket_up_cos')
    anchor = 'pose_error_rad=contact_quality_pose_error('
    if source.count(anchor) != 1:
        raise ValueError('Frozen contact-quality pose reward anchor changed')
    source = source.replace(anchor,'pose_error_rad=self.wind_pose_quality_error(wind_pose_state,wind_reference_tilt_deg,')
    namespace = dict(reward.__globals__)
    exec(compile(source, '<wind_pose_reward>', 'exec'), namespace)
    return projection_fn, step_source, namespace[reward.__name__]


def flatness_cost(env, state, up_cos, reference_tilt_deg=None):
    import jax.numpy as jnp
    from fixed_contact_plane_training import motion_penalty_cost, racket_flatness_penalty_cost
    cfg = env.cfg
    old = racket_flatness_penalty_cost(up_cos,
        use_angle=cfg.racket_flatness_use_angle_penalty,
        cos_target=cfg.racket_flatness_target_cos, cos_sigma=cfg.racket_flatness_sigma,
        angle_deadband_rad=cfg.racket_flatness_angle_deadband_rad,
        angle_scale_rad=cfg.racket_flatness_angle_scale_rad)
    angle = jnp.arccos(jnp.clip(up_cos, -1.+1e-7, 1.-1e-7))
    deadband = env.wind_pose_deadband(state, cfg.racket_flatness_angle_deadband_rad,reference_tilt_deg)
    new = motion_penalty_cost(jnp.maximum(angle-deadband, 0.) /
                              float(cfg.racket_flatness_angle_scale_rad), linear_tail=True)
    return jnp.where(shelter_gain(state, env.dt, xp=jnp) > 0., new, old)
