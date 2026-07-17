#!/usr/bin/env python3
"""Ball-state estimator: 60 Hz update / 200 Hz predict Kalman filter.

Unifies three problems the 200 Hz control loop has with the ~60 Hz vision
feed into ONE filter, instead of solving frequency-bridging, latency
compensation, and loss robustness separately:

  1. Frequency bridging. The detector emits a fresh measurement at most at
     the camera rate (~60 Hz); the control loop runs at 200 Hz. Two selectable
     output contracts are supported: ``continuous`` evaluates a ballistic
     prediction every control tick, while ``predict_on_frame_hold`` predicts
     once from the newest accepted measurement and holds that snapshot until
     the next configured output refresh.

  2. Perception-latency compensation (the key trick). Each measurement is
     stamped with the image timestamp ``t_img`` (the instant the ball was
     actually at that position), NOT the time detection finished. The
     committed filter state therefore carries an effective time of ``t_img``.
     When the control loop asks for the state at ``t_now`` via ``predict_to``,
     the elapsed ``(t_now - t_img)`` already includes the ~21-40 ms perception
     pipeline latency, so the extrapolated state is de-aged to the present
     with NO hand-tuned latency constant.

     This compensates ONLY perception (sensing) latency. The execution
     latency (command -> racket in place) is intentionally NOT compensated
     here: it is already modeled by the actuator model used in RL training,
     so the policy expects its commands to act with that lag. Adding a
     forward lead would double-count it. ``prediction_horizon_s`` exists for
     experiments but defaults to 0.0 and should stay there unless the
     actuator model changes.

  3. Loss robustness. A graded timeline replaces a single coarse timeout:
     short losses trust the ballistic extrapolation outright; medium losses
     keep extrapolating but progressively discount the observation (a growing
     reported age the policy already knows how to consume); long losses
     declare the state invalid so the controller's ball-lost failsafe takes
     over.

Pure NumPy, no ROS. One instance per node lifetime; default OFF at the node.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Gravity magnitude in mm/s^2 (positive scalar).
_G_MM_S2 = 9810.0

CONTROL_OUTPUT_CONTINUOUS = 'continuous'
CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD = 'predict_on_frame_hold'
CONTROL_OUTPUT_MODES = (
    CONTROL_OUTPUT_CONTINUOUS,
    CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD,
)


def gravity_for_base_link() -> Tuple[float, float, float]:
    """Gravity vector when state lives in base_link (Z up)."""
    return (0.0, 0.0, -_G_MM_S2)


def gravity_for_camera_frame(camera_tilt_deg: float) -> Tuple[float, float, float]:
    """Gravity vector in the camera optical frame (X right, Y down, Z fwd).

    Matches ``ball_detector.KalmanFilter._compute_gravity``: a positive
    ``camera_tilt_deg`` means the camera looks downward.
    """
    rad = float(np.deg2rad(camera_tilt_deg))
    return (0.0, _G_MM_S2 * float(np.cos(rad)), _G_MM_S2 * float(np.sin(rad)))


@dataclass
class BallStateEstimatorConfig:
    """Tunable parameters for :class:`BallStateEstimator`.

    Distances in mm, velocities in mm/s, times in seconds.
    """

    # Gravity vector in the SAME frame as the (position, velocity) passed to
    # update()/returned by predict_to(). Default assumes base_link, Z up.
    gravity_mm_s2: Tuple[float, float, float] = (0.0, 0.0, -_G_MM_S2)

    # --- Process noise (constant-velocity model uncertainty) ---
    # Position/velocity spectral densities. The velocity term dominates for a
    # fast ball; tuned so the filter tracks pingpong dynamics within a few
    # frames without chattering.
    q_pos_mm2_s: float = 500.0
    q_vel_mult: float = 100.0
    # Depth-axis process noise multipliers. Q is built in the camera frame as
    # diag(lateral, lateral, depth), then rotated into the estimator frame just
    # like R. D455 depth jitter mostly projects into base_link Y for this rig, so
    # keeping velocity process noise lower on the camera-Z axis prevents noisy
    # depth residuals from becoming large vy jumps while leaving lateral response
    # intact.
    q_depth_pos_mult: float = 0.50
    q_depth_vel_mult: float = 0.20
    # Multiplier applied to Q for a few ticks right after a detected bounce/hit
    # so the filter snaps to the post-event measurement instead of fighting it.
    event_q_boost: float = 50.0
    event_boost_ticks: int = 4

    # --- Measurement noise (anisotropic: depth axis is the worst) ---
    # The monocular depth (from circle radius) is far noisier than the lateral
    # pixel center, so R is built in the CAMERA frame with a large depth sigma
    # and rotated into the estimator frame. lateral applies to camera X/Y,
    # depth to camera Z (optical axis).
    r_lateral_mm: float = 20.0
    r_depth_mm: float = 60.0
    # Down-weight (inflate sigma) for a measurement that passed the innovation
    # gate only loosely; also used while reacquiring after a long loss.
    r_reacquire_mult: float = 4.0

    # --- Innovation gate (reject outliers) with an escape hatch ---
    max_measurement_jump_mm: float = 250.0
    max_depth_jump_mm: float = 180.0
    # After this many CONSECUTIVE gated rejects, force a re-init at the latest
    # measurement. Prevents a real bounce/hit (a legitimate velocity jump)
    # from being rejected forever by the gate.
    gate_escape_rejects: int = 3

    # --- Extrapolation limits ---
    # Hard cap on how far predict_to() will roll the committed state forward.
    # Beyond this the returned state is clamped (held at the cap) and flagged.
    # Sized for a ball-juggling toss: when the ball flies above the camera FOV
    # it is in pure free flight (no contact), so the gravity+drag ballistic
    # model stays accurate across the whole out-of-view arc — the cap is bounded
    # by depth-axis velocity noise growth, not the dynamics. A 0.5 m toss leaves
    # the FOV for ~0.3-0.4 s, so 0.45 s covers it with margin. (Was 0.20 s,
    # tuned for short single-frame drops, which truncated a real toss.)
    max_extrapolation_s: float = 0.45
    # While a bounce/hit is suspected imminent (ball near the table), shorten
    # the cap to avoid extrapolating across the discontinuity.
    near_event_extrapolation_s: float = 0.05

    # --- Graded loss timeline (time since last REAL measurement) ---
    # < trust_loss_s:    full trust, reported age ~ true elapsed.
    # < degrade_loss_s:  keep extrapolating, inflate reported age toward the
    #                    policy's age clip so it discounts the stale obs.
    # >= invalid_after_s: declare invalid; controller failsafe takes over.
    # invalid_after_s sits above a full toss (~0.3-0.4 s out of view) so the
    # estimator keeps a valid extrapolation across the apex instead of dropping
    # to the failsafe mid-juggle.
    trust_loss_s: float = 0.10
    degrade_loss_s: float = 0.40
    invalid_after_s: float = 0.60
    # Reported age (s) handed to the policy once in the degrade band, ramped
    # linearly from true-elapsed at degrade start up to this ceiling. Should be
    # >= the policy's ball_obs_age_clip so age_norm saturates.
    degraded_age_ceiling_s: float = 0.20

    # --- Bounce / hit detection ---
    # A sign flip of the vertical velocity (component anti-parallel to gravity)
    # plus a speed jump flags a bounce/hit. table_z arms the sign-flip guard
    # near the table. When table_z is unknown, sign flips alone are ignored so
    # the normal toss apex does not look like a bounce/hit; large speed jumps
    # still flag racket/table impacts.
    table_z_mm: Optional[float] = None
    near_table_band_mm: float = 80.0
    bounce_speed_jump_mm_s: float = 800.0

    # --- Forward lead (experiments only) ---
    # Extra time the state is led PAST t_now. Keep 0.0: execution latency is
    # handled by the RL actuator model, so leading here double-counts it.
    prediction_horizon_s: float = 0.0

    # --- Prediction (ballistic extrapolation) master switch ---
    # When True (default) predict_to() rolls the committed state forward to the
    # control tick: it bridges the ~60 Hz feed to 200 Hz AND de-ages the
    # perception latency (the whole point of the estimator).
    #
    # When False, predict_to() performs NO extrapolation: it returns the last
    # FUSED measurement verbatim (dt=0, reported age = true elapsed since the
    # measurement, extrapolation_s=0). This is the "no prediction, just publish
    # the latest ball state at 200 Hz" mode: the Kalman update() still fuses and
    # smooths every real detection at the camera rate, but the control loop only
    # ever sees held (zero-order) measurements between frames instead of a
    # forward-propagated one. Use this when the incoming ball state does not need
    # frequency up-conversion by prediction (e.g. matching a sim that updates the
    # ball state at the camera rate and merely re-issues it at 200 Hz). The loss
    # timeline (trust/degrade/invalid) is unchanged, so a truly lost ball still
    # ages out and hands control to the failsafe.
    enable_prediction: bool = True

    # --- Policy-facing output scheduling ---
    # continuous: preserve the original behavior and predict to every control
    # tick. predict_on_frame_hold: predict once after a new accepted measurement
    # at no more than frame_hold_output_rate_hz, then zero-order hold that
    # position/velocity snapshot between policy ticks. The latter matches the
    # deployed checkpoint's 200 Hz policy / 50 Hz held-ball observation path.
    control_output_mode: str = CONTROL_OUTPUT_CONTINUOUS
    frame_hold_output_rate_hz: float = 50.0

    # --- Dynamics phase ---
    # True preserves the original free-flight gravity/drag model. False uses a
    # constant-velocity model, intended for the operator-held phase before a
    # release gate clears. Controllers can switch this on at release and ask the
    # estimator to reseed velocity from recent real measurements.
    enable_ballistic_dynamics: bool = True
    dynamics_reseed_history_size: int = 8
    dynamics_reseed_window_s: float = 0.12

    # --- Aerodynamics (quadratic drag + optional Magnus) ---
    # A 2.7 g / 40 mm ball has a very high drag-to-mass ratio: ignoring drag
    # overshoots a fast ball by ~80 mm over a 150 ms detector dropout (8 m/s).
    # Modeled as a_drag = -drag_k_per_mm * |v| * v (v in mm/s, a in mm/s^2).
    # Default derived from F_drag = 0.5*rho*A*Cd*|v|*v with the ITTF ball and
    # standard air: rho=1.20 kg/m^3, A=pi*(0.02 m)^2, Cd=0.40, m=2.7 g
    #   k_SI = 0.5*rho*A*Cd/m = 0.1117 /m  ->  k_per_mm = k_SI/1000 = 1.117e-4.
    # Terminal velocity sqrt(g/k_SI) = 9.37 m/s matches the measured ~9.5 m/s.
    enable_aero_drag: bool = True
    drag_k_per_mm: float = 1.117e-4

    # Magnus (spin) lift. OFF by default and inert at zero spin: this rig has
    # no spin sensor, and inferring 3-axis spin from short, noisy position arcs
    # is weakly observable and destabilizes the filter. Left as a config hook
    # for experiments with a KNOWN serve spin only. a_magnus = magnus_k * (w x v)
    # with w in rad/s and v in mm/s; magnus_k carries the 0.5*rho*A*Cl*r/m group
    # in mm units. Keep enable_magnus False unless spin is externally supplied.
    enable_magnus: bool = False
    magnus_k: float = 0.0
    spin_rad_s: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Max integration sub-step for the drag/Magnus RK4. The dynamics are
    # nonlinear so a long predict_to() is split into <= this many-second steps
    # for accuracy; the no-drag closed form needs no sub-stepping.
    aero_substep_s: float = 0.005

    # --- Adaptive innovation gate ---
    # After a detector dropout the ball has moved far, so the first reacquired
    # detection legitimately sits far from the (now stale) prediction and the
    # fixed gate rejects it -- costing extra loss before tracking resumes.
    # When enabled the position/depth gates grow with the time since the last
    # accepted measurement: gate *= 1 + gate_loss_growth_per_s * loss_s,
    # capped at gate_loss_max_mult. Set growth 0 to recover the fixed gate.
    #
    # Default OFF: replaying real logs showed it admits depth outliers the
    # fixed gate correctly rejected, worsening long-gap prediction. Enable only
    # with a cleaner depth source. The fixed gate already has its own escape
    # hatch (gate_escape_rejects) for legitimate bounce/hit velocity jumps.
    adaptive_gate: bool = False
    gate_loss_growth_per_s: float = 6.0
    gate_loss_max_mult: float = 4.0


def _propagate(pos: np.ndarray, vel: np.ndarray, gravity: np.ndarray,
               dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form constant-accel ballistic step (no drag).

    Used when aerodynamics are disabled. With drag/Magnus the motion has no
    closed form -- see :func:`_propagate_aero`.
    """
    dt = max(float(dt), 0.0)
    if dt <= 0.0:
        return pos.copy(), vel.copy()
    new_pos = pos + vel * dt + 0.5 * gravity * (dt ** 2)
    new_vel = vel + gravity * dt
    return new_pos, new_vel


def _aero_accel(vel: np.ndarray, gravity: np.ndarray, drag_k: float,
                magnus_k: float, spin: Optional[np.ndarray]) -> np.ndarray:
    """Acceleration under gravity + quadratic drag + optional Magnus lift.

    a = g - drag_k*|v|*v + magnus_k*(spin x v). All in the estimator frame,
    distances in mm so |v| is mm/s and a is mm/s^2.
    """
    speed = float(np.linalg.norm(vel))
    a = gravity.copy()
    if drag_k > 0.0 and speed > 1e-6:
        a = a - drag_k * speed * vel
    if magnus_k != 0.0 and spin is not None:
        a = a + magnus_k * np.cross(spin, vel)
    return a


def _propagate_aero(pos: np.ndarray, vel: np.ndarray, gravity: np.ndarray,
                    dt: float, drag_k: float, magnus_k: float,
                    spin: Optional[np.ndarray],
                    substep_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """RK4 integration of the drag/Magnus dynamics over ``dt``.

    The dynamics are nonlinear (drag ~ |v|*v), so a long ``dt`` is split into
    fixed sub-steps of at most ``substep_s`` for accuracy. Returns the new
    (position, velocity). Reduces to gravity-only if drag/Magnus are off.
    """
    dt = max(float(dt), 0.0)
    if dt <= 0.0:
        return pos.copy(), vel.copy()
    substep = max(float(substep_s), 1e-4)
    n = max(1, int(np.ceil(dt / substep)))
    h = dt / n
    p = pos.astype(float).copy()
    v = vel.astype(float).copy()
    for _ in range(n):
        # RK4 on (p, v) with dp/dt = v, dv/dt = a(v).
        a1 = _aero_accel(v, gravity, drag_k, magnus_k, spin)
        k1p, k1v = v, a1
        a2 = _aero_accel(v + 0.5 * h * k1v, gravity, drag_k, magnus_k, spin)
        k2p, k2v = v + 0.5 * h * k1v, a2
        a3 = _aero_accel(v + 0.5 * h * k2v, gravity, drag_k, magnus_k, spin)
        k3p, k3v = v + 0.5 * h * k2v, a3
        a4 = _aero_accel(v + h * k3v, gravity, drag_k, magnus_k, spin)
        k4p, k4v = v + h * k3v, a4
        p = p + (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return p, v


# State-machine labels (plain strings; callers need not import them).
STATE_TRACKING = 'tracking'
STATE_PREDICTING = 'predicting'
STATE_LOST = 'lost'


class BallStateEstimator:
    """Single filter: 60 Hz Kalman update + 200 Hz ballistic predict.

    Threading: ``update`` runs on the vision callback thread, ``predict_to``
    on the 200 Hz control thread. The instance guards its own state with a
    lock, so a node may call the two from different threads safely.

    Intended use::

        est = BallStateEstimator(config, frame_rotation=R_base_camera)
        # vision callback, on a REAL detection:
        est.update(position_mm, t_img)
        # control tick:
        out = est.predict_to(t_now)
        if out['valid']:
            feed out['position'] / out['velocity'] / out['age_s'] to policy
    """

    def __init__(self, config: Optional[BallStateEstimatorConfig] = None,
                 frame_rotation: Optional[np.ndarray] = None):
        self.config = config or BallStateEstimatorConfig()
        self._gravity = np.asarray(self.config.gravity_mm_s2, dtype=float)
        self.control_output_mode = str(
            self.config.control_output_mode).strip().lower()
        if self.control_output_mode not in CONTROL_OUTPUT_MODES:
            raise ValueError(
                f'Unsupported control_output_mode={self.control_output_mode!r}; '
                f'expected one of {CONTROL_OUTPUT_MODES}')
        frame_hold_rate_hz = float(self.config.frame_hold_output_rate_hz)
        if not np.isfinite(frame_hold_rate_hz) or frame_hold_rate_hz <= 0.0:
            raise ValueError(
                'frame_hold_output_rate_hz must be finite and positive')
        self._frame_hold_output_period_s = 1.0 / frame_hold_rate_hz
        self._ballistic_dynamics_enabled = bool(
            self.config.enable_ballistic_dynamics)

        # Cached aero parameters (resolved once; 0 disables a term).
        self._drag_k = (float(self.config.drag_k_per_mm)
                        if self.config.enable_aero_drag else 0.0)
        self._magnus_k = (float(self.config.magnus_k)
                          if self.config.enable_magnus else 0.0)
        self._spin = np.asarray(self.config.spin_rad_s, dtype=float)
        self._aero_on = self._drag_k > 0.0 or self._magnus_k != 0.0

        # R_base_camera (3x3): rotates a camera-frame covariance into the
        # estimator frame so the noisy depth axis stays aligned with the
        # optical axis. None -> isotropic measurement noise.
        self._frame_rotation = (
            np.asarray(frame_rotation, dtype=float)
            if frame_rotation is not None else None)
        self._R_meas = self._build_measurement_noise()
        self._Q_pos_axis, self._Q_vel_axis = self._build_process_noise_axes()

        self._lock = threading.Lock()
        self._reset_locked()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_measurement_noise(self) -> np.ndarray:
        """3x3 measurement covariance, anisotropic on the camera depth axis.

        Built in the camera frame as diag(lateral, lateral, depth)^2, then
        rotated into the estimator frame via R_base_camera so the large depth
        variance follows the optical axis even when output is base_link.
        """
        lat = float(self.config.r_lateral_mm) ** 2
        dep = float(self.config.r_depth_mm) ** 2
        r_cam = np.diag([lat, lat, dep]).astype(float)
        if self._frame_rotation is None:
            return r_cam
        R = self._frame_rotation
        if R.shape != (3, 3) or not np.all(np.isfinite(R)):
            return r_cam
        # Covariance transform: R_meas_frame = Rot @ R_cam @ Rot^T.
        return R @ r_cam @ R.T

    def _build_process_noise_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Unit-dt process-noise axis matrices in the estimator frame.

        Q is anisotropic on the camera optical/depth axis, then rotated into the
        estimator frame. With no rotation, the estimator is assumed to already be
        in camera optical coordinates, so the depth axis is local Z.
        """
        pos_mult = max(float(self.config.q_depth_pos_mult), 1e-6)
        vel_mult = max(float(self.config.q_depth_vel_mult), 1e-6)
        q_pos_cam = np.diag([1.0, 1.0, pos_mult]).astype(float)
        q_vel_cam = np.diag([1.0, 1.0, vel_mult]).astype(float)
        R = self._frame_rotation
        if R is None or R.shape != (3, 3) or not np.all(np.isfinite(R)):
            return q_pos_cam, q_vel_cam
        return R @ q_pos_cam @ R.T, R @ q_vel_cam @ R.T

    def _depth_component(self, vec: np.ndarray) -> float:
        """Camera-depth component of an estimator-frame vector."""
        v = np.asarray(vec, dtype=float).reshape(3)
        R = self._frame_rotation
        if R is None or R.shape != (3, 3) or not np.all(np.isfinite(R)):
            return float(v[2])
        # R maps camera -> estimator, so camera_z = R[:, 2] dot vec_estimator.
        return float(np.dot(R[:, 2], v))

    def _reset_locked(self) -> None:
        """Drop all state; the next update() re-initializes. Caller holds lock."""
        self._ballistic_dynamics_enabled = bool(
            self.config.enable_ballistic_dynamics)
        self._pos = np.zeros(3)          # committed position at _filter_time
        self._vel = np.zeros(3)          # committed velocity at _filter_time
        self._P = np.eye(6) * 1e6        # committed covariance
        self._filter_time: Optional[float] = None      # == t_img of last update
        self._last_meas_time: Optional[float] = None    # last REAL measurement
        self._pending_init = None        # (pos, t) awaiting a 2nd point to seed v
        self._initialized = False
        self._consecutive_rejects = 0
        self._reject_samples: list = []  # (pos, t) of gated rejects, for re-seed
        self._event_boost_remaining = 0
        self._prev_vert_vel = 0.0        # vertical velocity sign tracker
        self._state = STATE_TRACKING
        self.last_event: Optional[str] = None
        # Policy-facing snapshot state. Measurement sequence advances only for
        # accepted/reinitialized real observations, never for gated rejects.
        self._measurement_sequence = 0
        self._frame_hold_cached_sequence = -1
        self._frame_hold_snapshot = None
        self._frame_hold_last_refresh_time: Optional[float] = None
        self._measurement_history = deque(
            maxlen=max(2, int(self.config.dynamics_reseed_history_size)))
        # Diagnostics.
        self.updates_accepted = 0
        self.updates_rejected = 0
        self.reinits = 0
        self.bounces_detected = 0
        self.frame_hold_snapshots = 0
        self.dynamics_reseeds = 0

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()
            self.last_event = 'reset'

    def set_ballistic_dynamics(
            self, enabled: bool,
            *, reinitialize_from_history: bool = False) -> None:
        """Switch between held/static CV and free-flight ballistic dynamics.

        The original estimator behavior is ``enabled=True`` for its entire
        lifetime. Release-gated controllers may start with it disabled, then
        enable it once real measurement history confirms that the ball moved.
        """
        with self._lock:
            enabled = bool(enabled)
            if (enabled == self._ballistic_dynamics_enabled
                    and not reinitialize_from_history):
                return
            self._ballistic_dynamics_enabled = enabled

            reseeded = False
            if enabled and reinitialize_from_history and self._measurement_history:
                latest_pos, latest_t = self._measurement_history[-1]
                window_s = max(
                    0.0, float(self.config.dynamics_reseed_window_s))
                samples = [
                    (pos, stamp)
                    for pos, stamp in self._measurement_history
                    if latest_t - stamp <= window_s + 1e-9
                ]
                velocity = self._velocity_from_samples(samples)
                self._init_at(latest_pos, latest_t, velocity=velocity)
                self.reinits += 1
                self.dynamics_reseeds += 1
                reseeded = True

            # Force a fresh policy snapshot on the next control tick even when
            # the configured 50 Hz output period has not elapsed yet.
            self._frame_hold_cached_sequence = -1
            self._frame_hold_last_refresh_time = None
            self.last_event = (
                'ballistic_enabled_reseed' if reseeded
                else ('ballistic_enabled' if enabled else 'ballistic_disabled'))

    # ------------------------------------------------------------------
    # 60 Hz update (vision callback thread)
    # ------------------------------------------------------------------

    def update(self, position_mm, t_img: float,
               measurement_noise_mult: float = 1.0) -> None:
        """Fuse one REAL detection, stamped at its image time ``t_img``.

        Args:
            position_mm: measured [x, y, z] in the estimator frame (mm).
            t_img: image timestamp the measurement belongs to (seconds, same
                clock domain the control loop will pass to ``predict_to``).
            measurement_noise_mult: optional inflation of R for this update
                (e.g. a loosely-trusted measurement).
        """
        z = np.asarray(position_mm, dtype=float).reshape(-1)
        if z.shape != (3,) or not np.all(np.isfinite(z)):
            return
        t = float(t_img)
        if not np.isfinite(t):
            return

        with self._lock:
            if (not self._measurement_history
                    or t > self._measurement_history[-1][1] + 1e-9):
                self._measurement_history.append((z.copy(), t))
            # First-ever measurement (or after a reset): stash it and wait for
            # a second point to seed velocity by finite difference (two-point
            # initialization). A CV filter started from v=0 takes ~5 frames to
            # converge velocity, which is too slow for a fast ball.
            if not self._initialized or self._filter_time is None:
                if self._pending_init is None:
                    self._pending_init = (z.copy(), t)
                    self.last_event = 'init_pending'
                    return
                z0, t0 = self._pending_init
                self._pending_init = None
                dt0 = t - t0
                if dt0 <= 1e-4 or dt0 > 0.2:
                    # Bad spacing (duplicate stamp or huge gap): restart the
                    # two-point seed from this measurement.
                    self._pending_init = (z.copy(), t)
                    return
                v_seed = (z - z0) / dt0
                self._init_at(z, t, velocity=v_seed)
                return

            dt = t - self._filter_time
            if dt < 0.0:
                # Out-of-order frame: ignore rather than rewind the filter.
                self.last_event = 'stale_measurement'
                return
            dt = max(dt, 1e-4)

            # Predict the committed state forward to the measurement time.
            pred_pos, pred_vel = self._step(self._pos, self._vel, dt)
            F = self._transition(dt, self._vel)
            Q = self._process_noise(dt)
            pred_P = F @ self._P @ F.T + Q

            # Innovation gate with an escape hatch for real bounces/hits.
            # After a detector dropout the ball has moved far, so the first
            # reacquired detection legitimately sits far from the (stale)
            # prediction. Grow the gate with the time since the last accepted
            # measurement so reacquisition isn't rejected as an outlier.
            gate_mult = 1.0
            if self.config.adaptive_gate and self._last_meas_time is not None:
                loss = max(t - self._last_meas_time, 0.0)
                gate_mult = min(
                    1.0 + float(self.config.gate_loss_growth_per_s) * loss,
                    float(self.config.gate_loss_max_mult))
            jump_gate = self.config.max_measurement_jump_mm * gate_mult
            depth_gate = self.config.max_depth_jump_mm * gate_mult

            innov = z - pred_pos
            innov_norm = float(np.linalg.norm(innov))
            depth_innov = float(abs(self._depth_component(innov)))
            gated = (innov_norm > jump_gate or depth_innov > depth_gate)

            if gated:
                self._consecutive_rejects += 1
                self.updates_rejected += 1
                self._reject_samples.append((z.copy(), t))
                if self._consecutive_rejects >= max(
                        1, int(self.config.gate_escape_rejects)):
                    # Escape: the "outliers" are consistent -> a real event
                    # (bounce/hit). Re-seed velocity from the reject trend so
                    # the post-event trajectory is tracked immediately instead
                    # of restarting from rest.
                    v_seed = self._velocity_from_samples(self._reject_samples)
                    self._init_at(z, t, velocity=v_seed)
                    self._reject_samples = []
                    self._event_boost_remaining = max(
                        self._event_boost_remaining,
                        int(self.config.event_boost_ticks))
                    self.reinits += 1
                    self.bounces_detected += 1
                    self.last_event = 'gate_escape_reinit'
                else:
                    # Reject: keep the committed state but still advance time
                    # to the measurement so extrapolation stays anchored, and
                    # leave the loss clock untouched (this was not a good obs).
                    self._pos, self._vel, self._P = pred_pos, pred_vel, pred_P
                    self._filter_time = t
                    self.last_event = 'gated_reject'
                return

            self._consecutive_rejects = 0
            self._reject_samples = []

            # Detect bounce/hit BEFORE correcting, using the predicted vertical
            # velocity vs. its previous sign; arms the event Q-boost.
            self._maybe_flag_event(pred_pos, pred_vel)

            # Kalman correction (position-only measurement).
            R = self._R_meas * float(max(measurement_noise_mult, 1e-3))
            H = np.zeros((3, 6))
            H[:, :3] = np.eye(3)
            x_pred = np.concatenate([pred_pos, pred_vel])
            S = H @ pred_P @ H.T + R
            try:
                K = pred_P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                self._pos, self._vel, self._P = pred_pos, pred_vel, pred_P
                self._filter_time = t
                return
            x_upd = x_pred + K @ innov
            P_upd = (np.eye(6) - K @ H) @ pred_P

            self._pos = x_upd[:3]
            self._vel = x_upd[3:]
            self._P = P_upd
            self._filter_time = t
            self._last_meas_time = t
            self._prev_vert_vel = float(np.dot(self._vel, self._up_dir()))
            self._state = STATE_TRACKING
            self.updates_accepted += 1
            self._mark_measurement_accepted_locked()
            self.last_event = 'updated'

    # ------------------------------------------------------------------
    # 200 Hz predict (control thread)
    # ------------------------------------------------------------------

    def predict_to(self, t_now: float) -> dict:
        """Extrapolate the committed state to ``t_now`` for the control loop.

        The committed filter state (pos, vel, P, time) is never modified, so the
        next update() still fuses against the real last measurement. In
        ``continuous`` mode, the elapsed (t_now - t_img) is propagated on every
        call. In ``predict_on_frame_hold`` mode, the first due control tick after
        a new accepted measurement creates one de-aged snapshot and subsequent
        calls return that fixed position/velocity until the next refresh.

        Returns a dict shaped like a detector observation:
            position [x,y,z] mm, velocity [vx,vy,vz] mm/s, valid (bool),
            age_s (graded age for the policy), detected (== valid), and
            diagnostics: state, extrapolation_s, loss_s, clamped, predicted.
        """
        t = float(t_now)
        with self._lock:
            if not self._initialized or self._filter_time is None:
                return self._invalid_out(reason='uninitialized')

            # Time since the last REAL measurement governs the loss timeline.
            loss_s = (t - self._last_meas_time
                      if self._last_meas_time is not None else 0.0)
            loss_s = max(loss_s, 0.0)

            if loss_s >= self.config.invalid_after_s:
                self._state = STATE_LOST
                return self._invalid_out(reason='loss_timeout', loss_s=loss_s)

            output_mode = self.control_output_mode
            snapshot_sequence = self._measurement_sequence
            snapshot_new = False
            snapshot_hold_s = 0.0

            if not self.config.enable_prediction:
                # No-prediction mode: return the last FUSED measurement held at
                # its committed time. No ballistic roll-forward, no latency
                # de-aging — just the smoothed 60 Hz state re-issued at 200 Hz.
                output_mode = 'no_prediction'
                dt = 0.0
                clamped = False
                pos, vel = self._pos.copy(), self._vel.copy()
                age_s = self._graded_age(loss_s)
                control_age_s = loss_s
            elif output_mode == CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD:
                has_new_measurement = (
                    self._measurement_sequence
                    != self._frame_hold_cached_sequence)
                refresh_due = (
                    self._frame_hold_last_refresh_time is None
                    or t - self._frame_hold_last_refresh_time
                    >= self._frame_hold_output_period_s - 1e-9)
                if (self._frame_hold_snapshot is None
                        or (has_new_measurement and refresh_due)):
                    pos, vel, dt, clamped = self._predict_committed_to_locked(t)
                    self._frame_hold_snapshot = {
                        'position': pos.copy(),
                        'velocity': vel.copy(),
                        'extrapolation_s': float(dt),
                        'clamped': bool(clamped),
                        'control_time_s': float(t),
                        'measurement_sequence': int(self._measurement_sequence),
                    }
                    self._frame_hold_cached_sequence = self._measurement_sequence
                    self._frame_hold_last_refresh_time = float(t)
                    self.frame_hold_snapshots += 1
                    snapshot_new = True

                snapshot = self._frame_hold_snapshot
                pos = snapshot['position'].copy()
                vel = snapshot['velocity'].copy()
                dt = float(snapshot['extrapolation_s'])
                clamped = bool(snapshot['clamped'])
                snapshot_sequence = int(snapshot['measurement_sequence'])
                snapshot_hold_s = max(
                    0.0, t - float(snapshot['control_time_s']))
                # The checkpoint treats normal inter-frame zero-order hold as
                # fresh. Age starts only when the next 50 Hz refresh is overdue.
                overdue_s = max(
                    0.0,
                    snapshot_hold_s - self._frame_hold_output_period_s)
                age_s = self._graded_age(overdue_s)
                # Safety gating must account for both the prediction used to
                # make the snapshot and how long that snapshot has been held.
                control_age_s = float(dt) + snapshot_hold_s
            else:
                pos, vel, dt, clamped = self._predict_committed_to_locked(t)
                age_s = self._graded_age(loss_s)
                control_age_s = float(dt)

            prediction_active = bool(self.config.enable_prediction)
            if self._state != STATE_LOST:
                # With prediction disabled the output is a held fused
                # measurement, never a predicted state.  Keep loss_s/age_s for
                # stale-observation safety, but do not report "predicting".
                self._state = (
                    STATE_TRACKING
                    if (not prediction_active
                        or loss_s < self.config.trust_loss_s)
                    else STATE_PREDICTING)

            return {
                'position': pos.tolist(),
                'velocity': vel.tolist(),
                'detected': True,
                'valid': True,
                'age_s': float(age_s),
                'predicted': bool(
                    prediction_active
                    and loss_s >= self.config.trust_loss_s),
                'state': self._state,
                'extrapolation_s': float(dt),
                'loss_s': float(loss_s),
                'clamped': bool(clamped),
                'output_mode': output_mode,
                'snapshot_sequence': int(snapshot_sequence),
                'snapshot_new': bool(snapshot_new),
                'snapshot_hold_s': float(snapshot_hold_s),
                'control_age_s': float(control_age_s),
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_at(self, z: np.ndarray, t: float,
                 velocity: Optional[np.ndarray] = None) -> None:
        """(Re)initialize the committed state at a measurement. Holds lock.

        ``velocity`` seeds the velocity estimate (mm/s); when None the ball is
        assumed at rest and the velocity covariance is left large so the next
        few updates pull it in.
        """
        self._pos = z.copy()
        if velocity is not None and np.all(np.isfinite(velocity)):
            self._vel = np.asarray(velocity, dtype=float).reshape(3).copy()
            vel_var = (self.config.r_depth_mm * 4.0) ** 2
        else:
            self._vel = np.zeros(3)
            vel_var = 1e6
        self._P = np.eye(6) * 1e6
        # Position is well-known at init; velocity per the seed confidence.
        self._P[:3, :3] = self._R_meas.copy()
        self._P[3:, 3:] = np.eye(3) * vel_var
        self._filter_time = t
        self._last_meas_time = t
        self._pending_init = None
        self._initialized = True
        self._consecutive_rejects = 0
        self._prev_vert_vel = float(np.dot(self._vel, self._up_dir()))
        self._state = STATE_TRACKING
        self._mark_measurement_accepted_locked()

    def _mark_measurement_accepted_locked(self) -> None:
        """Advance the version consumed by frame-held policy snapshots."""
        self._measurement_sequence += 1

    def _predict_committed_to_locked(
            self, control_time_s: float
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Predict committed state to one control target without mutating it."""
        target = float(control_time_s) + max(
            float(self.config.prediction_horizon_s), 0.0)
        raw_dt = target - float(self._filter_time)

        cap = float(self.config.max_extrapolation_s)
        if self._near_event(self._pos):
            cap = min(cap, float(self.config.near_event_extrapolation_s))
        clamped = raw_dt > cap
        dt = min(max(raw_dt, 0.0), cap)
        pos, vel = self._step(self._pos, self._vel, dt)
        return pos, vel, float(dt), bool(clamped)

    def _velocity_from_samples(self, samples: list) -> Optional[np.ndarray]:
        """Least-squares velocity (mm/s) from a list of (pos, t) reject points.

        Used at a gate-escape re-init so the post-bounce/hit trajectory is
        tracked immediately. Returns None when there are too few points or the
        time span is degenerate (the caller then re-inits from rest).
        """
        if len(samples) < 2:
            return None
        times = np.array([s[1] for s in samples], dtype=float)
        positions = np.array([s[0] for s in samples], dtype=float)
        t0 = times[0]
        t = times - t0
        if float(np.ptp(t)) < 1e-4:
            return None
        A = np.stack([np.ones_like(t), t], axis=1)
        try:
            coeffs, *_ = np.linalg.lstsq(A, positions, rcond=None)
        except np.linalg.LinAlgError:
            return None
        vel = coeffs[1]
        return vel if np.all(np.isfinite(vel)) else None

    def _step(self, pos: np.ndarray, vel: np.ndarray,
              dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate (pos, vel) by dt using the active dynamics model.

        Dispatches to the RK4 drag/Magnus integrator when aerodynamics are on,
        otherwise the closed-form gravity-only step. Single source of truth so
        update() and predict_to() always integrate identically.
        """
        if not self._ballistic_dynamics_enabled:
            return pos + vel * dt, vel.copy()
        if self._aero_on:
            return _propagate_aero(
                pos, vel, self._gravity, dt, self._drag_k, self._magnus_k,
                self._spin if self._magnus_k != 0.0 else None,
                self.config.aero_substep_s)
        return _propagate(pos, vel, self._gravity, dt)

    def _transition(self, dt: float, vel: Optional[np.ndarray] = None
                    ) -> np.ndarray:
        """State-transition Jacobian F (6x6) for covariance propagation.

        With no drag this is the constant-velocity block matrix. With drag the
        velocity dynamics depend on v (a = -k|v|v), so we linearize about the
        current velocity: d(a)/d(v) = -k*(|v| I + v v^T / |v|). The discrete
        Jacobian uses a first-order (Euler) expansion of that block, which is
        accurate for the small dt of a single update/predict tick and keeps the
        covariance consistent with the nonlinear mean integrated by _step().
        """
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        if (not self._ballistic_dynamics_enabled
                or not self._aero_on or vel is None):
            return F
        v = np.asarray(vel, dtype=float).reshape(3)
        speed = float(np.linalg.norm(v))
        if self._drag_k > 0.0 and speed > 1e-6:
            # Jacobian of drag accel wrt velocity.
            Jv = -self._drag_k * (speed * np.eye(3) + np.outer(v, v) / speed)
        else:
            Jv = np.zeros((3, 3))
        if self._magnus_k != 0.0:
            # d/dv (magnus_k * spin x v) = magnus_k * [spin]_x.
            sx, sy, sz = self._spin
            skew = np.array([[0.0, -sz, sy], [sz, 0.0, -sx], [-sy, sx, 0.0]])
            Jv = Jv + self._magnus_k * skew
        # Velocity-update block: v_{k+1} ~= (I + Jv dt) v_k.
        F[3:, 3:] = np.eye(3) + Jv * dt
        # Position picks up half the velocity-change curvature over dt.
        F[:3, 3:] = np.eye(3) * dt + 0.5 * Jv * (dt ** 2)
        return F

    def _process_noise(self, dt: float) -> np.ndarray:
        q_pos = float(self.config.q_pos_mm2_s)
        q_vel = q_pos * float(self.config.q_vel_mult)
        if self._event_boost_remaining > 0:
            q_pos *= float(self.config.event_q_boost)
            q_vel *= float(self.config.event_q_boost)
            self._event_boost_remaining -= 1
        Q = np.zeros((6, 6))
        Q[:3, :3] = self._Q_pos_axis * (q_pos * dt)
        Q[3:, 3:] = self._Q_vel_axis * (q_vel * dt)
        return Q

    def _up_dir(self) -> np.ndarray:
        gnorm = float(np.linalg.norm(self._gravity))
        if gnorm < 1e-6:
            return np.array([0.0, 0.0, 1.0])
        return -self._gravity / gnorm

    def _maybe_flag_event(self, pred_pos: np.ndarray,
                          pred_vel: np.ndarray) -> None:
        """Flag a bounce/hit from a vertical-velocity sign flip + speed jump.

        Arms the process-noise boost so the next few updates snap to the new
        (post-event) trajectory instead of being held back by the gate/filter.
        """
        up = self._up_dir()
        vert = float(np.dot(pred_vel, up))
        prev = self._prev_vert_vel
        sign_flip = (prev < -1e-3 and vert > 1e-3) or (prev > 1e-3 and vert < -1e-3)
        speed_jump = abs(vert - prev) > self.config.bounce_speed_jump_mm_s
        near = self._near_event(pred_pos)
        # A bounce off the table: was going down, now up, near the table.
        # A hit: large vertical-speed jump. Without table_z, a normal toss apex
        # is also a sign flip, so only speed jumps can flag events.
        is_event = (sign_flip and near) or speed_jump
        if is_event:
            self._event_boost_remaining = max(
                self._event_boost_remaining,
                int(self.config.event_boost_ticks))
            self.bounces_detected += 1
            self.last_event = 'bounce_or_hit'

    def _near_event(self, pos: np.ndarray) -> bool:
        """True if the ball is within the bounce band of the table plane."""
        if self.config.table_z_mm is None:
            return False
        up = self._up_dir()
        # Height of the ball above the table along the up axis.
        height = float(np.dot(pos, up) - np.dot(
            np.asarray(self._table_point(), dtype=float), up))
        return abs(height) <= float(self.config.near_table_band_mm)

    def _table_point(self) -> np.ndarray:
        """A point on the table plane along the up axis (table_z on up axis)."""
        up = self._up_dir()
        return up * float(self.config.table_z_mm or 0.0)

    def _graded_age(self, loss_s: float) -> float:
        """Map time-since-last-measurement to the age handed to the policy.

        Trust band -> the true elapsed (de-aged to ~0 on a fresh frame).
        Degrade band -> ramp from true elapsed up to the ceiling so age_norm
        saturates and the policy discounts the increasingly stale obs.
        """
        if loss_s < self.config.trust_loss_s:
            return float(loss_s)
        d0 = self.config.trust_loss_s
        d1 = max(self.config.degrade_loss_s, d0 + 1e-3)
        frac = min(max((loss_s - d0) / (d1 - d0), 0.0), 1.0)
        ceil = max(self.config.degraded_age_ceiling_s, loss_s)
        return float(loss_s + frac * (ceil - loss_s))

    def _invalid_out(self, reason: str, loss_s: float = 0.0) -> dict:
        output_mode = (
            self.control_output_mode
            if self.config.enable_prediction else 'no_prediction')
        return {
            'position': [0.0, 0.0, 0.0],
            'velocity': [0.0, 0.0, 0.0],
            'detected': False,
            'valid': False,
            'age_s': self.config.degraded_age_ceiling_s,
            'predicted': False,
            'state': self._state,
            'extrapolation_s': 0.0,
            'loss_s': float(loss_s),
            'clamped': False,
            'reject_reason': reason,
            'output_mode': output_mode,
            'snapshot_sequence': int(self._measurement_sequence),
            'snapshot_new': False,
            'snapshot_hold_s': 0.0,
            'control_age_s': float(loss_s),
        }

    def diagnostics(self) -> dict:
        """Snapshot of counters/state for throttled logging (thread-safe)."""
        with self._lock:
            return {
                'state': self._state,
                'last_event': self.last_event,
                'updates_accepted': self.updates_accepted,
                'updates_rejected': self.updates_rejected,
                'reinits': self.reinits,
                'bounces_detected': self.bounces_detected,
                'initialized': self._initialized,
                'control_output_mode': self.control_output_mode,
                'measurement_sequence': self._measurement_sequence,
                'frame_hold_snapshots': self.frame_hold_snapshots,
                'ballistic_dynamics_enabled': self._ballistic_dynamics_enabled,
                'dynamics_reseeds': self.dynamics_reseeds,
            }
