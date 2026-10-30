"""Causal ball observation updates with separate bearing/range validity.

All units are metres/seconds in a frame whose Z axis points up. Only trusted
3D measurements enter the bounded flight fit. Bearing updates constrain the
transverse state, never renew the last real range timestamp, and cannot seed
a trajectory by themselves. Repeated control queries do not update history.
"""
from collections import deque
from time import perf_counter_ns

import numpy as np
from pingpong_controller.ball_contact_observation import BallContactObservation
from pingpong_controller.ball_radius_observation import update_radius_observation
from pingpong_controller.ball_observation_noise import ObservationNoiseProfile, default_profile


class BallObservationModel:
    MAX_RANGE_AGE_S = .200
    MAX_UNCERTAINTY_M = .025
    # A missing image cannot establish lift-off. Briefly coast the last known
    # support motion, including two 90 Hz misses plus normal delivery latency.
    # Longer unobserved support motion needs fresh evidence, not a 320 ms coast.
    MAX_SUPPORTED_PREDICTION_S = .050
    HISTORY_WINDOW_S = .120
    GRAVITY = np.array([0., 0., -9.8])

    def __init__(self, gravity=9.8, prediction_floor_m=None, noise_profile=None):
        self.gravity = np.array([0., 0., -float(gravity)])
        self.prediction_floor_m = prediction_floor_m
        self.noise_profile = noise_profile or ObservationNoiseProfile()
        self.reset()

    def reset(self):
        self.history = deque(maxlen=10)
        self.state = None
        self.covariance = None
        self.state_time = None
        self.last_frame_time = None
        self.last_range_time = None
        self.last_position = None
        self.position_valid = False
        self.velocity_valid = False
        self.supported = False
        self.segment = 0
        self.fast_segment = False
        self.contact_velocity_prior = None
        self.output_fit_weight = 0.
        self.reacquisition_filter_enabled = False
        self.axis_refinement_enabled = False
        self.fit_history_samples = 0
        self.fit_span_s = 0.
        self.filter_update = 'none'
        self.horizontal_noise_scale = 1.
        self.robust_noise_inflation = 1.
        self.noise_contact_active = False
        self.last_radius_time = None
        self.last_radius_accepted_time = None
        self.current_raw = {}
        self.reason = 'insufficient_history'
        self.source = 'missing'
        self.fit_rmse = None
        self.bearing_accepted = False
        self.range_rejected = False
        self.pending_range = None
        self.discontinuity_pending = False

    def _new_segment(self, start_time):
        # Retain only already-seen measurements explicitly belonging to the
        # new segment. No old velocity/covariance or prediction survives.
        keep = [(t, p, s) for t, p, s in self.history if t >= start_time-1e-6]
        self.history.clear()
        self.history.extend(keep)
        self.state = self.covariance = None
        self.state_time = self.last_range_time = None
        self.velocity_valid = False
        self.fit_rmse = None
        self.fast_segment = True
        self.fit_history_samples = 0
        self.fit_span_s = 0.
        self.contact_velocity_prior = None
        self.last_radius_time = None
        self.last_radius_accepted_time = None
        self.pending_range = None
        self.segment += 1

    def _propagated(self, timestamp):
        if self.state is None:
            return None
        dt = float(timestamp-self.state_time)
        if dt < -1e-6:
            return None
        dt = max(0., dt)
        state = self.state.copy()
        acceleration = np.zeros(3) if self.supported else self.gravity
        state[:3] += dt*state[3:] + .5*dt*dt*acceleration
        state[3:] += dt*acceleration
        transform = np.eye(6)
        transform[:3, 3:] = dt*np.eye(3)
        covariance = transform @ self.covariance @ transform.T
        # Continuous acceleration uncertainty; small matrices, bounded work.
        noise = np.diag([.10**2, .10**2, .15**2]) if self.reacquisition_filter_enabled else .15**2*np.eye(3)
        if (self.reacquisition_filter_enabled and self.axis_refinement_enabled and self.noise_profile.filter_enabled
                and not (self.noise_profile.guard_contact and self.noise_contact_active)):
            noise = self.noise_profile.process_matrix
        covariance[:3, :3] += noise*dt**3/3
        covariance[:3, 3:] += noise*dt**2/2
        covariance[3:, :3] += noise*dt**2/2
        covariance[3:, 3:] += noise*dt
        return state, covariance

    def _fit(self, now):
        samples = [s for s in self.history if now-s[0] <= self.HISTORY_WINDOW_S+1e-6]
        self.fit_history_samples = len(samples)
        self.fit_span_s = samples[-1][0]-samples[0][0] if samples else 0.
        if not samples:
            return False
        if not self.reacquisition_filter_enabled:
            return self._fit_ballistic(now)
        contact = (self.contact_velocity_prior is not None
            and 0 <= now-self.contact_velocity_prior[0] <= .045)
        prior = self._propagated(now) if self.velocity_valid else None
        short = len(samples) < 4 or self.fit_span_s < .030
        # An established flight must not replace its velocity with a two-point
        # difference after a gap. A new contact still needs its fast initial fit.
        fitted = False if prior is not None and short and not contact else self._fit_ballistic(now)
        if prior is None or contact:
            self.filter_update = 'ballistic_fit' if fitted else 'initializing'
            return fitted

        vertical_state = self.state[[2, 5]].copy()
        vertical_covariance = self.covariance[np.ix_([2, 5], [2, 5])].copy()
        state, covariance = prior
        _, position, sigma = samples[-1]
        sd = max(.0015, min(.006, float(sigma)))
        factor = 1.
        depth = self.current_raw.get('depth_mm')
        # Approaching the near-depth boundary, small biased image centres can
        # produce a false lateral velocity immediately before a long dropout.
        # Keep this a continuous noise weight, not a hard detection rejection.
        if depth is not None and np.isfinite(depth) and state[5] > .2:
            factor += 2.*float(np.clip((420.-depth)/100., 0., 1.))
        self.horizontal_noise_scale = factor
        noise = np.diag([max(.001, sd*.5)*factor]*2+[sd])**2
        self.robust_noise_inflation = 1.
        if (self.axis_refinement_enabled and self.noise_profile.filter_enabled
                and not (self.noise_profile.guard_contact and self.noise_contact_active)):
            noise, self.robust_noise_inflation = self.noise_profile.measurement_noise(
                sd, factor, covariance, position-state[:3],
                robust_allowed=not contact and self.fit_history_samples >= 4 and self.fit_span_s >= .030)
        gain = np.linalg.solve(covariance[:3, :3]+noise, covariance[:3]).T
        residual_transform = np.eye(6)
        residual_transform[:, :3] -= gain
        self.state = state+gain@(position-state[:3])
        self.covariance = (residual_transform@covariance@residual_transform.T
                           + gain@noise@gain.T)
        self.state_time = now
        if fitted:
            # Retain the validated vertical flight/contact fit. The lateral
            # Kalman block uses only this new real range; it never re-inserts
            # the rolling fit as a statistically independent measurement.
            self.state[[2, 5]] = vertical_state
            self.covariance[[2, 5], :] = 0.
            self.covariance[:, [2, 5]] = 0.
            self.covariance[np.ix_([2, 5], [2, 5])] = vertical_covariance
        self.filter_update = 'lateral_kalman' if fitted else 'reacquisition_kalman'
        return True

    def _fit_ballistic(self, now):
        samples = [s for s in self.history if now-s[0] <= self.HISTORY_WINDOW_S+1e-6]
        minimum = 2 if self.fast_segment else 4
        if len(samples) < minimum:
            return False
        times = np.array([s[0]-now for s in samples])
        if np.ptp(times) < (.009 if self.fast_segment else .030):
            return False
        points = np.array([s[1] for s in samples])
        sigmas = np.array([s[2] for s in samples])
        target = points-.5*times[:, None]**2*self.gravity
        design = np.column_stack((np.ones(len(times)), times))
        weights = (np.min(sigmas)/sigmas)**2
        for _ in range(3):
            root = np.sqrt(weights)
            coefficients = np.linalg.lstsq(design*root[:, None], target*root[:, None], rcond=None)[0]
            residual = np.linalg.norm(target-design@coefficients, axis=1)
            weights = (np.min(sigmas)/sigmas)**2*np.minimum(1., .004/np.maximum(residual, 1e-12))
        rmse = float(np.sqrt(np.mean(residual**2)))
        if rmse > .004 or residual[-1] > .012 or np.linalg.norm(coefficients[1]) > 5.:
            self.reason = 'trajectory_fit_rejected'
            return False
        normal = design.T @ (weights[:, None]*design)
        temporal_covariance = np.linalg.inv(normal)*max(.0015, rmse)**2
        prior=self.contact_velocity_prior
        if prior is not None and 0<=now-prior[0]<=.045:
            expected=prior[1]+(now-prior[0])*self.gravity
            # The first two real post-contact samples give a noisy velocity.
            # Bound the prior correction continuously: a hard innovation gate
            # would turn all three constraints off at once and create spikes.
            # Validate the real fit above first; keep its covariance unchanged.
            # Two points fit exactly, so residual alone cannot measure their
            # velocity noise. Retain a 3 mm observation floor for the first
            # three samples when balancing the outgoing-velocity prior.
            velocity_variance=temporal_covariance[1,1]*(
                (.003/max(.0015,rmse))**2 if len(samples)<=3 else 1.)
            weight=velocity_variance/(velocity_variance+.25**2)
            coefficients[1]+=weight*np.clip(expected-coefficients[1],-.8,.8)
        self.state = np.concatenate(coefficients)
        self.state_time = now
        self.covariance = np.kron(temporal_covariance, np.eye(3))
        if prior is not None and 0<=now-prior[0]<=.045 and len(samples)<=3:
            covariance=np.kron(np.linalg.inv(normal)*max(.003,rmse)**2,np.eye(3))
            gain=np.linalg.solve(covariance[3:,3:]+.25**2*np.eye(3),covariance[3:]).T
            self.covariance=covariance-gain@covariance[3:]
        self.velocity_valid = True
        self.fit_rmse = rmse
        return True

    def _bearing_update(self, now, origin, direction, sigma_rad):
        if self.axis_refinement_enabled and origin is None:
            from pingpong_controller.ball_axis_observation import update_horizontal_edge
            return update_horizontal_edge(self, now)
        predicted = self._propagated(now)
        if predicted is None or origin is None or direction is None:
            return False
        origin, direction = np.asarray(origin, float), np.asarray(direction, float)
        if origin.shape != (3,) or direction.shape != (3,) or not np.isfinite([*origin, *direction, sigma_rad]).all():
            return False
        norm = np.linalg.norm(direction)
        if norm < .5:
            return False
        direction = direction/norm
        state, covariance = predicted
        distance = float((state[:3]-origin)@direction)
        if not .1 <= distance <= 2.:
            return False
        # Two orthogonal image-plane constraints H p = H camera_origin.
        axis = np.eye(3)[int(np.argmin(np.abs(direction)))]
        first = np.cross(direction, axis)
        first /= np.linalg.norm(first)
        basis = np.stack((first, np.cross(direction, first)))
        measurement = np.zeros((2, 6))
        measurement[:, :3] = basis
        innovation = basis@(origin-state[:3])
        bearing_scale = .8 if self.axis_refinement_enabled else 1.
        noise = max(.001, distance*max(.0005, float(sigma_rad))*bearing_scale)**2*np.eye(2)
        innovation_covariance = measurement@covariance@measurement.T+noise
        if np.linalg.norm(innovation) > .020 or innovation@np.linalg.solve(innovation_covariance, innovation) > 16.:
            return False
        gain = np.linalg.solve(innovation_covariance, measurement@covariance).T
        self.state = state+gain@innovation
        residual_transform = np.eye(6)-gain@measurement
        self.covariance = residual_transform@covariance@residual_transform.T+gain@noise@gain.T
        self.state_time = now
        # Use radius only after the independent bearing innovation was accepted.
        update_radius_observation(self,self.current_raw,now,origin,direction)
        return True

    def step(self, position, velocity, measured_valid, measurement_time, query_time, *,
             velocity_valid=False, sigma_m=.003, bearing_origin=None,
             bearing_direction=None, bearing_sigma_rad=.003,
             segment_reset=False, segment_age_s=0., contact_pending=False,
             supported=False, contact_model_active=False, allow_discontinuous_reacquisition=True):
        now, query = float(measurement_time), float(query_time)
        if not np.isfinite([now, query]).all():
            raise ValueError('finite observation times required')
        position, velocity = np.asarray(position, float), np.asarray(velocity, float)
        valid = bool(measured_valid and position.shape == (3,) and np.isfinite(position).all())
        if self.last_frame_time is not None and now < self.last_frame_time-1e-6:
            self.reset()
        fresh = self.last_frame_time is None or now-self.last_frame_time > 1e-6
        if fresh:
            self.last_frame_time = now
            self.position_valid = False
            self.bearing_accepted = self.range_rejected = False
            self.discontinuity_pending = False
            self.filter_update = 'prediction'
            self.horizontal_noise_scale = 1.
            self.robust_noise_inflation = 1.
            if segment_reset or contact_pending:
                start = now-max(0., min(.050, float(segment_age_s))) if segment_reset else now
                contact_prior=self.contact_velocity_prior
                self._new_segment(start)
                # A second detector reset for the same contact must not erase
                # its bounded velocity prior. Real sample segmentation remains.
                if contact_prior is not None and -.010<=start-contact_prior[0]<=.050:
                    self.contact_velocity_prior=contact_prior
            if self.supported and valid and not supported and not (segment_reset or contact_pending):
                release_prior = self._propagated(now) if self.velocity_valid else None
                release_age = float('inf') if self.last_range_time is None else now-self.last_range_time
                last_range_time = self.last_range_time
                self._new_segment(now-.025)
                # Release has continuous velocity; a collision does not. If a
                # short dropout leaves fewer than two real release samples,
                # retain this bounded prior until fresh ranges fit the flight.
                # Predicted positions never enter the new segment's history.
                if (release_prior is not None and release_age <= self.MAX_SUPPORTED_PREDICTION_S
                        and (len(self.history) < 2
                             or self.history[-1][0]-self.history[0][0] < .009)):
                    self.state, self.covariance = release_prior
                    self.covariance[3:, 3:] += .30**2*np.eye(3)
                    self.state_time = now
                    self.last_range_time = last_range_time
                    self.velocity_valid = True
            # Absent support metadata on a missing frame is not evidence of
            # release. Explicit contact/reset evidence still owns segmentation.
            if valid or segment_reset or contact_pending:
                self.supported = bool(supported and valid)
            predicted = self._propagated(now)
            if valid and predicted is not None and not self.supported:
                state, covariance = predicted
                error = np.linalg.norm(position-state[:3])
                uncertainty = 2*np.sqrt(max(0., float(np.linalg.eigvalsh(covariance[:3, :3])[-1])))
                # New segment evidence has already invalidated the old fit.
                # Otherwise a doubtful range must not contaminate its history.
                gate=max(.020, uncertainty+.010)
                if not allow_discontinuous_reacquisition:
                    gate=min(.040,gate)
                if error > gate:
                    previous = self.pending_range
                    self.pending_range = (now, position.copy(), max(.0015, float(sigma_m)))
                    dt = now-previous[0] if previous is not None else 0.
                    if (allow_discontinuous_reacquisition and previous is not None
                            and .005 <= dt <= .025 and np.linalg.norm(position-previous[1])/dt <= 5.):
                        # Independent fresh 3D evidence establishes a new
                        # segment. Never keep rejecting it against an old arc.
                        self._new_segment(now)
                        self.history.append(previous)
                        predicted = None
                        self.pending_range = None
                    else:
                        valid = False
                        self.range_rejected = True
                        self.discontinuity_pending = error > .040
                else:
                    self.pending_range = None
            if valid:
                self.position_valid = True
                self.last_position = position.copy()
                self.last_range_time = now
                self.history.append((now, position.copy(), max(.0015, float(sigma_m))))
                self.source, self.reason = 'measured', 'measured'
                if self.supported and velocity_valid and np.isfinite(velocity).all():
                    self.state = np.r_[position, velocity]
                    self.covariance = np.diag([sigma_m**2]*3+[.02**2]*3)
                    self.state_time = now
                    self.velocity_valid = True
                    self.filter_update = 'supported_measurement'
                elif not self._fit(now):
                    if predicted is not None and not contact_pending:
                        # Fresh range can update a still-valid same-segment
                        # prior without putting any predicted point in the fit.
                        state, covariance = predicted
                        noise = max(.0015, float(sigma_m))**2*np.eye(3)
                        gain = np.linalg.solve(covariance[:3, :3]+noise, covariance[:3]).T
                        self.state = state+gain@(position-state[:3])
                        h = np.zeros((3, 6));h[:, :3] = np.eye(3)
                        j = np.eye(6)-gain@h
                        self.covariance = j@covariance@j.T+gain@noise@gain.T
                        self.state_time = now
                    else:
                        self.velocity_valid = False
                        self.reason = 'measured_position_only'
            else:
                self.source = 'prediction'
                if not contact_pending and self.last_range_time is not None and now-self.last_range_time <= self.MAX_RANGE_AGE_S:
                    self.bearing_accepted = self._bearing_update(now, bearing_origin, bearing_direction, bearing_sigma_rad)
                self.source = 'bearing' if self.bearing_accepted else 'prediction'
                if self.bearing_accepted:
                    self.filter_update = 'bearing_kalman'
                self.reason = ('range_innovation_rejected' if self.range_rejected else
                               'bearing_update' if self.bearing_accepted else 'bounded_prediction')
            if contact_pending:
                self.velocity_valid = False
                self.reason = 'contact_pending'
        age = float('inf') if self.last_range_time is None else max(0., query-self.last_range_time)
        predicted = self._propagated(query)
        uncertainty = None
        if predicted is not None:
            state, covariance = predicted
            uncertainty = 2*np.sqrt(max(0., float(np.linalg.eigvalsh(covariance[:3, :3])[-1])))
        usable = bool(predicted is not None and self.velocity_valid and age <= self.MAX_RANGE_AGE_S+1e-9
                      and uncertainty <= self.MAX_UNCERTAINTY_M and not self.discontinuity_pending)
        supported_expired = bool(self.supported and age > self.MAX_SUPPORTED_PREDICTION_S+1e-9)
        usable = usable and not supported_expired
        below_contact_band = bool(predicted is not None and not self.position_valid
            and not contact_model_active and self.prediction_floor_m is not None
            and state[2] < self.prediction_floor_m and age > .060)
        usable = usable and not below_contact_band
        position_valid = bool(usable or (self.position_valid and query-now <= .025+1e-9))
        output_position = state[:3].copy() if usable else (self.last_position.copy() if position_valid else np.zeros(3))
        # Blend only the current causal same-segment fit with the true range.
        # Neither the blended point nor a predicted point enters fit history.
        if self.position_valid and usable:
            dt = max(0., query-now)
            frame_state = self._propagated(now)[0]
            acceleration = np.zeros(3) if self.supported else self.gravity
            weight=self.output_fit_weight
            recent_contact = (self.contact_velocity_prior is not None
                and 0 <= now-self.contact_velocity_prior[0] <= .045)
            if self.reacquisition_filter_enabled and not recent_contact:
                weight = np.array([.85, .85, self.output_fit_weight])
            output_position = (1-weight)*self.last_position+weight*frame_state[:3]+dt*frame_state[3:]+.5*dt*dt*acceleration
        reason = self.reason
        if not usable and not self.position_valid:
            reason = 'prediction_timeout' if age > self.MAX_RANGE_AGE_S else 'prediction_uncertainty_limit' if predicted is not None else 'insufficient_history'
            if self.discontinuity_pending:reason = 'new_segment_confirming'
            elif below_contact_band:reason = 'prediction_below_contact_band'
            elif supported_expired:reason = 'supported_prediction_timeout'
        return dict(position=output_position, velocity=state[3:].copy() if usable else np.zeros(3),
                    valid=usable, position_valid=position_valid, reason=reason,
                    source=self.source, position_predicted=bool(usable and (not self.position_valid or query-now > 1e-6)),
                    velocity_predicted=usable, timestamp=query if usable else now,
                    range_age_s=age if np.isfinite(age) else None, uncertainty_m=uncertainty,
                    fit_rmse_m=self.fit_rmse, history_samples=len(self.history), segment=self.segment,
                    bearing_accepted=self.bearing_accepted, range_rejected=self.range_rejected)


class BallObservationProcessor:
    """Producer-only fitting; emit a small immutable scalar observation snapshot.

    Online camera detection and offline ReplaySession both call this class.
    The receiver and control consumer never fit a model or wait for it.
    """
    SUPPORTS_RACKET_CONTEXT = True
    def __init__(self, sim_offset_m=(0., 0., 0.), prediction_floor_m=1.24, noise_profile=None):
        self.noise_profile = default_profile() if noise_profile is None else noise_profile
        self.model = BallObservationModel(prediction_floor_m=prediction_floor_m-float(sim_offset_m[2]),
                                          noise_profile=self.noise_profile)
        self.contact = BallContactObservation()
        self.acquisition=deque(maxlen=8)
        self.default_floor=self.model.prediction_floor_m

    def _supported_velocity(self, position, timestamp, raw):
        """Detect slow rolling/support from real ball samples near the racket.

        A gravity-only model is invalid while the ball rests or rolls on the
        racket. Do not repeatedly reject those good positions against free fall.
        """
        if self.contact.motion is None:
            return None
        racket_position,racket_velocity,_=self.contact.motion
        separation=position-racket_position
        height=float(raw.get('ball_contact_height_m',.0082))
        if not height-.020<=separation[2]<=height+.012 or np.linalg.norm(separation[:2])>.110:
            return None
        samples=[(t,p) for t,p,_ in self.model.history if timestamp-t<=.055]
        samples.append((timestamp,position))
        if len(samples)<4 or timestamp-samples[0][0]<.025:
            return None
        times=np.array([t-timestamp for t,_ in samples]);points=np.array([p for _,p in samples])
        design=np.column_stack((np.ones(len(times)),times))
        coefficients=np.linalg.lstsq(design,points,rcond=None)[0]
        if (np.max(np.linalg.norm(points-design@coefficients,axis=1))>.002
                or np.linalg.norm(coefficients[1])>1.
                or np.linalg.norm(coefficients[1]-racket_velocity)>.35):
            return None
        return coefficients[1]

    def process(self, observation, timestamp, racket_state=None, racket_context=None):
        raw = observation.get('raw_measurement')
        if not isinstance(raw, dict) or raw.get('ball_observation_version') != 1:
            return
        if raw.get('observation_processed'):
            return
        started = perf_counter_ns()
        shift=float(raw.get('ball_calibration_time_shift_s') or 0.)
        timestamp=float(timestamp)+shift
        calibrated=bool(raw.get('ball_calibration_applied'))
        if self.model.last_frame_time is not None and timestamp<self.model.last_frame_time-1e-6:
            self.contact=BallContactObservation()
            self.acquisition.clear()
            self.model.reset()
        self.model.MAX_RANGE_AGE_S=.320 if calibrated else .200
        self.model.output_fit_weight=.35 if calibrated else 0.
        self.model.reacquisition_filter_enabled=calibrated
        self.model.axis_refinement_enabled = bool(calibrated and raw.get('ball_axis_refinement'))
        raw['observation_radius_range_accepted']=False
        valid = bool(observation.get('detected'))
        position = np.asarray(observation['position'], float)/1000. if valid else np.zeros(3)
        velocity = np.asarray(observation['velocity'], float)/1000. if valid else np.zeros(3)
        origin = direction = None
        if raw.get('bearing_valid'):
            origin = np.array([raw.get('bearing_origin_'+k+'_m', np.nan) for k in 'xyz'])
            direction = np.array([raw.get('bearing_direction_'+k, np.nan) for k in 'xyz'])
        model_flags=raw
        if calibrated:
            # Preserve detector evidence in logs; contact ownership changes
            # only the flags passed into the unified state estimator.
            model_flags=raw.copy()
            self.contact.context=racket_context
            self.contact.prepare(self.model,model_flags,position,valid,timestamp,racket_state)
            for key in ('observation_contact_refined','observation_contact_time_correction_s',
                        'observation_contact_real_samples'):
                raw[key]=model_flags.get(key)
        raw['observation_algorithm_version']='camera-smooth-v1.3.0-rc1-20261019'
        if self.model.axis_refinement_enabled:
            raw['observation_algorithm_version']='camera-axis-v1.4.0-rc1-20261019'
        raw['observation_racket_context_clock']=(racket_context or {}).get('clock')
        raw['observation_racket_context_age_s']=(
            None if racket_state is None else timestamp-float(racket_state[0]))
        supported=raw.get('velocity_motion_state') == 'supported'
        velocity_ready=bool(raw.get('velocity_ready'))
        if calibrated and valid:
            support_velocity=self._supported_velocity(position,timestamp,raw)
            if support_velocity is not None:
                supported=True;velocity_ready=True;velocity=support_velocity
        if valid and raw.get('detection_method')=='ir_sphere_surface':
            predicted=self.model._propagated(timestamp)
            if predicted is not None and np.linalg.norm(position-predicted[0][:3])>.015:
                valid=False
        allow_reacquisition=True
        if calibrated:
            near_racket=False
            if valid and self.contact.motion is not None:
                separation=position-self.contact.motion[0]
                near_racket=bool(np.linalg.norm(separation[:2])<=.130 and -.030<=separation[2]<=.060)
            allow_reacquisition=near_racket
            # The legacy detector can call a stationary bright object
            # "supported" without fresh racket feedback. That flag must not
            # bypass the range innovation gate or establish ball identity.
            if not near_racket:
                supported=False
            if model_flags.get('flight_segment_reset') and not near_racket:
                model_flags['flight_segment_reset']=False
            # Initial acquisition needs physical context or a moving ballistic
            # sequence. A stationary bright surface cannot acquire ball identity.
            if valid and self.model.state is None and not near_racket:
                self.acquisition.append((timestamp,position.copy()))
                samples=[(t,p) for t,p in self.acquisition if timestamp-t<=.100]
                acquired=False
                if len(samples)>=7 and timestamp-samples[0][0]>=.060:
                    times=np.array([t-timestamp for t,_ in samples]);points=np.array([p for _,p in samples])
                    design=np.column_stack((np.ones(len(times)),times))
                    target=points-.5*times[:,None]**2*self.model.gravity
                    fit=np.linalg.lstsq(design,target,rcond=None)[0]
                    acquired=bool(np.max(np.linalg.norm(target-design@fit,axis=1))<.004
                        and np.linalg.norm(points[-1]-points[0])>.015 and np.linalg.norm(fit[1])<5.)
                    if acquired:
                        self.model._new_segment(samples[0][0])
                        self.model.history.extend((t,p,.003) for t,p in samples[:-1])
                if not acquired:valid=False
        recent_radius=(self.model.last_radius_accepted_time is not None
                       and timestamp-self.model.last_radius_accepted_time<.033)
        if calibrated and recent_radius:self.model.MAX_RANGE_AGE_S=.360
        self.model.prediction_floor_m=self.default_floor
        if calibrated and recent_radius and self.contact.motion is not None and raw.get('bearing_valid'):
            self.model.prediction_floor_m=min(self.default_floor,self.contact.motion[0][2]+.0382)
        self.model.current_raw=raw
        plan=self.contact.plan if calibrated else None
        contact_active=bool(plan is not None and -.080<=timestamp-plan['time']<=.040)
        self.model.noise_contact_active = contact_active
        result = self.model.step(position, velocity, valid, timestamp, timestamp,
            velocity_valid=velocity_ready,
            sigma_m=float(raw.get('range_sigma_mm') or 6.)/1000.,
            bearing_origin=origin, bearing_direction=direction,
            bearing_sigma_rad=float(raw.get('bearing_sigma_rad') or .003),
            segment_reset=bool(model_flags.get('flight_segment_reset')),
            segment_age_s=float(raw.get('flight_segment_age_s') or 0.),
            contact_pending=bool(model_flags.get('contact_pending') or model_flags.get('observation_state') == 'impact_confirming'),
            supported=supported,
            contact_model_active=contact_active,
            allow_discontinuous_reacquisition=allow_reacquisition)
        raw.pop('_horizontal_edge_observation', None)
        raw.update(observation_processed=True, observation_position_valid=result['position_valid'],
            observation_velocity_valid=result['valid'], observation_source=result['source'],
            observation_model_reason=result['reason'], observation_range_age_s=result['range_age_s'],
            observation_uncertainty_m=result['uncertainty_m'], observation_fit_rmse_m=result['fit_rmse_m'],
            observation_history_samples=result['history_samples'], observation_segment=result['segment'],
            observation_bearing_accepted=result['bearing_accepted'], observation_range_rejected=result['range_rejected'],
            observation_supported=self.model.supported)
        raw.update(observation_fit_history_samples=self.model.fit_history_samples,
            observation_fit_span_s=self.model.fit_span_s,
            observation_filter_update=self.model.filter_update,
            observation_horizontal_noise_scale=self.model.horizontal_noise_scale)
        raw['observation_time_shift_s']=shift
        raw['observation_prediction_floor_m']=self.model.prediction_floor_m
        raw['observation_radius_age_s']=(None if self.model.last_radius_accepted_time is None
                                       else timestamp-self.model.last_radius_accepted_time)
        raw['observation_prediction_budget_s']=(min(self.model.MAX_RANGE_AGE_S,
            self.model.MAX_SUPPORTED_PREDICTION_S) if self.model.supported else self.model.MAX_RANGE_AGE_S)
        raw['observation_contact_model_active']=contact_active
        raw['observation_contact_model_until_s']=.040-(timestamp-plan['time']) if contact_active else 0.
        for i, axis in enumerate('xyz'):
            raw['observation_position_'+axis+'_m'] = float(result['position'][i])
            raw['observation_velocity_'+axis+'_m_s'] = float(result['velocity'][i])
        predicted = self.model._propagated(timestamp)
        if predicted is not None:
            covariance = predicted[1]
            raw['observation_position_variance'] = max(0., float(np.linalg.eigvalsh(covariance[:3, :3])[-1]))
            raw['observation_velocity_variance'] = max(0., float(np.linalg.eigvalsh(covariance[3:, 3:])[-1]))
            raw['observation_cross_covariance'] = float(np.linalg.norm(covariance[:3, 3:], 2))
        raw['observation_contact_horizon_s'] = None
        if result['valid'] and racket_state is not None and not calibrated:
            racket_time, racket_position_mm, racket_velocity_mm_s = racket_state
            age = float(timestamp-racket_time)
            racket_velocity = np.asarray(racket_velocity_mm_s, float)/1000.
            racket_position = np.asarray(racket_position_mm, float)/1000.+age*racket_velocity
            separation = result['position']-racket_position
            relative_velocity = result['velocity']-racket_velocity
            # A flight extrapolation must end before plausible contact. The
            # 40 mm plane includes ball radius and 20 mm pose/range margin;
            # it does not invent a post-impact velocity or a bounce.
            if (abs(age) <= .060 and np.isfinite([*separation, *relative_velocity]).all()
                    and relative_velocity[2] < -.20 and -.04 <= separation[2] <= .35):
                height = max(0., separation[2]-.040)
                horizon = (relative_velocity[2]+np.sqrt(relative_velocity[2]**2+19.6*height))/9.8
                lateral = separation[:2]+horizon*relative_velocity[:2]
                if horizon <= .080 and np.linalg.norm(lateral) <= .130:
                    raw['observation_contact_horizon_s'] = float(horizon)
        if calibrated:
            self.contact.publish(self.model,raw,timestamp)
        if calibrated and self.model.axis_refinement_enabled and self.noise_profile.enabled:
            self.noise_profile.correct_snapshot(raw)
            raw['observation_algorithm_version'] = self.noise_profile.VERSION
            raw['observation_noise_profile'] = self.noise_profile.name
            raw['observation_noise_robust_inflation'] = self.model.robust_noise_inflation
            for i, axis in enumerate('xyz'):
                raw['observation_bias_correction_'+axis+'_m'] = -float(self.noise_profile.bias[i])
        raw['observation_compute_ms'] = (perf_counter_ns()-started)/1e6


def query_observation_snapshot(raw, measurement_time, query_time, offset_m, *, max_prediction_s=.320, prediction_floor_m=1.24):
    """Constant-size arithmetic only: no histories, fitting, covariance updates or I/O."""
    measurement_time=float(measurement_time)+float(raw.get('observation_time_shift_s') or 0.)
    signed_elapsed=float(query_time-measurement_time)
    elapsed = max(0., signed_elapsed)
    position = np.array([raw['observation_position_'+k+'_m'] for k in 'xyz'])+np.asarray(offset_m, float)
    velocity = np.array([raw['observation_velocity_'+k+'_m_s'] for k in 'xyz'])
    range_age = raw.get('observation_range_age_s')
    age = float('inf') if range_age is None else float(range_age)+elapsed
    variance = (float(raw.get('observation_position_variance') or 0.)
                + 2*elapsed*float(raw.get('observation_cross_covariance') or 0.)
                + elapsed**2*float(raw.get('observation_velocity_variance') or 0.)
                + .15**2*elapsed**3/3)
    uncertainty = 2*np.sqrt(max(0., variance))
    radius_age=raw.get('observation_radius_age_s')
    fresh_radius=radius_age is not None and float(radius_age)+elapsed<=.033
    budget=min(.360 if fresh_radius else .320,float(raw.get('observation_prediction_budget_s') or .200),max_prediction_s)
    if raw.get('observation_supported'):
        budget=min(budget,BallObservationModel.MAX_SUPPORTED_PREDICTION_S)
    valid = bool(raw.get('observation_velocity_valid') and age <= budget+1e-9
                 and uncertainty <= .025 and np.isfinite([*position, *velocity]).all())
    if valid:
        acceleration = np.zeros(3) if raw.get('observation_supported') else BallObservationModel.GRAVITY
        position += elapsed*velocity+.5*elapsed**2*acceleration
        velocity += elapsed*acceleration
    observed_floor=raw.get('observation_prediction_floor_m')
    if observed_floor is not None and fresh_radius:
        prediction_floor_m=min(prediction_floor_m,float(observed_floor)+float(offset_m[2]))
    below_contact_band = bool(age > .060 and position[2] < prediction_floor_m
        and not (raw.get('observation_contact_model_active')
                 and elapsed<=float(raw.get('observation_contact_model_until_s') or 0.)))
    if below_contact_band:valid = False
    contact_horizon = raw.get('observation_contact_horizon_s')
    contact_limit = bool(contact_horizon is not None and elapsed >= float(contact_horizon))
    if contact_limit:valid = False
    # Position-only samples remain useful at their exposure time, but cannot
    # masquerade as a current-time position while endpoint velocity is unknown.
    position_valid = bool(valid or (raw.get('observation_position_valid') and not raw.get('observation_velocity_valid') and elapsed <= 1e-6))
    if contact_limit:
        position_valid = bool(raw.get('observation_position_valid') and elapsed <= 1e-6)
    reason = raw.get('observation_model_reason', 'missing')
    if not position_valid:
        if age > budget:reason = 'prediction_timeout'
        elif below_contact_band:reason = 'prediction_below_contact_band'
        elif raw.get('observation_velocity_valid'):reason = 'prediction_uncertainty_limit'
    if contact_limit:reason = 'contact_prediction_limit'
    confirmation_deadline=raw.get('observation_contact_confirm_deadline_s')
    if confirmation_deadline is not None and elapsed>float(confirmation_deadline):
        valid=position_valid=False;reason='contact_confirmation_timeout'
    bounce_horizon=raw.get('observation_bounce_horizon_s')
    if bounce_horizon is not None and elapsed>=float(bounce_horizon):
        dt=elapsed-float(bounce_horizon)
        if dt<=.040 and age<=budget:
            position=np.array([raw['observation_bounce_position_'+k+'_m'] for k in 'xyz'])+np.asarray(offset_m,float)
            velocity=np.array([raw['observation_bounce_velocity_'+k+'_m_s'] for k in 'xyz'])
            position+=dt*velocity+.5*dt**2*BallObservationModel.GRAVITY
            velocity+=dt*BallObservationModel.GRAVITY
            uncertainty=2*np.sqrt(.004**2+(.20*dt)**2)
            valid=position_valid=bool(uncertainty<=.025 and np.isfinite([*position,*velocity]).all())
            reason='contact_prediction'
        else:
            valid=position_valid=False;reason='contact_prediction_timeout'
    if signed_elapsed < -1e-6:
        valid=position_valid=False;reason='future_observation'
    if position_valid and raw.get('ball_axis_refinement'):
        from pingpong_controller.ball_axis_observation import apply_contact_timing_mean
        apply_contact_timing_mean(raw, elapsed, position, velocity)
    return dict(position=position if position_valid else np.zeros(3),
        velocity=velocity if valid else np.zeros(3), valid=valid, position_valid=position_valid,
        timestamp=query_time if valid else measurement_time, reason=reason,
        source='contact_prediction' if reason=='contact_prediction' else raw.get('observation_source', 'missing'), range_age_s=age if np.isfinite(age) else None,
        uncertainty_m=uncertainty, position_predicted=bool(valid and (elapsed > 1e-6 or raw.get('observation_source') != 'measured')))
