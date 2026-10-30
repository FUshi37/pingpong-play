"""Bounded horizontal evidence from a vertically truncated IR silhouette.

This is a rank-one image-plane constraint, not a range measurement. Only X
and its velocity may change; uncertainty in the other coordinates remains in
the innovation variance. No range age, contact, or acquisition gate is renewed.
"""
import numpy as np
import math


def retain_horizontal_edge(detector, cx, cy, radius, shape):
    if not getattr(detector, '_axis_refinement_enabled', False):
        return
    height, width = shape[:2]
    if not np.isfinite([cx, cy, radius]).all() or not radius + 4 < cx < width - 1 - radius - 4:
        return
    candidates = detector._horizontal_edges
    candidate = (float(cx), float(cy), float(radius))
    if any(np.hypot(cx - u, cy - v) < 3 for u, v, _ in candidates):
        return
    if len(candidates) >= 8:
        detector._horizontal_edges_ambiguous = True
        return
    candidates.append(candidate)


def attach_horizontal_edges(detector, raw):
    calibration = detector.observation_calibration
    if not detector._horizontal_edges or detector._horizontal_edges_ambiguous:
        return
    # Cached geometry is producer-private and is removed from the snapshot
    # after processing. Nothing is added to the IPC/control packet schema.
    if getattr(detector, '_horizontal_geometry_calibration', None) is not calibration:
        transform = detector.T_base_camera @ detector.T_ir_to_rgb
        rotation = np.asarray(calibration.get('rotation', np.eye(3)))
        detector._horizontal_geometry = dict(
            inverse=np.linalg.inv(rotation @ transform[:3, :3]),
            origin=rotation @ transform[:3, 3] + np.asarray(calibration['translation_m']),
            fx=detector.ir_fx, fy=detector.ir_fy, cx=detector.ir_cx, cy=detector.ir_cy)
        detector._horizontal_geometry_calibration = calibration
    raw['_horizontal_edge_observation'] = (detector._horizontal_geometry, tuple(detector._horizontal_edges))


def update_horizontal_edge(model, now):
    evidence = model.current_raw.get('_horizontal_edge_observation')
    prior = model._propagated(now)
    if evidence is None or prior is None:
        return False
    geometry, candidates = evidence
    state, covariance = prior
    inverse, origin = geometry['inverse'], geometry['origin']
    camera = inverse @ (state[:3] - origin)
    if not np.isfinite(camera).all() or not .1 < camera[2] < 2.:
        return False
    u = geometry['cx'] + geometry['fx'] * camera[0] / camera[2]
    v = geometry['cy'] + geometry['fy'] * camera[1] / camera[2]
    near = [c for c in candidates if abs(c[0] - u) < 20 and abs(c[1] - v) < max(30, 2 * c[2])]
    if len(near) != 1:
        return False
    normal = inverse.T @ np.array([1., 0., -(near[0][0] - geometry['cx']) / geometry['fx']])
    normal /= np.linalg.norm(normal)
    measurement = np.r_[normal, np.zeros(3)]
    innovation = float(normal @ (origin - state[:3]))
    noise = max(.001, camera[2] * 3. / geometry['fx']) ** 2
    variance = float(measurement @ covariance @ measurement + noise)
    if not np.isfinite([innovation, variance]).all() or variance <= 0:
        return False
    if abs(innovation) > .020 or innovation ** 2 / variance > 16.:
        return False
    gain = covariance @ measurement / variance
    gain[[1, 2, 4, 5]] = 0.
    residual = np.eye(6) - np.outer(gain, measurement)
    model.state = state + gain * innovation
    model.covariance = residual @ covariance @ residual.T + noise * np.outer(gain, gain)
    model.state_time = now
    model.current_raw['observation_horizontal_edge_accepted'] = True
    return True


def apply_contact_timing_mean(raw, elapsed, position, velocity):
    """Mean of the existing two flight branches over a small timing kernel.

    The 3.2 ms width is approximately one 90 Hz frame / sqrt(12). This is a
    bounded observation correction, not evidence of a detected collision and
    not an extension of any contact/range validity deadline. Real post-impact
    samples continue to determine the new flight. Only the vertical marginal
    is corrected; lateral state and contact decisions are left unchanged.
    """
    horizon = raw.get('observation_bounce_horizon_s')
    if horizon is None:
        return
    dt = elapsed - float(horizon)
    sigma = .0032
    if abs(dt) >= 4 * sigma:
        return
    u = abs(dt) / sigma
    tail = .5 * math.erfc(u / math.sqrt(2.))
    density = math.exp(-.5 * u * u) / math.sqrt(2. * math.pi)
    jump = float(raw['observation_bounce_velocity_z_m_s']) - (
        float(raw['observation_velocity_z_m_s']) - 9.8 * horizon)
    position[2] += jump * (sigma * density - abs(dt) * tail)
    velocity[2] += jump * tail * (1. if dt < 0 else -1.)
