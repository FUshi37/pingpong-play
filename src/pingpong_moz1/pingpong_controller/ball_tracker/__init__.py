"""Hybrid ball state estimation and racket-impact prediction."""

from .ball_model import ball_dynamics, jacobian
from .config import BallTrackerConfig
from .ekf_tracker import (
    MODE_CONTACT_CANDIDATE,
    MODE_FLIGHT,
    MODE_POST_IMPACT,
    MODE_UNINITIALIZED,
    BallEKFTracker,
)
from .estimator import (
    BallEstimate,
    BallEstimatorBackend,
    BallStateEstimator,
    EKFEstimatorBackend,
)
from .experiment_logger import BallExperimentLogger, RLPerformanceLogger
from .impact_predictor import (
    RacketPlane,
    predict_contact,
    predict_impact,
    racket_plane_from_pose,
)
from .rl_observation import (
    BallObservationConfig,
    BallPolicyObservationAdapter,
    ball_observation_field_names,
    build_rl_observation,
)

__all__ = [
    'BallEKFTracker',
    'BallEstimate',
    'BallExperimentLogger',
    'BallEstimatorBackend',
    'BallObservationConfig',
    'BallPolicyObservationAdapter',
    'BallStateEstimator',
    'BallTrackerConfig',
    'EKFEstimatorBackend',
    'MODE_CONTACT_CANDIDATE',
    'MODE_FLIGHT',
    'MODE_POST_IMPACT',
    'MODE_UNINITIALIZED',
    'RacketPlane',
    'RLPerformanceLogger',
    'ball_dynamics',
    'ball_observation_field_names',
    'build_rl_observation',
    'jacobian',
    'predict_contact',
    'predict_impact',
    'racket_plane_from_pose',
]
