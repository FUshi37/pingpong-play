"""ROS-package import shim for MJX/JAX PPO real-time policy inference.

The implementation lives in ``tools.rl_2real.mjx_policy_controller`` so it can
also be run directly as a standalone smoke-test script.  This module provides a
stable import path for ``pingpong_node.py`` after the package is installed with
colcon.
"""

from pingpong_controller.tools.rl_2real.mjx_policy_controller import (  # noqa: F401
    MJXPolicyController,
    NumpyMJXActor,
    load_mjx_checkpoint,
)

__all__ = [
    "MJXPolicyController",
    "NumpyMJXActor",
    "load_mjx_checkpoint",
]

