"""pingpong_controller package init.

Pin BLAS / OpenMP thread pools to a single thread *before* NumPy (and the
OpenBLAS / MKL it links) get imported anywhere in this process.

Why this lives here and runs this early:
  The node runs a 200 Hz RL control loop whose per-tick work is tiny
  matrix math — the MJX policy MLP (NumPy matmuls in mjx_rl_policy.py) and
  one MuJoCo ``mj_forward`` for racket FK (rl_policy.py). On a 32-core box
  OpenBLAS defaults to 32 threads, so each of those calls forks/syncs a
  large thread pool ~1000x/second. For matrices this small that buys
  nothing and actively hurts: the pool churn periodically preempts the
  YOLO detection callback (same process, GIL released only in native
  sections), which showed up as the detection rate sagging from 60 Hz to
  ~30 Hz for a few hundred ms and recovering, on a loop. Forcing single
  threaded BLAS makes the small ops fastest anyway and removes the
  contention — verified to hold detection steady at 60 Hz.

The setuptools console-script entry point imports this package (running
this file) before importing ``pingpong_node`` and thus before its
``import numpy``, so setting the env vars here takes effect when OpenBLAS
loads. ``setdefault`` is used so an explicit value passed on the command
line (e.g. ``OPENBLAS_NUM_THREADS=4 ros2 run ...``) still wins.
"""

import os

# One thread each for the common BLAS / OpenMP backends NumPy may link
# against. setdefault: do not override anything the operator set explicitly.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
