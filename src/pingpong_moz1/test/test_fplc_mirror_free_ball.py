"""Physical free-ball regression for checkpoint mass edits in the CPU mirror."""
import numpy as np
import mujoco as mj
import pytest

from pingpong_controller.fplc_deployment import fplc_assets
from pingpong_controller.sim_mirror import RealBallSimMirror,SimMirrorConfig


@pytest.mark.parametrize('profile',['fplc_gpu0_v325','fplc_v321_s4_u2097_20261029_r1'])
def test_free_ball_uses_checkpoint_mass_and_nominal_gravity(profile):
    checkpoint,xml=fplc_assets(profile)
    mirror=RealBallSimMirror(SimMirrorConfig(model_path=checkpoint,
        policy_xml_path=xml,sim_xml_path=xml,policy_obs_dim=114 if '325' in profile else 72,
        control_dt=.005,policy_profile=profile,enable_video_frames=False,use_mujoco_renderer=False))
    try:
        mirror.reset(ball_pos=np.array([3.,3.,3.]),ball_vel=np.zeros(3),ball_valid=False)
        mj.mj_forward(mirror.model,mirror.data)
        ball_mass=mirror.model.body_mass[mirror.ball_body_id]
        assert ball_mass==pytest.approx(.004)
        np.testing.assert_allclose(mirror.model.dof_M0[mirror.ball_vadr:mirror.ball_vadr+3],
                                   np.full(3,ball_mass),rtol=0,atol=1e-12)
        np.testing.assert_allclose(mirror.data.qacc[mirror.ball_vadr:mirror.ball_vadr+3],
                                   mirror.model.opt.gravity,rtol=0,atol=1e-9)
        for _ in range(10):mj.mj_step(mirror.model,mirror.data)
        np.testing.assert_allclose(mirror.data.qvel[mirror.ball_vadr:mirror.ball_vadr+3],
                                   mirror.model.opt.gravity*.01,rtol=0,atol=1e-7)
    finally:mirror.close()
